#!/usr/bin/env python3

import argparse
import json
import multiprocessing as mp
import os
import random
import subprocess
import sys

sys.setrecursionlimit(100000)
sys.path.append(os.path.normpath(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))
)

import numpy as np
import pandas as pd
import torch
from agent import Agent
from models.prover import Prover
from multiprocess_utils import MPSelections, mp_iter_libs
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("method", type=str)
    parser.add_argument("eval_id", type=str)
    parser.add_argument("--datapath", type=str, default="../data")
    parser.add_argument("--projs_split", type=str, default="../projs_split.json")
    parser.add_argument(
        "--split", choices=["train", "valid", "test"], type=str, default="test"
    )
    parser.add_argument("--file", type=str)
    parser.add_argument("--proof", type=str)
    parser.add_argument("--filter", type=str)
    parser.add_argument("--path", type=str)
    parser.add_argument("--output_dir", type=str, default="evaluation")
    parser.add_argument("--max_num_tactics", type=int, default=300)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--hammer_timeout", type=int, default=100)
    parser.add_argument("--depth_limit", type=int, default=50)
    parser.add_argument(
        "--beam_width", type=int, default=20
    )  # lots of timeout when >200
    parser.add_argument("--num_tactic_candidates", type=int, default=20)
    parser.add_argument(
        "--lens_norm", type=float, default=0.5, help="lengths normalization"
    )
    parser.add_argument("--tac_grammar", type=str, default="tactics.ebnf")
    parser.add_argument("--term_embedding_dim", type=int, default=256)
    parser.add_argument("--size_limit", type=int, default=50)
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=256,
        help="dimension of the grammar embeddings",
    )
    parser.add_argument(
        "--symbol_dim",
        type=int,
        default=256,
        help="dimension of the terminal/nonterminal symbol embeddings",
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=256, help="dimension of the LSTM controller"
    )

    # term encoder
    parser.add_argument("--model_type", type=str, default="GraphSage")
    parser.add_argument("--dropout", type=int, default=0.5)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--nonterminals_feature_dim", type=int, default=32)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="")

    # Skip proofs csv
    parser.add_argument(
        "--skip",
        type=str,
        default="",
        help="CSV file with columns [project, lib, proof] to skip",
    )

    parser.add_argument("-e", "--extra", help="extra args", default="--num_heads 2")
    parser.add_argument("--n_cpu", default=mp.cpu_count() - 1, type=int)

    args = parser.parse_args()
    if args.device not in ["cuda", "cpu"]:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = torch.device(args.device)
    if args.device.type == "cpu":
        print("using CPU", "WARNING")

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Skip option field validation
    skip_projects = []
    skip_libs = []
    args.skip_proofs = []
    if args.skip and os.path.exists(args.skip):
        skip_df = pd.read_csv(args.skip)
        if any(field not in skip_df for field in ["project", "lib", "proof"]):
            print(
                "Invalid skip csv, skipping nothing. Requires columns: project, lib, proof",
                "WARNING",
            )
        else:
            skip_projects = skip_df[skip_df["project"].notnull()]["project"].to_list()
            skip_libs = skip_df[skip_df["lib"].notnull()]["lib"].to_list()
            args.skip_proofs = skip_df[skip_df["proof"].notnull()]["proof"].to_list()

    args.skip_projects = skip_projects
    args.skip_libs = skip_libs
    print("skip projects:", args.skip_projects)
    print("skip libs:", args.skip_libs)
    print("skip proofs:", args.skip_proofs)
    return args


def evaluate_wrapper(project, filename, _, args, _process_list):
    # Get model and agent
    if "ours" in args.method:
        model = Prover(args)
        # print("loading model checkpoint from %s.." % args.path)
        if args.device.type == "cpu":
            checkpoint = torch.load(args.path, map_location="cpu")
        else:
            checkpoint = torch.load(args.path)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(args.device)
    else:
        model = None

    agent = Agent(model, None, None, args)
    try:
        return agent.evaluate(filename, args.proof, _process_list=_process_list)
    except Exception as e:
        tqdm.write(f"Error in {filename}", sys.stderr)
        tqdm.write(str(e), sys.stderr)
        raise


def project_level_aggregation_wrapper(file_results):
    results = []
    errors = []
    for r, e in file_results.values():
        results.extend(r)
        errors.extend(e)
    return results, errors


def main(args):
    filters = MPSelections(name="filters")
    skips = MPSelections(name="skips")
    if args.filter:
        filters.add_projects(args.filter)
    print(filters)
    print(skips)

    # Multiprocess over proofs
    _, proj_level_results = mp_iter_libs(
        evaluate_wrapper,
        [args],
        {},
        project_level_aggregation_wrapper,
        split=args.split,
        filters=filters,
        skips=skips,
        n_cpu=args.n_cpu,
        # log_file="mp_test.log",
        mute=True,
    )

    args = vars(args)
    del args["device"]
    oup_dir = os.path.join(args["output_dir"], args["eval_id"])
    err_dir = os.path.join(oup_dir, "errors")
    if not os.path.exists(oup_dir):
        os.makedirs(oup_dir)
    if not os.path.exists(err_dir):
        os.mkdir(err_dir)
    for proj, (results, errors) in proj_level_results.items():
        json.dump(
            {"options": args, "results": results},
            open(os.path.join(oup_dir, f"{proj}.json"), "w"),
        )
        if errors:
            json.dump(errors, open(os.path.join(err_dir, f"{proj}.json"), "w"))


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
