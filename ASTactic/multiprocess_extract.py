#!/usr/bin/env python3
import argparse
import asyncio
import json
import multiprocessing as mp
import subprocess
import sys
import os
from tqdm import tqdm

sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))
)

from utils import update_env


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog=argv[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--n_cpu", default=mp.cpu_count(), type=int)
    parser.add_argument(
        "-s",
        "--split",
        type=str,
        default="train",
        choices=["train", "valid", "train_valid"],
    )
    parser.add_argument(
        "-f",
        "--filter",
        type=str,
        default="",
        help="Filter proof extraction by project name. Multiple projects can be separated by comma.",
    )
    parser.add_argument("-o", "--output", type=str, default="./proof_steps_gnn")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--data_path", type=str, default="../data")
    parser.add_argument("--proj_splits_file", type=str, default="../projs_split.json")
    parser.add_argument("-l", "--log", type=str, default="")
    args = parser.parse_args(argv[1:])
    if args.split not in ["train", "valid", "train_valid"]:
        raise ValueError(f"Invalid split {args.split}")
    if args.split == "train_valid":
        args.splits = ["train", "valid"]
    else:
        args.splits = [args.split]

    if not os.path.exists(args.output):
        os.makedirs(args.output)
        for split in args.splits:
            os.makedirs(os.path.join(args.output, split))
    if args.verbose:
        print(args)
    args.mute = not args.verbose
    if args.filter:
        args.filter = args.filter.split(",")
    else:
        args.filter = []
    return args


def process_lib(project, file_name, proofs, output, split, _process_list):
    env = {"constants": [], "inductives": []}
    id = _process_list.index(os.getpid())
    # Process proof data
    pbar = tqdm(
        total=len(proofs),
        desc=file_name[file_name.find(project) :],
        position=id + 1,
        leave=False,
    )
    # tqdm.write(f"{id + 1} {file_name[file_name.find(project) :]}", sys.stdout)
    for proof_data in proofs:
        env = update_env(env, proof_data["env_delta"])
        del proof_data["env_delta"]
        proof_data["env"] = env
        process_proof(project, file_name, proof_data, output, split)
        pbar.update()


if __name__ == "__main__":
    args = parse_args(sys.argv)
    print(args)
    # rage(args.n_cpu)
    from multiprocess_utils import MPSelections, mp_iter_libs
    from extract_proof_steps import process_proof

    filters = MPSelections(args.filter)
    for split in args.splits:
        process_proof_args = [
            args.output,
            split,
        ]
        print(process_proof_args, filters)
        mp_iter_libs(
            process_lib,
            process_proof_args,
            n_cpu=args.n_cpu,
            filters=filters,
            data_path=args.data_path,
            proj_splits_file=args.proj_splits_file,
            mute=args.mute,
            split=split,
            log_file=args.log,
        )
