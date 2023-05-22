import os
import random
import sys

import networkx as nx
import torch
from torch.utils.data import DataLoader
from options import parse_args
from progressbar import ProgressBar
from torch_geometric.data import Data, Dataset, Batch

sys.setrecursionlimit(100000)
import pdb
from collections import defaultdict
from glob import glob
import re

import numpy as np


class ProofStepsData(Dataset):
    def __init__(self, split, opts):
        super().__init__()
        self.opts = opts

        if split in ["train", "valid"]:
            proof_steps = glob(os.path.join(opts.datapath, split, "*.pt"))
        elif split == "train_valid":
            proof_steps = glob(os.path.join(opts.datapath, "train/*.pt")) + glob(
                os.path.join(opts.datapath, "valid/*.pt")
            )
        else:
            raise ValueError("Invalid split")
        # Assuming proof steps are named as "project-proof-####.pt"
        # Filter out skipped proofs
        self.proof_steps = []
        for step in proof_steps:
            matches = re.match(
                r"([^.]+)--([^.]+)--([^.]+)-[0-9]{8,}.pt",
                step.split(os.path.sep)[-1],
            )
            if not matches:
                print(step)
                continue
            # Check for filters
            if (
                matches[1] in opts.skip_projects
                or matches[2] in opts.skip_libs
                or matches[3] in opts.skip_proofs
            ):
                continue
            self.proof_steps.append(step)

        random.shuffle(self.proof_steps)
        print("%d proof steps in %s" % (len(self), split))

    def __len__(self):
        return len(self.proof_steps)

    def len(self):
        return self.__len__()

    def __getitem__(self, idx):
        """
        step = {
            'file': STR,
            'proof_name': STR,
            'n_step': INT,
            'env': [{
                'qualid': STR,
                'ast': LARK.TREE.TREE,
                'x': TENSOR.TENSOR
                'edge_index': TENSOR.TENSOR
            }],
            'local_context': [{
                'ident': STR,
                'ast': LARK.TREE.TREE,
                'x': TENSOR.TENSOR
                'edge_index': TENSOR.TENSOR
            }],
            'goal': {
                "id": STR,
                "text": STR,
                "ast": LARK.TREE.TREE,
                "x": TENSOR.TENSOR,
                "edge_index": TENSOR.TENSOR,
            },
            'tactic_actions':  [INT|STR],
            'tactic_str': STR,
        }
        """
        try:
            data = torch.load(self.proof_steps[idx])
        except:
            print("Error loading proof step %s" % self.proof_steps[idx])
            raise
        return data

    def get(self, idx):
        return self.__getitem__(idx)


def merge(batch_list):
    fields = [
        "file",
        "proof_name",
        "n_step",
        "env",
        "local_context",
        "goal",
        "is_synthetic",
        "tactic_actions",
        "tactic_str",
    ]
    data_batch = {key: [] for key in fields}
    for proof_step in batch_list:
        for key, value in proof_step._store.items():
            if key not in fields:
                continue
            data_batch[key].append(value)
    batch = Batch.from_data_list(batch_list)
    for k, v in data_batch.items():
        batch[k] = v
    return batch


def create_dataloader(split, opts):
    ds = ProofStepsData(split, opts)
    return DataLoader(  # Use original pytorch data loader since we want to define custom collate_fn
        ds,
        opts.batchsize,
        shuffle=split.startswith("train"),
        collate_fn=merge,
        num_workers=opts.num_workers,
    )


if __name__ == "__main__":
    import json

    opts = parse_args()
    loader = create_dataloader("train", opts)
    bar = ProgressBar(max_value=len(loader))
    num_tactics_with_goal_idents = 0
    idents_per_goal = defaultdict(int)
    for i, data_batch in enumerate(loader):
        if i == 0:
            print(data_batch)
            # pickle.dump(data_batch, open("data_batch.pickle", "wb"))
        for goal, tactic_str in zip(data_batch.goal, data_batch.tactic_str):
            idents_per_goal[len(goal["idents"])] += 1
            for ident in goal["idents"]:
                if ident[1] in set(tactic_str.split(" ")):
                    num_tactics_with_goal_idents += 1
        bar.update(i)
    print(idents_per_goal)
    json.dump(idents_per_goal, open("idents_per_goal.json", "w"))
