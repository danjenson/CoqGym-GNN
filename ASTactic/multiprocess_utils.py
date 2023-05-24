import json
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict
from copy import deepcopy
from functools import partial
from inspect import signature
from itertools import repeat
from pathlib import Path
from typing import Callable, Iterable, Optional, Tuple, overload, Union

from tqdm import tqdm

sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))
)

from utils import update_env

__all__ = ["mp_iter_proofs", "mp_iter_libs", "MPSelections"]

#! IMPORTANT: Use tqdm.write() for printouts in multiprocessing for thread safety


class MPSelections(dict):
    def __init__(
        self,
        projects: Optional[Union[str, Iterable[str]]] = None,
        libs: Optional[Union[str, Iterable[str]]] = None,
        proofs: Optional[Union[str, Iterable[str]]] = None,
        name: Optional[str] = None,
    ):
        super().__init__()
        self["projects"] = []
        self["libs"] = []
        self["proofs"] = []
        self.name = name if name is not None else ""
        if projects is not None:
            self.add_projects(projects)
        if libs is not None:
            self.add_libs(libs)
        if proofs is not None:
            self.add_proofs(proofs)

    def __repr__(self):
        name = "default" if self.name == "" else self.name
        return f"MPSelections[{self.name}](projects={self['projects']}, libs={self['libs']}, proofs={self['proofs']})"

    def merge(self, selections: "MPSelections"):
        self.add_projects(selections["projects"])
        self.add_libs(selections["libs"])
        self.add_proofs(selections["proofs"])
        return self

    def add_projects(self, projects: Union[str, Iterable[str]]):
        if isinstance(projects, str):
            projects = [projects]
        self["projects"].extend(projects)
        return self

    def add_libs(self, libs: Union[str, Iterable[str]]):
        if isinstance(libs, str):
            libs = [libs]
        self["libs"].extend(libs)
        return self

    def add_proofs(self, proofs: Union[str, Iterable[str]]):
        if isinstance(proofs, str):
            proofs = [proofs]
        self["proofs"].extend(proofs)
        return self


DEFAULT_FILTERS = MPSelections(
    [],  # Projects
    [],  # Libs
    [],  # Proofs
)

DEFAULT_SKIPS = MPSelections(
    [],  # Projects
    [],  # Libs
    [],  # Proofs
)

SPLIT_CONVERSION = {
    "train": "projs_train",
    "valid": "projs_valid",
    "test": "projs_test",
}


def filter_projects(proj_splits_file, split, filters, skips):
    splits = json.load(open(proj_splits_file))
    if split is not None:
        if split not in SPLIT_CONVERSION:
            raise ValueError(
                f"Invalid split {split}. Must be one of {list(splits.keys())}"
            )
        projects = splits[SPLIT_CONVERSION[split]]
    else:
        projects = [project for split in splits.values() for project in split]
    # Filter projects if there are project filters but no lib or proof filters
    if filters["projects"] and (not filters["libs"] or not filters["proofs"]):
        projects = [p for p in projects if p in filters["projects"]]
    # Filter all skipped projects
    projects = [p for p in projects if p not in skips["projects"]]
    return projects


def get_init(log_file, mute):
    init = None
    if log_file:
        init = _init
    elif mute:
        init = _init
        log_file = os.devnull
    return init, log_file


@overload
def mp_iter_proofs(
    fn: Callable,
    args: Iterable = [],
    kwargs: dict = {},
    *,  # Force keyword-only arguments
    split: Optional[str] = None,
    filters: dict = DEFAULT_FILTERS,
    skips: dict = DEFAULT_SKIPS,
    mute: bool = False,
    n_cpu: int = mp.cpu_count(),
    proj_splits_file: str = "../projs_split.json",
    data_path: str = "../data",
    log_file: str = "",
) -> dict:
    ...


@overload
def mp_iter_proofs(
    fn: Callable,
    args: Iterable = [],
    kwargs: dict = {},
    proj_callback: Callable = ...,
    proj_callback_args: Iterable = ...,
    proj_callback_kwargs: dict = ...,
    *,  # Force keyword-only arguments
    split: Optional[str] = None,
    filters: dict = DEFAULT_FILTERS,
    skips: dict = DEFAULT_SKIPS,
    mute: bool = False,
    n_cpu: int = mp.cpu_count(),
    proj_splits_file: str = "../projs_split.json",
    data_path: str = "../data",
    log_file: str = "",
) -> Tuple[dict, dict]:
    ...


def mp_iter_proofs(
    fn: Callable,
    args: Iterable = [],
    kwargs: dict = {},
    proj_callback: Optional[Callable] = None,
    proj_callback_args: Optional[Iterable] = None,
    proj_callback_kwargs: Optional[dict] = None,
    *,  # Force keyword-only arguments
    split: Optional[str] = None,
    filters: dict = DEFAULT_FILTERS,
    skips: dict = DEFAULT_SKIPS,
    mute: bool = False,
    n_cpu: int = mp.cpu_count(),
    proj_splits_file: str = "../projs_split.json",
    data_path: str = "../data",
    log_file: str = "",
):
    """
    Applies `fn(project, library, proof_data, *args, **kwargs)` in parallel using multiprocessing. Can whitelist/blacklist projects, proof files (libs), and individual proofs.

    List of process IDs is stored in an optional kwarg '_process_list' in kwargs. This can be used to uniquely identify each process running in parallel

    Filters and skips are a dictionary of the form:
        {
            "projects": list[str],
            "libs":     list[str],
            "proofs":   list[str],
        }

    Filtering is applied AND-wise (i.e. take all projects in filters["projects"] AND all libs in filters["libs"] AND all proofs in filters["proofs"])
    THEN, apply all skips AND-wise (i.e. skip all projects in skips["projects"] AND all libs in skips["libs"] AND all proofs in skips["proofs"])

    Project callback is applied after obtaining results from all proofs in a project.

    Output is of the form:
    {
        project: {
            lib: {
                proof: result
                ...
            }
            ...
        }
        ...
    }
    """
    out = defaultdict(lambda: defaultdict(dict))
    proj_callback_results = defaultdict()

    projects = filter_projects(proj_splits_file, split, filters, skips)
    init, log_file = get_init(log_file, mute)
    tqdm.set_lock(mp.Lock())
    with mp.Pool(n_cpu, initializer=init, initargs=(tqdm.get_lock(), log_file)) as pool:
        process_list = [c.pid for c in mp.active_children()]
        extras = dict(
            _process_list=process_list,
        )
        # TODO: Rewrite this as a generator outputting (project, lib, proof_data, *args, **kwargs)
        # Load all async tasks (assumes all proofs can be loaded into memory)
        for project in tqdm(projects, desc=f"{fn.__name__} Projects", position=1):
            for j in tqdm(
                list(str(p) for p in (Path(data_path) / project).glob("**/*.json")),
                desc=project,
                position=0,
            ):
                lib = j.split(os.path.sep)[-1]
                if lib in skips["libs"]:
                    continue
                if filters["libs"] and lib not in filters["libs"]:
                    continue
                proofs = json.load(open(j))["proofs"]
                env = {"constants": [], "inductives": []}
                # Process proof data
                run_proofs = []
                for proof_data in proofs:
                    env = update_env(env, proof_data["env_delta"])
                    del proof_data["env_delta"]
                    proof_data["env"] = json.loads(json.dumps(env))
                    # Filters
                    if (
                        filters["proofs"]
                        and proof_data["name"] not in filters["proofs"]
                    ):
                        continue
                    # Skips
                    if proof_data["name"] in skips["proofs"]:
                        continue
                    run_proofs.append(proof_data)

                fn_args = ((project, j, proof_data, *args) for proof_data in run_proofs)
                r = pool.starmap_async(
                    _fn_wrapper,
                    zip(
                        repeat(fn),
                        fn_args,
                        repeat(kwargs),
                        repeat(extras),
                    ),
                )
                for proof_data, (result, p, f) in zip(run_proofs, r.get()):
                    out[p][f][proof_data["name"]] = result
                if proj_callback is not None:
                    if proj_callback_args is None:
                        proj_callback_args = []
                    if proj_callback_kwargs is None:
                        proj_callback_kwargs = {}
                    proj_callback_results[project] = proj_callback(
                        out[project], *proj_callback_args, **proj_callback_kwargs
                    )

    if proj_callback is not None:
        return dict(out), dict(proj_callback_results)
    return dict(out)


def _fn_wrapper(fn, args, kwargs, all_extras):
    sig = signature(fn)
    p, j = args[:2]
    extras = {k: v for k, v in all_extras.items() if k in sig.parameters}
    if extras:
        return fn(*args, **kwargs, **extras), p, j
    return fn(*args, **kwargs), p, j


def _init(lock, log_file):
    sys.stdout = open(log_file, "w")
    tqdm.set_lock(lock)


@overload
def mp_iter_libs(
    fn: Callable,
    args: Iterable = [],
    kwargs: dict = {},
    *,  # Force keyword-only arguments
    split: Optional[str] = None,
    filters: dict = DEFAULT_FILTERS,
    skips: dict = DEFAULT_SKIPS,
    mute: bool = False,
    n_cpu: int = mp.cpu_count(),
    proj_splits_file: str = "../projs_split.json",
    data_path: str = "../data",
    log_file: str = "",
) -> dict:
    ...


@overload
def mp_iter_libs(
    fn: Callable,
    args: Iterable = [],
    kwargs: dict = {},
    proj_callback: Callable = ...,
    proj_callback_args: Iterable = ...,
    proj_callback_kwargs: dict = ...,
    *,  # Force keyword-only arguments
    split: Optional[str] = None,
    filters: dict = DEFAULT_FILTERS,
    skips: dict = DEFAULT_SKIPS,
    mute: bool = False,
    n_cpu: int = mp.cpu_count(),
    proj_splits_file: str = "../projs_split.json",
    data_path: str = "../data",
    log_file: str = "",
) -> Tuple[dict, dict]:
    ...


def mp_iter_libs(
    fn: Callable,
    args: Iterable = [],
    kwargs: dict = {},
    proj_callback: Optional[Callable] = None,
    proj_callback_args: Iterable = [],
    proj_callback_kwargs: dict = {},
    *,  # Force keyword-only arguments
    split: Optional[str] = None,
    filters: dict = DEFAULT_FILTERS,
    skips: dict = DEFAULT_SKIPS,
    mute: bool = False,
    n_cpu: int = mp.cpu_count(),
    proj_splits_file: str = "../projs_split.json",
    data_path: str = "../data",
    log_file: str = "",
):
    out = defaultdict(dict)
    proj_callback_results = defaultdict()

    projects = filter_projects(proj_splits_file, split, filters, skips)
    init, log_file = get_init(log_file, mute)
    tqdm.set_lock(mp.Lock())
    with mp.Pool(n_cpu, initializer=init, initargs=(tqdm.get_lock(), log_file)) as pool:
        process_list = [c.pid for c in mp.active_children()]
        print(process_list)
        extras = dict(
            _process_list=process_list,
        )
        # Load all async tasks (assumes all proofs can be loaded into memory)
        to_eval = []
        start = time.time()
        skipped = 0
        for project in tqdm(
            projects, desc=f"{fn.__name__} Loading...", position=0, leave=False
        ):
            for j in list(
                str(p) for p in (Path(data_path) / project).glob("**/*.json")
            ):
                lib = j[j.find(project) + len(project) + 1 :]
                if lib in skips["libs"]:
                    skipped += 1
                    continue
                if filters["libs"] and lib not in filters["libs"]:
                    skipped += 1
                    continue
                to_eval.append((project, lib, j))
        fn_args = ((p, j, json.load(open(j))["proofs"], *args) for p, l, j in to_eval)
        # Start mapping
        results_pbar = tqdm(
            total=len(to_eval), desc=f"Gathering Results", position=0, leave=False
        )
        # Collect results
        for result, project, j in pool.imap_unordered(
            partial(_fn_wrapper, fn, kwargs=kwargs, all_extras=extras),
            fn_args,
        ):
            out[project][j] = result
            results_pbar.update()
        # Process callbacks
        if proj_callback is not None:
            for project, lib_dict in tqdm(out.items(), desc="Project Callback"):
                if proj_callback_args is None:
                    proj_callback_args = []
                if proj_callback_kwargs is None:
                    proj_callback_kwargs = {}
                proj_callback_results[project] = proj_callback(
                    lib_dict, *proj_callback_args, **proj_callback_kwargs
                )
    print(f"\nSummary for multiprocessing {fn.__name__} on all libraries:")
    print(f"\tTotal time: {time.time() - start} seconds")
    print(f"\tTotal number of projects processed: {len(out)}")
    print(f"\tTotal number of libraries processed: {len(to_eval)}")
    print(f"\tTotal number of libraries skipped: {len(skips['libs'])}")

    if proj_callback is not None:
        return dict(out), dict(proj_callback_results)
    return dict(out)


if __name__ == "__main__":

    def print_name(proj, lib, proof, *args, **kwargs):
        print(proof["name"], args, kwargs)

    def sleep(_, __, p, **kwargs):
        time.sleep(0.1)
        return p["name"]

    def sum_proofs(_, j, p, *args, **kwargs):
        return len(p)

    def sum_values(lib_dict: dict):
        return sum(v for v in lib_dict.values())

    filter = {
        "projects": [
            "twoSquare",
            "QuickChick",
            "regexp",
            "domain-theory",
            "tarski-geometry",
            "ieee754",
            "ctltctl",
        ],
        "libs": [],
        "proofs": [],
    }
    kwargs = {"a": 1, "b": 2}
    # mp_iter_proofs(print_name, filters=filter)
    # result = mp_iter_proofs(print_name, kwargs=kwargs, filters=filter, n_cpu=6)
    _, project_results = mp_iter_libs(
        sum_proofs, [], {}, sum_values, [], {}, split="test"
    )
    print(sum_values(project_results))
    print(project_results)
    print(len(project_results))
