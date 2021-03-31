import json
from collections import OrderedDict
from datetime import datetime
from itertools import cycle
from pathlib import Path

import networkx as nx
import numpy as np
from numba import njit, prange


class Timer(object):
    def __init__(self):
        self.cache = datetime.now()

    def check(self):
        now = datetime.now()
        duration = now - self.cache
        self.cache = now
        return duration.total_seconds()

    def reset(self):
        self.cache = datetime.now()


class InfiniteKIterator(object):
    def __init__(self, iterable):
        self.iter = cycle(iterable)

    def get_k_item(self, k):
        ret = list()
        for _ in range(k):
            ret.append(next(self.iter))
        return ret


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def get_level(graph: nx.DiGraph, node):
    if node.level > 0:
        return node.level
    if len(list(graph.predecessors(node))) > 0:
        current_level = list()
        for parent in graph.predecessors(node):
            current_level.append(get_level(graph, parent) + 1)
        node.level = min([l for l in current_level if l >= 0])
        if len(current_level) > 1:
            print(
                f"Current node: {node}, potential levels: {', '.join([str(l) for l in current_level])}, take minimum.")
        return node.level
    else:
        node.level = 0
        return node.level


@njit(parallel=True)
def get_batch_scores_for_validation(world_size, pathfinder_batch, stopper_batch, label_batch):
    pathfinder_score = np.zeros((world_size * pathfinder_batch[0].shape[0],))
    stopper_score = np.zeros((world_size * stopper_batch[0].shape[0], 3))
    label = np.zeros((world_size * label_batch[0].shape[0],))

    for i in prange(world_size):
        pathfinder_score[i::world_size] += pathfinder_batch[i]
        stopper_score[i::world_size, :] += stopper_batch[i]
        label[i::world_size] += label_batch[i]

    return pathfinder_score, stopper_score, label
