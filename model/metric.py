import networkx as nx
import numpy as np
from networkx.algorithms import tree_all_pairs_lowest_common_ancestor


def obtain_ranks(fitting_scores: np.ndarray, labels: np.ndarray):
    """ 
    fitting_scores : ndarray of size (batch_size, seed_size), calculated fitting scores
    labels : ndarray of size (batch_size, seed_size), labels

    rankings: ndarray of size(batch_size, 1), fitting score rankings of ground truth anchor
    """
    gt_score = fitting_scores[labels == 1]
    assert gt_score.shape[0] == fitting_scores.shape[0], 'Each node should have one and only one parent'
    for i in range(fitting_scores.shape[0]):
        assert np.in1d(gt_score[i], fitting_scores[i, ...].squeeze())
    rankings = np.sum(fitting_scores > gt_score[..., np.newaxis], axis=1) + 1
    seeds_count = np.array([fitting_scores.shape[1]])
    rankings2 = seeds_count - np.sum(fitting_scores <= gt_score[..., np.newaxis], axis=1) + 1
    print(rankings)
    print(rankings2)
    return rankings


def micro_mr(all_ranks: np.ndarray):
    return np.mean(all_ranks)


def hit_at_1(all_ranks: np.ndarray):
    return np.sum(all_ranks <= 1) / all_ranks.size


def hit_at_3(all_ranks: np.ndarray):
    return np.sum(all_ranks <= 3) / all_ranks.size


def hit_at_5(all_ranks: np.ndarray):
    return np.sum(all_ranks <= 5) / all_ranks.size


def mrr(all_ranks: np.ndarray):
    return np.mean(1.0 / all_ranks)


def mrr_scaled_10(all_ranks: np.ndarray):
    # Scaled MRR score, check eq. (2) in the PinSAGE paper: https://arxiv.org/pdf/1806.01973.pdf
    return np.mean(1.0 / np.ceil(all_ranks / 10))


def wu_palmer(fitting_scores: np.ndarray, labels: np.ndarray, seed: nx.DiGraph):
    seed_nodes = list(seed.nodes)
    pred_indices = fitting_scores.argmax(axis=1)
    gt_indices = labels.argmax(axis=1)
    node_pairs = [(seed_nodes[pred_idx], seed_nodes[gt_idx]) for pred_idx, gt_idx in zip(pred_indices, gt_indices)]
    lcas = tree_all_pairs_lowest_common_ancestor(seed, pairs=node_pairs)

    def calc_wu_palmer(y, y_star, lca):
        return 2.0 * lca.level / (y.level + y_star.level + 0.000001)

    ret = [calc_wu_palmer(*pair, lca) for pair, lca in lcas]
    return np.array(ret).mean()


def combined_metrics(all_ranks):
    # combination of three metrics, used in early stop
    score = micro_mr(all_ranks) * (1.0 / max(mrr_scaled_10(all_ranks), 0.0001)) * (
            1.0 / max(hit_at_3(all_ranks), 0.0001)) * (1.0 / max(hit_at_1(all_ranks), 0.0001))
    return score
