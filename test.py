import argparse
import collections
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from apex import amp
from apex.parallel import DistributedDataParallel
from numba.typed import List
from tqdm import tqdm

import data_loader.data_loaders as module_dataloader
import data_loader.dataset as module_dataset
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from utils.util import get_batch_scores_for_validation


def main(config):
    torch.tensor([1], device=torch.device('cuda'))  # This line is only for occupying gpus earlier >w<
    assert not (config.case_study and config['trainer']['abl_current_only'])
    logger = config.get_logger('test')

    # setup data_loader instances
    full_dataset = config.initialize('dataset', module_dataset)
    test_data_loader = config.initialize('test_data_loader', module_dataloader, full_dataset,
                                         distributed=config.distributed)
    logger.info(test_data_loader)

    # get function handles of loss and metrics
    metrics = [getattr(module_metric, met) for met in config['metrics']]
    pre_metric = module_metric.obtain_ranks

    # build model architecture
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = config.initialize('arch', module_arch, full_dataset.emb, full_dataset.max_level * 3)
    model = amp.initialize(model.to(device), opt_level='O0')
    if config.distributed:
        model = DistributedDataParallel(model)
    logger.info(model)

    # load saved model
    logger.info(f'Loading checkpoint: {config.resume} ...')
    checkpoint = torch.load(config.resume, map_location={'cuda:%d' % 0: 'cuda:%d' % config.local_rank})
    if config.distributed:
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])
    amp.load_state_dict(checkpoint['amp'])
    torch.cuda.synchronize()

    # prepare model for testing
    model.eval()

    # print test details
    query_num = len(test_data_loader.matching_dataset.node_list)
    anchor_num = len(test_data_loader.matching_dataset.seed_nodes)
    adj_matrix = test_data_loader.matching_dataset.adj_matrix
    print(f'total queries: {query_num}, total seed nodes: {anchor_num}')

    # begin testing.
    logger.info('Begin testing')
    start = time.time()
    with torch.no_grad():
        if config.distributed:
            # DDP
            if config.local_rank == 0:
                pathfinder_scores, stopper_scores, labels = list(), list(), list()
            for batch_example in tqdm(test_data_loader):
                x, abs_level_seq, rel_level_seq, segment_seq, node_batch_lengths, tree_batch_mask, types, tags = [
                    tensor.cuda() for tensor in batch_example]
                dist.barrier()
                _, _, pathfinder_s, stopper_s = model(x, abs_level_seq, rel_level_seq, segment_seq,
                                                      node_batch_lengths, tree_batch_mask, types)

                pathfinder_score_list = [torch.zeros((tags.shape[0])).cuda() for _ in range(config.world_size)]
                stopper_score_list = [torch.zeros((tags.shape[0], 3)).cuda() for _ in range(config.world_size)]
                labels_list = [torch.zeros((tags.shape[0])).cuda() for _ in range(config.world_size)]

                # ProcessGroup NCCL does not support gather() yet, so use all_gather instead
                dist.barrier()
                dist.all_gather(pathfinder_score_list, pathfinder_s)
                dist.all_gather(stopper_score_list, stopper_s)
                dist.all_gather(labels_list, tags.type(torch.float))

                if config.local_rank == 0:
                    pathfinder_score_batch = List()
                    stopper_scores_batch = List()
                    labels_batch = List()
                    [pathfinder_score_batch.append(p_s.cpu().detach().numpy()) for p_s in pathfinder_score_list]
                    [stopper_scores_batch.append(s_s.cpu().detach().numpy()) for s_s in stopper_score_list]
                    [labels_batch.append(l.cpu().detach().numpy()) for l in labels_list]

                    pathfinder_score, stopper_score, label = get_batch_scores_for_validation(config.world_size,
                                                                                             pathfinder_score_batch,
                                                                                             stopper_scores_batch,
                                                                                             labels_batch)

                    pathfinder_scores.append(pathfinder_score)
                    stopper_scores.append(stopper_score)
                    labels.append(label.astype(np.int))
        else:
            # Single-GPU
            pathfinder_scores, stopper_scores, labels = list(), list(), list()
            with torch.no_grad():
                for batch_example in tqdm(test_data_loader):
                    x, abs_level_seq, rel_level_seq, segment_seq, node_batch_lengths, tree_batch_mask, types, tags = [
                        tensor.cuda() for tensor in batch_example]
                    _, _, pathfinder_s, stopper_s = model(x, abs_level_seq, rel_level_seq, segment_seq,
                                                          node_batch_lengths, tree_batch_mask, types)
                    pathfinder_scores.append(pathfinder_s.cpu().detach().numpy())
                    stopper_scores.append(stopper_s.cpu().detach().numpy())
                    labels.append(tags.cpu().detach().numpy())

        def get_case(idx, case_type):
            seed_list = test_data_loader.matching_dataset.seed_nodes
            taxon = test_data_loader.matching_dataset.node_list[idx]
            # get predicted parent and ground-truth parent
            predicted_idx = np.argmax(fitting_scores[idx])
            predicted = seed_list[predicted_idx]
            ground_truth = taxon.parents[0]
            ground_truth_idx = seed_list.index(ground_truth)

            # get fitting scores and predicted/ground-truth's parent & child with max Pathfinder Score
            predicted_fitting_score = fitting_scores[idx, predicted_idx]
            predicted_pathfinder = pathfinder_scores[idx, predicted_idx]
            predicted_parent_forward = parent_forward[idx, predicted_idx]
            predicted_current = current_scores[idx, predicted_idx]
            predicted_child_backward = child_backward[idx, predicted_idx]
            predicted_max_ps_child = seed_list[backward_idx[idx, predicted_idx]] if predicted.children else None
            try:
                predicted_parent = predicted.parents[0]
            except IndexError:
                predicted_parent = None
            ground_truth_ranking = all_ranks[idx]
            ground_truth_fitting_score = fitting_scores[idx, ground_truth_idx]
            ground_truth_pathfinder = pathfinder_scores[idx, ground_truth_idx]
            ground_truth_parent_forward = parent_forward[idx, ground_truth_idx]
            ground_truth_current = current_scores[idx, ground_truth_idx]
            ground_truth_child_backward = child_backward[idx, ground_truth_idx]
            ground_truth_max_ps_child = seed_list[
                backward_idx[idx, ground_truth_idx]] if ground_truth.children else None
            try:
                ground_truth_parent = ground_truth.parents[0]
            except IndexError:
                ground_truth_parent = None

            ret = list()
            ret.append(f'{case_type} Case:\n')
            ret.append(f'Query: {taxon}\n')
            ret.append(f'Predicted: {predicted}, Fitting Score: {predicted_fitting_score},'
                       f'Pathfinder Score: {predicted_pathfinder}, Current Score: {predicted_current}\n')
            ret.append(f'Predicted\'s parent: {predicted_parent}, '
                       f'Forward Score: {predicted_parent_forward}\n')
            ret.append(f'Predicted\'s child w/ max PS: {predicted_max_ps_child}, '
                       f'Backward Score: {predicted_child_backward}\n\n')
            ret.append(f'Ground Truth: {ground_truth}, Rank: {ground_truth_ranking},'
                       f'Fitting Score: {ground_truth_fitting_score},'
                       f'Pathfinder Score: {ground_truth_pathfinder}, Current Score: {ground_truth_current}\n')
            ret.append(f'Ground Truth\'s parent: {ground_truth_parent}, '
                       f'Forward Score: {ground_truth_parent_forward}\n')
            ret.append(f'Ground Truth\'s child w/ max PS: {ground_truth_max_ps_child}, '
                       f'Backward Score: {ground_truth_child_backward}\n')
            ret.append('----------------------------------\n')
            return ret

        # if DDP, metric calculation is only performed once on process 0
        if not config.distributed or config.local_rank == 0:
            pathfinder_scores = np.concatenate(pathfinder_scores, axis=0).reshape((-1))[
                                :query_num * anchor_num].reshape((query_num, -1))
            stopper_scores = np.concatenate(stopper_scores, axis=0).reshape((-1, 3))[:query_num * anchor_num,
                             :].reshape((query_num, anchor_num, 3))
            labels = np.concatenate(labels, axis=0).reshape((-1))[:query_num * anchor_num].reshape((query_num, -1))
            forward_scores = stopper_scores[..., 0]
            current_scores = stopper_scores[..., 1]
            backward_scores = stopper_scores[..., 2]
            print(np.sum(labels, axis=1))
            print(np.sum(labels))

            if not config.distributed:
                f_fitting_score = model.fitting_score
            else:
                f_fitting_score = model.module.fitting_score

            if config['trainer']['abl_current_only']:
                fitting_scores = current_scores
            else:
                fitting_scores, parent_forward, child_backward, backward_idx = f_fitting_score(
                    adj_matrix,
                    pathfinder_scores,
                    forward_scores,
                    current_scores,
                    backward_scores)

            all_ranks = pre_metric(fitting_scores, labels)

            for i, metric in enumerate(metrics):
                if metric.__name__ == 'wu_palmer':
                    m = metric(fitting_scores, labels, test_data_loader.matching_dataset.seed)
                else:
                    m = metric(all_ranks)
                logger.info(f'{metric.__name__}: {m}')

            # case_study or not
            if config.case_study:
                assert isinstance(config.resume, Path)
                case_study_dir = Path(f'case_studies/{config.resume.parent.name}/')
                case_study_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f'Performing case study, results will be saved in {case_study_dir}')
                # hit cases
                hit_idx = np.flatnonzero(all_ranks == 1)
                if hit_idx.size != 0:
                    hit_idx = np.random.choice(hit_idx, min(10, hit_idx.size), replace=False)
                else:
                    hit_idx = np.array([])
                # worst cases
                worst_case_num = all_ranks.size - 10
                worst_case_num = all_ranks.size if worst_case_num < 0 else worst_case_num
                worst_idx = np.argpartition(all_ranks, worst_case_num)[worst_case_num:]
                # random cases
                other_idx = np.arange(all_ranks.size)
                other_idx = other_idx[~(all_ranks == 1) & ~np.in1d(other_idx, worst_idx)]
                if other_idx.size != 0:
                    other_idx = np.random.choice(other_idx, min(10, other_idx.size), replace=False)
                else:
                    other_idx = np.array([])

                print_list = list()
                for idx in hit_idx:
                    print_list += get_case(idx, "Hit")
                for idx in worst_idx:
                    print_list += get_case(idx, "Worst")
                for idx in other_idx:
                    print_list += get_case(idx, "Other Random")
                with (case_study_dir / f'case_study_{config.resume.parent.name}.txt').open('w') as f:
                    f.writelines(print_list)
        else:
            logger.info("no need to save case study results")

    logger.info(f'Finish testing in {time.time() - start} seconds')
    if config.distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='HEF testing phase')
    args.add_argument('-r', '--resume', required=True, type=str, help='path to latest checkpoint')
    args.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    args.add_argument('-s', '--case_study', action='store_true', help='whether case study is needed')
    args.add_argument("--local_rank", default=0, type=int)

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--batch_size'], type=int, target=('test_data_loader', 'args', 'node_batch_size'))
    ]
    config = ConfigParser(args, options)
    main(config)
