import os
import numpy as np
import torch
import torch.nn as nn
from apex import amp
from numba.typed import List
from torch.distributed import get_rank, barrier, all_gather, get_world_size
from tqdm import tqdm

from base.base_trainer import BaseTrainer
from data_loader.data_loaders import TreeMatchingDataLoader
from utils.util import get_batch_scores_for_validation


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, model, loss, metrics, pre_metric, optimizer, config,
                 data_loader: TreeMatchingDataLoader = None, valid_data_loader: TreeMatchingDataLoader = None,
                 lr_scheduler=None, abl_current_only=False):
        super(Trainer, self).__init__(model, loss, metrics, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.val_freq = self.config['trainer']['val_freq']
        self.grad_clip = self.config['trainer']['grad_clip']
        self.accumulation_steps = self.config['trainer']['accumulation_steps']
        self.pre_metric = pre_metric
        self.abl_current_current = abl_current_only  # for ablation study
        self.step = 0
        self.writer.add_text('Text', 'Model Architecture: {}'.format(self.config['arch']), 0)
        self.writer.add_text('Text', 'Training Data Loader: {}'.format(self.config['train_data_loader']), 0)
        self.writer.add_text('Text', 'Loss Function: {}'.format(self.config['loss']), 0)
        self.writer.add_text('Text', 'Optimizer: {}'.format(self.config['optimizer']), 0)

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        all_ranks = self.pre_metric(output, target)
        for i, metric in enumerate(self.metrics):
            if metric.__name__ == 'wu_palmer':
                acc_metrics[i] += metric(output, target, self.valid_data_loader.matching_dataset.seed)
            else:
                acc_metrics[i] += metric(all_ranks)
            self.writer.add_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        self.optimizer.zero_grad()
        if self.distributed:
            self.data_loader.sampler.set_epoch(epoch)
        for batch_idx, batch_example in enumerate(self.data_loader):
            x, abs_level_seq, rel_level_seq, segment_seq, node_batch_lengths, tree_batch_mask, types, pathfinder_tags, stopper_tags = [
                tensor.cuda() for tensor in batch_example]

            prediction = self.model(x, abs_level_seq, rel_level_seq, segment_seq, node_batch_lengths,
                                    tree_batch_mask, types)  # contains pathfinder scores & stopper scores
            loss = self.loss(*prediction[:2], pathfinder_tags, stopper_tags)
            fp16_loss = loss
            total_loss += loss

            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)
            self.writer.add_scalar('loss', fp16_loss)

            # gradient accumulation
            if len(self.data_loader) - batch_idx < self.accumulation_steps:
                accumulation_steps = len(self.data_loader) - batch_idx
            else:
                accumulation_steps = self.accumulation_steps
            loss /= accumulation_steps

            # apex loss scale
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()

            if (batch_idx + 1) % self.accumulation_steps == 0 or batch_idx + 1 == len(self.data_loader):
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.lr_scheduler is not None:
                    # print(self.step)
                    self.step += 1
                    self.lr_scheduler.step()

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch,
                    batch_idx * self.data_loader.batch_size,
                    self.data_loader.n_samples,
                    100.0 * batch_idx / len(self.data_loader),
                    fp16_loss))

        log = {'loss': total_loss / len(self.data_loader)}

        # Validation stage
        if self.do_validation and epoch % self.val_freq == 0:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        torch.cuda.synchronize()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training val_freq epochs

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        query_num = len(self.valid_data_loader.matching_dataset.node_list)
        anchor_num = len(self.valid_data_loader.matching_dataset.seed_nodes)
        adj_matrix = self.valid_data_loader.matching_dataset.adj_matrix
        print(f'total queries: {query_num}, total seed nodes: {anchor_num}')

        if 'WORLD_SIZE' in os.environ:
            # DDP
            model = self.model.module
            model.eval()
            if get_rank() == 0:
                pathfinder_scores, stopper_scores, labels = list(), list(), list()
            with torch.no_grad():
                for batch_example in tqdm(self.valid_data_loader):
                    x, abs_level_seq, rel_level_seq, segment_seq, node_batch_lengths, tree_batch_mask, types, tags = [
                        tensor.cuda() for tensor in batch_example]
                    _, _, pathfinder_s, stopper_s = model(x, abs_level_seq, rel_level_seq, segment_seq,
                                                          node_batch_lengths, tree_batch_mask, types)

                    pathfinder_score_list = [torch.zeros((tags.shape[0])).cuda() for _ in range(get_world_size())]
                    stopper_score_list = [torch.zeros((tags.shape[0], 3)).cuda() for _ in range(get_world_size())]
                    labels_list = [torch.zeros((tags.shape[0])).cuda() for _ in range(get_world_size())]

                    # ProcessGroup NCCL does not support gather() yet, so use all_gather instead.
                    barrier()
                    all_gather(pathfinder_score_list, pathfinder_s)
                    all_gather(stopper_score_list, stopper_s)
                    all_gather(labels_list, tags.type(torch.float))

                    if get_rank() == 0:
                        pathfinder_score_batch = List()
                        stopper_scores_batch = List()
                        labels_batch = List()
                        [pathfinder_score_batch.append(p_s.cpu().detach().numpy()) for p_s in pathfinder_score_list]
                        [stopper_scores_batch.append(s_s.cpu().detach().numpy()) for s_s in stopper_score_list]
                        [labels_batch.append(l.cpu().detach().numpy()) for l in labels_list]

                        pathfinder_score, stopper_score, label = get_batch_scores_for_validation(get_world_size(),
                                                                                                 pathfinder_score_batch,
                                                                                                 stopper_scores_batch,
                                                                                                 labels_batch)
                        pathfinder_scores.append(pathfinder_score)
                        stopper_scores.append(stopper_score)
                        labels.append(label.astype(np.int))

            # metric calculation is only performed once on process 0
            if get_rank() == 0:
                return self._evaluate_scores(query_num, anchor_num, adj_matrix, model, epoch, pathfinder_scores,
                                             stopper_scores, labels)
            else:
                return {}

        else:
            # Single-GPU
            model = self.model
            model.eval()
            pathfinder_scores, stopper_scores, labels = list(), list(), list()
            with torch.no_grad():
                for batch_example in tqdm(self.valid_data_loader):
                    x, abs_level_seq, rel_level_seq, segment_seq, node_batch_lengths, tree_batch_mask, types, tags = [
                        tensor.cuda() for tensor in batch_example]
                    _, _, pathfinder_s, stopper_s = model(x, abs_level_seq, rel_level_seq, segment_seq,
                                                          node_batch_lengths, tree_batch_mask, types)
                    pathfinder_scores.append(pathfinder_s.cpu().detach().numpy())
                    stopper_scores.append(stopper_s.cpu().detach().numpy())
                    labels.append(tags.cpu().detach().numpy())
            return self._evaluate_scores(query_num, anchor_num, adj_matrix, model, epoch, pathfinder_scores,
                                         stopper_scores, labels)

    def _evaluate_scores(self, query_num, anchor_num, adj_matrix, model, epoch, pathfinder_scores, stopper_scores,
                         labels):
        pathfinder_scores = np.concatenate(pathfinder_scores, axis=0).reshape((-1))[
                            :query_num * anchor_num].reshape((query_num, -1))
        stopper_scores = np.concatenate(stopper_scores, axis=0).reshape((-1, 3))[:query_num * anchor_num,
                         ...].reshape((query_num, anchor_num, 3))
        labels = np.concatenate(labels, axis=0).reshape((-1))[:query_num * anchor_num].reshape((query_num, -1))
        forward_scores = stopper_scores[..., 0]
        current_scores = stopper_scores[..., 1]
        backward_scores = stopper_scores[..., 2]
        print(np.sum(labels, axis=1))
        if self.abl_current_current:
            fitting_scores = current_scores
        else:
            fitting_scores, _, _, _ = model.fitting_score(adj_matrix, pathfinder_scores, forward_scores,
                                                          current_scores, backward_scores)

        self.writer.set_step(epoch - 1, 'valid')
        total_val_metrics = self._eval_metrics(fitting_scores, labels)

        # add histogram of model parameters to the tensorboard
        for name, p in model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        return {'val_metrics': total_val_metrics.tolist()}
