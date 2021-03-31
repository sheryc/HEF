from functools import partial
from itertools import chain

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import DistilBertTokenizer

from .dataset import TaxonomyDataset, TreeMatchingDataset


class TreeMatchingDataLoader(DataLoader):
    def __init__(self, dataset: TaxonomyDataset, mode, node_batch_size=256, exact_total_size=True, tree_batch_size=32,
                 pos_current_num=-1, pos_backward_num=4, pos_forward_num=-1, min_neg_num=20, expand_factor=3,
                 neg_path='backward', top_k_similarity=True, shuffle=True, num_workers=16, cross=False,
                 distributed=False, abl_use_egonet=False, abl_name_only=False):
        assert mode in ['train', 'val_full', 'test_full',
                        'test_hierarchical'], 'mode must be train, val_full, test_full or test_hierarchical'

        # Batching mode
        self.mode = mode

        # Batch sizes. A batch contains several node batches,
        # each contains a split of different tags of total tree_batch_size
        self.node_batch_size = node_batch_size
        self.tree_batch_size = tree_batch_size
        self.exact_total_size = exact_total_size

        # Split in tree batches
        self.pos_current_num = pos_current_num
        self.pos_backward_num = pos_backward_num
        self.pos_forward_num = pos_forward_num
        self.min_neg_num = min_neg_num

        # Other options for TreeMatchingDataset
        self.expand_factor = expand_factor
        self.neg_path = neg_path
        self.top_k_similarity = top_k_similarity

        # Other options for DataLoader
        self.shuffle = shuffle
        self.num_workers = num_workers

        self.full_dataset = dataset
        self.matching_dataset = TreeMatchingDataset(self.full_dataset, mode=mode, exact_total_size=exact_total_size,
                                                    pos_current_num=pos_current_num, pos_backward_num=pos_backward_num,
                                                    pos_forward_num=pos_forward_num, min_neg_num=min_neg_num,
                                                    total_size=tree_batch_size, expand_factor=expand_factor,
                                                    neg_path=neg_path, cross=cross,
                                                    top_k_similarity=top_k_similarity, abl_use_egonet=abl_use_egonet,
                                                    abl_name_only=abl_name_only)
        if distributed:
            self.sampler = DistributedSampler(self.matching_dataset, shuffle=(shuffle and 'full' not in self.mode))
        else:
            self.sampler = None

        if 'full' in self.mode:
            self.collate_fn = partial(collate_single_pair, tokenizer=self.full_dataset.tokenizer)
        else:
            self.collate_fn = partial(collate_tree_batches, tokenizer=self.full_dataset.tokenizer)

        self.num_workers = num_workers
        super(TreeMatchingDataLoader, self).__init__(dataset=self.matching_dataset,
                                                     batch_size=self.node_batch_size,
                                                     shuffle=self.shuffle if not distributed else False,
                                                     sampler=self.sampler,
                                                     collate_fn=self.collate_fn, num_workers=self.num_workers,
                                                     pin_memory=True)
        self.n_samples = len(self.matching_dataset)

    @property
    def _adj_matrix(self):
        return self.matching_dataset.adj_matrix


def collate_tree_batches(samples, tokenizer):
    # tag splits into pathfinder tag and stopper tag
    texts, abs_level_seq, rel_level_seq, segment_seq, pathfinder_tag, stopper_tag = map(list, zip(*chain(*samples)))
    processed = process_input(texts, abs_level_seq, rel_level_seq, segment_seq, tokenizer)
    pathfinder_tags = torch.tensor(pathfinder_tag)
    stopper_tags = torch.tensor(stopper_tag)
    return (*processed, pathfinder_tags, stopper_tags)


def collate_single_pair(samples, tokenizer):
    # each sample only contains one subtree-query pair
    # only contains one tag, 1 iff anchor is ground-truth
    texts, abs_level_seq, rel_level_seq, segment_seq, tag = list(zip(*chain(samples)))
    processed = process_input(texts, abs_level_seq, rel_level_seq, segment_seq, tokenizer)
    tags = torch.tensor(tag)
    return (*processed, tags)


def process_input(texts, abs_level_seq, rel_level_seq, segment_seq, tokenizer: DistilBertTokenizer):
    node_batch_lengths = torch.tensor(
        [len(subtree_seq[0]) for subtree_seq in texts])  # To be split in higher transformer
    processed = tokenizer(*[list(chain(*k)) for k in list(zip(*chain(texts)))], padding='max_length', truncation=True,
                          max_length=32, return_attention_mask=True, return_token_type_ids=True)
    x, tree_batch_mask, types = torch.tensor(processed['input_ids']), torch.tensor(
        processed['attention_mask']), torch.tensor(processed['token_type_ids'])
    abs_level_seq_cat = torch.cat(abs_level_seq, 0)  # [node_batch * tree_batch, 1]
    rel_level_seq_cat = torch.cat(rel_level_seq, 0)  # [node_batch * tree_batch, 1]
    segment_seq_cat = torch.cat(segment_seq, 0)  # [node_batch * tree_batch, 1]
    return x, abs_level_seq_cat, rel_level_seq_cat, segment_seq_cat, node_batch_lengths, tree_batch_mask, types
