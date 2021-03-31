import re
from functools import reduce
from typing import List

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix
from transformers import DistilBertModel

from base.base_model import BaseModel
from .model_zoo import ConstPositionalEncoding, LearnablePositionalEncoding, LevelEncoding, SegmentEncoding


class HEFBase(BaseModel):
    def __init__(self, max_level, **options):
        super(HEFBase, self).__init__()
        self.options = options
        self.emb_dim = options['emb_dim']

        self.higher_segment_emb = SegmentEncoding(self.emb_dim)
        if options['abs_level']:
            self.abs_level = True
            self.higher_abs_level_emb = LevelEncoding(max_level, self.emb_dim)
        else:
            self.abs_level = False
        if options['rel_level']:
            self.rel_level = True
            self.higher_rel_level_emb = LevelEncoding(max_level, self.emb_dim)
        else:
            self.rel_level = False

        self.layer_norm = nn.LayerNorm(self.emb_dim)
        higher_transformer_layer = nn.TransformerEncoderLayer(d_model=self.emb_dim, nhead=options['higher_nhead'],
                                                              dim_feedforward=options['dim_feedforward'],
                                                              dropout=options['higher_drop'])
        self.higher_transformer = nn.TransformerEncoder(higher_transformer_layer, num_layers=options['higher_layers'],
                                                        norm=self.layer_norm)

        self.dropout = nn.Dropout(p=options['dropout'])

        self.pathfinder = nn.Sequential(
            nn.Linear(self.emb_dim, options['pathfinder_hidden']),
            nn.Tanh(),
            self.dropout,
            nn.Linear(options['pathfinder_hidden'], 1)
        )

        self.stopper = nn.Sequential(
            nn.Linear(self.emb_dim, options['stopper_hidden']),
            nn.Tanh(),
            self.dropout,
            nn.Linear(options['stopper_hidden'], 3)
        )

        self.higher_cls = nn.Parameter(torch.randn(1, self.emb_dim, requires_grad=True))
        self.register_buffer('higher_padding', torch.zeros(1, self.emb_dim))

        self.pathfinder_final = nn.Sigmoid()
        self.stopper_final = nn.Softmax(dim=-1)

        # About fitting score calculation
        self.root_forward = options['root_forward']
        self.leaf_backward = options['leaf_backward']

        self.fs_no_pathfinder = options.get('fs_no_pathfinder', False)
        self.fs_no_forward = options.get('fs_no_forward', False)
        self.fs_no_backward = options.get('fs_no_backward', False)

    def forward(self, x: torch.Tensor, abs_levels: torch.Tensor, rel_levels: torch.Tensor, segments: torch.Tensor,
                node_batch_lengths: torch.Tensor, tree_batch_mask: torch.Tensor, types: torch.Tensor):
        # x: [node_batch*tree_batch, seq_len]
        # levels, segments: [node_batch*tree_batch, 1]

        # lower: [node_batch*tree_batch, emb_dim]
        lower = self.get_lower_repr(x, tree_batch_mask, segments, types, rel_levels, abs_levels)

        node_batch_lengths_tuple = tuple(node_batch_lengths.tolist())
        higher_s = torch.split(lower, node_batch_lengths_tuple, dim=0)  # [tree_batch, emb_dim]
        higher_segments_s = [self.higher_segment_emb(l).squeeze() for l in
                             torch.split(segments, node_batch_lengths_tuple, dim=0)]
        if self.rel_level:
            higher_rel_levels_s = [self.higher_rel_level_emb(l).squeeze() for l in
                                   torch.split(rel_levels, node_batch_lengths_tuple, dim=0)]
        if self.abs_level:
            higher_abs_levels_s = [self.higher_abs_level_emb(l).squeeze() for l in
                                   torch.split(abs_levels, node_batch_lengths_tuple, dim=0)]

        higher_repr = self.padding(higher_s, node_batch_lengths, self.higher_cls)  # [node_batch, tree_batch, emb_dim]
        higher_repr += self.padding(higher_segments_s, node_batch_lengths, self.higher_padding)
        if self.rel_level:
            higher_repr += self.padding(higher_rel_levels_s, node_batch_lengths, self.higher_padding)
        if self.abs_level:
            higher_repr += self.padding(higher_abs_levels_s, node_batch_lengths, self.higher_padding)

        higher_repr.transpose_(0, 1)
        lengths = self.length_to_mask(node_batch_lengths + 2)  # + 2 * cls

        # higher: [node_batch, emb_dim]
        higher = self.higher_transformer(higher_repr, src_key_padding_mask=lengths)

        pathfinder_score = self.pathfinder(higher[0, :, :])
        stopper_score = self.stopper(higher[1, :, :])
        return pathfinder_score, stopper_score, self.pathfinder_final(pathfinder_score), self.stopper_final(
            stopper_score)

    def get_lower_repr(self, x, tree_batch_mask, segments, types, rel_levels, abs_levels):
        raise NotImplementedError

    @staticmethod
    def length_to_mask(lengths: torch.Tensor):
        max_len = lengths.max()
        return torch.arange(max_len.item()).to(lengths.device).unsqueeze(0) >= lengths.unsqueeze(1)

    def padding(self, lower_repr: List[torch.Tensor], lengths: torch.Tensor, init: torch.Tensor):
        max_len = lengths.max()

        higher_repr = [
            torch.cat([init, init, tree_batch, *[self.higher_padding] * (max_len - tree_batch.shape[0])],
                      0).unsqueeze_(0) for tree_batch in lower_repr]
        higher_repr = torch.cat(higher_repr, 0)
        return higher_repr

    def fitting_score(self, adj_matrix: csr_matrix, pathfinder_s: np.ndarray, forward_s: np.ndarray,
                      current_s: np.ndarray,
                      backward_s: np.ndarray) -> np.ndarray:
        # adj_matrix: [seed_num, seed_num]
        # score_matrix: [query_num, seed_num]
        parent_forward = forward_s * adj_matrix
        parent_forward[parent_forward == 0] = self.root_forward

        current_current = current_s

        child_backward_matrix_idx = adj_matrix.T.multiply(pathfinder_s[..., np.newaxis]).argmax(axis=1).squeeze()
        child_backward_matrix_mask = adj_matrix.multiply(np.eye(adj_matrix.shape[0])[child_backward_matrix_idx])
        child_backward = (child_backward_matrix_mask @ backward_s[..., np.newaxis]).squeeze()
        child_backward[child_backward == 0] = self.leaf_backward

        # sanity check of fitting score's components
        print('Sanity check in fitting score calculation:')
        print(
            f'Pathfinder Scores: max: {pathfinder_s.max()}, min: {pathfinder_s.min()}, mean: {pathfinder_s.mean()}, #NANs: {np.isnan(pathfinder_s).sum()}')
        print(
            f'Parent\'s Forward Scores: max: {parent_forward.max()}, min: {parent_forward.min()}, mean: {parent_forward.mean()}, #NANs: {np.isnan(parent_forward).sum()}')
        print(
            f'Current\'s Current Scores: max: {current_s.max()}, min: {current_s.min()}, mean: {current_s.mean()}, #NANs: {np.isnan(current_s).sum()}')
        print(
            f'Child with max Pathfinder Score\'s Backward Score: max: {child_backward.max()}, min: {child_backward.min()}, mean: {child_backward.mean()}, #NANs: {np.isnan(child_backward).sum()}')

        score_items = [current_current]
        if not self.fs_no_pathfinder:
            score_items.append(pathfinder_s)
        if not self.fs_no_forward:
            score_items.append(parent_forward)
        if not self.fs_no_backward:
            score_items.append(child_backward)

        return reduce(np.multiply, score_items), parent_forward, child_backward, child_backward_matrix_idx


class HEFTransformer(HEFBase):
    def __init__(self, emb: nn.Embedding, max_level, **options):
        super(HEFTransformer, self).__init__(max_level, **options)
        self.lower_token_emb = emb

        if options['const_positional_emb']:
            self.lower_positional_emb = ConstPositionalEncoding(self.emb_dim)
        else:
            self.lower_positional_emb = LearnablePositionalEncoding(self.emb_dim)

        if options['share_level_emb']:
            self.lower_segment_emb = self.higher_segment_emb
        else:
            self.lower_segment_emb = SegmentEncoding(self.emb_dim)

        if options['abs_level']:
            if options['share_level_emb']:
                self.lower_abs_level_emb = self.higher_abs_level_emb
            else:
                self.lower_abs_level_emb = LevelEncoding(max_level, self.emb_dim)

        if options['rel_level']:
            if options['share_level_emb']:
                self.lower_rel_level_emb = self.higher_rel_level_emb
            else:
                self.lower_rel_level_emb = LevelEncoding(max_level, self.emb_dim)

        lower_transformer_layer = nn.TransformerEncoderLayer(d_model=self.emb_dim, nhead=options['lower_nhead'],
                                                             dim_feedforward=options['dim_feedforward'],
                                                             dropout=options['lower_drop'])
        self.lower_transformer = nn.TransformerEncoder(lower_transformer_layer, num_layers=options['lower_layers'],
                                                       norm=self.layer_norm)

    def get_lower_repr(self, x, mask, segments, types, rel_levels, abs_levels):
        emb = self.lower_token_emb(x)  # [node_batch*tree_batch, seq_len, emb_dim]
        emb += self.lower_positional_emb(x)
        emb += self.lower_segment_emb(segments)
        if self.rel_level:
            emb += self.lower_rel_level_emb(rel_levels)
        if self.abs_level:
            emb += self.lower_abs_level_emb(abs_levels)
        emb = self.layer_norm(emb)
        emb.transpose_(0, 1)  # [seq_len, node_batch*tree_batch, emb_dim]
        mask = (mask == 0)

        # lower: [node_batch*tree_batch, emb_dim]
        return self.lower_transformer(emb, src_key_padding_mask=mask)[0, :, :]


class HEFBert(HEFBase):
    def __init__(self, emb: nn.Embedding, max_level, **options):
        super(HEFBert, self).__init__(max_level, **options)
        self.lower_bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        if options['init_layers'] > 0:
            self._re_initialize_top_layers(options['init_layers'])

    def get_lower_repr(self, x, mask, segments, types, rel_levels, abs_levels):
        # return self.lower_bert(input_ids=x, attention_mask=mask, token_type_ids=types, return_dict=True)[
        #            'last_hidden_state'][:, 0, :]
        return self.lower_bert(input_ids=x, attention_mask=mask, return_dict=True)['last_hidden_state'][:, 0, :]

    def _re_initialize_top_layers(self, init_layer_num):
        # reinitialize the top layers of BERT according to its original initialization.
        # this approach has been tested effective for fine-tuning on several tasks.
        # check https://arxiv.org/pdf/2006.05987.pdf
        bert_layer_num = self.lower_bert.config.num_hidden_layers
        for name, tensor in self.lower_bert.named_parameters():
            search = re.search('\d+', name)
            if search is not None and int(search.group(0)) in range(bert_layer_num - init_layer_num, bert_layer_num):
                if 'LayerNorm.weight' in name:
                    tensor.data.fill_(1.0)
                elif 'weight' in name:
                    tensor.data.normal_(mean=0, std=0.02)
                elif 'bias' in name:
                    tensor.data.zero_()

        # For ablation study
        if init_layer_num == bert_layer_num:
            self.lower_bert.embeddings.word_embeddings.weight.data.normal_(mean=0, std=1)

    def generate_decayed_param_list(self, initial_lr, init_layer, decay_rate, weight_decay):
        bert_layer = self.lower_bert.config.num_hidden_layers
        no_decay = ['bias', 'LayerNorm.weight']
        lrs = [{
            'params': [p for n, p in next(self.lower_bert.transformer.layer.modules())[x].named_parameters() if
                       any(nd in n for nd in no_decay)],
            'lr': initial_lr * (decay_rate ** max(0, bert_layer - init_layer - x)),
            'weight_decay': 0.0
        } for x in range(bert_layer)]
        lrs += [{
            'params': [p for n, p in next(self.lower_bert.transformer.layer.modules())[x].named_parameters() if
                       not any(nd in n for nd in no_decay)],
            'lr': initial_lr * (decay_rate ** max(0, bert_layer - init_layer - x)),
            'weight_decay': weight_decay
        } for x in range(bert_layer)]
        lrs += [{
            'params': [p for n, p in self.named_parameters() if
                       p.requires_grad and 'lower_bert' not in n and any(nd in n for nd in no_decay)],
            'lr': initial_lr,
            'weight_decay': 0.0
        }]
        lrs += [{
            'params': [p for n, p in self.named_parameters() if
                       p.requires_grad and 'lower_bert' not in n and not any(nd in n for nd in no_decay)],
            'lr': initial_lr,
            'weight_decay': weight_decay
        }]
        return lrs
