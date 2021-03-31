import heapq
import pathlib
import pickle
import random
import time
from math import ceil
from random import sample, shuffle
from typing import Dict

import networkx as nx
import spacy
import torch
from nltk.corpus import wordnet as wn
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import DistilBertTokenizer, DistilBertModel

from utils.util import InfiniteKIterator, get_level

nlp = spacy.load('en_core_web_md')


class Taxon(object):
    def __init__(self, tx_id, display_name, level=-1, parents=None, children=None):
        self.tx_id = tx_id
        self.display_name = display_name
        self.level = level  # typically assigned when the whole graph is constructed
        self.parents = parents if parents is not None else list()
        self.children = children if children is not None else list()
        self.description = ""
        self.desc_emb = None

    @staticmethod
    def _generate_single_description(word, root_desc: str = None, pos=None):
        synsets = wn.synsets(word, pos=pos)
        if len(synsets) == 0:
            return word
        for synset in synsets:
            if root_desc is None:
                if word in synset.name():
                    return synset.definition()
            else:
                nlp_root_desc = nlp(root_desc)
                try:
                    if nlp(best_desc).similarity(nlp_root_desc) < nlp(synset.definition()).similarity(nlp_root_desc):
                        best_desc = synset.definition()
                except NameError:
                    best_desc = synset.definition()
                return best_desc
        return synsets[0].definition()

    def generate_description(self, prefix='display_name', root_desc: str = None):
        # Algorithm 2 in the paper
        pos_dict = {
            # 'ADJ': wn.ADJ,
            'NOUN': wn.NOUN
        }
        synsets = wn.synsets(self.display_name.replace(' ', '_'), pos=wn.NOUN)
        description = ''
        if len(synsets) == 0:
            if ' ' in self.display_name:
                spacy_processed = nlp(self.display_name)
                words, poses = self._generate_wordnet_ngrams([token.text for token in spacy_processed],
                                                             [token.pos_ for token in spacy_processed])

                for word, pos in zip(words, poses):
                    description += f' {Taxon._generate_single_description(word, root_desc, pos_dict.get(pos, wn.NOUN))}'
            else:
                description = self.display_name
        else:
            if root_desc is None:
                description = synsets[0].definition()
            else:
                nlp_root_desc = nlp(root_desc)
                for synset in synsets:
                    try:
                        if nlp(best_desc).similarity(nlp_root_desc) < nlp(synset.definition()).similarity(
                                nlp_root_desc):
                            best_desc = synset.definition()
                    except NameError:
                        best_desc = synset.definition()
                description = best_desc
        if prefix is not None:
            description = getattr(self, prefix) + f' is {description}'
        self.description = description

    @staticmethod
    def _generate_wordnet_ngrams(str_list, pos_list):
        # DP for wordnet concept splitting
        def score_fn(x):
            return x ** 2 + 1  # to penalize shorter ngrams

        scores = [0] * (len(str_list) + 1)  # how many tokens exists
        begin_indices = [0] * len(str_list)
        for idx, word in enumerate(str_list):
            for begin_idx in range(idx + 1):
                new_score = score_fn(len(str_list[begin_idx:idx + 1])) if len(
                    wn.synsets('_'.join(str_list[begin_idx:idx + 1]))) > 0 else 1
                if new_score + scores[begin_idx] > scores[idx + 1]:
                    scores[idx + 1] = new_score + scores[begin_idx]
                    begin_indices[idx] = begin_idx
        ptr = len(str_list) - 1
        split = list()
        pos = list()
        while ptr != -1:
            split.append('_'.join(str_list[begin_indices[ptr]:ptr + 1]))
            pos.append(pos_list[ptr])
            ptr = begin_indices[ptr] - 1
        return split[::-1], pos[::-1]

    def add_parent(self, parent):
        if isinstance(parent, Taxon):
            self.parents.append(parent)
        else:
            self.parents += parent

    def add_child(self, child):
        if isinstance(child, Taxon):
            self.children.append(child)
        else:
            self.children += child

    def del_parent(self, parent):
        self.parents.remove(parent)

    def del_child(self, child):
        self.children.remove(child)

    def __str__(self):
        return f'Taxon {self.tx_id} (name: {self.display_name}, level: {self.level}, description: {self.description})'

    def __lt__(self, another_taxon):
        return self.level < another_taxon.level


class TaxonomyDataset(object):
    def __init__(self, name, data_dir, embed="bert", existing_partition=False, prune_to_tree=False, max_length=None):
        """ Raw dataset class for Taxonomy dataset

        Parameters
        ----------
        name : str
            taxonomy name
        data_dir : str
            path to dataset, if raw=True, this is the directory path to dataset, if raw=False, this is the pickle path
        embed : str
            suffix of embedding file name, by default ""
        existing_partition : bool, optional
            whether to use the existing train/validation/test partitions or randomly sample new ones, by default False
        prune_to_tree : bool, optional
            whether prune the dataset to a tree, setting to True will result in removing some of the redundant edges.
            by default False
        """
        self.name = name  # taxonomy name
        self.embed = embed
        self.existing_partition = existing_partition
        self.prune_to_tree = prune_to_tree
        self.max_length = max_length
        self.graph = nx.DiGraph()  # full graph, including masked train/val node indices, assigned in loading phase
        self.vocab = []  # from node_id to human-readable concept string
        self.train_node_tx_ids = []  # a list of train node_ids
        self.val_node_tx_ids = []  # a list of validation node_ids
        self.test_node_tx_ids = []  # a list of test node_ids
        data_path = pathlib.Path(data_dir)

        if self.embed == "":
            output_pickle_file_name = data_path / f'{self.name}.pickle.bin'
        else:
            output_pickle_file_name = data_path / f'{self.name}.{self.embed}.pickle.bin'

        if output_pickle_file_name.is_file():
            self._load_dataset_pickled(output_pickle_file_name)
        else:
            self._load_dataset_raw(data_dir)

    def _load_dataset_pickled(self, pickle_path):
        print('Loading pickled dataset...')
        with open(pickle_path, "rb") as fin:
            data = pickle.load(fin)
        self.name = data["name"]
        self.graph = data["graph"]
        self.vocab = data["vocab"]
        self.train_nodes = data["train_nodes"]
        self.val_nodes = data["val_nodes"]
        self.test_nodes = data["test_nodes"]
        self.emb = data["emb"]
        self.tokenizer = data["tokenizer"]
        self.max_level = data["max_level"]

    def _load_dataset_raw(self, dir_path):
        """ Load data from three separated files, generate train/validation/test partitions, and save to pickle.
        Please refer to the README.md file for details.
        Parameters
        ----------
        dir_path : str
            The path to a directory containing three input files.
        """
        print('Loading raw dataset...')
        data_path = pathlib.Path(dir_path)
        node_file_name = data_path / f'{self.name}.terms'
        edge_file_name = data_path / f'{self.name}.taxo'
        if self.embed == "":
            output_pickle_file_name = data_path / f'{self.name}.pickle.bin'
        else:
            output_pickle_file_name = data_path / f'{self.name}.{self.embed}.pickle.bin'

        if self.existing_partition:
            train_node_file_name = data_path / f'{self.name}.terms.train'
            validation_node_file_name = data_path / f'{self.name}.terms.validation'
            test_file_name = data_path / f'{self.name}.terms.test'

            print("Loading existing train/validation/test partitions")
            self.train_node_tx_ids = self._load_node_list(train_node_file_name)
            self.val_node_tx_ids = self._load_node_list(validation_node_file_name)
            self.test_node_tx_ids = self._load_node_list(test_file_name)

        self.tx_id2taxon = {}  # taxon id in data file to Taxon

        # load nodes
        with open(node_file_name, "r") as fin:
            for line in tqdm(fin, desc="Loading terms"):
                line = line.strip()
                if line:
                    segs = line.split("\t")
                    assert len(segs) == 2, f"Wrong number of segmentations {line}"
                    taxon = Taxon(tx_id=segs[0], display_name=segs[1])
                    self.tx_id2taxon[segs[0]] = taxon
                    self.graph.add_node(taxon)

        # load edges
        with open(edge_file_name, "r") as fin:
            for line in tqdm(fin, desc="Loading relations"):
                line = line.strip()
                if line:
                    segs = line.split("\t")
                    assert len(segs) == 2, f"Wrong number of segmentations {line}"
                    parent_taxon = self.tx_id2taxon[segs[0]]
                    child_taxon = self.tx_id2taxon[segs[1]]
                    self.tx_id2taxon[segs[0]].add_child(self.tx_id2taxon[segs[1]])
                    self.tx_id2taxon[segs[1]].add_parent(self.tx_id2taxon[segs[0]])
                    self.graph.add_edge(parent_taxon, child_taxon)

        # remove unconnected nodes
        unconnected = [node for node in self.graph.nodes if not node.parents and not node.children]
        self.graph.remove_nodes_from(unconnected)

        # get the level of nodes
        for node in tqdm(self.graph.nodes, desc="Calculating node level"):
            node.level = get_level(self.graph, node)

        # remove edges that forms cycle
        if self.prune_to_tree:
            pruned = False
            for node in self.graph.nodes:
                if self.graph.in_degree(node) > 1:
                    max_level = -1
                    for parent in node.parents:
                        if parent.level > max_level:
                            max_level = parent.level
                            save_parent = parent
                    for parent in node.parents:
                        if parent != save_parent:
                            self.graph.remove_edge(parent, node)
                            parent.del_child(node)
                    node.parents = [save_parent]
                    pruned = True

            if pruned:
                for node in self.graph.nodes:
                    node.level = -1
                for node in tqdm(self.graph.nodes, desc="Recalculating node level"):
                    node.level = get_level(self.graph, node)
            print(f'Pruned to tree: {nx.is_arborescence(self.graph)}')
            print(
                f'leaf / non-leafs: {len([node for node in self.graph.nodes if not node.children and len(node.parents[0].children) == 1]) / len(self.graph.nodes)}')

        # get root and max level
        for node in self.graph.nodes:
            if node.level == 0:
                self.root = node
        self.max_level = max([node.level for node in self.graph.nodes])

        # get node definitions
        self.root.generate_description()
        for node in tqdm(self.graph.nodes, desc='Generating node definitions'):
            if node != self.root:
                node.generate_description(root_desc=self.root.description)

        # construct vocab and embeddings
        print('Constructing vocab and embeddings...')
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.emb = DistilBertModel.from_pretrained('distilbert-base-uncased').get_input_embeddings()

        # generate validation/test node_indices using either existing partitions or randomly sampled partition
        print('Generating train/val/test splits...')
        if self.existing_partition:
            self.val_nodes = [self.tx_id2taxon[idx] for idx in self.val_node_tx_ids]
            self.test_nodes = [self.tx_id2taxon[idx] for idx in self.test_node_tx_ids]
            self.train_nodes = [self.tx_id2taxon[idx] for idx in self.train_node_tx_ids]
        else:
            leaf_nodes = [node for node in self.graph.nodes if self.graph.out_degree(node) == 0]

            random.seed(42)
            random.shuffle(leaf_nodes)
            test_size = min(len(leaf_nodes), ceil(0.2 * len(self.graph.nodes)))
            validation_size = min(len(leaf_nodes) - test_size, 10)
            test_size -= validation_size

            self.val_nodes = leaf_nodes[:validation_size]
            self.test_nodes = leaf_nodes[validation_size:(validation_size + test_size)]
            self.train_nodes = [node for node in self.graph.nodes if
                                node not in self.val_nodes and node not in self.test_nodes]

        # save to pickle for faster loading next time
        print("start saving pickle data...")
        with open(output_pickle_file_name, 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            data = {
                "name": self.name,
                "graph": self.graph,
                "vocab": self.vocab,
                "train_nodes": self.train_nodes,
                "val_nodes": self.val_nodes,
                "test_nodes": self.test_nodes,
                "emb": self.emb,
                "tokenizer": self.tokenizer,
                "max_level": self.max_level
            }
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print(f"Save pickled dataset to {output_pickle_file_name}")

    @staticmethod
    def _load_node_list(file_path):
        node_list = []
        with open(file_path, "r") as fin:
            for line in fin:
                line = line.strip()
                if line:
                    node_list.append(line)
        return node_list


class TreeMatchingDataset(Dataset):
    """ Dataset for generating (matching_seq, path_score, stopper_score) tuples."""

    def __init__(self, graph_dataset: TaxonomyDataset, mode="train", exact_total_size=True, pos_current_num=-1,
                 pos_backward_num=4, pos_forward_num=-1, min_neg_num=20, total_size=32, expand_factor=3, cross=False,
                 neg_path="backward", top_k_similarity=True, abl_use_egonet=False, abl_name_only=False):

        assert mode in ['train', 'val_full', 'test_full',
                        'test_hierarchical'], 'mode must be train, val_full, test_full or test_hierarchical'
        assert sum([n for n in [pos_current_num, pos_backward_num, pos_forward_num, min_neg_num] if
                    n != -1]) <= total_size, 'sum of assigned num for each split should <= total_size'
        assert neg_path in ['forward', 'current', 'backward',
                            'random'], 'neg_path must be forward/current/backward/random'

        print(f'Constructing TreeMatchingDataset for mode {mode}...')
        start = time.time()
        self.mode = mode
        self.full_graph = graph_dataset
        self.exact_total_size = exact_total_size  # Whether the total_size is strictly
        self.pos_current_num = pos_current_num  # Ground truth node
        self.pos_backward_num = pos_backward_num  # Descendant nodes of ground truth
        self.pos_forward_num = pos_forward_num  # Ascendant nods of ground truth
        self.min_neg_num = min_neg_num  # Minimum negative samples
        self.total_size = total_size  # total_size = pos_current + pos_backward + pos_forward + neg_backward
        self.expand_factor = expand_factor  # How many potential siblings are contained in subtree
        self.cross = cross  # Whether apply cross-attention in lower transformer
        self.neg_path = neg_path  # Whether stopper tag for negative path score is fixed with "backward"
        self.top_k_similarity = top_k_similarity  # Whether potential siblings are chosen according to similarity
        self.abl_use_egonet = abl_use_egonet  # for ablation study
        self.abl_name_only = abl_name_only  # for ablation study

        self.spacy = spacy.load('en_core_web_md')  # For calculating similarity in subtree construction

        # Construct subgraph for corresponding mode
        self.node_list = getattr(self.full_graph, f'{mode.split("_")[0]}_nodes').copy()  # Potential query nodes
        self.graph = self.full_graph.graph.subgraph(list(set(graph_dataset.train_nodes + self.node_list))).copy()

        # No training for connecting root node(s)
        self.node_list = [node for node in self.node_list if self.graph.in_degree(node) > 0]

        # Generate scope for 4 kinds of training data of each node
        self.pos_current, self.pos_forward, self.pos_backward, self.neg = dict(), dict(), dict(), dict()
        for query in self.node_list:
            # pos_current contains query's direct parent(s)
            self.pos_current[query] = query.parents

            # pos_forward contains anchor's ascendants
            for parent in query.parents:
                ascendants = [node for node in self.graph.nodes if nx.has_path(self.graph, node, parent)]
                try:
                    self.pos_forward[query] += ascendants
                except KeyError:
                    self.pos_forward[query] = ascendants

            # pos_backward contains anchor's descendants
            for parent in query.parents:
                descendants = [node for node in self.graph.nodes if nx.has_path(self.graph, parent, node)]
                try:
                    self.pos_backward[query] += descendants
                except KeyError:
                    self.pos_backward[query] = descendants

            # neg contains other nodes
            self.neg[query] = [node for node in self.graph.nodes if (
                    node not in self.pos_current[query] and
                    node not in self.pos_forward[query] and
                    node not in self.pos_backward[query])]

        # Initialize infinite iterators for each node for creating batch
        self.pos_current_iter, self.pos_forward_iter, self.pos_backward_iter, self.neg_iter = dict(), dict(), dict(), dict()

        # Remove edges towards validation or test nodes
        remove_edges = [(u, v) for u, v in self.graph.edges if mode != 'train' and v in self.node_list]
        for u, v in remove_edges:
            u.del_child(v)
        self.graph.remove_edges_from(remove_edges)

        # Initialize cache for subtree construction
        self.ascendants_cache, self.descendents_cache = dict(), dict()

        # Initialize subtree seq cache
        self.seq_cache = dict()

        # Create seed taxonomy for constructing adj matrix
        if self.mode != 'train':
            self.seed = self.graph.subgraph([node for node in self.graph.nodes if node not in self.node_list])
        else:
            self.seed = self.graph
        self.seed_nodes = list(self.seed.nodes)

        self.adj_matrix = nx.adj_matrix(self.seed)

        print(f'Finish constructing dataset for {self.mode} in {time.time() - start}s.')

    def __len__(self):
        if self.mode == 'train':  # return tree-batch with tree_batch subtree-query pairs
            return len(self.node_list)
        elif self.mode in ['val_full', 'test_full']:  # return single subtree-query pair, batching in DataLoader
            return len(self.node_list) * len(self.seed.nodes)

    def __getitem__(self, index):
        """

        In train mode, __getitem__() returns a batch of subtree-query pairs with query fixed in all pairs.
        Instances returned are split according to the pos_current, pos_backward, pos_forward, neg nums in param.
        Such instances form a batch called 'tree batch' in other modules, to distinguish from 'node batch' formed
        in dataloader, where each 'instance' is a tree batch with different query node.

        In val_full or test_full mode, HEF needs to compare all nodes in seed taxonomy for fitting score rankings.
        In this mode, __getitem__() only returns a single subtree-query pair, and batches are formed in dataloader.
        The splits and batch size set in dataset param are not considered. __getitem__ will return subtree-query pairs
        in dictionary order of (query, anchor). This order is for future calculation of fitting scores in trainer.

        """
        if self.mode == 'train':
            query = self.node_list[index]
            seed_nodes = list(self.seed.nodes)
            ret = list()
            samples_idx = list()
            # When calling __getitem__, the function returns total_size instances if self.exact_total_size
            # Each instance is [texts, abs_levels, rel_levels, segments, path_score, stopper_score]

            # Calculate exact split for current query node
            if self.exact_total_size:
                if self.min_neg_num > -1:
                    pos_remaining = self.total_size - self.min_neg_num
                else:
                    pos_remaining = self.total_size
            else:
                pos_remaining = 2 * self.total_size

            if pos_remaining > 0:
                if self.pos_current_num < 0:
                    pos_current_num = min([len(self.pos_current[query]), pos_remaining])
                else:
                    pos_current_num = min([len(self.pos_current[query]), pos_remaining, self.pos_current_num])
                pos_remaining -= pos_current_num
            else:
                pos_current_num = 0
                print('Error where max count of instances with positive path score is initialized as 0')

            if pos_remaining > 0:
                if self.pos_forward_num < 0:
                    pos_forward_num = min([len(self.pos_forward[query]), pos_remaining])
                else:
                    pos_forward_num = min([len(self.pos_forward[query]), pos_remaining, self.pos_forward_num])
                pos_remaining -= pos_forward_num
            else:
                pos_forward_num = 0
                print(f'Dataset getitem stops after getting pos_current. Query: {query}')

            if pos_remaining > 0:
                if self.pos_backward_num < 0:
                    pos_backward_num = min([len(self.pos_backward[query]), pos_remaining])
                else:
                    pos_backward_num = min([len(self.pos_backward[query]), pos_remaining, self.pos_backward_num])
                pos_remaining -= pos_backward_num
            else:
                pos_backward_num = 0
                print(f'Dataset getitem stops after getting pos_current and pos_forward. Query: {query}')

            if self.exact_total_size:
                neg_num = pos_remaining + self.min_neg_num
            else:
                neg_num = self.min_neg_num

            # Generate pos_current instances. path score = 1, stopper score = 1
            tag = [1, 1]
            anchors = self._sample_anchors(query, pos_current_num, self.pos_current, self.pos_current_iter)
            ret += [self._generate_subtree_sequences(anchor, query) + tag for anchor in anchors]

            # Generate pos_forward instances. path score = 1, stopper score = 0
            tag = [1, 0]
            anchors = self._sample_anchors(query, pos_forward_num, self.pos_forward, self.pos_forward_iter)
            ret += [self._generate_subtree_sequences(anchor, query) + tag for anchor in anchors]

            # Generate pos_backward instances. path score = 1, stopper score = 2
            tag = [1, 2]
            anchors = self._sample_anchors(query, pos_backward_num, self.pos_backward, self.pos_backward_iter)
            ret += [self._generate_subtree_sequences(anchor, query) + tag for anchor in anchors]

            # Generate neg instances. path score = 0, stopper score = 2 or random.
            anchors = self._sample_anchors(query, neg_num, self.neg, self.neg_iter)
            if self.neg_path == 'random':
                ret += [self._generate_subtree_sequences(anchor, query) + [0, random.randint(0, 2)] for anchor in
                        anchors]
            else:
                path_tag = ['forward', 'current', 'backward'].index(self.neg_path)
                ret += [self._generate_subtree_sequences(anchor, query) + [0, path_tag] for anchor in anchors]
            shuffle(ret)
            return tuple(ret)

        elif self.mode in ['val_full', 'test_full']:
            query = self.node_list[index // len(self.seed.nodes)]
            anchor = self.seed_nodes[index % len(self.seed.nodes)]
            return (self._generate_subtree_sequences(anchor, query) + [int(anchor in self.pos_current[query])])

        else:  # self.mode == 'test_hierarchical'
            raise NotImplementedError

    @staticmethod
    def _sample_anchors(query: Taxon, anchor_num: int, total_anchors: Dict, total_anchors_iter: Dict):
        if 0 < anchor_num < len(total_anchors[query]):
            try:
                anchors = total_anchors_iter[query].get_k_item(anchor_num)
            except KeyError:
                total_anchors_iter[query] = InfiniteKIterator(total_anchors[query])
                anchors = total_anchors_iter[query].get_k_item(anchor_num)
        else:
            anchors = total_anchors[query]
        return anchors

    def _get_subtree(self, anchor: Taxon, query: Taxon):
        # Add anchor
        subtree_nodes = [anchor]

        # Add all ascendants
        try:
            subtree_nodes += self.ascendants_cache[anchor]
        except KeyError:
            if self.abl_use_egonet:
                ascendants = anchor.parents
            else:
                pointer = anchor
                ascendants = list()
                while len(pointer.parents) != 0:
                    ascendants += pointer.parents
                    pointer = pointer.parents[0]
            subtree_nodes += ascendants
            self.ascendants_cache[anchor] = ascendants

        # Add one layer of children
        if len(anchor.children) <= self.expand_factor or self.expand_factor == -1:
            subtree_nodes += anchor.children
        elif self.expand_factor > 0:
            # Choose self.expand_factor children nodes
            if self.top_k_similarity:
                try:
                    subtree_nodes += self.descendents_cache[(anchor, query)]
                except KeyError:
                    # Choose top k children according to display name average cosine similarity
                    query_vec = self.spacy(query.display_name)
                    similarity = [(query_vec.similarity(self.spacy(child.display_name)), child) for child in
                                  anchor.children]
                    heapq.heapify(similarity)
                    descendents = [child for _, child in heapq.nlargest(self.expand_factor, similarity)]
                    subtree_nodes += descendents
                    self.descendents_cache[(anchor, query)] = descendents
            else:
                # Sample children randomly
                subtree_nodes += [c for c in sample(query.children, self.expand_factor)]

        return nx.subgraph(self.full_graph.graph, subtree_nodes)

    def _generate_subtree_sequences(self, anchor: Taxon, query: Taxon):
        try:
            ret = self.seq_cache[(anchor, query)]
        except KeyError:
            attr = 'display_name' if self.abl_name_only else 'description'
            subtree = self._get_subtree(anchor, query)
            query_level = anchor.level + 1
            node_seq = sorted(list(subtree.nodes), key=lambda node: node.level)

            description_seq = [getattr(node, attr) for node in node_seq]
            description_seq.append(getattr(query, attr))

            description_seq = [description_seq]

            if self.cross:
                description_seq.append([getattr(query, attr)] * (len(node_seq) + 1))

            abs_level_seq = [[node.level] for node in node_seq]
            abs_level_seq.append([query_level])

            rel_level_seq = [[query_level - node.level] for node in node_seq]  # rel_lv > 0
            rel_level_seq.append([0])

            segment_seq = [[0] for _ in node_seq]
            anchor_idx = node_seq.index(anchor)
            segment_seq[anchor_idx] = [1]
            segment_seq.append([2])

            x = description_seq  # List of tokenized descriptions. Padding & numericalize in dataloader.
            abs_level_seq = torch.tensor(abs_level_seq)  # [tree_batch, 1]
            rel_level_seq = torch.tensor(rel_level_seq)  # [tree_batch, 1]
            segment_seq = torch.tensor(segment_seq)  # [tree_batch, 1]

            ret = [x, abs_level_seq, rel_level_seq, segment_seq]
            self.seq_cache[(anchor, query)] = ret

        return ret
