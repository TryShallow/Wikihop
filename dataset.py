import pickle
import os
import torch
import numpy as np
import scipy.sparse

from config import Config
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, config: Config, graph_file, elmo_file=None, glove_file=None,
                 bert_file=None, extra_file=None, is_test=0):
        self.graph_file = graph_file
        self.config = config
        self.max_nodes = config.max_nodes
        self.max_query_size = config.max_query_size
        self.max_candidates = config.max_candidates
        self.max_candidates_len = config.max_candidates_len
        self.use_elmo = config.use_elmo
        self.use_glove = config.use_glove
        self.use_bert = config.use_bert
        self.use_extra = config.use_extra

        if config.use_elmo:
            self.elmo_file = elmo_file
            self.data_elmo = None
        if config.use_glove:
            self.glove_file = glove_file
            self.data_glove = None
        if config.use_bert:
            self.bert_file = bert_file
            self.data_bert = None
        if config.use_extra:
            self.extra_file = extra_file
            self.data_extra = None
            self.pos_dict = None
            self.ner_dict = None

        self.data = None
        self.is_test = is_test
        if is_test == 0:
            self._init_data()
        else:
            self._init_data_test()

    def _init_data_test(self):
        self.graph_file = "data_gen/dev.json.preprocessed.pickle"
        self.elmo_file = "data_gen/dev.json.elmo.preprocessed.pickle"
        self.bert_file = "data_gen/dev.json.bert.pickle"
        self.glove_file = "data_gen/dev.json.glove.preprocessed.pickle"
        self.extra_file = "data_gen/dev.json.extra.preprocessed.pickle"
        with open(self.graph_file, 'rb') as f:
            self.data = [d for d in pickle.load(f) if len(d['nodes_candidates_id']) > 0]
            if self.is_test == 1:
                self.data = self.data[: 1000]
            elif self.is_test == 2:
                self.data = self.data[1000: 2000]
            print("Load graph data file:", self.graph_file, "len:", len(self.data))
        if self.use_elmo:
            with open(self.elmo_file, "rb") as f:
                self.data_elmo = [d for d in pickle.load(f) if len(d['nodes_elmo']) > 0]
                if self.is_test == 1:
                    self.data_elmo = self.data_elmo[: 1000]
                elif self.is_test == 2:
                    self.data_elmo = self.data_elmo[1000: 2000]
                print("Load elmo data file:", self.elmo_file, "len:", len(self.data_elmo))
        if self.use_glove:
            with open(self.glove_file, "rb") as f:
                self.data_glove = [d for d in pickle.load(f) if len(d['nodes_glove']) > 0]
                if self.is_test == 1:
                    self.data_glove = self.data_glove[: 1000]
                elif self.is_test == 2:
                    self.data_glove = self.data_glove[1000: 2000]
                print("Load glove data file:", self.glove_file, "len:", len(self.data_glove))
        if self.use_bert:
            with open(self.bert_file, "rb") as f:
                self.data_bert = [d for d in pickle.load(f) if len(d['nodes_bert']) > 0]
                if self.is_test == 1:
                    self.data_bert = self.data_bert[: 1000]
                elif self.is_test == 2:
                    self.data_bert = self.data_bert[1000: 2000]
                print("Load bert data file:", self.bert_file, "len:", len(self.data_bert))
        if self.use_extra:
            with open(self.extra_file, "rb") as f:
                self.data_extra = [d for d in pickle.load(f) if len(d['nodes_pos']) > 0]
                self.ner_dict = pickle.load(open('data/ner_dict.pickle', 'rb'))
                self.pos_dict = pickle.load(open('data/pos_dict.pickle', 'rb'))
                self.config.ner_dict_size = len(self.ner_dict)
                self.config.pos_dict_size = len(self.pos_dict)
                if self.is_test == 1:
                    self.data_extra = self.data_extra[: 1000]
                elif self.is_test == 2:
                    self.data_extra = self.data_extra[1000: 2000]
                print("Load extra data file:", self.extra_file, "len:", len(self.data_extra))

    def _init_data(self):
        assert os.path.isfile(self.graph_file) and os.path.isfile(self.elmo_file)
        with open(self.graph_file, 'rb') as f:
            self.data = [d for d in pickle.load(f) if len(d['nodes_candidates_id']) > 0]
        if self.use_elmo:
            with open(self.elmo_file, "rb") as f:
                self.data_elmo = [d for d in pickle.load(f) if len(d['nodes_elmo']) > 0]
        if self.use_glove:
            with open(self.glove_file, "rb") as f:
                self.data_glove = [d for d in pickle.load(f) if len(d['nodes_glove']) > 0]
        if self.use_bert:
            with open(self.bert_file, "rb") as f:
                self.data_bert = [d for d in pickle.load(f) if len(d['nodes_bert']) > 0]
        if self.use_extra:
            with open(self.extra_file, "rb") as f:
                self.data_extra = [d for d in pickle.load(f) if len(d['nodes_pos']) > 0]
                self.ner_dict = pickle.load(open('data/ner_dict.pickle', 'rb'))
                self.pos_dict = pickle.load(open('data/pos_dict.pickle', 'rb'))
                self.config.ner_dict_size = len(self.ner_dict)
                self.config.pos_dict_size = len(self.pos_dict)

    def __getitem__(self, index):
        item = dict()
        data_item = self.data[index]
        if self.use_elmo:
            data_elmo_item = self.data_elmo[index]
            # max_nodes*3*1024,   max_query_size*3*1024
            item["nodes_elmo"], item["query_elmo"] = self.build_elmo_data(data_elmo_item)
        if self.use_glove:
            data_glove_item = self.data_glove[index]
            # max_nodes*300,    max_query_size*300
            item["nodes_glove"], item["query_glove"] = self.build_glove_data(data_glove_item)
        if self.use_bert:
            data_bert_item = self.data_bert[index]
            # max_nodes*768,   max_query_size*768
            item["nodes_bert"], item["query_bert"] = self.build_bert_data(data_bert_item)
        if self.use_extra:
            data_extra_item = self.data_extra[index]
            item["nodes_ner"], item["nodes_pos"], item["query_ner"], item["query_pos"] = self.\
                build_extra_data(data_extra_item)
        nodes_length = self.truncate_nodes_and_edges(data_item)
        query_length = self.truncate_query(data_item)
        # 3*max_nodes*max_nodes
        adj = self.build_edge(data_item)
        mask = np.pad(
            np.array([(i == np.array(data_item["nodes_candidates_id"])).astype(np.uint8)
                      for i in range(len(data_item["candidates"]))]),
            ((0, self.max_candidates - len(data_item["candidates"]) - 1),
             (0, self.max_nodes - len(data_item["nodes_candidates_id"]))),
            mode="constant"
        )
        item["id"] = index
        item["nodes_length"] = nodes_length
        item["query_length"] = query_length
        item["adj"] = adj
        item["mask"] = mask
        item["answer_index"] = data_item["answer_candidate_id"]
        return item

    @staticmethod
    def to(batch, device):
        batch['query_length'] = batch['query_length'].to(device)
        batch['nodes_length'] = batch['nodes_length'].to(device)
        batch['adj'] = batch['adj'].type(torch.float32).to(device)
        batch['mask'] = batch['mask'].type(torch.float32).to(device)
        if batch.__contains__("nodes_elmo"):
            batch['nodes_elmo'] = batch['nodes_elmo'].to(device)
        if batch.__contains__("query_elmo"):
            batch['query_elmo'] = batch['query_elmo'].to(device)
        if batch.__contains__("nodes_bert"):
            batch['nodes_bert'] = batch['nodes_bert'].to(device)
        if batch.__contains__("query_bert"):
            batch['query_bert'] = batch['query_bert'].to(device)
        if batch.__contains__("nodes_glove"):
            batch['nodes_glove'] = batch['nodes_glove'].to(device)
        if batch.__contains__("query_glove"):
            batch['query_glove'] = batch['query_glove'].to(device)
        if batch.__contains__("nodes_ner"):
            batch['nodes_ner'] = batch['nodes_ner'].to(device)
        if batch.__contains__("nodes_pos"):
            batch['nodes_pos'] = batch['nodes_pos'].to(device)
        if batch.__contains__("query_ner"):
            batch['query_ner'] = batch['query_ner'].to(device)
        if batch.__contains__("query_pos"):
            batch['query_pos'] = batch['query_pos'].to(device)
        return batch

    def __len__(self):
        return len(self.data)

    def build_elmo_data(self, data_elmo_item):
        node_rep = lambda x: np.array([x[:, 0].mean(0), x[0, 1], x[-1, 2]])
        data_elmo_item['nodes_elmo'] = data_elmo_item['nodes_elmo'][: self.max_nodes]
        nodes_elmo = np.pad(np.array([node_rep(x) for x in data_elmo_item['nodes_elmo']]),
                            ((0, self.max_nodes - len(data_elmo_item['nodes_elmo'])),
                             (0, 0), (0, 0)), mode='constant').astype(np.float32)
        data_elmo_item['query_elmo'] = data_elmo_item['query_elmo'][: self.max_query_size]
        query_elmo = np.pad(data_elmo_item['query_elmo'],
                            ((0, self.max_query_size - data_elmo_item['query_elmo'].shape[0]),
                             (0, 0), (0, 0)), mode='constant').astype(np.float32)
        return nodes_elmo, query_elmo

    def build_glove_data(self, data_glove_item):
        node_rep = lambda x: np.array(x.mean(0))
        data_glove_item['nodes_glove'] = data_glove_item['nodes_glove'][: self.max_nodes]
        nodes_glove = np.pad(
            np.array([node_rep(x) for x in data_glove_item['nodes_glove']]),
            ((0, self.max_nodes - len(data_glove_item['nodes_glove'])),
             (0, 0)), mode='constant'
        ).astype(np.float32)
        data_glove_item['query_glove'] = data_glove_item['query_glove'][: self.max_query_size]
        query_glove = np.pad(
            data_glove_item['query_glove'],
            ((0, self.max_query_size - data_glove_item['query_glove'].shape[0]),
             (0, 0)), mode='constant'
        )
        return nodes_glove, query_glove

    def build_bert_data(self, data_bert_item):
        node_rep = lambda x: np.array(x.mean(0))
        data_bert_item['nodes_bert'] = data_bert_item['nodes_bert'][: self.max_nodes]
        nodes_bert = np.pad(
            np.array([node_rep(x) for x in data_bert_item['nodes_bert']]),
            ((0, self.max_nodes - len(data_bert_item['nodes_bert'])),
             (0, 0)), mode='constant'
        )
        data_bert_item['query_bert'] = data_bert_item['query_bert'][: self.max_query_size]
        query_bert = np.pad(
            data_bert_item['query_bert'],
            ((0, self.max_query_size - data_bert_item['query_bert'].shape[0]),
             (0, 0)), mode='constant'
        )
        return nodes_bert, query_bert

    def build_extra_data(self, data_extra_item):
        node_rep = lambda x: np.argmax(np.bincount(x))
        data_extra_item['nodes_ner'] = data_extra_item['nodes_ner'][: self.max_nodes]
        nodes_ner = np.pad(
            np.array([node_rep(x) for x in data_extra_item['nodes_ner']]),
            (0, self.max_nodes - len(data_extra_item['nodes_ner'])), mode='constant'
        )
        data_extra_item['nodes_pos'] = data_extra_item['nodes_pos'][: self.max_nodes]
        nodes_pos = np.pad(
            np.array([node_rep(x) for x in data_extra_item['nodes_pos']]),
            (0, self.max_nodes - len(data_extra_item['nodes_pos'])), mode='constant'
        )
        data_extra_item['query_ner'] = data_extra_item['query_ner'][: self.max_query_size]
        query_ner = np.pad(
            data_extra_item['query_ner'],
            (0, self.max_query_size - len(data_extra_item['query_ner'])), mode='constant'
        )
        data_extra_item['query_pos'] = data_extra_item['query_pos'][: self.max_query_size]
        query_pos = np.pad(
            data_extra_item['query_pos'],
            (0, self.max_query_size - len(data_extra_item['query_pos'])), mode='constant'
        )
        return nodes_ner, nodes_pos, query_ner, query_pos

    def truncate_nodes_and_edges(self, data):
        nodes_length = len(data['nodes_candidates_id'])
        is_exceed = nodes_length > self.max_nodes
        if is_exceed:
            data['edges_in'] = self.truncate_edges(data['edges_in'])
            data['edges_out'] = self.truncate_edges(data['edges_out'])
            data['nodes_candidates_id'] = data['nodes_candidates_id'][: self.max_nodes]

        return min(nodes_length, self.max_nodes)

    def truncate_edges(self, edges):
        truncated_edges = []
        for edge_pair in edges:
            if edge_pair[0] >= self.max_nodes:
                continue
            if edge_pair[1] < self.max_nodes:
                truncated_edges.append(edge_pair)
        return truncated_edges

    def truncate_query(self, data_item):
        query_length = len(data_item['query'])
        is_exceed = query_length > self.max_query_size
        if is_exceed:
            data_item['query'] = data_item['query'][: self.max_query_size]
        return min(query_length, self.max_query_size)

    def build_edge(self, data_item):
        len_nodes = len(data_item['nodes_candidates_id'])
        adj_ = []
        adj_.append(self._build_edge(data_item["edges_in"]))
        adj_.append(self._build_edge(data_item["edges_out"]))
        adj = np.pad(
            np.ones((len_nodes, len_nodes)), ((0, self.max_nodes - len_nodes),
                                              (0, self.max_nodes - len_nodes)), mode='constant'
        ) - adj_[0] - adj_[1] - np.pad(
            np.eye(len_nodes), ((0, self.max_nodes - len_nodes),
                                (0, self.max_nodes - len_nodes)), mode='constant'
        )
        adj_.append(np.clip(adj, 0, 1))
        adj = np.stack(adj_, 0)
        d_ = adj.sum(-1)
        d_[np.nonzero(d_)] **= -1
        adj = adj * np.expand_dims(d_, -1)
        return adj

    def _build_edge(self, edges):
        if len(edges) == 0:
            return np.zeros((self.max_nodes, self.max_nodes))
        else:
            return scipy.sparse.coo_matrix(
                (np.ones(len(edges)), np.array(edges).T), shape=(self.max_nodes, self.max_nodes)
            ).toarray()

