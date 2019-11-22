import pickle
import torch
import os
import time
import attention
import numpy as np
import scipy.sparse
import torch.nn as nn
import logging

from torch import optim
from torch.utils.data import Dataset, DataLoader


class Config(object):
    def __init__(self):
        self.device = "cuda:3"
        # self.device_ids = [0, 3]
        # self.device = "cpu"
        self.map_location = None  # {'cuda:2': 'cuda:3'}

        self.use_elmo = True
        # self.use_elmo = False
        self.use_bert = False

        self.max_query_size = 25
        self.max_nodes = 500
        self.max_candidates = 80
        self.max_candidates_len = 10

        self.query_encoding_type = "linear"  # "lstm"
        self.encoding_size = 512
        self.dropout = 0.8
        self.hops = 5

        self.contains_query_node = False
        self.add_query_self_att = False

        self.batch_size = 16
        self.num_epochs = 5000
        self.model_path = "model_saved"
        self.learning_rate = 1e-4
        self.momentum = 0.9

        # self.model_prefix = "wiki_mha_"
        # self.model_prefix = "wiki_bert_"
        # self.model_prefix = "wiki_bert_satt_"
        # self.model_prefix = "wiki_bert_mha"
        self.model_prefix = "wiki_org_"
        self.add_candidates_multi_att = False
        # self.add_candidates_multi_att = False
        self.add_candidates_self_att = False
        # self.add_candidates_self_att = True
        self.pretrained_parameters = None  # "wiki_org_0.6042113472411776"


class MyDataset(Dataset):
    def __init__(self, config, graph_file, elmo_file=None, bert_file=None, is_test=0):
        self.graph_file = graph_file
        self.config = config
        self.max_nodes = config.max_nodes
        self.max_query_size = config.max_query_size
        self.max_candidates = config.max_candidates
        self.max_candidates_len = config.max_candidates_len
        self.use_elmo = config.use_elmo
        self.use_bert = config.use_bert

        if config.use_elmo:
            self.elmo_file = elmo_file
            self.data_elmo = []
        if config.use_bert:
            self.bert_file = bert_file
            self.data_bert = []

        self.data = []
        self.is_test = is_test
        if is_test == 0:
            self._init_data()
        else:
            self._init_data_test()

    def _init_data_test(self):
        self.graph_file = "data_gen/train.json.preprocessed.pickle"
        self.elmo_file = "data_gen/train.json.elmo.preprocessed.pickle"
        self.bert_file = "data_gen/train.json.bert.pickle"
        self.glove_file = "data_gen/train.json.glove.preprocessed.pickle"
        # self.graph_file = "data_gen/train.json.graph_bert.pickle"
        # self.elmo_file = "data_gen/train.json.elmo_bert.pickle"
        # self.bert_file = "data_gen/train.json.bert.pickle"
        # self.graph_file = "data_gen/train.json.graph.pickle"
        # self.elmo_file = "data_gen/train.json.elmo.pickle"
        if self.use_bert:
            with open(self.bert_file, "rb") as f:
                self.data_bert = [d for d in pickle.load(f) if len(d['nodes_bert']) > 0]
                if self.is_test == 1:
                    self.data_bert = self.data_bert[: 1000]
                elif self.is_test == 2:
                    self.data_bert = self.data_bert[1000: 2000]
                print("Load bert data file:", self.bert_file, "len:", len(self.data_bert))
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

    def _init_data(self):
        assert os.path.isfile(self.graph_file) and os.path.isfile(self.elmo_file)
        with open(self.graph_file, 'rb') as f:
            self.data = [d for d in pickle.load(f) if len(d['nodes']) > 0]
            # self.data = self.data[:1000]
            # print("Load graph data file:", self.graph_file, "len:", len(self.data))
        if self.use_elmo:
            with open(self.elmo_file, "rb") as f:
                self.data_elmo = [d for d in pickle.load(f) if len(d['nodes_elmo']) > 0]
            # self.data_elmo = self.data_elmo[:1000]
            # print("Load elmo data file:", self.elmo_file, "len:", len(self.data_elmo))
        if self.use_bert:
            with open(self.bert_file, "rb") as f:
                self.data_bert = [d for d in pickle.load(f) if len(d['nodes_bert']) > 0]

    def __getitem__(self, index):
        item = dict()
        data_item = self.data[index]
        # nodes_elmo, nodes_bert, query_elmo, query_bert = None, None, None, None
        if self.use_elmo:
            data_elmo_item = self.data_elmo[index]
            # max_nodes*3*1024,   max_query_size*3*1024
            nodes_elmo, query_elmo = self.build_elmo_data(data_elmo_item)
            item["nodes_elmo"] = nodes_elmo
            item["query_elmo"] = query_elmo
        if self.use_bert:
            data_bert_item = self.data_bert[index]
            # max_nodes*768,   max_query_size*768
            nodes_bert, query_bert = self.build_bert_data(data_bert_item)
            item["nodes_bert"] = nodes_bert
            item["query_bert"] = query_bert
        nodes_length = self.truncate_nodes_and_edges(data_item)
        query_length = self.truncate_query(data_item)
        # 3*max_nodes*max_nodes
        adj = self.build_edge(data_item)
        if self.config.contains_query_node:
            mask = np.pad(
                np.array([(i == np.array(data_item["nodes_candidates_id"])).astype(np.uint8)
                          for i in range(- data_item["query_node_count"], len(data_item["candidates"]))]),
                ((0, self.max_candidates - len(data_item["candidates"]) - data_item["query_node_count"]),
                 (0, self.max_nodes - len(data_item["nodes_candidates_id"]))),
                mode="constant"
            )
        else:
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
        # return {
        #     "id": index,
        #     "nodes_elmo": nodes_elmo,
        #     "query_elmo": query_elmo,
        #     "nodes_bert": nodes_bert,
        #     "query_bert": query_bert,
        #     "nodes_length": nodes_length,
        #     "query_length": query_length,
        #     "adj": adj,
        #     "mask": mask,
        #     "answer_index": data_item["answer_index"],
        # }

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
        # batch['answer_index'] = batch['answer_index'].to(device)
        return batch

    def __len__(self):
        return len(self.data)

    def build_elmo_data(self, data_elmo_item):
        # node_rep = lambda x: x.mean(0)
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

    def truncate_nodes_and_edges(self, data):
        nodes_length = len(data['nodes_candidates_id'])
        is_exceed = nodes_length > self.max_nodes
        if is_exceed:
            data['edges_in'] = self.truncate_edges(data['edges_in'])
            data['edges_out'] = self.truncate_edges(data['edges_out'])
            data['nodes_candidates_id'] = data['nodes_candidates_id'][: self.max_nodes]
            # data['nodes'] = data['nodes'][: self.max_nodes]

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


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.use_elmo = config.use_elmo
        self.use_bert = config.use_bert

        # feature layer
        input_size = 0
        if config.use_elmo:
            input_size += (3 * 1024)
        if config.use_bert:
            input_size += 768
        self.query_linear = nn.Linear(input_size, config.encoding_size)
        self.query_lstm = nn.LSTM(input_size, config.encoding_size // 2, 2, bidirectional=True, batch_first=True)
        self.nodes_linear = nn.Linear(input_size, config.encoding_size)
        # gcn layer
        self.nodes_dropout = nn.Dropout(self.config.dropout)
        self.hidden_linears = nn.ModuleList([nn.Linear(512, 512)] * 4)
        self.combined_linear = nn.Linear(1024, 512)
        # bi_attention layer
        self.attention_linear = nn.Linear(512 * 3, 1, False)
        self.mha = attention.MultiHeadAttention(config.encoding_size, 8)
        # output layer
        self.out_att1 = nn.Linear(2048, 128)
        self.out_att2 = nn.Linear(128, 1)

    def forward(self, x):
        nodes_length = x['nodes_length']
        query_length = x['query_length']
        adj = x['adj'].type(torch.float32)
        mask = x['mask']
        # if self.use_elmo:
        #     nodes_elmo = x['nodes_elmo']
        #     query_elmo = x['query_elmo']
        # if self.use_bert:
        #     nodes_bert = x['nodes_bert']
        #     query_bert = x['query_bert']
        # [batch_size, max_nodes, 512]  [batch_size, max_query_size, 512]
        nodes_compress, query_compress = self.feature_layer(x)
        # nodes_compress, query_compress = self.feature_layer(nodes_elmo, query_elmo,
        #                                                     nodes_bert, query_bert)
        # nodes_mask = torch.arange(self.config.max_nodes).to(self.config.device).unsqueeze(0)
        # nodes_mask = nodes_mask.repeat((nodes_length.size(0), 1)) < nodes_length.unsqueeze(-1)
        # nodes_mask = nodes_mask.type(torch.float32).unsqueeze(-1)
        nodes_mask = self.build_mask(self.config.max_nodes, nodes_length)
        nodes = nodes_compress * nodes_mask
        nodes = self.nodes_dropout(nodes)
        if self.config.add_candidates_self_att:
            nodes = attention.attention(nodes, nodes, nodes)
        elif self.config.add_candidates_multi_att:
            nodes = self.mha(nodes, nodes, nodes)
        if self.config.add_query_self_att:
            # query_mask = self.build_mask(self.config.max_query_size, query_length)
            query_compress = attention.attention(query_compress, query_compress, query_compress)
        last_hop = nodes
        for _ in range(self.config.hops):
            last_hop = self.gcn_layer(adj, last_hop, nodes_mask)
        bi_attention = self.bi_attention_layer(query_compress, nodes_compress, last_hop)
        return self.output_layer(bi_attention, mask)

    def feature_layer(self, x):  # nodes_elmo=None, query_elmo=None, nodes_bert=None, query_bert=None):
        query_flat, nodes_flat = None, None
        if self.use_elmo:
            query_flat = torch.reshape(x["query_elmo"], (-1, self.config.max_query_size, 3 * 1024))
            nodes_flat = torch.reshape(x["nodes_elmo"], (-1, self.config.max_nodes, 3 * 1024))
        if self.use_bert:
            query_flat = x["query_bert"] if query_flat is None \
                else torch.cat([query_flat, x["query_bert"]], -1)
            nodes_flat = x["nodes_bert"] if nodes_flat is None \
                else torch.cat([nodes_flat, x["nodes_bert"]], -1)
        if self.config.query_encoding_type == 'lstm':
            query_compress, (h1, c1) = self.query_lstm(query_flat)
        elif self.config.query_encoding_type == 'linear':
            query_compress = self.query_linear(query_flat)
        nodes_compress = torch.tanh(self.nodes_linear(nodes_flat))
        return nodes_compress, query_compress

    def build_mask(self, max_nodes, nodes_length):
        nodes_mask = torch.arange(max_nodes).to(self.config.device).unsqueeze(0)
        nodes_mask = nodes_mask.repeat((nodes_length.size(0), 1)) < nodes_length.unsqueeze(-1)
        nodes_mask = nodes_mask.type(torch.float32).unsqueeze(-1)
        return nodes_mask

    def gcn_layer(self, adj, hidden_tensor, hidden_mask):
        adjacency_tensor = adj
        # [batch_size, 3, max_nodes, max_nodes]
        hidden_tensors = torch.stack([
            self.hidden_linears[i](hidden_tensor) for i in range(adj.size(1))
        ], 1) * hidden_mask.unsqueeze(1)
        # hidden_tensors = torch.stack([
        #     nn.Linear(hidden_tensor.size(-1), hidden_tensor.size(-1))(hidden_tensor) for _ in range(adj.size(1))
        # ], 1) * hidden_mask.unsqueeze(1)

        update = torch.sum(torch.matmul(adjacency_tensor, hidden_tensors), 1) +\
            self.hidden_linears[adj.size(1)](hidden_tensor) * hidden_mask
        update_combined = torch.cat((update, hidden_tensor), -1)
        att = torch.sigmoid(self.combined_linear(update_combined)) * hidden_mask
        return att * torch.tanh(update) + (1 - att) * hidden_tensor

    def bi_attention_layer(self, query_compress, nodes_compress, last_hop):
        expanded_query = query_compress.unsqueeze(-3).repeat((1, self.config.max_nodes, 1, 1))
        expanded_nodes = last_hop.unsqueeze(-2).repeat((1, 1, self.config.max_query_size, 1))
        context_query_similarity = expanded_nodes * expanded_query
        # [batch_size, max_nodes, max_query, d * 3]
        concat_attention_data = torch.cat((expanded_nodes, expanded_query, context_query_similarity), -1)
        # [batch_size, max_nodes, max_query]
        similarity = torch.mean(self.attention_linear(concat_attention_data), -1)
        # [batch_size, max_nodes, d]
        nodes2query = torch.matmul(nn.Softmax(-1)(similarity), query_compress)
        # [batch_size, max_nodes]
        b = nn.Softmax(-1)(similarity.max(-1)[0])
        # [batch_size, max_nodes, d]
        query2nodes = torch.matmul(b.unsqueeze(1), nodes_compress).repeat((1, self.config.max_nodes, 1))
        g = torch.cat((nodes_compress, nodes2query, nodes_compress * nodes2query, nodes_compress * query2nodes), -1)
        return g

    def output_layer(self, bi_attention, mask):
        raw_predictions = torch.tanh(self.out_att1(bi_attention))
        raw_predictions = self.out_att2(raw_predictions).squeeze(-1)
        predictions2 = mask.type(torch.float32) * raw_predictions.unsqueeze(1)
        predictions2[predictions2 == 0] = - np.inf
        predictions2 = predictions2.max(-1)[0]
        return predictions2


class Trainer(object):
    def __init__(self, train_data, dev_data, logger, config):
        self.logger = logger
        self.config = config

        self.batch_size = config.batch_size
        self.num_epochs = config.num_epochs
        self.model_path = config.model_path
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        self.train_data_loader = DataLoader(train_data, self.batch_size)
        self.dev_data_loader = DataLoader(dev_data, self.batch_size)
        self.model = Model(config)
        self.load_parameters(self.config.pretrained_parameters)
        self.model.to(config.device)

        # self.optimizer = optim.SGD(self.model.parameters(), lr=config.learning_rate, momentum=config.momentum)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)

    def load_parameters(self, model_file=None):
        if model_file is not None:
            self.model.load_state_dict(torch.load(self.model_path + '/' + model_file,
                                                  map_location=self.config.map_location))
            self.logger.info('Load parameters file ' + model_file)

    def train(self):
        self.model.train()
        best_acc = 0.0
        save_model_prefix = os.path.join(self.model_path, self.config.model_prefix)
        for epoch in range(self.num_epochs):
            self.logger.info("Epoch %d/%d" % (epoch + 1, self.num_epochs))
            start_time = time.time()
            for batch in self.train_data_loader:
                output = self.model(MyDataset.to(batch, self.config.device))
                self.model.zero_grad()
                loss = self._calc_loss(output, batch)
                loss.backward()
                self.optimizer.step()

            time_diff = time.time() - start_time
            self.logger.info("epoch %d time consumed: %dm%ds." % (epoch + 1, time_diff // 60, time_diff % 60))

            # evaluate model
            cur_acc = self.eval_dev(self.dev_data_loader)
            self.model.train()
            self.logger.info("Current accuracy: %.3f" % cur_acc)
            if cur_acc > best_acc:  # and epoch > 10:
                save_filename = save_model_prefix + str(cur_acc)
                torch.save(self.model.state_dict(), save_filename)
                best_acc = cur_acc

    def eval(self):
        self.model.eval()

    def eval_dev(self, dev_data_loader):
        self.model.eval()
        correct_count = 0
        total_count = 0
        for batch in dev_data_loader:
            output = self.model(MyDataset.to(batch, self.config.device))
            pred = torch.argmax(output, 1)
            correct_count += (pred.cpu().detach().numpy() == batch['answer_index'].numpy()).sum()
            total_count += len(batch['query_length'])
        return float(correct_count) / total_count

    def _calc_loss(self, output, batch):
        cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        return cross_entropy(output, batch['answer_index'].to(self.config.device))


def config_logger(log_prefix):
    logger_prepared = logging.getLogger()
    logger_prepared.setLevel(logging.INFO)
    # logger_prepared.setLevel(logging.ERROR)
    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(pathname)s[line:%(lineno)d]: %(message)s')
    # write to terminal
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(log_format)
    logger_prepared.addHandler(stream_handler)
    # write to file
    # rq = time.strftime('%Y-%m-%d %H-%M', time.localtime(time.time()))
    # log_path = os.getcwd() + '/logs/' + log_prefix + "/"
    # if not os.path.exists(log_path):
    #     os.makedirs(log_path)
    # log_filename = log_path + rq + '.log'
    # file_handler = logging.FileHandler(log_filename, 'a', encoding='utf-8')
    # file_handler.setLevel(logging.INFO)
    # file_handler.setFormatter(log_format)
    # logger_prepared.addHandler(file_handler)
    return logger_prepared


config = Config()
logger = config_logger("main")
logger.info('Load preprocessed train and dev file.')
train_dataset = MyDataset(config, "", "", is_test=1)
dev_dataset = MyDataset(config, "", "", is_test=2)
# train_graph_file = "data_gen/train.json.graph_bert_all.pickle"
# train_elmo_file = "data_gen/train.json.elmo_bert_all.pickle"
# train_bert_file = "data_gen/train.json.bert_all.pickle"
# dev_graph_file = "data_gen/dev.json.graph_bert_all.pickle"
# dev_elmo_file = "data_gen/dev.json.elmo_bert_all.pickle"
# dev_bert_file = "data_gen/dev.json.bert_all.pickle"
# train_dataset = MyDataset(config, train_graph_file, train_elmo_file, train_bert_file)
# dev_dataset = MyDataset(config, dev_graph_file, dev_elmo_file, dev_bert_file)
logger.info("Data has prepared, train: %d, dev: %d." % (len(train_dataset), len(dev_dataset)))
trainer = Trainer(train_dataset, dev_dataset, logger, config)
trainer.train()
