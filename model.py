import torch
import torchsnooper
import torch.nn as nn

from config import Config


class Model(nn.Module):
    def __init__(self, config: Config):
        super(Model, self).__init__()
        self.config = config
        self.use_elmo = config.use_elmo
        self.use_glove = config.use_glove
        self.use_bert = config.use_bert
        self.use_extra = config.use_extra

        # feature layer
        input_size = 0
        if config.use_elmo:
            input_size += (3 * 1024)
        if config.use_glove:
            input_size += 300
        if config.use_bert:
            input_size += 768
        if config.query_encoding_type == 'linear':
            self.query_linear = nn.Linear(input_size, config.encoding_size)
        if config.query_encoding_type == 'lstm':
            self.query_lstm = nn.LSTM(input_size, config.encoding_size // 2, 2,
                                      bidirectional=True, batch_first=True)
        self.nodes_linear = nn.Linear(input_size, config.encoding_size)
        self.nodes_dropout = nn.Dropout(config.dropout)
        if config.use_extra:
            self.nodes_ner_embed = nn.Embedding(config.ner_dict_size, config.ner_embedding_size)
            self.nodes_pos_embed = nn.Embedding(config.pos_dict_size, config.pos_embedding_size)
            self.query_ner_embed = nn.Embedding(config.ner_dict_size, config.ner_embedding_size)
            self.query_pos_embed = nn.Embedding(config.pos_dict_size, config.pos_embedding_size)

        # gcn layer
        encoding_size = config.encoding_size
        if config.use_extra:
            encoding_size += config.ner_embedding_size + config.pos_embedding_size
        self.hidden_linears = nn.ModuleList([nn.Linear(encoding_size, encoding_size)] * 4)
        self.combined_linear = nn.Linear(encoding_size * 2, encoding_size)

        # bi_attention layer
        self.attention_linear = nn.Linear(encoding_size * 3, 1, False)

        # output layer
        self.out_att1 = nn.Linear(encoding_size * 4, 128)
        self.out_att2 = nn.Linear(128, 1)

    # @torchsnooper.snoop()
    def forward(self, x):
        nodes_length = x['nodes_length']
        adj = x['adj'].type(torch.float32)
        mask = x['mask']
        # [batch_size, max_nodes, 512]  [batch_size, max_query_size, 512]
        nodes_compress, query_compress = self.feature_layer(x)
        nodes_mask = self.build_mask(self.config.max_nodes, nodes_length)
        nodes = nodes_compress * nodes_mask
        nodes = self.nodes_dropout(nodes)
        last_hop = nodes
        for _ in range(self.config.hops):
            last_hop = self.gcn_layer(adj, last_hop, nodes_mask)
        bi_attention = self.bi_attention_layer(query_compress, nodes_compress, last_hop)
        return self.output_layer(bi_attention, mask)

    def feature_layer(self, x):
        query_flat, nodes_flat = None, None
        if self.use_elmo:
            query_flat = torch.reshape(x["query_elmo"],
                                       (-1, self.config.max_query_size, 3 * 1024))
            nodes_flat = torch.reshape(x["nodes_elmo"],
                                       (-1, self.config.max_nodes, 3 * 1024))
        if self.use_glove:
            query_flat = x["query_glove"] if query_flat is None \
                else torch.cat((query_flat, x["query_glove"]), -1)
            nodes_flat = x["nodes_glove"] if nodes_flat is None \
                else torch.cat((nodes_flat, x["nodes_glove"]), -1)
        if self.use_bert:
            query_flat = x["query_bert"] if query_flat is None \
                else torch.cat((query_flat, x["query_bert"]), -1)
            nodes_flat = x["nodes_bert"] if nodes_flat is None \
                else torch.cat((nodes_flat, x["nodes_bert"]), -1)

        nodes_compress = torch.tanh(self.nodes_linear(nodes_flat))
        if self.config.query_encoding_type == 'lstm':
            query_compress, _ = self.query_lstm(query_flat)
        else:
            query_compress = self.query_linear(query_flat)
        if self.use_extra:
            nodes_ner = self.nodes_ner_embed(x["nodes_ner"])
            nodes_pos = self.nodes_pos_embed(x["nodes_pos"])
            query_ner = self.query_ner_embed(x["query_ner"])
            query_pos = self.query_pos_embed(x["query_pos"])
            nodes_compress = torch.cat((nodes_compress, nodes_ner, nodes_pos), -1)
            query_compress = torch.cat((query_compress, query_ner, query_pos), -1)
        return nodes_compress, query_compress

    @staticmethod
    def build_mask(max_nodes, nodes_length):
        nodes_mask = torch.arange(max_nodes).unsqueeze(0).to(nodes_length.device)
        nodes_mask = nodes_mask.repeat((nodes_length.size(0), 1)) < nodes_length.unsqueeze(-1)
        nodes_mask = nodes_mask.type(torch.float32).unsqueeze(-1)
        return nodes_mask

    def gcn_layer(self, adj, hidden_tensor, hidden_mask):
        adjacency_tensor = adj
        # [batch_size, 3, max_nodes, max_nodes]
        hidden_tensors = torch.stack(
            tuple([self.hidden_linears[i](hidden_tensor) for i in range(adj.size(1))]), 1
        ) * hidden_mask.unsqueeze(1)

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
        concat_attention_data = torch.cat((expanded_nodes, expanded_query,
                                           context_query_similarity), -1)
        # [batch_size, max_nodes, max_query]
        similarity = torch.mean(self.attention_linear(concat_attention_data), -1)
        # [batch_size, max_nodes, d]
        nodes2query = torch.matmul(nn.Softmax(-1)(similarity), query_compress)
        # [batch_size, max_nodes]
        b = nn.Softmax(-1)(similarity.max(-1)[0])
        # [batch_size, max_nodes, d]
        query2nodes = torch.matmul(b.unsqueeze(1), nodes_compress).repeat(
            (1, self.config.max_nodes, 1))
        g = torch.cat((nodes_compress, nodes2query, nodes_compress * nodes2query,
                       nodes_compress * query2nodes), -1)
        return g

    def output_layer(self, bi_attention, mask):
        raw_predictions = torch.tanh(self.out_att1(bi_attention))
        raw_predictions = self.out_att2(raw_predictions).squeeze(-1)
        predictions2 = mask.type(torch.float32) * raw_predictions.unsqueeze(1)
        # predictions2[predictions2 == 0] = - np.inf
        predictions2[predictions2 == 0] = - 1e6
        predictions2 = predictions2.max(-1)[0]
        return predictions2