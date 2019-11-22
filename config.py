class Config(object):
    def __init__(self):
        self.device = "cuda:3"
        self.map_location = None  # {'cuda:2': 'cuda:3'}

        self.is_test = True
        self.use_elmo = True
        self.use_glove = True
        self.use_bert = False
        self.use_extra = True

        self.max_query_size = 25
        self.max_nodes = 500
        self.max_candidates = 80
        self.max_candidates_len = 10

        self.query_encoding_type = "lstm"  # "lstm"
        self.encoding_size = 512
        self.dropout = 0.8
        self.hops = 5
        self.ner_embedding_size = 8
        self.pos_embedding_size = 8
        self.ner_dict_size = 0
        self.pos_dict_size = 0

        self.contains_query_node = False
        self.add_query_self_att = False

        self.batch_size = 16
        self.num_epochs = 5000
        self.model_path = "model_saved"
        self.learning_rate = 1e-4
        self.momentum = 0.9
        self.optim = "sgd"

        self.model_prefix = "wiki_org_ge_"
        self.pretrained_parameters = None

        self.log_dir = 'tensorboardX_log/'
        self.train_graph_file = "data_gen/train.json.preprocessed.pickle"
        self.train_elmo_file = "data_gen/train.json.elmo.preprocessed.pickle"
        self.train_glove_file = "data_gen/train.json.glove.preprocessed.pickle"
        self.train_extra_file = "data_gen/train.json.extra.preprocessed.pickle"
        self.dev_graph_file = "data_gen/dev.json.preprocessed.pickle"
        self.dev_elmo_file = "data_gen/dev.json.elmo.preprocessed.pickle"
        self.dev_glove_file = "data_gen/dev.json.glove.preprocessed.pickle"
        self.dev_extra_file = "data_gen/dev.json.extra.preprocessed.pickle"

    def print(self):
        print("Running parameters:")
        print("\tis_test:", self.is_test)
        print("\tuse_elmo:", self.use_elmo)
        print("\tuse_glove:", self.use_glove)
        print("\tuse_bert:", self.use_bert)
        print("\tuse_extra:", self.use_extra)
        print("\tdevice:", self.device)
        print("\toptim:", self.optim)
        print("\tlearning_rate:", self.learning_rate)
        print("\tbatch_size:", self.batch_size)
        print("\tmodel_prefix:", self.model_prefix)
        if self.pretrained_parameters is not None:
            print("\tpre-trained parameters:", self.pretrained_parameters)

        print()