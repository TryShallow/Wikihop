import os
import json
import pickle
import numpy as np

from allennlp.commands.elmo import ElmoEmbedder
from progressbar import ProgressBar, Percentage, Bar, Timer, ETA
from utils.ConfigLogger import config_logger
from nltk.tokenize import TweetTokenizer
from bert_serving.client import BertClient


class Preprocesser:
    def __init__(self, file_name, logger, is_masked=False, is_part=False):
        self.file_name = file_name
        self.dir_gen = "data_gen/"
        self.graph_file = self.dir_gen + self.file_name + ".graph_bert_all.pickle"
        self.elmo_file = self.dir_gen + self.file_name + ".elmo_bert_all.pickle"
        self.bert_file = self.dir_gen + self.file_name + ".bert_all.pickle"
        self.is_masked = is_masked
        self.use_elmo = True
        self.use_bert = True
        self.tokenizer = TweetTokenizer()
        self.logger = logger
        self.elmo_split_interval = 8
        self.max_support_length = 510
        self.contains_query_node = False
        self.is_part = is_part

    """ check whether string of current span matches the candidate
    """

    def check(self, support, word_index, candidate, for_unmarked=False):
        if for_unmarked:
            return sum(
                [self.is_contain_special_symbol(c_, support[word_index + j].lower()) for j, c_ in enumerate(candidate)
                 if
                 word_index + j < len(support)]) == len(candidate)
        else:
            return sum([support[word_index + j].lower() == c_ for j, c_ in enumerate(candidate) if
                        word_index + j < len(support)]) == len(candidate)

    def is_contain_special_symbol(self, candidate_tok, support_tok):
        if candidate_tok.isdigit():
            return support_tok.find(candidate_tok) >= 0
        else:
            return support_tok == candidate_tok or candidate_tok + 's' == support_tok or \
                   (support_tok.find(candidate_tok) >= 0 and (
                           support_tok.find('-') > 0 or support_tok.find('\'s') > 0 or
                           support_tok.find(',') > 0))

    """ Check whether the mask is valid via its length
    """

    def check_masked(self, support, word_index, candidate):
        return sum([support[word_index + j] == c_ for j, c_ in enumerate(candidate) if
                    word_index + j < len(support)]) == len(candidate)

    """ generating index for candidates in the original document
    """
    def ind(self, support_index, word_index, candidate_index, candidate, marked_candidate):
        marked_candidate[candidate_index] = True
        return [[support_index, word_index + i, candidate_index] for i in range(len(candidate))]

    """ some candidates may not be found in the original document so we have to merge it with the node masks who were 
        found in original document
    """
    def merge_two_masks(self, mask, unmarked_mask):
        for i in range(len(mask)):
            if len(unmarked_mask[i]) != 0:
                if len(mask[i]) == 0:
                    mask[i] = unmarked_mask[i]
                else:
                    for unmarked_index in range(len(unmarked_mask[i])):
                        mask[i].append(unmarked_mask[i][unmarked_index])
                    mask[i].sort(key=lambda x: x[0][1])
        return mask

    """ if some new POS or NER tags are found in data, we need to merge it with previous POS or NER dict
    """

    def mergeTwoDictFile(self, file_name, dict):
        with open(file_name, 'rb') as f:
            prev_dict = pickle.load(f)
        for name in dict:
            if not prev_dict.__contains__(name):
                prev_dict[name] = len(prev_dict)
        with open(file_name, 'wb') as f:
            pickle.dump(prev_dict, f)

    """ The main function to pre-processing WIKIHOP dataset and save it as several pickle files
    """

    def preprocess(self):
        supports = self.doPreprocessForGraph()
        with open(self.graph_file, 'rb') as f:
            data_graph = [d for d in pickle.load(f)]
        # text data including supporting documents, queries and node mask
        text_data = []
        for index, graph_d in enumerate(data_graph):
            tmp = {}
            tmp['query'] = graph_d['query']
            tmp['query_full_token'] = graph_d['query_full_token']
            tmp['nodes_mask'] = graph_d['nodes_mask']
            tmp['candidates'] = graph_d['candidates']
            tmp['nodes'] = graph_d['nodes']
            tmp['supports'] = supports[index]['supports']
            text_data.append(tmp)
        # process elmo
        if self.use_elmo:
            self.doPreprocessForElmo(text_data)
        if self.use_bert:
            self.doPreprocessForBert(text_data)

    """ Build entity graph base on input json data and save graph as a pickle
    """

    def doPreprocessForGraph(self):
        with open(self.file_name, 'r') as f:
            data = json.load(f)
            if self.is_part:
                data = data[: 2000]
            self.logger.info('Load json file: ' + self.file_name + " len: " + str(len(data)))
            supports = self.doPreprocess(data, mode='supports')

        if not os.path.isfile(self.graph_file):
            self.logger.info('Preprocsssing Json data for Graph....')
            data = self.doPreprocess(data, mode='graph', supports=supports)
            self.logger.info('Preprocessing Graph data finished')
            with open(self.graph_file, 'wb') as f:
                pickle.dump(data, f)
                self.logger.info('Save preprocessed Graph data file %s', self.graph_file)
        else:
            self.logger.info('Preprocessed Graph data is already existed.')
        return supports

    """ Generating pickle file for ELMo embeddings of queries and nodes in graph
    """

    def doPreprocessForElmo(self, text_data):
        if not os.path.isfile(self.elmo_file):
            elmoEmbedder = ElmoEmbedder(cuda_device=2)
            self.logger.info('Preprocsssing Json data for Elmo....')
            data = self.doPreprocess(text_data, mode='elmo', ee=elmoEmbedder)
            self.logger.info('Preprocessing Elmo data finished.')
            with open(self.elmo_file, 'wb') as f:
                pickle.dump(data, f)
                self.logger.info('Save preprocessed Elmo data file %s', self.elmo_file)
        else:
            self.logger.info('Preprocessed Elmo data is already existed.')

    def doPreprocessForBert(self, text_data):
        if not os.path.isfile(self.bert_file):
            bc = BertClient()
            self.logger.info("Preprocessing Json data for Bert....")
            data = self.doPreprocess(text_data, mode='bert', bc=bc)
            self.logger.info("Preprocessing Bert data finished.")
            with open(self.bert_file, 'wb') as f:
                pickle.dump(data, f)
                self.logger.info("Save preprocessed Bert data file %s", self.bert_file)
        else:
            self.logger.info("Preprocessed Bert data is already existed.")

    def doPreprocess(self, data_mb, mode, supports=None, ee=None, bc=None):
        data_gen = []
        widgets = ['Progress: ', Percentage(), ' ', Bar('#'), ' ', Timer(),
                   ' ', ETA(), ' ']
        pbar = ProgressBar(widgets=widgets, maxval=len(data_mb)).start()

        data_count = 0
        for index, data in enumerate(data_mb):
            if mode == 'supports':
                tmp = {}
                tmp['supports'] = [self.tokenizer.tokenize(support) for support in data['supports']]
                for index in range(len(tmp['supports'])):
                    if len(tmp['supports'][index]) > self.max_support_length:
                        tmp['supports'][index] = tmp['supports'][index][:self.max_support_length]
                data_gen.append(tmp)
            elif mode == 'graph':
                preprocessGraphData = self.preprocessForGraph(data, supports[index]['supports'])
                data_gen.append(preprocessGraphData)
            elif mode == 'elmo':
                preprocessElmoData = self.preprocessForElmo(data, ee)
                data_gen.append(preprocessElmoData)
            elif mode == 'bert':
                preprocessBertData = self.preprocessForBert(data, bc)
                data_gen.append(preprocessBertData)
            data_count += 1
            pbar.update(data_count)
        pbar.finish()
        return data_gen

    def preprocessForGraph(self, data, supports):
        if data.__contains__('annotations'):
            data.pop('annotations')

        ## The first token in the query is combined with underline so we have to divided it into several words by
        ## removing underlines
        first_blank_pos = data['query'].find(' ')
        if first_blank_pos > 0:
            first_token_in_query = data['query'][:first_blank_pos]
        else:
            first_token_in_query = data['query']
        query = data['query'].replace('_', ' ')
        data['query'] = self.tokenizer.tokenize(query)
        ## query_full_token means split the relation word in query based on "_"
        data['query_full_token'] = query

        candidates_orig = list(data['candidates'])

        data['candidates'] = [self.tokenizer.tokenize(candidate) for candidate in data['candidates']]

        marked_candidate = {}

        ## find all matched candidates in documents and mark their positions
        if self.is_masked:
            mask = [[self.ind(sindex, windex, cindex, candidate, marked_candidate)
                     for windex, word_support in enumerate(support) for cindex, candidate in
                     enumerate(data['candidates'])
                     if self.check_masked(support, windex, candidate)] for sindex, support in
                    enumerate(supports)]
        else:
            mask = [[self.ind(sindex, windex, cindex, candidate, marked_candidate)
                     for windex, word_support in enumerate(support) for cindex, candidate in
                     enumerate(data['candidates'])
                     if self.check(support, windex, candidate)] for sindex, support in enumerate(supports)]
            tok_unmarked_candidates = []
            unmarked_candidates_index_map = {}
            for candidate_index in range(len(data['candidates'])):
                if not marked_candidate.__contains__(candidate_index):
                    tok_unmarked_candidates.append(data['candidates'][candidate_index])
                    unmarked_candidates_index_map[len(tok_unmarked_candidates) - 1] = candidate_index
            if len(tok_unmarked_candidates) != 0:
                unmarked_mask = [
                    [self.ind(sindex, windex, unmarked_candidates_index_map[cindex], candidate, marked_candidate)
                     for windex, word_support in enumerate(support) for cindex, candidate in
                     enumerate(tok_unmarked_candidates)
                     if self.check(support, windex, candidate, for_unmarked=True)] for sindex, support in
                    enumerate(supports)]
                mask = self.merge_two_masks(mask, unmarked_mask)

        nodes_id_name = []
        count = 0
        for e in [[[x[-1] for x in c][0] for c in s] for s in mask]:
            u = []
            for f in e:
                u.append((count, f))
                count += 1

            nodes_id_name.append(u)

        #         data['nodes_candidates_id'] = [[node_triple[-1] for node_triple in node][0]
        #                                        for nodes_in_a_support in mask for node in nodes_in_a_support]
        data['nodes'] = [[node_triple[-1] for node_triple in node][0]
                         for nodes_in_a_support in mask for node in nodes_in_a_support]

        ## find two kinds of edges between nodes
        ## edges_in means nodes within a document, edges_out means nodes with same string across different document
        edges_in, edges_out = [], []
        for e0 in nodes_id_name:
            for f0, w0 in e0:
                for f1, w1 in e0:
                    if f0 != f1:
                        edges_in.append((f0, f1))

                for e1 in nodes_id_name:
                    for f1, w1 in e1:
                        if e0 != e1 and w0 == w1:
                            edges_out.append((f0, f1))

        data['edges_in'] = edges_in
        data['edges_out'] = edges_out

        data['nodes_mask'] = mask

        data['relation_index'] = len(first_token_in_query)
        for index, answer in enumerate(candidates_orig):
            if answer == data['answer']:
                # data['answer_candidate_id'] = index
                data['answer_index'] = index
                break
        return data

    """ gerating ELMo embeddings for nodes and query
    """

    def preprocessForElmo(self, text_data, ee):
        data_elmo = {}

        mask_ = [[x[:-1] for x in f] for e in text_data['nodes_mask'] for f in e]
        supports, query = text_data['supports'], text_data['query']
        query_full_tokens = text_data['query_full_token']
        first_tokens_in_query = query[0].split('_')

        query, _ = ee.batch_to_embeddings([query])
        query = query.data.cpu().numpy()
        data_elmo['query_elmo'] = (query.transpose((0, 2, 1, 3))).astype(np.float32)[0]
        if len(first_tokens_in_query) == 1:
            data_elmo['query_full_token_elmo'] = data_elmo['query_elmo']
        else:
            print(query)
            print(query_full_tokens)
            query_full_tokens, _ = ee.batch_to_embeddings([first_tokens_in_query])
            query_full_tokens = query_full_tokens.cpu().numpy()
            data_elmo['query_full_token_elmo'] = np.concatenate(
                (query_full_tokens.transpose((0, 2, 1, 3)).astype(np.float32)[0],
                 data_elmo['query_elmo'][1:, :, :]), 0
            )

        split_interval = self.elmo_split_interval
        if len(supports) <= split_interval:
            candidates, _ = ee.batch_to_embeddings(supports)
            candidates = candidates.data.cpu().numpy()
        else:
            # split long support data into several parts to avoid possible OOM
            count = 0
            candidates = None
            while count < len(supports):
                current_candidates, _ = \
                    ee.batch_to_embeddings(supports[count:min(count + split_interval, len(supports))])
                current_candidates = current_candidates.data.cpu().numpy()
                if candidates is None:
                    candidates = current_candidates
                else:
                    if candidates.shape[2] > current_candidates.shape[2]:
                        current_candidates = np.pad(current_candidates,
                                                    ((0, 0), (0, 0),
                                                     (0, candidates.shape[2] - current_candidates.shape[2]), (0, 0)),
                                                    'constant')
                    elif current_candidates.shape[2] > candidates.shape[2]:
                        candidates = np.pad(candidates,
                                            ((0, 0), (0, 0), (0, current_candidates.shape[2] - candidates.shape[2]),
                                             (0, 0)), 'constant')
                    candidates = np.concatenate((candidates, current_candidates))
                count += split_interval

        data_elmo['nodes_elmo'] = [(candidates.transpose((0, 2, 1, 3))[tuple(np.array(m).T)]).astype(np.float32)
                                   for m in mask_]
        return data_elmo

    def preprocessForBert(self, text_data, bc):
        data_bert = {}

        mask_ = [[x[:-1] for x in f] for e in text_data['nodes_mask'] for f in e]
        supports, query = text_data['supports'], text_data['query']
        query_full_tokens = text_data['query_full_token']
        first_tokens_in_query = query[0].split('_')

        query = bc.encode([query], is_tokenized=True)
        data_bert['query_bert'] = query[0][1: len(query) + 1].astype(np.float32)
        if len(first_tokens_in_query) == 1:
            data_bert['query_full_token_bert'] = data_bert['query_bert']
        else:
            print(query)
            print(query_full_tokens)

        candidates = bc.encode(supports, is_tokenized=True)
        data_bert['nodes_bert'] = [candidates[tuple(np.array(m).T)].astype(np.float32)
                                   for m in mask_]
        return data_bert
        # data_elmo['nodes_elmo'] = [(candidates.transpose((0, 2, 1, 3))[tuple(np.array(m).T)]).astype(np.float32)
        #                            for m in mask_]
        # split_interval = self.elmo_split_interval
        # if len(supports) <= split_interval:
        #     candidates, _ = ee.batch_to_embeddings(supports)
        #     candidates = candidates.data.cpu().numpy()
        # else:
        #     # split long support data into several parts to avoid possible OOM
        #     count = 0
        #     candidates = None
        #     while count < len(supports):
        #         current_candidates, _ = \
        #             ee.batch_to_embeddings(supports[count:min(count + split_interval, len(supports))])
        #         current_candidates = current_candidates.data.cpu().numpy()
        #         if candidates is None:
        #             candidates = current_candidates
        #         else:
        #             if candidates.shape[2] > current_candidates.shape[2]:
        #                 current_candidates = np.pad(current_candidates,
        #                                             ((0, 0), (0, 0),
        #                                              (0, candidates.shape[2] - current_candidates.shape[2]), (0, 0)),
        #                                             'constant')
        #             elif current_candidates.shape[2] > candidates.shape[2]:
        #                 candidates = np.pad(candidates,
        #                                     ((0, 0), (0, 0), (0, current_candidates.shape[2] - candidates.shape[2]),
        #                                      (0, 0)), 'constant')
        #             candidates = np.concatenate((candidates, current_candidates))
        #         count += split_interval
        #
        # data_elmo['nodes_elmo'] = [(candidates.transpose((0, 2, 1, 3))[tuple(np.array(m).T)]).astype(np.float32)
        #                            for m in mask_]


logger = config_logger('Preprocess')
file_name = "train.json"
# file_name = "dev.json"
processer = Preprocesser(file_name, logger, is_masked=False, is_part=False)
processer.preprocess()