import itertools
import json
import pickle
import os

import numpy as np
import torch
from tqdm import tqdm
from transformers import (OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP, BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
                          XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP, ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP)
from transformers import (OpenAIGPTTokenizer, BertTokenizer, XLNetTokenizer, RobertaTokenizer)
try:
    from transformers import ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP
    from transformers import AlbertTokenizer
except:
    pass

from preprocess_utils import conceptnet
from utils import utils
import spacy
import scispacy
from scispacy.linking import EntityLinker
from spacy.matcher import Matcher
import re

MODEL_CLASS_TO_NAME = {
    'gpt': list(OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'bert': list(BERT_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'xlnet': list(XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'roberta': list(ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'lstm': ['lstm'],
}
try:
    MODEL_CLASS_TO_NAME['albert'] =  list(ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP.keys())
except:
    pass

MODEL_NAME_TO_CLASS = {model_name: model_class for model_class, model_name_list in MODEL_CLASS_TO_NAME.items() for model_name in model_name_list}

#Add SapBERT configuration
model_name = 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext'
MODEL_NAME_TO_CLASS[model_name] = 'bert'

GPT_SPECIAL_TOKENS = ['_start_', '_delimiter_', '_classify_']


class MultiGPUSparseAdjDataBatchGenerator(object):
    """A data generator that batches the data and moves them to the corresponding devices."""
    def __init__(self, device0, device1, batch_size, indexes, qids, labels,
                 tensors0=[], lists0=[], tensors1=[], lists1=[], adj_data=None,
                 linked_data=None):
        self.device0 = device0
        self.device1 = device1
        self.batch_size = batch_size
        self.indexes = indexes
        self.qids = qids
        self.labels = labels
        self.tensors0 = tensors0
        self.lists0 = lists0
        self.tensors1 = tensors1
        self.lists1 = lists1
        self.adj_data = adj_data
        self.linked_data = linked_data

    def __len__(self):
        return (self.indexes.size(0) - 1) // self.batch_size + 1

    def __iter__(self):
        bs = self.batch_size
        n = self.indexes.size(0)
        for a in range(0, n, bs):
            b = min(n, a + bs)
            batch_indexes = self.indexes[a:b]
            batch_qids = [self.qids[idx] for idx in batch_indexes]
            batch_labels = self._to_device(self.labels[batch_indexes], self.device1)
            batch_tensors0 = [self._to_device(x[batch_indexes], self.device1) for x in self.tensors0]
            batch_tensors1 = [self._to_device(x[batch_indexes], self.device1) for x in self.tensors1]
            batch_tensors1[0] = batch_tensors1[0].to(self.device0)
            batch_lists0 = [self._to_device([x[i] for i in batch_indexes], self.device0) for x in self.lists0]
            batch_lists1 = [self._to_device([x[i] for i in batch_indexes], self.device1) for x in self.lists1]

            edge_index_all, edge_type_all = self.adj_data
            #edge_index_all: nested list of shape (n_samples, num_choice), where each entry is tensor[2, E]
            #edge_type_all:  nested list of shape (n_samples, num_choice), where each entry is tensor[E, ]
            edge_index = self._to_device([edge_index_all[i] for i in batch_indexes], self.device1)
            edge_type  = self._to_device([edge_type_all[i] for i in batch_indexes], self.device1)

            if self.linked_data is not None:
                linked_span = [self.linked_data[0][idx] for idx in batch_indexes]
                linked_ids = [self.linked_data[1][idx] for idx in batch_indexes]
                yield tuple([batch_qids, batch_labels, *batch_tensors0,
                             *batch_lists0, *batch_tensors1, *batch_lists1, edge_index,
                             edge_type, linked_span, linked_ids])
            else:
                yield tuple([batch_qids, batch_labels, *batch_tensors0, *batch_lists0,
                             *batch_tensors1, *batch_lists1, edge_index, edge_type])

    def _to_device(self, obj, device):
        if isinstance(obj, (tuple, list)):
            return [self._to_device(item, device) for item in obj]
        else:
            return obj.to(device)


class GreaseLM_DataLoader(object):

    def __init__(self, train_statement_path, train_adj_path,
                 dev_statement_path, dev_adj_path,
                 test_statement_path, test_adj_path,
                 batch_size, eval_batch_size, device, model_name, max_node_num=200, max_seq_length=128,
                 is_inhouse=False, inhouse_train_qids_path=None,
                 subsample=1.0, n_train=-1, debug=False, cxt_node_connects_all=False, kg="cpnet",
                 align_mask=False,
                 train_span_path=None, train_ids_path=None,
                 dev_span_path=None, dev_ids_path=None,
                 test_span_path=None, test_ids_path=None):
        super().__init__()
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.device0, self.device1 = device
        self.is_inhouse = is_inhouse
        self.debug = debug
        self.model_name = model_name
        self.max_node_num = max_node_num
        self.debug_sample_size = 32
        self.cxt_node_connects_all = cxt_node_connects_all

        self.model_type = MODEL_NAME_TO_CLASS[model_name]
        self.load_resources(kg)

        self.align_mask = align_mask
        # Load training data
        print ('train_statement_path', train_statement_path)
        if not self.align_mask:
            self.train_qids, self.train_labels, self.train_encoder_data, \
            train_concepts_by_sents_list = self.load_input_tensors(train_statement_path, max_seq_length)
        else:
            self.train_qids, self.train_labels, self.train_encoder_data, train_concepts_by_sents_list,\
            (self.train_linked_span, self.train_linked_ids) = self.load_input_tensors(train_statement_path,
                                                                                      max_seq_length,
                                                                                      train_span_path,
                                                                                      train_ids_path)
        num_choice = self.train_encoder_data[0].size(1)
        self.num_choice = num_choice
        print ('num_choice', num_choice)
        *self.train_decoder_data, self.train_adj_data = self.load_sparse_adj_data_with_contextnode(train_adj_path, max_node_num, train_concepts_by_sents_list)
        if not debug:
            assert all(len(self.train_qids) == len(self.train_adj_data[0]) == x.size(0) for x in [self.train_labels] + self.train_encoder_data + self.train_decoder_data)

        print("Finish loading training data.")

        # Load dev data
        if not self.align_mask:
            self.dev_qids, self.dev_labels, self.dev_encoder_data, dev_concepts_by_sents_list = self.load_input_tensors(dev_statement_path, max_seq_length)
        else:
            self.dev_qids, self.dev_labels, self.dev_encoder_data, dev_concepts_by_sents_list, \
            (self.dev_linked_span, self.dev_linked_ids) = self.load_input_tensors(dev_statement_path,
                                                                                  max_seq_length,
                                                                                  dev_span_path,
                                                                                  dev_ids_path)
        *self.dev_decoder_data, self.dev_adj_data = self.load_sparse_adj_data_with_contextnode(dev_adj_path, max_node_num, dev_concepts_by_sents_list)
        if not debug:
            assert all(len(self.dev_qids) == len(self.dev_adj_data[0]) == x.size(0) for x in [self.dev_labels] + self.dev_encoder_data + self.dev_decoder_data)

        print("Finish loading dev data.")

        # Load test data
        if test_statement_path is not None:
            if not self.align_mask:
                self.test_qids, self.test_labels, self.test_encoder_data, test_concepts_by_sents_list = self.load_input_tensors(test_statement_path, max_seq_length)
            else:
                self.test_qids, self.test_labels, self.test_encoder_data, test_concepts_by_sents_list, \
                (self.test_linked_span, self.test_linked_ids) = self.load_input_tensors(test_statement_path,
                                                                                        max_seq_length,
                                                                                        test_span_path,
                                                                                        test_ids_path)
            *self.test_decoder_data, self.test_adj_data = self.load_sparse_adj_data_with_contextnode(test_adj_path, max_node_num, test_concepts_by_sents_list)
            if not debug:
                assert all(len(self.test_qids) == len(self.test_adj_data[0]) == x.size(0) for x in [self.test_labels] + self.test_encoder_data + self.test_decoder_data)

            print("Finish loading test data.")

        # If using inhouse split, we split the original training set into an inhouse training set and an inhouse test set.
        if self.is_inhouse:
            with open(inhouse_train_qids_path, 'r') as fin:
                inhouse_qids = set(line.strip() for line in fin)
            self.inhouse_train_indexes = torch.tensor([i for i, qid in enumerate(self.train_qids) if qid in inhouse_qids])
            self.inhouse_test_indexes = torch.tensor([i for i, qid in enumerate(self.train_qids) if qid not in inhouse_qids])

        # Optionally we can subsample the training set.
        assert 0. < subsample <= 1.
        if subsample < 1. or n_train >= 0:
            # n_train will override subsample if the former is not None
            if n_train == -1:
                n_train = int(self.train_size() * subsample)
            assert n_train > 0
            if self.is_inhouse:
                self.inhouse_train_indexes = self.inhouse_train_indexes[:n_train]
            else:
                self.train_qids = self.train_qids[:n_train]
                self.train_labels = self.train_labels[:n_train]
                self.train_encoder_data = [x[:n_train] for x in self.train_encoder_data]
                self.train_decoder_data = [x[:n_train] for x in self.train_decoder_data]
                self.train_adj_data = self.train_adj_data[:n_train]
                assert all(len(self.train_qids) == len(self.train_adj_data[0]) == x.size(0) for x in [self.train_labels] + self.train_encoder_data + self.train_decoder_data)
            assert self.train_size() == n_train

    def train_size(self):
        return self.inhouse_train_indexes.size(0) if self.is_inhouse else len(self.train_qids)

    def dev_size(self):
        return len(self.dev_qids)

    def test_size(self):
        if self.is_inhouse:
            return self.inhouse_test_indexes.size(0)
        else:
            return len(self.test_qids) if hasattr(self, 'test_qids') else 0

    def train(self):
        if self.debug:
            train_indexes = torch.arange(self.debug_sample_size)
        elif self.is_inhouse:
            n_train = self.inhouse_train_indexes.size(0)
            train_indexes = self.inhouse_train_indexes[torch.randperm(n_train)]
        else:
            train_indexes = torch.randperm(len(self.train_qids))
        return MultiGPUSparseAdjDataBatchGenerator(self.device0,
                                                   self.device1,
                                                   self.batch_size,
                                                   train_indexes,
                                                   self.train_qids,
                                                   self.train_labels,
                                                   tensors0=self.train_encoder_data,
                                                   tensors1=self.train_decoder_data,
                                                   adj_data=self.train_adj_data,
                                                   linked_data=(self.train_linked_span, self.train_linked_ids) if self.align_mask else None)

    def train_eval(self):
        exit('unknown')
        return MultiGPUSparseAdjDataBatchGenerator(self.device0,
                                                   self.device1,
                                                   self.eval_batch_size,
                                                   torch.arange(len(self.train_qids)),
                                                   self.train_qids,
                                                   self.train_labels,
                                                   tensors0=self.train_encoder_data,
                                                   tensors1=self.train_decoder_data,
                                                   adj_data=self.train_adj_data)

    def dev(self):
        if self.debug:
            dev_indexes = torch.arange(self.debug_sample_size)
        else:
            dev_indexes = torch.arange(len(self.dev_qids))
        return MultiGPUSparseAdjDataBatchGenerator(self.device0,
                                                   self.device1,
                                                   self.eval_batch_size,
                                                   dev_indexes,
                                                   self.dev_qids,
                                                   self.dev_labels,
                                                   tensors0=self.dev_encoder_data,
                                                   tensors1=self.dev_decoder_data,
                                                   adj_data=self.dev_adj_data,
                                                   linked_data=None)

    def test(self):
        if self.debug:
            test_indexes = torch.arange(self.debug_sample_size)
        elif self.is_inhouse:
            test_indexes = self.inhouse_test_indexes
        else:
            test_indexes = torch.arange(len(self.test_qids))
        if self.is_inhouse:
            return MultiGPUSparseAdjDataBatchGenerator(self.device0,
                                                       self.device1,
                                                       self.eval_batch_size,
                                                       test_indexes,
                                                       self.train_qids,
                                                       self.train_labels,
                                                       tensors0=self.train_encoder_data,
                                                       tensors1=self.train_decoder_data,
                                                       adj_data=self.train_adj_data,
                                                       linked_data=None)
        else:
            return MultiGPUSparseAdjDataBatchGenerator(self.device0,
                                                       self.device1,
                                                       self.eval_batch_size, test_indexes,
                                                       self.test_qids,
                                                       self.test_labels,
                                                       tensors0=self.test_encoder_data,
                                                       tensors1=self.test_decoder_data,
                                                       adj_data=self.test_adj_data,
                                                       linked_data=None)

    def train_dev_test(self):
        merged_linked_data = None
        if self.debug or self.is_inhouse:
            n_train = self.inhouse_train_indexes.size(0)
            train_indexes = self.inhouse_train_indexes[torch.randperm(n_train)]
            dev_indexes = torch.arange(len(self.dev_qids)) + len(train_indexes)
            test_indexes = self.inhouse_test_indexes + len(train_indexes) + len(dev_indexes)

            merged_indexes = torch.cat((train_indexes, dev_indexes, test_indexes), dim=0)

            merged_qids = self.train_qids + self.dev_qids + self.train_qids

            merged_labels = torch.cat((self.train_labels, self.dev_labels, self.train_labels), dim=0)

            merged_encoder_data = [torch.cat((i, j, k), dim=0) for i, j, k in zip(self.train_encoder_data,
                                                                                  self.dev_encoder_data,
                                                                                  self.train_encoder_data)]

            merged_decoder_data = [
                torch.cat((i, j, k), dim=0) for i, j, k in zip(self.train_decoder_data,
                                                               self.dev_decoder_data,
                                                               self.train_decoder_data)
            ]
            merged_edge_idx = self.train_adj_data[0] + self.dev_adj_data[0] + self.train_adj_data[0]
            merged_edge_type = self.train_adj_data[1] + self.dev_adj_data[1] + self.train_adj_data[1]
            merged_adj_data = (merged_edge_idx, merged_edge_type)
            if self.align_mask:
                merged_linked_span = self.train_linked_span + self.dev_linked_span + self.train_linked_span
                merged_linked_ids = self.train_linked_ids + self.dev_linked_ids + self.train_linked_ids
                merged_linked_data = (merged_linked_span, merged_linked_ids)

        else:
            train_indexes = torch.randperm(len(self.train_qids))
            dev_indexes = torch.randperm(len(self.dev_qids)) + len(train_indexes)
            test_indexes = torch.randperm(len(self.test_qids)) + len(train_indexes) + len(dev_indexes)
            merged_indexes = torch.cat((train_indexes, dev_indexes, test_indexes), dim=0)

            merged_qids = self.train_qids + self.dev_qids + self.test_qids

            merged_labels = torch.cat((self.train_labels, self.dev_labels, self.test_labels), dim=0)

            merged_encoder_data = [torch.cat((i, j, k), dim=0) for i, j, k in zip(self.train_encoder_data,
                                                                                  self.dev_encoder_data,
                                                                                  self.test_encoder_data)]

            merged_decoder_data = [
                torch.cat((i, j, k), dim=0) for i, j, k in zip(self.train_decoder_data,
                                                               self.dev_decoder_data,
                                                               self.test_decoder_data)
            ]
            merged_edge_idx = self.train_adj_data[0] + self.dev_adj_data[0] + self.test_adj_data[0]
            merged_edge_type = self.train_adj_data[1] + self.dev_adj_data[1] + self.test_adj_data[1]
            merged_adj_data = (merged_edge_idx, merged_edge_type)
            if self.align_mask:
                merged_linked_span = self.train_linked_span + self.dev_linked_span + self.test_linked_span
                merged_linked_ids = self.train_linked_ids + self.dev_linked_ids + self.test_linked_ids
                merged_linked_data = (merged_linked_span, merged_linked_ids)

        return MultiGPUSparseAdjDataBatchGenerator(self.device0, self.device1, self.batch_size, merged_indexes,
                                                   merged_qids, merged_labels,
                                                   tensors0=merged_encoder_data, tensors1=merged_decoder_data,
                                                   adj_data=merged_adj_data,
                                                   linked_data=merged_linked_data)

    def load_resources(self, kg):
        # Load the tokenizer
        try:
            tokenizer_class = {'bert': BertTokenizer, 'xlnet': XLNetTokenizer, 'roberta': RobertaTokenizer, 'albert': AlbertTokenizer}.get(self.model_type)
        except:
            tokenizer_class = {'bert': BertTokenizer, 'xlnet': XLNetTokenizer, 'roberta': RobertaTokenizer}.get(self.model_type)
        tokenizer = tokenizer_class.from_pretrained(self.model_name)
        self.tokenizer = tokenizer

        if kg == "cpnet":
            # Load cpnet
            cpnet_vocab_path = "data/cpnet/concept.txt"
            with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
                self.id2concept = [w.strip() for w in fin]
            self.concept2id = {w: i for i, w in enumerate(self.id2concept)}
            self.id2relation = conceptnet.merged_relations
        elif kg == "ddb":
            cpnet_vocab_path = "data/ddb/vocab.txt"
            with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
                self.id2concept = [w.strip() for w in fin]
            self.concept2id = {w: i for i, w in enumerate(self.id2concept)}
            self.id2relation = [
                'belongstothecategoryof',
                'isacategory',
                'maycause',
                'isasubtypeof',
                'isariskfactorof',
                'isassociatedwith',
                'maycontraindicate',
                'interactswith',
                'belongstothedrugfamilyof',
                'child-parent',
                'isavectorfor',
                'mabeallelicwith',
                'seealso',
                'isaningradientof',
                'mabeindicatedby'
            ]
        else:
            raise ValueError("Invalid value for kg.")

    def load_input_tensors(self, input_jsonl_path, max_seq_length, span_path=None,
                           ids_path=None):
        """Construct input tensors for the LM component of the model."""
        cache_path = input_jsonl_path + "-sl{}".format(max_seq_length) + \
                     (("-" + self.model_type) if self.model_type != "roberta" else "") + \
                     ("-align_mask" if self.align_mask else "") + '.loaded_cache'
        use_cache = True

        if use_cache and not os.path.exists(cache_path):
            use_cache = False

        if use_cache:
            with open(cache_path, 'rb') as f:
                input_tensors = utils.CPU_Unpickler(f).load()
        else:
            if self.model_type in ('lstm',):
                raise NotImplementedError
            elif self.model_type in ('gpt',):
                input_tensors = load_gpt_input_tensors(input_jsonl_path, max_seq_length)
            elif self.model_type in ('bert', 'xlnet', 'roberta', 'albert'):
                if not self.align_mask:
                    input_tensors = load_bert_xlnet_roberta_input_tensors(input_jsonl_path, max_seq_length, self.debug, self.tokenizer, self.debug_sample_size)
                else:
                    input_tensors = load_bert_xlnet_roberta_input_tensors_with_linking_mask(input_jsonl_path,
                                                                                            max_seq_length,
                                                                                            self.debug,
                                                                                            self.tokenizer,
                                                                                            self.debug_sample_size,
                                                                                            linked_span_save_path=span_path,
                                                                                            linked_ids_save_path=ids_path)
            if not self.debug:
                utils.save_pickle(input_tensors, cache_path)
        return input_tensors

    def load_sparse_adj_data_with_contextnode(self, adj_pk_path, max_node_num, concepts_by_sents_list):
        """Construct input tensors for the GNN component of the model."""
        print("Loading sparse adj data...")
        cache_path = adj_pk_path + "-nodenum{}".format(max_node_num) + ("-cntsall" if self.cxt_node_connects_all else "") + '.loaded_cache'
        use_cache = True

        if use_cache and not os.path.exists(cache_path):
            use_cache = False

        if use_cache:
            with open(cache_path, 'rb') as f:
                adj_lengths_ori, concept_ids, node_type_ids, node_scores, adj_lengths, edge_index, edge_type, half_n_rel, special_nodes_mask = utils.CPU_Unpickler(f).load()
        else:
            # Set special nodes and links
            context_node = 0
            n_special_nodes = 1
            cxt2qlinked_rel = 0
            cxt2alinked_rel = 1
            half_n_rel = len(self.id2relation) + 2
            if self.cxt_node_connects_all:
                cxt2other_rel = half_n_rel
                half_n_rel += 1

            adj_concept_pairs = []
            with open(adj_pk_path, "rb") as in_file:
                try:
                    while True:
                        ex = pickle.load(in_file)
                        if type(ex) == dict:
                            adj_concept_pairs.append(ex)
                        elif type(ex) == list:
                            adj_concept_pairs.extend(ex)
                        else:
                            raise TypeError("Invalid type for ex.")
                except EOFError:
                    pass

            n_samples = len(adj_concept_pairs) #this is actually n_questions x n_choices
            edge_index, edge_type = [], []
            adj_lengths = torch.zeros((n_samples,), dtype=torch.long)
            concept_ids = torch.full((n_samples, max_node_num), 1, dtype=torch.long)
            node_type_ids = torch.full((n_samples, max_node_num), 2, dtype=torch.long) #default 2: "other node"
            node_scores = torch.zeros((n_samples, max_node_num, 1), dtype=torch.float)
            special_nodes_mask = torch.zeros(n_samples, max_node_num, dtype=torch.bool)

            adj_lengths_ori = adj_lengths.clone()
            if not concepts_by_sents_list:
                concepts_by_sents_list = itertools.repeat(None)
            for idx, (_data, cpts_by_sents) in tqdm(enumerate(zip(adj_concept_pairs, concepts_by_sents_list)), total=n_samples, desc='loading adj matrices'):
                if self.debug and idx >= self.debug_sample_size * self.num_choice:
                    break
                adj, concepts, qm, am, cid2score = _data['adj'], _data['concepts'], _data['qmask'], _data['amask'], _data['cid2score']
                #adj: e.g. <4233x249 (n_nodes*half_n_rels x n_nodes) sparse matrix of type '<class 'numpy.bool'>' with 2905 stored elements in COOrdinate format>
                #concepts: np.array(num_nodes, ), where entry is concept id
                #qm: np.array(num_nodes, ), where entry is True/False
                #am: np.array(num_nodes, ), where entry is True/False
                assert len(concepts) == len(set(concepts))
                qam = qm | am
                #sanity check: should be T,..,T,F,F,..F
                assert qam[0] == True
                F_start = False
                for TF in qam:
                    if TF == False:
                        F_start = True
                    else:
                        assert F_start == False

                assert n_special_nodes <= max_node_num
                special_nodes_mask[idx, :n_special_nodes] = 1
                num_concept = min(len(concepts) + n_special_nodes, max_node_num) #this is the final number of nodes including contextnode but excluding PAD
                adj_lengths_ori[idx] = len(concepts)
                adj_lengths[idx] = num_concept

                #Prepare nodes
                concepts = concepts[:num_concept - n_special_nodes]
                concept_ids[idx, n_special_nodes:num_concept] = torch.tensor(concepts + 1)  #To accomodate contextnode, original concept_ids incremented by 1
                concept_ids[idx, 0] = context_node #this is the "concept_id" for contextnode

                #Prepare node scores
                if cid2score is not None:
                    if -1 not in cid2score:
                        cid2score[-1] = 0
                    for _j_ in range(num_concept):
                        _cid = int(concept_ids[idx, _j_]) - 1 # Now context node is -1
                        node_scores[idx, _j_, 0] = torch.tensor(cid2score[_cid])

                #Prepare node types
                node_type_ids[idx, 0] = 3 # context node
                node_type_ids[idx, 1:n_special_nodes] = 4 # sent nodes
                node_type_ids[idx, n_special_nodes:num_concept][torch.tensor(qm, dtype=torch.bool)[:num_concept - n_special_nodes]] = 0
                node_type_ids[idx, n_special_nodes:num_concept][torch.tensor(am, dtype=torch.bool)[:num_concept - n_special_nodes]] = 1

                #Load adj
                ij = torch.tensor(adj.row, dtype=torch.int64) #(num_matrix_entries, ), where each entry is coordinate
                k = torch.tensor(adj.col, dtype=torch.int64)  #(num_matrix_entries, ), where each entry is coordinate
                n_node = adj.shape[1]
                assert len(self.id2relation) == adj.shape[0] // n_node
                i, j = ij // n_node, ij % n_node

                #Prepare edges
                i += 2; j += 1; k += 1  # **** increment coordinate by 1, rel_id by 2 ****
                extra_i, extra_j, extra_k = [], [], []
                for _coord, q_tf in enumerate(qm):
                    _new_coord = _coord + n_special_nodes
                    if _new_coord > num_concept:
                        break
                    if q_tf:
                        extra_i.append(cxt2qlinked_rel) #rel from contextnode to question concept
                        extra_j.append(0) #contextnode coordinate
                        extra_k.append(_new_coord) #question concept coordinate
                    elif self.cxt_node_connects_all:
                        extra_i.append(cxt2other_rel) #rel from contextnode to other concept
                        extra_j.append(0) #contextnode coordinate
                        extra_k.append(_new_coord) #other concept coordinate
                for _coord, a_tf in enumerate(am):
                    _new_coord = _coord + n_special_nodes
                    if _new_coord > num_concept:
                        break
                    if a_tf:
                        extra_i.append(cxt2alinked_rel) #rel from contextnode to answer concept
                        extra_j.append(0) #contextnode coordinate
                        extra_k.append(_new_coord) #answer concept coordinate
                    elif self.cxt_node_connects_all:
                        extra_i.append(cxt2other_rel) #rel from contextnode to other concept
                        extra_j.append(0) #contextnode coordinate
                        extra_k.append(_new_coord) #other concept coordinate

                # half_n_rel += 2 #should be 19 now
                if len(extra_i) > 0:
                    i = torch.cat([i, torch.tensor(extra_i)], dim=0)
                    j = torch.cat([j, torch.tensor(extra_j)], dim=0)
                    k = torch.cat([k, torch.tensor(extra_k)], dim=0)
                ########################

                mask = (j < max_node_num) & (k < max_node_num)
                i, j, k = i[mask], j[mask], k[mask]
                i, j, k = torch.cat((i, i + half_n_rel), 0), torch.cat((j, k), 0), torch.cat((k, j), 0)  # add inverse relations
                edge_index.append(torch.stack([j,k], dim=0)) #each entry is [2, E]
                edge_type.append(i) #each entry is [E, ]

            if not self.debug:
                with open(cache_path, 'wb') as f:
                    pickle.dump([adj_lengths_ori, concept_ids, node_type_ids, node_scores, adj_lengths, edge_index, edge_type, half_n_rel, special_nodes_mask], f)


        ori_adj_mean  = adj_lengths_ori.float().mean().item()
        ori_adj_sigma = np.sqrt(((adj_lengths_ori.float() - ori_adj_mean)**2).mean().item())
        print('| ori_adj_len: mu {:.2f} sigma {:.2f} | adj_len: {:.2f} |'.format(ori_adj_mean, ori_adj_sigma, adj_lengths.float().mean().item()) +
            ' prune_rate： {:.2f} |'.format((adj_lengths_ori > adj_lengths).float().mean().item()) +
            ' qc_num: {:.2f} | ac_num: {:.2f} |'.format((node_type_ids == 0).float().sum(1).mean().item(),
                                                        (node_type_ids == 1).float().sum(1).mean().item()))

        edge_index = list(map(list, zip(*(iter(edge_index),) * self.num_choice))) #list of size (n_questions, n_choices), where each entry is tensor[2, E] #this operation corresponds to .view(n_questions, n_choices)
        edge_type = list(map(list, zip(*(iter(edge_type),) * self.num_choice))) #list of size (n_questions, n_choices), where each entry is tensor[E, ]

        concept_ids, node_type_ids, node_scores, adj_lengths, special_nodes_mask = [x.view(-1, self.num_choice, *x.size()[1:]) for x in (concept_ids, node_type_ids, node_scores, adj_lengths, special_nodes_mask)]
        #concept_ids: (n_questions, num_choice, max_node_num)
        #node_type_ids: (n_questions, num_choice, max_node_num)
        #node_scores: (n_questions, num_choice, max_node_num)
        #adj_lengths: (n_questions,　num_choice)
        return concept_ids, node_type_ids, node_scores, adj_lengths, special_nodes_mask, (edge_index, edge_type) #, half_n_rel * 2 + 1


def load_gpt_input_tensors(statement_jsonl_path, max_seq_length):
    def _truncate_seq_pair(tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def load_qa_dataset(dataset_path):
        """ Output a list of tuples(story, 1st continuation, 2nd continuation, label) """
        with open(dataset_path, "r", encoding="utf-8") as fin:
            output = []
            for line in fin:
                input_json = json.loads(line)
                label = ord(input_json.get("answerKey", "A")) - ord("A")
                output.append((input_json['id'], input_json["question"]["stem"], *[ending["text"] for ending in input_json["question"]["choices"]], label))
        return output

    def pre_process_datasets(encoded_datasets, num_choices, max_seq_length, start_token, delimiter_token, clf_token):
        """ Pre-process datasets containing lists of tuples(story, 1st continuation, 2nd continuation, label)

            To Transformer inputs of shape (n_batch, n_alternative, length) comprising for each batch, continuation:
            input_ids[batch, alternative, :] = [start_token] + story[:cap_length] + [delimiter_token] + cont1[:cap_length] + [clf_token]
        """
        tensor_datasets = []
        for dataset in encoded_datasets:
            n_batch = len(dataset)
            input_ids = np.zeros((n_batch, num_choices, max_seq_length), dtype=np.int64)
            mc_token_ids = np.zeros((n_batch, num_choices), dtype=np.int64)
            lm_labels = np.full((n_batch, num_choices, max_seq_length), fill_value=-1, dtype=np.int64)
            mc_labels = np.zeros((n_batch,), dtype=np.int64)
            for i, data, in enumerate(dataset):
                q, mc_label = data[0], data[-1]
                choices = data[1:-1]
                for j in range(len(choices)):
                    _truncate_seq_pair(q, choices[j], max_seq_length - 3)
                    qa = [start_token] + q + [delimiter_token] + choices[j] + [clf_token]
                    input_ids[i, j, :len(qa)] = qa
                    mc_token_ids[i, j] = len(qa) - 1
                    lm_labels[i, j, :len(qa) - 1] = qa[1:]
                mc_labels[i] = mc_label
            all_inputs = (input_ids, mc_token_ids, lm_labels, mc_labels)
            tensor_datasets.append(tuple(torch.tensor(t) for t in all_inputs))
        return tensor_datasets

    def tokenize_and_encode(tokenizer, obj):
        """ Tokenize and encode a nested object """
        if isinstance(obj, str):
            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
        elif isinstance(obj, int):
            return obj
        else:
            return list(tokenize_and_encode(tokenizer, o) for o in obj)

    tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
    tokenizer.add_tokens(GPT_SPECIAL_TOKENS)
    special_tokens_ids = tokenizer.convert_tokens_to_ids(GPT_SPECIAL_TOKENS)

    dataset = load_qa_dataset(statement_jsonl_path)
    examples_ids = [data[0] for data in dataset]
    dataset = [data[1:] for data in dataset]  # discard example ids
    num_choices = len(dataset[0]) - 2

    encoded_dataset = tokenize_and_encode(tokenizer, dataset)

    (input_ids, mc_token_ids, lm_labels, mc_labels), = pre_process_datasets([encoded_dataset], num_choices, max_seq_length, *special_tokens_ids)
    return examples_ids, mc_labels, input_ids, mc_token_ids, lm_labels


def load_bert_xlnet_roberta_input_tensors(statement_jsonl_path, max_seq_length, debug, tokenizer, debug_sample_size):
    class InputExample(object):

        def __init__(self, example_id, question, contexts, endings, label=None):
            self.example_id = example_id
            self.question = question
            self.contexts = contexts
            self.endings = endings
            self.label = label

    class InputFeatures(object):

        def __init__(self, example_id, choices_features, label):
            self.example_id = example_id
            self.choices_features = [
                {
                    'input_ids': input_ids,
                    'input_mask': input_mask,
                    'segment_ids': segment_ids,
                    'output_mask': output_mask,
                }
                for input_ids, input_mask, segment_ids, output_mask in choices_features
            ]
            self.label = label

    def read_examples(input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            examples = []
            for line in f.readlines():
                json_dic = json.loads(line)
                label = ord(json_dic["answerKey"]) - ord("A") if 'answerKey' in json_dic else 0
                contexts = json_dic["question"]["stem"]
                if "para" in json_dic:
                    contexts = json_dic["para"] + " " + contexts
                if "fact1" in json_dic:
                    contexts = json_dic["fact1"] + " " + contexts
                examples.append(
                    InputExample(
                        example_id=json_dic["id"],
                        contexts=[contexts] * len(json_dic["question"]["choices"]),
                        question="",
                        endings=[ending["text"] for ending in json_dic["question"]["choices"]],
                        label=label
                    ))
        return examples

    def simple_convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
        """ Loads a data file into a list of `InputBatch`s
            `cls_token_at_end` define the location of the CLS token:
                - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
                - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
            `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
        """
        label_map = {label: i for i, label in enumerate(label_list)}

        features = []
        concepts_by_sents_list = []
        for ex_index, example in tqdm(enumerate(examples), total=len(examples), desc="Converting examples to features"):
            if debug and ex_index >= debug_sample_size:
                break
            choices_features = []
            for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)):
                ans = example.question + " " + ending

                encoded_input = tokenizer(context, ans, padding="max_length", truncation=True, max_length=max_seq_length, return_token_type_ids=True, return_special_tokens_mask=True)
                input_ids = encoded_input["input_ids"]
                output_mask = encoded_input["special_tokens_mask"]
                input_mask = encoded_input["attention_mask"]
                segment_ids = encoded_input["token_type_ids"]

                assert len(input_ids) == max_seq_length
                assert len(output_mask) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length

                choices_features.append((input_ids, input_mask, segment_ids, output_mask))
            label = label_map[example.label]
            features.append(InputFeatures(example_id=example.example_id, choices_features=choices_features, label=label))

        return features, concepts_by_sents_list

    def select_field(features, field):
        return [[choice[field] for choice in feature.choices_features] for feature in features]

    def convert_features_to_tensors(features):
        all_input_ids = torch.tensor(select_field(features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(features, 'segment_ids'), dtype=torch.long)
        all_output_mask = torch.tensor(select_field(features, 'output_mask'), dtype=torch.bool)
        all_label = torch.tensor([f.label for f in features], dtype=torch.long)
        return all_input_ids, all_input_mask, all_segment_ids, all_output_mask, all_label

    examples = read_examples(statement_jsonl_path)
    features, concepts_by_sents_list = simple_convert_examples_to_features(examples, list(range(len(examples[0].endings))), max_seq_length, tokenizer)
    example_ids = [f.example_id for f in features]
    *data_tensors, all_label = convert_features_to_tensors(features)
    return example_ids, all_label, data_tensors, concepts_by_sents_list


def load_bert_xlnet_roberta_input_tensors_with_linking_mask(statement_jsonl_path,
                                                            max_seq_length,
                                                            debug,
                                                            tokenizer,
                                                            debug_sample_size,
                                                            linking_mask_save_path=None,
                                                            linked_span_save_path=None,
                                                            linked_ids_save_path=None,
                                                            preprocess=True,
                                                            is_med=False):
    def load_matcher(nlp, pattern_path):
        with open(pattern_path, "r", encoding="utf8") as fin:
            all_patterns = json.load(fin)

        matcher = Matcher(nlp.vocab)
        for concept, pattern in all_patterns.items():
            matcher.add(concept, [pattern])
        return matcher

    def lemmatize(nlp, concept):

        doc = nlp(concept.replace("_", " "))
        lcs = set()
        lcs.add("_".join([token.lemma_ for token in doc]))  # all lemma
        return lcs

    def entity_linking_to_umls(sentence, nlp, linker):
        doc = nlp(sentence)
        entities = doc.ents
        all_entities_results = []
        for mm in range(len(entities)):
            entity_text = entities[mm].text
            entity_start = entities[mm].start
            entity_end = entities[mm].end
            all_linked_entities = entities[mm]._.kb_ents
            all_entity_results = []
            for ii in range(len(all_linked_entities)):
                curr_concept_id = all_linked_entities[ii][0]
                curr_score = all_linked_entities[ii][1]
                curr_scispacy_entity = linker.kb.cui_to_entity[all_linked_entities[ii][0]]
                curr_canonical_name = curr_scispacy_entity.canonical_name
                curr_TUIs = curr_scispacy_entity.types
                curr_entity_result = {"Canonical Name": curr_canonical_name, "Concept ID": curr_concept_id,
                                      "TUIs": curr_TUIs, "Score": curr_score}
                all_entity_results.append(curr_entity_result)
            curr_entities_result = {"text": entity_text, "start": entity_start, "end": entity_end,
                                    "start_char": entities[mm].start_char, "end_char": entities[mm].end_char,
                                    "linking_results": all_entity_results}
            all_entities_results.append(curr_entities_result)
        return doc, all_entities_results

    def map_to_ddb(ent_obj):
        res = []
        for ent_cand in ent_obj['linking_results']:
            CUI = ent_cand['Concept ID']
            name = ent_cand['Canonical Name']
            if CUI in umls_to_ddb:
                ddb_cid = umls_to_ddb[CUI]
                res.append((ddb_cid, name))
        return res

    if os.path.exists(linked_span_save_path) and os.path.exists(linked_ids_save_path):
        preprocess = False
    if preprocess:
        if not is_med:
            nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])
            # nlp.add_pipe(nlp.create_pipe('sentencizer'))
            nlp.add_pipe('sentencizer')
            matcher = load_matcher(nlp, './data/cpnet/matcher_patterns.json')
            with open('./data/cpnet/concept.txt', "r", encoding="utf8") as fin:
                id2concept = [w.strip() for w in fin]
            concept2id = {w: i for i, w in enumerate(id2concept)}
        else:
            nlp = spacy.load("en_core_sci_sm")
            linker = EntityLinker(
                resolve_abbreviations=True,
                name="umls",
                threshold=0.90)
            nlp.add_pipe(linker)

            umls_to_ddb = {}
            with open('./data/ddb/ddb_to_umls_cui.txt') as f:
                for line in f.readlines()[1:]:
                    elms = line.split("\t")
                    umls_to_ddb[elms[2]] = elms[1]

            with open('./data/ddb/ptrs.txt', "r", encoding="utf8") as fin:
                id2concept = [w.strip() for w in fin]
            concept2id = {w: i for i, w in enumerate(id2concept)}

    class InputExample(object):

        def __init__(self, example_id, question, contexts, endings, label=None):
            self.example_id = example_id
            self.question = question
            self.contexts = contexts
            self.endings = endings
            self.label = label

    class InputFeatures(object):

        def __init__(self, example_id, choices_features, label):
            self.example_id = example_id
            self.choices_features = [
                {
                    'input_ids': input_ids,
                    'input_mask': input_mask,
                    'segment_ids': segment_ids,
                    'output_mask': output_mask,
                }
                for input_ids, input_mask, segment_ids, output_mask in choices_features
            ]
            self.label = label

    def read_examples(input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            examples = []
            for line in f.readlines():
                json_dic = json.loads(line)
                label = ord(json_dic["answerKey"]) - ord("A") if 'answerKey' in json_dic else 0
                contexts = json_dic["question"]["stem"]
                if "para" in json_dic:
                    contexts = json_dic["para"] + " " + contexts
                if "fact1" in json_dic:
                    contexts = json_dic["fact1"] + " " + contexts
                examples.append(
                    InputExample(
                        example_id=json_dic["id"],
                        contexts=[contexts] * len(json_dic["question"]["choices"]),
                        question="",
                        endings=[ending["text"] for ending in json_dic["question"]["choices"]],
                        label=label
                    ))
        return examples

    def simple_convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
        """ Loads a data file into a list of `InputBatch`s
            `cls_token_at_end` define the location of the CLS token:
                - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
                - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
            `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
        """
        label_map = {label: i for i, label in enumerate(label_list)}

        features = []
        concepts_by_sents_list = []
        for ex_index, example in tqdm(enumerate(examples), total=len(examples), desc="Converting examples to features"):
            if debug and ex_index >= debug_sample_size:
                break
            choices_features = []
            for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)):
                ans = example.question + " " + ending

                encoded_input = tokenizer(context, ans, padding="max_length", truncation=True, max_length=max_seq_length, return_token_type_ids=True, return_special_tokens_mask=True)
                input_ids = encoded_input["input_ids"]
                output_mask = encoded_input["special_tokens_mask"]
                input_mask = encoded_input["attention_mask"]
                segment_ids = encoded_input["token_type_ids"]

                assert len(input_ids) == max_seq_length
                assert len(output_mask) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length

                choices_features.append((input_ids, input_mask, segment_ids, output_mask))
            label = label_map[example.label]
            features.append(InputFeatures(example_id=example.example_id, choices_features=choices_features, label=label))

        return features, concepts_by_sents_list

    def process_linking_mask(examples, label_list, max_seq_length, tokenizer):
        """ Loads a data file into a list of `InputBatch`s
            `cls_token_at_end` define the location of the CLS token:
                - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
                - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
            `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
        """
        def _func(input_str, is_ans=False):
            doc = nlp(input_str)
            context_matches = matcher(doc)
            context_matches = sorted(context_matches, key=lambda x: x[1])
            left = 0
            splits = []
            linked_entity = []
            linked_span = []
            cur_entities = None
            for match_id, start, end in context_matches:
                span = doc[start:end]
                idx = span[0].idx
                original_concept = nlp.vocab.strings[match_id]
                original_concept_set = set()
                original_concept_set.add(original_concept)

                if len(original_concept.split("_")) == 1:
                    original_concept_set.update(lemmatize(nlp, nlp.vocab.strings[match_id]))
                if idx <= left:
                    if cur_entities:
                        cur_entities.update(original_concept_set)
                    continue
                if idx > 0 and input_str[idx - 1] != ' ':
                    continue
                if idx + len(span.text) < len(input_str) - 1 and input_str[idx + len(span.text)] != ' ':
                    continue
                if cur_entities is not None:
                    linked_entity.append(cur_entities)
                linked_span.append([idx, idx + len(span.text)])
                cur_entities = original_concept_set
                splits.append((input_str[left: idx], False))
                splits.append((input_str[idx: idx + len(span.text)], True))
                left = idx + len(span.text)
            if left != len(input_str):
                splits.append((input_str[left:], False))
            if cur_entities is not None:
                linked_entity.append(cur_entities)

            linked_entity_ids = []
            for entity_set in linked_entity:
                linked_entity_ids.append([concept2id[_] for _ in list(entity_set) if _ in concept2id])

            assert len(linked_span) == len(linked_entity_ids)

            linked_input_ids_span = []
            linking_mask = []
            tokens = []
            for i, (text, is_linking) in enumerate(splits):
                text = text.strip(' ')
                if len(text) == 0:
                    continue
                flag = (is_ans or i != 0) and text[0] not in [',', '?', '!', '.', ';', ':']
                _tokens = tokenizer.tokenize(text, is_split_into_words=flag)
                if is_linking:
                    linked_input_ids_span.append([len(linking_mask), len(linking_mask) + len(_tokens)])
                    _mask = [1] * len(_tokens)
                else:
                    _mask = [0] * len(_tokens)
                tokens += _tokens
                linking_mask += _mask
            # print(linking_mask)
            # print(linked_input_ids_span)
            assert len(linked_input_ids_span) == len(linked_entity_ids)

            return splits, tokens, linking_mask, np.array(linked_input_ids_span), linked_entity_ids

        def _func_for_med(input_str, is_ans=False):
            # print(input_str)
            doc, all_ent_linkings = entity_linking_to_umls(input_str, nlp, linker)
            left = 0
            splits = []
            linked_entity = []
            linked_span = []
            linked_entity_ids = []
            for obj in all_ent_linkings:
                span = doc[obj['start']:obj['end']]
                idx = span[0].idx
                ddb_res = map_to_ddb(obj)
                if len(ddb_res):
                    if idx != left:
                        splits.append([input_str[left:idx], False])
                    # print(span.text, ddb_res)
                    linked_entity_ids.append([concept2id[_[0]] for _ in ddb_res])
                    linked_span.append([idx, idx + len(span.text)])
                    splits.append([input_str[idx: idx + len(span.text)], True])
                    left = idx + len(span.text)
            if left != len(input_str):
                splits.append((input_str[left:], False))
            # print(linked_span)
            # print(splits)

            linked_input_ids_span = []
            linking_mask = []
            tokens = []
            for i, (text, is_linking) in enumerate(splits):
                text = text.strip(' ')
                if len(text) == 0:
                    continue
                flag = (is_ans or i != 0) and text[0] not in [',', '?', '!', '.', ';', ':']
                _tokens = tokenizer.tokenize(text, is_split_into_words=flag)
                if is_linking:
                    linked_input_ids_span.append([len(linking_mask), len(linking_mask) + len(_tokens)])
                    _mask = [1] * len(_tokens)
                else:
                    _mask = [0] * len(_tokens)
                tokens += _tokens
                linking_mask += _mask
            # print(tokens)
            # print(linking_mask)
            # print(tokenizer.tokenize(input_str))
            assert len(linked_span) == len(linked_entity_ids)

            assert len(linked_input_ids_span) == len(linked_entity_ids)
            # exit('debug')
            return splits, tokens, linking_mask, np.array(linked_input_ids_span), linked_entity_ids
        exceed_cnt = 0
        linking_masks = []
        linked_span = []
        linked_ids = []
        illegal_cnt = 0
        for ex_index, example in tqdm(enumerate(examples), total=len(examples), desc="Converting examples to features"):
            if debug and ex_index >= debug_sample_size:
                break
            for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)):
                ans = example.question + " " + ending
                context = re.sub('\s+', ' ', context)
                ans = re.sub('\s+', ' ', ans)
                '''add linking mask'''
                if not is_med:
                    _splits1, context_tokens, context_linking_mask, context_linked_span, context_linked_ids = _func(context)
                    _splits2, ans_tokens, ans_linking_mask, ans_linked_span, ans_linked_ids = _func(ans, True)
                else:
                    _splits1, context_tokens, context_linking_mask, context_linked_span, context_linked_ids = _func_for_med(
                        context)
                    _splits2, ans_tokens, ans_linking_mask, ans_linked_span, ans_linked_ids = _func_for_med(ans, True)
                total_len = len(context_linking_mask) + len(ans_linking_mask) + tokenizer.num_special_tokens_to_add(
                    pair=True)
                exceed_flag = False
                if total_len > max_seq_length:
                    exceed_cnt += 1
                    print('max_length_exceed! {}'.format(exceed_cnt))
                    exceed_flag = True
                    context_linking_mask, ans_linking_mask, overflowing_tokens = tokenizer.truncate_sequences(
                        context_linking_mask,
                        pair_ids=ans_linking_mask,
                        num_tokens_to_remove=total_len - max_seq_length,
                        truncation_strategy="longest_first",
                        stride=0,
                    )
                context_linking_mask = [0] + context_linking_mask + [0]
                context_linked_span += 1
                if not is_med:
                    ans_linking_mask = [0] + ans_linking_mask + [0]
                    ans_linked_span += 1 + len(context_linking_mask)
                else:
                    ans_linking_mask = ans_linking_mask + [0]
                    ans_linked_span += len(context_linking_mask)
                merged_linking_mask = context_linking_mask + ans_linking_mask
                merged_linked_span = context_linked_span.tolist() + ans_linked_span.tolist()
                merged_linked_ids = context_linked_ids + ans_linked_ids

                if exceed_flag:
                    pos = None
                    for i, (start, end) in enumerate(merged_linked_span):
                        if end >= max_seq_length:
                            pos = i
                            break
                    if pos is not None:
                        merged_linked_span = merged_linked_span[:pos]
                        merged_linked_ids = merged_linked_ids[:pos]

                assert len(merged_linked_span) == len(merged_linked_ids)

                _encoded_input = tokenizer(context, ans, return_token_type_ids=True,
                                           return_special_tokens_mask=True)
                try:
                    assert len(merged_linking_mask) == len(_encoded_input['input_ids'])
                except AssertionError:
                    print(len(merged_linking_mask), len(_encoded_input['input_ids']))
                    print(tokenizer.convert_ids_to_tokens(_encoded_input['input_ids']))
                    print(context)
                    print(ans)
                    print(context_tokens + ans_tokens)
                    print(_splits1 + _splits2)
                    print(tokenizer.tokenize(context + ans))
                    illegal_cnt += 1
                    # exit('error')
                pad_length = max_seq_length - len(merged_linking_mask)
                merged_linking_mask += [0] * pad_length
                linking_masks.append(merged_linking_mask)
                linked_span.append(merged_linked_span)
                linked_ids.append(merged_linked_ids)
        if not debug:
            with open(linking_mask_save_path, 'w') as f:
                json.dump(linking_masks, f)
            with open(linked_span_save_path, 'w') as f:
                json.dump(linked_span, f)
            with open(linked_ids_save_path, 'w') as f:
                json.dump(linked_ids, f)
            with open('./process.log', 'a+') as f:
                f.write('illegal cnt: {}\n'.format(illegal_cnt))
        else:
            pass

    def select_field(features, field):
        return [[choice[field] for choice in feature.choices_features] for feature in features]

    def convert_features_to_tensors(features):
        all_input_ids = torch.tensor(select_field(features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(features, 'segment_ids'), dtype=torch.long)
        all_output_mask = torch.tensor(select_field(features, 'output_mask'), dtype=torch.bool)
        all_label = torch.tensor([f.label for f in features], dtype=torch.long)
        return all_input_ids, all_input_mask, all_segment_ids, all_output_mask, all_label

    examples = read_examples(statement_jsonl_path)
    if preprocess:
        print('pre-processing the align data...')
        process_linking_mask(examples, list(range(len(examples[0].endings))), max_seq_length, tokenizer)
    features, concepts_by_sents_list = simple_convert_examples_to_features(examples,
                                                                           list(range(len(examples[0].endings))),
                                                                           max_seq_length,
                                                                           tokenizer)
    with open(linked_span_save_path) as f:
        linked_span = json.load(f)
    with open(linked_ids_save_path) as f:
        linked_ids = json.load(f)
    example_ids = [f.example_id for f in features]
    *data_tensors, all_label = convert_features_to_tensors(features)
    num_choice = data_tensors[0].shape[1]
    if debug:
        linked_span = linked_span[:debug_sample_size * num_choice]
        linked_ids = linked_ids[:debug_sample_size * num_choice]
    print(data_tensors[0].shape)
    print(len(linked_span), len(linked_ids))
    assert len(linked_span) % num_choice == 0
    reshaped_span = []
    for i in range(len(linked_span) // num_choice):
        reshaped_span.append(linked_span[i * num_choice: (i + 1) * num_choice])
    assert len(linked_ids) % num_choice == 0
    reshaped_ids = []
    for i in range(len(linked_ids) // num_choice):
        reshaped_ids.append(linked_ids[i * num_choice: (i + 1) * num_choice])
    linked_span = reshaped_span
    linked_ids = reshaped_ids
    print(len(linked_span), len(linked_ids))
    # exit('debug')
    return example_ids, all_label, data_tensors, concepts_by_sents_list, (linked_span, linked_ids)
