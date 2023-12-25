import pickle
import os
import random
import numpy as np
import logging
logger = logging.getLogger("MAIN")
import json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
import itertools
from copy import deepcopy
import spacy

from utils.data_utils import *
from utils.other_utils import *


class BatchGenerator(object):
    def __init__(self, args, device, batch_size, input_data, tokenizer, is_shuffle=True):
        
        self.args = args
        self.n_samples = len(input_data['example_id'])
        self.n_batch = (self.n_samples-1)//batch_size + 1
        self.map_idx = list(range(self.n_samples))
        self.is_shuffle = is_shuffle
        if self.is_shuffle:
            self.shuffle_idx()
        
        self.device = device
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.pad_token_ids = tokenizer.pad_token_id
        self.input_data = input_data
        
    def __len__(self):
        return self.n_batch

    def __getitem__(self, batch_idx, is_skip_last_batch=False):
        # the idx-th batch
        assert batch_idx>=0 and batch_idx<self.n_batch, 'Invalid batch_idx: {batch_idx}'

        bg_idx = batch_idx*self.batch_size
        ed_idx = (batch_idx+1)*self.batch_size
        if ed_idx > self.n_samples:
            ed_idx = self.n_samples
        selected_index = self.map_idx[bg_idx:ed_idx]
        if self.is_shuffle:
            # all samples are generated 
            # or all samples except the last batch are generated (when is_skip_last_batch and the data is for training!)
            if ed_idx == self.n_samples or (is_skip_last_batch and batch_idx==(self.n_batch-2)):
                self.shuffle_idx()

        batch_data = deepcopy({k:[v[idx] for idx in selected_index] for k,v in self.input_data.items()})
        bs = len(batch_data['example_id'])
        is_load_ref_str = True if len(batch_data['ref_str'][0]) else False

        batch_keys = list(batch_data.keys())
        for k in batch_keys:
            if k in ['example_id','endings_label']:
                continue
            if k in ['example_label']:
                batch_data[k] = torch.tensor(batch_data[k], dtype=torch.long).to(self.device)
                continue
            if self.args.input_format=='each_option':
                # Enforce all samples in a batch have the same number of options!
                # pad to (bs*max_nc, )
                choice_num_lst = [len(_ending_label) for _ending_label in batch_data['endings_label']]
                max_nc = max(choice_num_lst)
                for tmp_i in range(len(choice_num_lst)):
                    while len(batch_data['input_str'][tmp_i]) < max_nc:
                        sample_wrong_choice = random.choice([choice_id for choice_id in range(choice_num_lst[tmp_i]) if choice_id!=batch_data['example_label'][tmp_i]])
                        batch_data['input_str'][tmp_i].append(batch_data['input_str'][tmp_i][sample_wrong_choice])
                        if batch_data.get('ref_str') is not None:
                            # (nc, topk) -> (max_nc, topk)
                            # (nc*topk,) -> (max_nc*topk,)
                            sample_wrong_choice_ref = batch_data['ref_str'][tmp_i][sample_wrong_choice*self.args.CET_topk:(sample_wrong_choice+1)*self.args.CET_topk]
                            batch_data['ref_str'][tmp_i].extend(sample_wrong_choice_ref)

                # (bs, max_nc) -> (bs*max_nc, seq_len)       
                if k in ['input_str']:
                    flatten_input_str = list(itertools.chain(*batch_data['input_str']))
                    batch_data['LM_input'] = self.tokenizer(
                                                    flatten_input_str, 
                                                    return_tensors="pt", 
                                                    padding='longest', 
                                                    truncation=True,
                                                    max_length=self.args.max_input_len
                                                ).to(self.device)
                
                # (bs, max_nc*topk) -> (max_nc*(ref_cnt_1+ref_cnt_2+...), seq_len)  
                elif self.args.is_CET and k in ['ref_str'] and is_load_ref_str:
                    if np.sum(batch_data['ref_cnt']) == 0:
                        batch_data['ref_LM_input'] = None
                    else:
                        flatten_ref_str = []
                        for tmp_i in range(bs):
                            ref_cnt = batch_data['ref_cnt'][tmp_i]
                            if ref_cnt==0:
                                continue
                            for tmp_j in range(max_nc):
                                flatten_ref_str.extend(
                                                    batch_data['ref_str'][tmp_i][tmp_j*self.args.CET_topk:tmp_j*self.args.CET_topk+ref_cnt] 
                                                )
                        batch_data['ref_LM_input'] = self.tokenizer(
                                                        flatten_ref_str, 
                                                        return_tensors="pt", 
                                                        padding='longest', 
                                                        truncation=True,
                                                        max_length=self.args.max_input_len
                                                    ).to(self.device)
                    
            elif self.args.input_format=='all_option':
                # pad to (bs, -1)
                if k in ['input_str']:
                    batch_data['LM_input'] = self.tokenizer(
                                                batch_data['input_str'], 
                                                return_tensors="pt", 
                                                padding='longest', 
                                                truncation=True,
                                                max_length=self.args.max_input_len
                                            ).to(self.device)

                elif self.args.is_CET and k in ['ref_str'] and is_load_ref_str:
                    if np.sum(batch_data['ref_cnt']) == 0:
                        batch_data['ref_LM_input'] = None
                    else:
                        flatten_ref_str = []
                        for tmp_i in range(bs):
                            ref_cnt = batch_data['ref_cnt'][tmp_i]
                            if ref_cnt==0:
                                continue
                            flatten_ref_str.extend(
                                                batch_data['ref_str'][tmp_i][:ref_cnt] 
                                            )
                        batch_data['ref_LM_input'] = self.tokenizer(
                                                        flatten_ref_str,
                                                        return_tensors="pt", 
                                                        padding='longest', 
                                                        truncation=True,
                                                        max_length=self.args.max_input_len
                                                    ).to(self.device)

                elif self.args.is_CET and k in ['ref_input_ids']:
                    ref_LM_input = {}
                    ref_max_len = max([len(one_lst) for lst in batch_data['ref_input_ids'] for one_lst in lst])
                    bs, topk = len(batch_data['ref_input_ids']), len(batch_data['ref_input_ids'][0])
                    ref_LM_input['input_ids'] = torch.ones(bs*topk,ref_max_len).long().to(self.device)*self.tokenizer.pad_token_id
                    ref_LM_input['attention_mask'] = torch.zeros_like(ref_LM_input['input_ids']).long().to(self.device)

                    for i, lst in enumerate(batch_data['ref_input_ids']):
                        for j, one_lst in enumerate(lst):
                            seq_len = len(one_lst)
                            ref_LM_input['input_ids'][i*topk+j][:seq_len] = torch.from_numpy(one_lst).long()
                            ref_LM_input['attention_mask'][i*topk+j][:seq_len] = torch.ones(seq_len).long()

                    batch_data['ref_LM_input'] = ref_LM_input

        return batch_data

    def shuffle_idx(self):
        random.shuffle(self.map_idx)

    def generate_refs(self, model=None, load_cache=True):

        input_data = self.input_data
        sim_thres = self.args.CET_sim_thres
        cache_path = os.path.join(self.args.dataset_dir,'ref_str_{}_{}_nsamples{}_top{}_{}.pk'.format(
                                    self.args.input_format,
                                    self.args.pretrain_model,
                                    len(input_data['input_str']),
                                    self.args.CET_topk,
                                    'thres%.2f'%sim_thres
                                    )
                                )

        if load_cache and os.path.isfile(cache_path):
            logger.info('Loading cache for ref str from %s'%cache_path)
            with open(cache_path,'rb') as f:
                cache_dict = pickle.load(f)
                input_data['ref_str'] = cache_dict.get('ref_str')
                input_data['ref_cnt'] = cache_dict.get('ref_cnt')
            return input_data

        nlp = spacy.load('en_core_web_lg')
        gt_answer_lst = [eds[i].strip().lower() for eds, i in zip(input_data['endings'], input_data['example_label'])]
        n_samples = len(gt_answer_lst)
        sim_matrix = np.zeros((n_samples, n_samples))
        doc_lst = [nlp(ans) for ans in gt_answer_lst]
        for i in range(n_samples):
            for j in range(n_samples):
                if i<j:
                    continue
                sim_score = doc_lst[i].similarity(doc_lst[j])
                sim_matrix[i][j] = sim_score
                sim_matrix[j][i] = sim_score
        sim_matrix = sim_matrix - np.eye(n_samples) * 1e8
        match_sim_matrix, match_id_matrix = torch.topk(torch.from_numpy(sim_matrix), 
                                                    k=self.args.CET_topk, 
                                                    largest=True, 
                                                    dim=1) # (num_sample, topk)                          

        n_samples = len(input_data['input_str'])
        ref_str_all = []
        ref_cnt_all = []
        for i in range(n_samples):
            if self.args.input_format=='each_option':
                ref_str_lst = [] # (n_option, topk) -> (n_option*topk,)
                n_option = len(input_data['input_str'][i])
                ref_cnt = 0
                # each context should be the same for each options
                for option_id in range(n_option):
                    option_str = input_data['endings'][i][option_id]
                    for k in range(self.args.CET_topk):              
                        match_sim = match_sim_matrix[i][k]
                        
                        # Note: Pad the number of ref samples to topK when not enough KNN are found
                        if sim_thres>0 and match_sim<sim_thres:
                            match_id = i
                        # Invalid similarity: the range should be [0,1]
                        elif match_sim>1.0:
                            match_id = i
                        else:
                            match_id = match_id_matrix[i][k]
                            # only count ref_cnt for the first option
                            ref_cnt = ref_cnt+1 if option_id==0 else ref_cnt

                        one_ref_question = input_data['contexts'][match_id][0]
                        ref_str_lst.append(one_ref_question+' '+option_str)
                ref_str_all.append(ref_str_lst)
                ref_cnt_all.append(ref_cnt)
            elif self.args.input_format=='all_option':
                ref_str_lst = [] # (topk,)
                n_option = len(input_data['endings'][i])
                ref_cnt = 0
                option_str = ' \\n '
                for ed_idx, ed in enumerate(input_data['endings'][i]):
                    option_str += '('+chr(ord('A')+ed_idx)+')'+' '+ed+' '

                for k in range(self.args.CET_topk):              
                    match_sim = match_sim_matrix[i][k]
                    
                    # Note: Pad the number of ref samples to topK when not enough KNN are found
                    if sim_thres>0 and match_sim<sim_thres:
                        match_id = i
                    # Invalid similarity: the range should be [0,1]
                    elif match_sim>1.0:
                        match_id = i
                    else:
                        match_id = match_id_matrix[i][k]
                        # only count ref_cnt for the first option
                        ref_cnt = ref_cnt+1
                    

                    one_ref_question = input_data['contexts'][match_id]
                    ref_str_lst.append(one_ref_question+' '+option_str)

                ref_str_all.append(ref_str_lst)
                ref_cnt_all.append(ref_cnt)
            else:
                raise Exception('Invalid input_format %s'%(self.args.input_format))

        input_data['ref_str'] = ref_str_all
        input_data['ref_cnt'] = ref_cnt_all

        with open(cache_path,'wb') as f:
            logger.info('Saving cache for ref str to %s'%(cache_path))
            pickle.dump({'ref_str':ref_str_all,
                        'ref_cnt':ref_cnt_all,
                        'match_sim_matrix':match_sim_matrix,
                        'match_id_matrix':match_id_matrix,},f)

        self.input_data = input_data

class InputExample(object):
    def __init__(self, example_id, contexts, endings, endings_label, label, input_str, ref_str):
        # General
        self.example_id = example_id
        self.contexts = contexts
        self.endings = endings
        self.endings_label = endings_label
        self.label = label
        self.input_str = input_str
        self.ref_str = ref_str

def read_statement_examples(input_file, args):
    with open(input_file, "r", encoding="utf-8") as f:
        examples = []
        for line in f.readlines():
            json_dic = json.loads(line)
            example_id = json_dic["id"]
            num_choice = len(json_dic['question']['choices'])
            # answer
            if 'answerKey' in json_dic:
                label = 0
                endings_label = [0]*num_choice
                if type(json_dic['answerKey']) is bool:
                    label = int(json_dic['answerKey'])
                elif json_dic['answerKey'].isalpha():
                    label = ord(json_dic["answerKey"]) - ord("A")
                elif json_dic['answerKey'].isdigit():
                    label = ord(json_dic["answerKey"]) - ord("1")
                else:
                    raise Exception("Invalid answerKey %s"%(json_dic['answerKey']))
                endings_label[label] = 1
            else:
                # test set
                label = None
                endings_label = None

            if args.input_format=='each_option':
                # context
                contexts = json_dic["question"]["stem"]
                # if "para" in json_dic:
                #     contexts = json_dic["para"] + " " + contexts
                # if "fact1" in json_dic:
                #     contexts = json_dic["fact1"] + " " + contexts
                contexts = [contexts] * num_choice
                # endings
                endings = [ending["text"] for ending in json_dic["question"]["choices"]]

                # input_str
                input_str = [ct+' '+ed for ct, ed in zip(contexts,endings)]
                # ref_str
                ref_str = json_dic.get("ref_ans",[])
            elif args.input_format=='all_option':
                # context
                contexts = json_dic["question"]["stem"]
                # if "para" in json_dic:
                #     contexts = json_dic["para"] + " " + contexts
                # if "fact1" in json_dic:
                #     contexts = json_dic["fact1"] + " " + contexts
                # endings
                endings = [ending["text"] for ending in json_dic["question"]["choices"]]

                # input_str
                input_str = contexts + ' \\n '
                for ed_idx, ed in enumerate(endings):
                    input_str += '('+chr(ord('A')+ed_idx)+')'+' '+ed+' '
                # ref_str
                ref_str = json_dic.get("ref_ans",[])
            else:
                raise Exception('Invalid input_format %s'%args.input_format)

            examples.append(
                InputExample(
                    example_id = example_id,
                    contexts = contexts,
                    endings = endings,
                    endings_label = endings_label,
                    label = label,
                    input_str = input_str,
                    ref_str = ref_str
                ))
    return examples

def load_input_data(split_name, args): 

    dataset_dir = args.dataset_dir
    cache_path = os.path.join(dataset_dir,'%s_%s_%s_%s_tensors.pk'%(args.dataset, split_name, args.pretrain_model.replace('/','_'), args.input_format))

    if os.path.exists(cache_path):
        with open(cache_path,'rb') as f:
            input_data = pickle.load(f)
            logger.info('Loading input data from %s'%cache_path)
        return input_data
    else:
        if split_name == 'train':
            statement_jsonl_path = args.train_statements
        elif split_name == 'dev':
            statement_jsonl_path = args.dev_statements
        elif split_name == 'test':
            statement_jsonl_path = args.test_statements
        else:
            raise Exception('Invalid split_name %s'%split_name)

        examples = read_statement_examples(statement_jsonl_path, args)

        input_data = {
            'example_id': [e.example_id for e in examples],
            'example_label': [e.label for e in examples],
            'contexts': [e.contexts for e in examples],
            'endings_label': [e.endings_label for e in examples],
            'endings': [e.endings for e in examples],
            'input_str': [e.input_str for e in examples],
            'ref_str': [e.ref_str for e in examples]
        }
        with open(cache_path,'wb') as f:
            pickle.dump(input_data,f,protocol=4)
            logger.info('Saving input data to %s'%cache_path)

        return input_data


class Basic_Dataloader(object):
    def __init__(self, args, devices):
        super().__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.eval_batch_size = args.eval_batch_size
        self.device = devices
        self.is_inhouse = args.inhouse
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrain_model)
        self.pad_token_ids = self.tokenizer.pad_token_id

        self.train_data = load_input_data('train', args)
        self.dev_data = load_input_data('dev', args)
        self.test_data = load_input_data('test', args)

        if self.args.few_shot and (0. < args.subsample and args.subsample < 1.):
            logger.info('Using Few Shot Setting: Ratio=%.2f'%(args.subsample))
            n_train = int(self.train_size() * args.subsample)
            assert n_train > 0
            # generate qids
            few_shot_qids_path = 'few_shot_qids_%.2f.txt'%(args.subsample)
            few_shot_qids_path_bk = os.path.join(args.save_dir, few_shot_qids_path)
            few_shot_qids_path = os.path.join(args.dataset_dir, few_shot_qids_path)
            few_shot_qids = []
            if os.path.exists(few_shot_qids_path):
                logger.info('Loading few shot qids from %s'%(few_shot_qids_path))
                with open(few_shot_qids_path,'r') as f:
                    few_shot_qids = list(set(line.strip() for line in f))
                selected_indexes = [self.train_data['example_id'].index(qid) for qid in few_shot_qids]
            else:
                selected_indexes = torch.randperm(len(self.train_data['example_id']))[:n_train]
                few_shot_qids = [self.train_data['example_id'][idx] for idx in selected_indexes]
                logger.info('Saving few shot qids to %s'%(few_shot_qids_path))
                with open(few_shot_qids_path,'w') as f:
                    for qid in few_shot_qids:
                        f.write('%s\n'%(qid))  
            logger.info('Saving few shot qids to %s'%(few_shot_qids_path_bk))
            with open(few_shot_qids_path_bk,'w') as f:
                for qid in few_shot_qids:
                    f.write('%s\n'%(qid))
            # select qids
            fewshot_train_data = {k:[v[idx] for idx in selected_indexes] for k,v in self.train_data.items()}
            self.train_data = fewshot_train_data

    def train_size(self):
        return len(self.train_data['example_id'])

    def dev_size(self):
        return len(self.dev_data['example_id'])

    def test_size(self):
        return len(self.test_data['example_id'])

    def train(self):
        return BatchGenerator(
                    self.args, 
                    self.device, 
                    self.batch_size, 
                    input_data=self.train_data, 
                    tokenizer=self.tokenizer, 
                    is_shuffle=True
                )

    def dev(self):
        return BatchGenerator(
                    self.args, 
                    self.device, 
                    self.eval_batch_size, 
                    input_data=self.dev_data, 
                    tokenizer=self.tokenizer,
                    is_shuffle=False
                )

    def test(self):
        return BatchGenerator(
                    self.args, 
                    self.device, 
                    self.eval_batch_size, 
                    input_data=self.test_data, 
                    tokenizer=self.tokenizer,
                    is_shuffle=False
                )
