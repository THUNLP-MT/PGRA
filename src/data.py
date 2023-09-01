# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import random
import json
import numpy as np


class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 n_context=None,
                 tasks = 'cola',
                 question_prefix='question:',
                 question_suffix=' This is <extra_id_0> .',
                 title_prefix='',
                 context_suffix=' This is',
                 passage_prefix='context:'):
        self.data = data
        self.n_context = n_context
        self.question_prefix = question_prefix
        self.question_suffix = question_suffix
        self.title_prefix = title_prefix
        self.context_suffix = context_suffix
        self.passage_prefix = passage_prefix
        self.tasks = tasks
        if self.tasks == "trec":
            self.question_suffix = '<extra_id_0> : '
        if self.tasks == "RTE" or "QNLI":
            self.question_suffix = ' ? <extra_id_0> , '
        self.label_lists = {"cola":{"1":"correct","0":"incorrect"},
                              "cr":{"1":"great","0":"terrible"},
                              "mr":{"1":"great","0":"terrible"},
                              "mpqa":{"1":"great","0":"terrible"},
                              "SST-2":{"1":"great","0":"terrible"},
                              "sst-5":{"0":"terrible","1":"bad","2":"okay","3":"good","4":"great",},
                              "subj":{"0":'subjective',"1":'objective'},
                              "trec":{"0":'Description',"1":'Entity',"2":'Expression',"3":'Human',"4":'Location',"5":'Number'},
                               "RTE":{'not_entailment':'No','entailment':'Yes'},
                               "QNLI":{'not_entailment':'No','entailment':'Yes'},
                               "MRPC":{"1":"No","0":"Yes"},
                               'fever': {'SUPPORTS': "Yes", "REFUTES": 'No', 'NOT ENOUGH INFO': "Not Sure"}
                               #"STS-B":{<=2.5:No,>2.5:Yes}
                              }
        self.reverse_lists = {"cola":{"correct":1,"incorrect":0},
                                   "cr":{"great":1,"terrible":0},
                                   "mr":{"great":1,"terrible":0},
                                   "mpqa":{"great":1,"terrible":0},
                                   "SST-2":{"great":1,"terrible":0},
                                   "sst-5":{"great":4,"good":3,"okay":2,"bad":1,"terrible":0},
                                   "subj":{'subjective':0,'objective':1},
                                   "trec":{'Description':0,'Entity':1,'Expression':2,'Human':3,'Location':4,'Number':5},
                                   "RTE":{"No":0,"Yes":1},
                                   "QNLI":{"No":0,"Yes":1}, 
                                   "MRPC":{"No":0,"Yes":1}, 
                                   "fever": {"Yes": 0, "No": 1, "Not Sure": 2}
                                   }
        self.sort_data()

    def idx2label(self,ans):
        if self.tasks == "STS-B":
            return "No" if float(ans) <= 2.5 else "Yes"
        else:
            return self.label_lists[self.tasks][ans]

    def str2labelidx(self,str_):
        if str_ in self.reverse_lists[self.tasks].keys():
            return self.reverse_lists[self.tasks][str_]
        else:
            return 0

    def __len__(self):
        return len(self.data)

    def get_target(self, example):
        if 'answers' in example:
            return '<extra_id_0> ' + self.idx2label(example['answers']) + ' <extra_id_1>'
        else:
            return None

    def __getitem__(self, index):
        example = self.data[index]
        
        if self.tasks == "trec":
            question = self.question_suffix + example['question']
        elif self.tasks == "RTE" or self.tasks == "QNLI" or self.tasks == "MRPC":
            question = example['question1'] + self.question_suffix + example['question2']
        else:
            question = example['question'] + self.question_suffix
        target = self.get_target(example)

        if 'ctxs' in example and self.n_context is not None:

            if self.tasks == "trec":
                f = "{}" + ' : ' + " {}"
                contexts = example['ctxs'][:self.n_context]
                passages = [f.format(self.idx2label(c['answers']), c['text']) + ' .'
                            if 'answers' in c and c['answers'] != '' else c['text']
                            for c in contexts]
            elif self.tasks == "RTE" or self.tasks == "QNLI" or self.tasks == "MRPC":
                f = "{}" + ' ? {} , ' + " {}"
                contexts = example['ctxs'][:self.n_context]
                passages = [f.format(c['text1'], self.idx2label(c['answers']), c['text2']) + ' .' 
                            if 'answers' in c and c['answers'] != '' else f.format(c['text1'], "", c['text2'])
                            for c in contexts]
            else:
                # f = self.passage_prefix + " {}" + self.context_suffix + " {}"
                f = "{}" + self.context_suffix + " {}"
                contexts = example['ctxs'][:self.n_context]
                passages = [f.format(c['text'], self.idx2label(c['answers'])) + ' .' 
                            if 'answers' in c and c['answers'] != '' else c['text']
                            for c in contexts]


            scores = [float(c['score']) for c in contexts]
            scores = torch.tensor(scores)
            # TODO(egrave): do we want to keep this?
            if len(contexts) == 0:
                contexts = [question]
        else:
            passages, scores = None, None


        return {
            'index' : index,
            'question' : question,
            'target' : target,
            'passages' : passages,
            'scores' : scores
        }

    def sort_data(self):
        if self.n_context is None or not 'score' in self.data[0]['ctxs'][0]:
            return
        for ex in self.data:
            ex['ctxs'].sort(key=lambda x: float(x['score']), reverse=True)

    def get_example(self, index):
        return self.data[index]

def encode_passages(batch_text_passages, tokenizer, max_length):
    passage_ids, passage_masks = [], []
    for k, text_passages in enumerate(batch_text_passages):
        p = tokenizer.batch_encode_plus(
            text_passages,
            max_length=max_length,
            pad_to_max_length=True,
            return_tensors='pt',
            truncation=True
        )
        passage_ids.append(p['input_ids'][None])
        passage_masks.append(p['attention_mask'][None])

    passage_ids = torch.cat(passage_ids, dim=0)
    passage_masks = torch.cat(passage_masks, dim=0)
    return passage_ids, passage_masks.bool()

class Collator(object):
    def __init__(self, text_maxlength, tokenizer, answer_maxlength=20):
        self.tokenizer = tokenizer
        self.text_maxlength = text_maxlength
        self.answer_maxlength = answer_maxlength

    def __call__(self, batch):
        assert(batch[0]['target'] != None)
        index = torch.tensor([ex['index'] for ex in batch])
        target = [ex['target'] for ex in batch]
        target = self.tokenizer.batch_encode_plus(
            target,
            max_length=self.answer_maxlength if self.answer_maxlength > 0 else None,
            pad_to_max_length=True,
            return_tensors='pt',
            truncation=True if self.answer_maxlength > 0 else False,
        )
        target_ids = target["input_ids"]
        target_mask = target["attention_mask"].bool()
        target_ids = target_ids.masked_fill(~target_mask, -100)

        def append_question(example):
            if example['passages'] is None:
                return [example['question']]
            ret=[]
            for t in example['passages']:
                ret.append(t + " " + example['question'])
            # return [example['question'] + " " + t for t in example['passages']]
            return ret
        text_passages = [append_question(example) for example in batch]
        passage_ids, passage_masks = encode_passages(text_passages,
                                                     self.tokenizer,
                                                     self.text_maxlength)

        return (index, target_ids, target_mask, passage_ids, passage_masks)


class RagCollator(object):
    def __init__(self, text_maxlength, tokenizer, answer_maxlength=20):
        self.tokenizer = tokenizer
        self.text_maxlength = text_maxlength
        self.answer_maxlength = answer_maxlength

    def __call__(self, batch):
        assert(batch[0]['target'] != None)
        index = torch.tensor([ex['index'] for ex in batch])
        self.tokenizer._switch_to_target_mode()
        target = [ex['target'] for ex in batch]
        target = self.tokenizer.current_tokenizer.batch_encode_plus(
            target,
            max_length=self.answer_maxlength if self.answer_maxlength > 0 else None,
            padding=True,
            return_tensors='pt',
            truncation=True if self.answer_maxlength > 0 else False,
        )
        target_ids = target["input_ids"]
        target_mask = target["attention_mask"].bool()
        # target_ids = target_ids.masked_fill(~target_mask, -100)

        # def append_question(example):
        #     if example['passages'] is None:
        #         return [example['question']]
        #     ret=[]
        #     for t in example['passages']:
        #         ret.append(t + " " + example['question'])
        #     # return [example['question'] + " " + t for t in example['passages']]
        #     return ret
        # text_passages = [append_question(example) for example in batch]
        self.tokenizer._switch_to_input_mode()
        questions = [ex['question'] for ex in batch]
        input = self.tokenizer.current_tokenizer.batch_encode_plus(
            questions,
            max_length=self.text_maxlength,
            padding=True,
            return_tensors='pt',
            truncation=True if self.text_maxlength > 0 else False,
        )
        passage_ids, passage_masks = input["input_ids"], input["attention_mask"].bool()
        
        # encode_passages(questions,
        #                                              self.tokenizer.current_tokenizer,
        #                                              self.text_maxlength)

        return (index, target_ids, target_mask, passage_ids, passage_masks)



def load_data(data_path=None, global_rank=-1, world_size=-1):
    assert data_path
    if data_path.endswith('.jsonl'):
        data = open(data_path, 'r')
    elif data_path.endswith('.json'):
        with open(data_path, 'r') as fin:
            data = json.load(fin)
    examples = []
    for k, example in enumerate(data):
        if global_rank > -1 and not k%world_size==global_rank:
            continue
        if data_path is not None and data_path.endswith('.jsonl'):
            example = json.loads(example)
        if not 'id' in example:
            example['id'] = k
        for c in example['ctxs']:
            if not 'score' in c:
                c['score'] = 1.0 / (k + 1)
        examples.append(example)
    ## egrave: is this needed?
    if data_path is not None and data_path.endswith('.jsonl'):
        data.close()

    return examples
