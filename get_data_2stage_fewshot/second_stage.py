import torch
from torch.nn.parallel import DistributedDataParallel
from datasets import load_dataset, Dataset
from prompt_utils import transform_dataset_with_prompt
# import accelerate
import faiss
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import numpy as np
import json
import time
from multiprocessing import Pool
from functools import partial
import os
import random
# from accelerate import Accelerator
# acc = Accelerator()

LABEL_LISTS = {"cola":{"1":"acceptable","0":"unacceptable"},
                        "cr":{"1":"positive","0":"negative"},
                        "mr":{"1":"positive","0":"negative"},
                        "mpqa":{"1":"positive","0":"negative"},
                        "SST-2":{"1":"positive","0":"negative"},
                        "sst-5":{"0":"terrible","1":"bad","2":"okay","3":"good","4":"great",},
                        "subj":{"0":'subjective',"1":'objective'},
                        "trec":{"0":'Description',"1":'Entity',"2":'Abbreviation',"3":'Human',"4":'Location',"5":'Number'},
                        "RTE":{'not_entailment':'No','entailment':'Yes'},
                        "QNLI":{'not_entailment':'No','entailment':'Yes'},
                        "MRPC":{"1":"No","0":"Yes"},
                        "fever": {'SUPPORTS': 'Yes', 'REFUTES': 'No'}
                        }

device_ids=[0,1,2,3]

def opt_encode(example, model, tokenizer):
    text = example if isinstance(example, list) or isinstance(example, str) or isinstance(example, tuple) else example['text']
    i_input = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    i_feature = encode_func(model, i_input)
    i_feature /= i_feature.norm(dim=-1, keepdim=True)
    embeddings = i_feature.squeeze().cpu()
    return embeddings

@torch.no_grad()
def encode_func(model, batch_input):
    output = model(batch_input['input_ids'].cuda(0), output_hidden_states=True, return_dict=True)
    hs = output.last_hidden_state

    input_ids = batch_input['input_ids']
    batch_size = input_ids.shape[0]

    if isinstance(model, torch.nn.DataParallel):
        module = model.module
    else:
        module = model
    if module.config.pad_token_id is None:
        sequence_lengths = -1
    else:
        if input_ids is not None:
            sequence_lengths = torch.ne(input_ids, module.config.pad_token_id).sum(-1) - 1
        else:
            sequence_lengths = -1
    pooled_hs = hs[torch.arange(batch_size, device=hs.device), sequence_lengths]

    return pooled_hs

def batch_encode(text_sets, tokenizer, model, batch_size):
    counter = 0
    batch_text = []
    all_i_features = []
    # Prepare the inputs

    for i_n in tqdm(text_sets):
        counter += 1
        batch_text.append(i_n)
        if counter % batch_size == 0 or counter >= len(text_sets):
            i_feature = opt_encode(batch_text, model, tokenizer)
            if len(i_feature.shape) == 1:
                i_feature = i_feature.unsqueeze(0)
            all_i_features.append(i_feature)
            batch_text = []
    returned_text_features = torch.cat(all_i_features)
    return returned_text_features


def batch_encode_multigpu(text_sets, tokenizer, model, batch_size):
    ds = list(zip(range(len(text_sets)), text_sets))
    sampler = torch.utils.data.SequentialSampler(ds)
    dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size, sampler=sampler, shuffle=False)

    model = torch.nn.DataParallel(model, device_ids=device_ids)
    result = []
    for batch in tqdm(dataloader):
        ids, batch_text = batch
        embeddings = opt_encode(batch_text, model, tokenizer)
        result += list(zip(ids.tolist(), embeddings))
    result =sorted(result, key=lambda tuple: tuple[0])
    embs = torch.cat([e.unsqueeze(0) for _, e in result])
    return embs
    
def get_text_from_list(ctxs):
    return [c['text'] for c in ctxs]

def calculate_context_embeddings(raw_dataset, task_name, batch_size, tokenizer, model, demos):
    ctxs_mapping = {}
    for item in raw_dataset['ctxs']:
        for c in item:
            id_key = (c['id1'], c['id2'])
            if id_key not in ctxs_mapping:
                ctxs_mapping[id_key] = (c['text1'], c['text2'])
    
    ids, texts1, texts2 = [], [], []
    for i, t in ctxs_mapping.items():
        ids.append(i)
        texts1.append(t[0])
        texts2.append(t[1])

    texts =  transform_dataset_with_prompt(text = texts1, task_name=task_name, text2 = texts2)
    texts = [demos + s + '\nThe answer is' for s in texts]
    print(f"ctxs ex: {texts[0]}")
    text_embeddings = batch_encode(texts, tokenizer, model, batch_size)

    ids_mapping = {k:i for k,i in zip(ids, range(len(ids)))}
    ctxs_embeddings = []
    for item in raw_dataset['ctxs']:
        ids = [(c['id1'], c['id2']) for c in item]
        embed_index = np.array([ids_mapping[i] for i in ids])
        ctxs_embeddings.append(text_embeddings[embed_index])
    return ctxs_embeddings


def rerank_context_list(i, sim_indices, raw_dataset):
    idx = sim_indices[i]
    if i== 0:
        print(idx)
    ctx_dataset = Dataset.from_list(raw_dataset['ctxs'][i])
    ctx_dataset = ctx_dataset.select(idx)
    ctx_dataset_list = list(ctx_dataset)
    return ctx_dataset_list

def rerank(raw_dataset, tokenizer, model, batch_size, task_name, split, save_folder, demos ):
    
    
    question = raw_dataset['question']
    question = transform_dataset_with_prompt(question, task_name=task_name)
    question = [demos + s + '\nThe answer is' for s in question]
    print(f'question ex: {question[0]}')
    question_embeddings = batch_encode(question, tokenizer, model, batch_size)
    save_path = os.path.join(save_folder, f'{task_name}-{split}_question_embeddings.pt')
    torch.save(question_embeddings, save_path)


    ctxs_embeddings = calculate_context_embeddings(raw_dataset=raw_dataset,
                                                    task_name=task_name,
                                                    batch_size=batch_size,
                                                    tokenizer=tokenizer,
                                                    model=model,
                                                    demos=demos)
    save_path = os.path.join(save_folder, f'{task_name}-{split}_ctxs_embeddings.pt')
    torch.save(ctxs_embeddings, save_path)


def get_dataset_from_json(filename):
    return load_dataset('json', data_files={'train': filename})['train'] # type: ignore

def get_demonstrations(TASK, n=8, src_tag='bert'):
    return get_demonstrations_single(TASK, n=n, src_tag=src_tag)

def get_demonstrations_single(TASK, n = 8, src_tag='bert'):
    train_dataset = get_dataset_from_json(f'/share/project/guozhicheng/FiD/get_data_{src_tag}/{TASK}.train.json')
    label_list = LABEL_LISTS[TASK]
    # reverse_list = REVERSE_LISTS[TASK]
    demo_list = {k:[] for k in label_list.values()}
    indices = []
    for idx, item in enumerate(train_dataset):
        answer = label_list[item['answers']]
        # print(answer)
        if len(demo_list[answer]) < int(n/len(label_list.values())) + 1:
            demo_list[answer].append(item['question'])
            indices.append(idx)
            
        if sum([len(v) for v in demo_list.values()]) >= n and min([len(v) for v in demo_list.values()]) > 0:
            break
    ret = []
    for k, v in demo_list.items():
        transformed_v = transform_dataset_with_prompt(v, TASK)
        transformed_v = [s.replace('\n\n', '\n') for s in transformed_v]
        transformed_v = [s + f'\nThe answer is {k}' for s in transformed_v]
        # transformed_v = [s + f'\nThe answer is {k}' for s in transformed_v]
        ret +=  transformed_v
    random.shuffle(ret)
    return 'Follow the demonstrations:\n' + '\n\n'.join(ret) + '\n\n', indices

def select_context(ex, n=16):
    if n < 0:
        return ex
    ex['ctxs'] = ex['ctxs'][:n]
    return ex

def pipeline(task, split, model_name_or_path, save_folder, src_tag, d):
    filename = f'./get_data_{src_tag}/{task}.{split}.json'
    dataset = get_dataset_from_json(filename)  # type: ignore
    dataset = dataset.map(partial(select_context, n=d))
    print(len(dataset['ctxs'][0]))
    demos, demo_train_idx = get_demonstrations(task, 8, src_tag=src_tag)
    torch.save(demo_train_idx, os.path.join(save_folder, 'train_idx'))

    opt = AutoModel.from_pretrained(model_name_or_path).half().eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    rerank(dataset, tokenizer, opt, 4, task, split, save_folder=save_folder, demos=demos)

if __name__ == "__main__":
    import sys, os
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='SST-2')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--model_name_or_path', type=str, default='opt-350m')
    parser.add_argument('--src_tag', type=str, default='bert')
    parser.add_argument('--save_tag', type=str, default='bert')
    parser.add_argument('--d', default=-1, type=int)
    args = parser.parse_args()
    print(args)

    model_name_or_path = args.model_name_or_path
    model = model_name_or_path.split('/')[-1]
    save_folder = f'./get_data_2stage_fewshot/{args.save_tag}_{model}'
    os.makedirs(save_folder, exist_ok=True)
    print(f'{args.task} {args.split}')
    start = time.time()
    pipeline(args.task, args.split, model_name_or_path, save_folder, src_tag=args.src_tag, d=args.d)
    print(f"Time elasped: {time.time() - start}")
