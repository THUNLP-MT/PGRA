from prompt_utils import *
from datasets import load_dataset
from second_stage import *
from transformers import AutoModel, AutoTokenizer
from datasets import Dataset
import sys
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='fever')
parser.add_argument('--split', type=str, default='test')
parser.add_argument('--model', type=str, default='opt-350m')
parser.add_argument('--src_tag', type=str, default='bert')
parser.add_argument('--embedding_tag', type=str, default='bert')
parser.add_argument('--save_tag', type=str, default='bert')
parser.add_argument('--d', default=-1, type=int)
args = parser.parse_args()
if args.embedding_tag is None:
    args.embedding_tag = args.src_tag

TASK, SPLIT, MODEL  = args.task, args.split, args.model
print(TASK, SPLIT, MODEL)
print(args)

def select_context(ex, n=16):
    if n < 0:
        return ex
    ex['ctxs'] = ex['ctxs'][:n]
    return ex

dataset = get_dataset_from_json(f'./get_data_{args.src_tag}/{TASK}.{SPLIT}.json')
dataset = dataset.map(partial(select_context, n=args.d))
_, demo_train_idx = get_demonstrations(TASK, src_tag=args.src_tag)
if SPLIT == 'train':
    index = set(range(len(dataset))).difference(set(demo_train_idx))
    dataset = dataset.select(index)




question_embeddings = torch.load(f'./get_data_2stage_fewshot/{args.embedding_tag}_{MODEL}/{TASK}-{SPLIT}_question_embeddings.pt')
ctxs_embeddings = torch.load(f'./get_data_2stage_fewshot/{args.embedding_tag}_{MODEL}/{TASK}-{SPLIT}_ctxs_embeddings.pt')
print(question_embeddings.shape, len(ctxs_embeddings), len(dataset))
if question_embeddings.shape[0] == len(dataset) + len(demo_train_idx):
    index = sorted(list(index))
    question_embeddings = question_embeddings[torch.tensor(index)]
    ctxs_embeddings = [ctxs_embeddings[i] for i in index]

assert question_embeddings.shape[0] == len(dataset)

similarities = []
for i in range(question_embeddings.shape[0]):
    similarities.append(torch.softmax(question_embeddings[i].type(torch.float32)  @(ctxs_embeddings[i].type(torch.float32)).T, -1))

sim_indices = [s.topk(s.shape[0])[1] for s in similarities]


def rerank(i):
    idx = sim_indices[i]
    if i== 0:
        print(idx)
    ctx_dataset = Dataset.from_list(dataset['ctxs'][i])
    ctx_dataset = ctx_dataset.select(idx)
    ctx_dataset_list = list(ctx_dataset)
    return ctx_dataset_list


reranked_ctxs = []
n = question_embeddings.shape[0]
with Pool(20) as p:
    with tqdm(total=n) as pbar:
        for ctx_dataset_list in p.imap(rerank, range(n)):
            reranked_ctxs.append(ctx_dataset_list)
            pbar.update()


def write_json(raw_dataset, reranked_ctxs, task, split, save_folder):
    train_json = []
    for i in range(len(raw_dataset)):
        single_json = {'question': raw_dataset[i]['question'], 'answers': raw_dataset[i]['answers'], 'ctxs': reranked_ctxs[i]}
        train_json.append(single_json)
    
    print(train_json[:1])
    path = os.path.join(save_folder, task + f'.{split}.json')
    with open(path, 'w') as f:
        train_json_str = json.dumps(train_json)
        f.write(train_json_str)
    return train_json_str


write_json(dataset, reranked_ctxs, task=TASK, split=SPLIT, save_folder=f'./get_data_2stage_fewshot/{args.save_tag}_{MODEL}')
