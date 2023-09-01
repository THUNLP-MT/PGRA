import time
import sys
import torch
import transformers
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler, Subset
from src.options import Options
from torch.distributed import barrier

# import src.slurm
import src.util
import src.evaluation
import src.data
import src.model
import datasets

from evaluate import load as load_metric
from src.slurm import init_distributed_mode, init_signal_handler
from tqdm import tqdm

def evaluate(model, dataset, tokenizer, collator, opt):
    if opt.tasks == 'cola':
        # glue_metric = datasets.load_metric('glue', 'cola')
        glue_metric = load_metric('glue', 'cola')
    else:
        glue_metric = load_metric('accuracy')

    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
                            sampler=sampler,
                            batch_size=opt.per_gpu_batch_size,
                            drop_last=False,
                            num_workers=10,
                            collate_fn=collator
                            )
    model.eval()
    total = 0
    exactmatch = []
    ref_ans = []
    ref_gold = []
    model = model.module if hasattr(model, "module") else model
    answers, golds = [], []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            (idx, _, _, context_ids, context_mask) = batch

            outputs = model.generate(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                max_length=3
            )

            for k, o in enumerate(outputs):
                # print(o)
                ans = tokenizer.decode(o, skip_special_tokens=True)

                gold = dataset.label_lists[opt.tasks][dataset.get_example(idx[k])['answers']]
                answers.append(ans)
                golds.append(gold)
                score = src.evaluation.ems(ans, gold)
                total += 1
                exactmatch.append(score)
                ref_ans.append(dataset.str2labelidx(ans))
                ref_gold.append(dataset.str2labelidx(gold))

    results = glue_metric.compute(predictions=ref_ans, references=ref_gold)

    # print(results)
    exactmatch, total = src.util.weighted_average(np.mean(exactmatch), total, opt)
    return exactmatch, results, answers, golds


if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_optim_options()
    opt = options.parse()
    # opt = options.get_options(use_reader=True, use_optim=True)

    torch.manual_seed(opt.seed)
    n_gpu = 1
    opt.n_nodes = 1
    opt.node_id = 0
    opt.local_rank = 0
    opt.global_rank = 0
    opt.world_size = n_gpu
    opt.n_gpu_per_node = n_gpu
    opt.is_distributed = False
    opt.is_main = True
    if opt.n_context == 0:
        opt.n_context = None

    checkpoint_path = Path(opt.checkpoint_dir) / (
                opt.name + '_lr' + str(opt.lr) + '_bsz' + str(opt.per_gpu_batch_size) + '_steps' + str(
            opt.total_steps) + '_k' + str(opt.n_context))
    checkpoint_exists = checkpoint_path.exists()
    if opt.is_distributed:
        torch.distributed.barrier()  # type: ignore
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    # if not checkpoint_exists and opt.is_main:
    #    options.print_options(opt)
    # checkpoint_path, checkpoint_exists = util.get_checkpoint_path(opt)


    model_name = opt.model_name_or_path
    tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)
    # model_name = 
    if opt.model_checkpoint is not None:
        model_name = opt.model_checkpoint
    model_class = src.model.FiDT5

    # load data
    collator = src.data.Collator(opt.text_maxlength, tokenizer, answer_maxlength=opt.answer_maxlength)

    # use golbal rank and world size to split the eval set on multiple gpus
    eval_examples = src.data.load_data(
        opt.eval_data,
        global_rank=opt.global_rank,
        world_size=opt.world_size,
    )
    eval_dataset = src.data.Dataset(eval_examples, opt.n_context, opt.tasks)

    if opt.test_mode:
        eval_dataset = Subset(eval_dataset, range(100))
    model = src.model.FiDT5.from_pretrained(model_name)
    model = model.to(opt.local_rank)
    optimizer, scheduler = src.util.set_optim(opt, model)
    step, best_dev_em = 0, 0.0



    _, results, answers, golds =  evaluate(model, eval_dataset, tokenizer, collator, opt)
    with open(checkpoint_path / 'preds.txt', 'w') as f:
        s = [a + ' ' + g  for a, g in zip(answers, golds)]
        f.write('\n'.join(s) + '\n')

    print(results)
            
