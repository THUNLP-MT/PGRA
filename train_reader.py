# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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


# import wandb


def train(model, optimizer, scheduler, step, train_dataset, eval_dataset, opt, collator, best_dev_em, checkpoint_path):
    if opt.is_main:
        try:
            tb_logger = torch.utils.tensorboard.SummaryWriter(Path(opt.checkpoint_dir) / opt.name)
        except:
            tb_logger = None
            logger.warning('Tensorboard is not available.')

    torch.manual_seed(opt.global_rank + opt.seed)  # different seed for different sampling depending on global_rank
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=True,
        num_workers=10,
        collate_fn=collator
    )
    progress_bar = tqdm(range(opt.total_steps), disable=not opt.is_main)
    loss, curr_loss = 0.0, 0.0
    epoch = 1
    model.train()
    while step < opt.total_steps:
        epoch += 1
        for i, batch in enumerate(train_dataloader):
            step += 1
            progress_bar.update(1)
            (idx, labels, _, context_ids, context_mask) = batch

            train_loss = model(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                labels=labels.cuda()
            )[0]

            train_loss.backward()

            if step % opt.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            train_loss = src.util.average_main(train_loss, opt)
            curr_loss += train_loss.item()

            if step % opt.eval_freq == 0:
                dev_em, dev_metric = evaluate(model, eval_dataset, tokenizer, collator, opt)
                # test_em, test_metric = evaluate(model, test_dataset, tokenizer, collator, opt)
                model.train()
                if opt.is_main:
                    if dev_em > best_dev_em:
                        best_dev_em = dev_em
                        src.util.save(model, optimizer, scheduler, step, best_dev_em,
                                      opt, checkpoint_path, 'best_dev')
                    log = f"{step} / {opt.total_steps} |"
                    log += f"train: {curr_loss / opt.eval_freq:.3f} |"
                    log += (str(dev_metric) + " eval|")
                    # log += (str(test_metric) + " test|")
                    log += f"evaluation: {100 * dev_em:.2f}EM |"
                    # log += f"test: {100 * test_em:.2f}EM |"
                    log += f"lr: {scheduler.get_last_lr()[0]:.5f}"
                    logger.info(log)
                    if tb_logger is not None:
                        tb_logger.add_scalar("Evaluation", dev_em, step)
                        tb_logger.add_scalar("Training", curr_loss / (opt.eval_freq), step)
                    curr_loss = 0.

            if opt.is_main and step % opt.save_freq == 0:
                src.util.save(model, optimizer, scheduler, step, best_dev_em,
                              opt, checkpoint_path, f"step-{step}")
            if step > opt.total_steps:
                break


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
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, disable=not opt.is_main)):
            (idx, _, _, context_ids, context_mask) = batch

            outputs = model.generate(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                max_length=3
            )

            for k, o in enumerate(outputs):
                ans = tokenizer.decode(o, skip_special_tokens=True)

                gold = dataset.label_lists[opt.tasks][dataset.get_example(idx[k])['answers']]
                score = src.evaluation.ems(ans, gold)
                total += 1
                exactmatch.append(score)
                ref_ans.append(dataset.str2labelidx(ans))
                ref_gold.append(dataset.str2labelidx(gold))

    results = glue_metric.compute(predictions=ref_ans, references=ref_gold)

    exactmatch, total = src.util.weighted_average(np.mean(exactmatch), total, opt)
    return exactmatch, results


if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_optim_options()
    opt = options.parse()

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
            opt.total_steps) + '_k' + str(opt.n_context) + '_seed' + str(opt.seed))
    checkpoint_exists = checkpoint_path.exists()
    if opt.is_distributed:
        torch.distributed.barrier()  # type: ignore
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    logger = src.util.init_logger(
        opt.is_main,
        opt.is_distributed,
        checkpoint_path / 'run.log'
    )

    model_class = src.model.FiDT5

    # load data
    tokenizer = transformers.T5Tokenizer.from_pretrained(opt.model_name_or_path)
    collator = src.data.Collator(opt.text_maxlength, tokenizer, answer_maxlength=opt.answer_maxlength)

    # use golbal rank and world size to split the eval set on multiple gpus
    train_examples = src.data.load_data(
        opt.train_data,
    )
    train_dataset = src.data.Dataset(train_examples, opt.n_context, opt.tasks)
    # use golbal rank and world size to split the eval set on multiple gpus
    eval_examples = src.data.load_data(
        opt.eval_data,
        # global_rank=opt.global_rank,
        # world_size=opt.world_size,
    )
    eval_dataset = src.data.Dataset(eval_examples, opt.n_context, opt.tasks)
    # if not checkpoint_exists and opt.model_path == "none":
    t5 = transformers.T5ForConditionalGeneration.from_pretrained(opt.model_name_or_path)
    model = src.model.FiDT5(t5.config)
    model.load_t5(t5.state_dict())
    model = model.cuda()
    optimizer, scheduler = src.util.set_optim(opt, model)
    step, best_dev_em = 0, 0.0

    model.set_checkpoint(opt.use_checkpoint)

    if opt.is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            find_unused_parameters=False,
        )

    logger.info("Start training")
    train(
        model,
        optimizer,
        scheduler,
        step,
        train_dataset,
        eval_dataset,
        opt,
        collator,
        best_dev_em,
        checkpoint_path,
    )
