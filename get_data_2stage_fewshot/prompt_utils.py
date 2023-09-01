from functools import partial
from promptsource.templates import DatasetTemplates, Template
from typing import Union, List
from multiprocessing import Pool
from tqdm import tqdm
from pathlib import Path

def format_data(text: List[str], prompt: Template, text2=None, num_processes=20, key='sentence',):
    # print('Starting formatting data with prompting')
    # if isinstance(text, str):
    #     text = [text]
    # if isinstance(text[0], str):
    if isinstance(key, str):
        text = [{key: s, 'label': ''} for s in text]  # type: ignore
    else:
        text = [{key[0]: s1, key[1]: s2, 'label': ''} for s1, s2 in zip(text, text2)]  # type: ignore

    # prompted_text = [prompt.apply(sentence) for sentence in text]
    prompted_text = []
    
    with Pool(num_processes) as p:
        # with tqdm(total=len(text)) as pbar:
        for prompted in p.imap(partial(get_prompted_text, prompt=prompt), text):
            prompted_text.append(prompted)
                # pbar.update()
    # print(f'Formatting finished! Example: {prompted_text[0]}')
    return prompted_text


def get_prompted_text(sentence, prompt):
    return prompt.apply(sentence)[0]

def transform_dataset_with_prompt(text, task_name, text2=None, n_procs=20):
    if task_name in ['sst-5', 'subj']:
        prompt = Prompt(task_name)
    elif task_name == 'trec':
        prompt = get_prompt_templates(*get_dataset_name(task_name))
        prompt = prompt['which_category_best_describes']
    else:
        prompt = get_prompt_templates(*get_dataset_name(task_name))  # type: ignore
        key = list(prompt.keys())[0]
        # print(f'Prompt key is {key}.')
        prompt = prompt[key]
    task_key=get_task_key(task_name)
    # assert len(key) == 1 or text2
    text = format_data(text, prompt, text2=text2, num_processes=n_procs, key=task_key)  # type: ignore
    # text = [sentence.replace('\n', ' ') for sentence in text]
    return text



def get_prompt_templates(dataset_name, subset_name):
    prompts = DatasetTemplates(dataset_name, subset_name)
    names = prompts.all_template_names
    return {name : prompts[name] for name in names}


def get_task_key(task_name):
    if task_name.lower() == 'sst-2':
        return 'sentence'
    elif task_name.lower() == 'mr':
        return 'sentence'
    elif task_name.lower() == 'mpqa':
        return 'sentence'
    elif task_name.lower() == 'cola':
        return 'sentence'
    elif task_name.lower() == 'cr':
        return 'sentence'
    elif task_name.lower() == 'sst-5':
        return 'sentence'
    elif task_name.lower() == 'subj':
        return 'sentence'
    elif task_name.lower() == 'trec':
        return 'text'
    return NotImplementedError

def get_dataset_name(task_name):
    if task_name.lower() == 'sst-2':
        return 'glue', 'sst2'
    elif task_name.lower() == 'mr':
        return 'glue', 'sst2'
    elif task_name.lower() == 'mpqa':
        return 'glue', 'sst2'
    elif task_name.lower() == 'cola':
        return 'glue', 'cola'
    elif task_name.lower() == 'cr':
        return 'glue', 'sst2'
    elif task_name.lower() == 'sst-5':
        return 'glue', 'sst2'
    elif task_name.lower() == 'subj':
        return
    elif task_name.lower() == 'trec':
        return 'trec', None
    return NotImplementedError


class Prompt:
    def __init__(self, dataset_name) -> None:
        self.dataset_name = dataset_name

    def apply(self, text):
        if self.dataset_name == 'sst-5':
            template = 'What sentiment does this sentence have? terrible, bad, okay, good or great\n'
        elif self.dataset_name == 'subj':
            template = 'Is this a subjective or objective description?\n'
        else:
            raise NotImplementedError
        # text = text['sentence'] if isinstance(text, dict) else text
        # ret = [template + t for t in text] 
        ret = template + text['sentence']
        return [ret]