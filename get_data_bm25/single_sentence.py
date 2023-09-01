from sbert import SentRetriever
import json

from tqdm import tqdm
import sys

TASK=sys.argv[1]
COLA_TRAIN_FILE = ""
COLA_TEST_FILE = ""
CR_TRAIN_FILE = ""
CR_TEST_FILE = ""
MPQA_TRAIN_FILE = ""
MPQA_TEST_FILE = ""
MR_TRAIN_FILE = ""
MR_TEST_FILE = ""
SST2_TRAIN_FILE = ""
SST2_TEST_FILE = ""
SST5_TRAIN_FILE = ""
SST5_TEST_FILE = ""
SUBJ_TRAIN_FILE = ""
SUBJ_TEST_FILE = ""
TREC_TRAIN_FILE = ""
TREC_TEST_FILE = ""


if TASK == 'cola':
    TRAIN_FILE=COLA_TRAIN_FILE
    TEST_FILE=COLA_TEST_FILE
    SENTIDX=3
    LABELIDX=1

    allsents=[]
    alllabels=[]
    with open(TRAIN_FILE,'r') as f:
        for line in f:
            if len(line.strip())>0:
                allsents.append(line.strip().split('\t')[SENTIDX].strip())
                alllabels.append(line.strip().split('\t')[LABELIDX].strip())

    print(allsents[:5])
    print(alllabels[:5])

    ###build test context
    test_allsents=[]
    test_alllabels=[]
    with open(TEST_FILE,'r') as f:
        for line in f:
            if len(line.strip())>0:
                test_allsents.append(line.strip().split('\t')[SENTIDX].strip())
                test_alllabels.append(line.strip().split('\t')[LABELIDX].strip())
elif TASK == 'cr':
    TRAIN_FILE=CR_TRAIN_FILE
    TEST_FILE= CR_TEST_FILE
    SENTIDX=1
    LABELIDX=0
    K=100
    SKIP_FIRST=False

    allsents=[]
    alllabels=[]
    with open(TRAIN_FILE,'r') as f:
        for line in f:
            if len(line.strip())>0:
                allsents.append(line.strip().split(',',1)[SENTIDX].strip())
                alllabels.append(line.strip().split(',',1)[LABELIDX].strip())

    ###build test context
    test_allsents=[]
    test_alllabels=[]
    with open(TEST_FILE,'r') as f:
        for line in f:
            if len(line.strip())>0:
                test_allsents.append(line.strip().split(',',1)[SENTIDX].strip())
                test_alllabels.append(line.strip().split(',',1)[LABELIDX].strip())

    if SKIP_FIRST:
        allsents=allsents[1:]
        alllabels=alllabels[1:]
        test_allsents=test_allsents[1:]
        test_alllabels=test_alllabels[1:]
elif TASK == 'mpqa':
    TRAIN_FILE=MPQA_TRAIN_FILE
    TEST_FILE= MPQA_TEST_FILE
    SENTIDX=1
    LABELIDX=0
    K=100
    SKIP_FIRST=False


    allsents=[]
    alllabels=[]
    with open(TRAIN_FILE,'r') as f:
        for line in f:
            if len(line.strip())>0:
                allsents.append(line.strip().split(',',1)[SENTIDX].strip()) #'\t'   ',',1
                alllabels.append(line.strip().split(',',1)[LABELIDX].strip())

    ###build test context
    test_allsents=[]
    test_alllabels=[]
    with open(TEST_FILE,'r') as f:
        for line in f:
            if len(line.strip())>0:
                test_allsents.append(line.strip().split(',',1)[SENTIDX].strip())
                test_alllabels.append(line.strip().split(',',1)[LABELIDX].strip())

    if SKIP_FIRST:
        allsents=allsents[1:]
        alllabels=alllabels[1:]
        test_allsents=test_allsents[1:]
        test_alllabels=test_alllabels[1:]
elif TASK == 'mr':
    TRAIN_FILE=MR_TRAIN_FILE
    TEST_FILE= MR_TEST_FILE
    SENTIDX=1
    LABELIDX=0
    K=100
    SKIP_FIRST=False


    allsents=[]
    alllabels=[]
    with open(TRAIN_FILE,'r') as f:
        for line in f:
            if len(line.strip())>0:
                allsents.append(line.strip().split(',',1)[SENTIDX].strip()) #'\t'   ',',1
                alllabels.append(line.strip().split(',',1)[LABELIDX].strip())

    ###build test context
    test_allsents=[]
    test_alllabels=[]
    with open(TEST_FILE,'r') as f:
        for line in f:
            if len(line.strip())>0:
                test_allsents.append(line.strip().split(',',1)[SENTIDX].strip())
                test_alllabels.append(line.strip().split(',',1)[LABELIDX].strip())

    if SKIP_FIRST:
        allsents=allsents[1:]
        alllabels=alllabels[1:]
        test_allsents=test_allsents[1:]
        test_alllabels=test_alllabels[1:]
elif TASK == 'SST-2':
    TRAIN_FILE=SST2_TRAIN_FILE
    TEST_FILE=SST2_TEST_FILE
    SENTIDX=0
    LABELIDX=1
    K=100
    SKIP_FIRST=True


    allsents=[]
    alllabels=[]
    with open(TRAIN_FILE,'r') as f:
        for line in f:
            if len(line.strip())>0:
                allsents.append(line.strip().split('\t')[SENTIDX].strip())
                alllabels.append(line.strip().split('\t')[LABELIDX].strip())

    ###build test context
    test_allsents=[]
    test_alllabels=[]
    with open(TEST_FILE,'r') as f:
        for line in f:
            if len(line.strip())>0:
                test_allsents.append(line.strip().split('\t')[SENTIDX].strip())
                test_alllabels.append(line.strip().split('\t')[LABELIDX].strip())

    if SKIP_FIRST:
        allsents=allsents[1:]
        alllabels=alllabels[1:]
        test_allsents=test_allsents[1:]
        test_alllabels=test_alllabels[1:]
elif TASK == 'sst-5':
    TRAIN_FILE=SST5_TRAIN_FILE
    TEST_FILE=SST5_TEST_FILE
    SENTIDX=1
    LABELIDX=0
    K=100
    SKIP_FIRST=True


    allsents=[]
    alllabels=[]
    with open(TRAIN_FILE,'r') as f:
        for line in f:
            if len(line.strip())>0:
                allsents.append(line.strip().split(',',1)[SENTIDX].strip()) #'\t'   ',',1
                alllabels.append(line.strip().split(',',1)[LABELIDX].strip())

    ###build test context
    test_allsents=[]
    test_alllabels=[]
    with open(TEST_FILE,'r') as f:
        for line in f:
            if len(line.strip())>0:
                test_allsents.append(line.strip().split(',',1)[SENTIDX].strip())
                test_alllabels.append(line.strip().split(',',1)[LABELIDX].strip())

    if SKIP_FIRST:
        allsents=allsents[1:]
        alllabels=alllabels[1:]
        test_allsents=test_allsents[1:]
        test_alllabels=test_alllabels[1:]
elif TASK == 'subj':
    TRAIN_FILE=SUBJ_TRAIN_FILE
    TEST_FILE=SUBJ_TEST_FILE
    SENTIDX=1
    LABELIDX=0
    K=100
    SKIP_FIRST=False

    allsents=[]
    alllabels=[]
    with open(TRAIN_FILE,'r') as f:
        for line in f:
            if len(line.strip())>0:
                allsents.append(line.strip().split(',',1)[SENTIDX].strip())
                alllabels.append(line.strip().split(',',1)[LABELIDX].strip())

    ###build test context
    test_allsents=[]
    test_alllabels=[]
    with open(TEST_FILE,'r') as f:
        for line in f:
            if len(line.strip())>0:
                test_allsents.append(line.strip().split(',',1)[SENTIDX].strip())
                test_alllabels.append(line.strip().split(',',1)[LABELIDX].strip())

    if SKIP_FIRST:
        allsents=allsents[1:]
        alllabels=alllabels[1:]
        test_allsents=test_allsents[1:]
        test_alllabels=test_alllabels[1:]
elif TASK == 'trec':
    TRAIN_FILE=TREC_TRAIN_FILE
    TEST_FILE=TREC_TEST_FILE
    SENTIDX=1
    LABELIDX=0
    K=100
    SKIP_FIRST=False

    allsents=[]
    alllabels=[]
    with open(TRAIN_FILE,'r') as f:
        for line in f:
            if len(line.strip())>0:
                allsents.append(line.strip().split(',',1)[SENTIDX].strip())
                alllabels.append(line.strip().split(',',1)[LABELIDX].strip())

    ###build test context
    test_allsents=[]
    test_alllabels=[]
    with open(TEST_FILE,'r') as f:
        for line in f:
            if len(line.strip())>0:
                test_allsents.append(line.strip().split(',',1)[SENTIDX].strip())
                test_alllabels.append(line.strip().split(',',1)[LABELIDX].strip())

    if SKIP_FIRST:
        allsents=allsents[1:]
        alllabels=alllabels[1:]
        test_allsents=test_allsents[1:]
        test_alllabels=test_alllabels[1:]
else:
    raise NotImplementedError
    
print(allsents[:5])
print(alllabels[:5])

sentRetriever = SentRetriever()

###build train context
# sentRetriever.retrieve("No reason to watch. It was terrible.") 
train_json=[]

for idx in tqdm(range(len(allsents))):
    single_json={'question': allsents[idx], 'answers': alllabels[idx], 'ctxs': []}
    text = allsents[idx]
    _, ret_idx = sentRetriever.retrieve(text) 
    for idx_ in ret_idx:
        single_json['ctxs'].append({'id':idx_,'text':sentRetriever.all_sents[idx_]})
    train_json.append(single_json)

# print(train_json[:1])
with open(TASK +'.train.json','w') as f:
    train_json_str = json.dumps(train_json)
    f.write(train_json_str)   



test_json=[]

for idx in tqdm(range(len(test_allsents))):
    single_json={'question': test_allsents[idx], 'answers': test_alllabels[idx], 'ctxs': []}
    text = test_allsents[idx]
    _, ret_idx = sentRetriever.retrieve(text) 
    for idx_ in ret_idx:
        single_json['ctxs'].append({'id':idx_,'text':sentRetriever.all_sents[idx_]})
    test_json.append(single_json)

print(test_json[:1])
with open(TASK +'.test.json','w') as f:
    test_json_str = json.dumps(test_json)
    f.write(test_json_str)