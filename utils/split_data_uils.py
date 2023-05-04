import json
import shutil
import random
import os

origin_dataset_name = 'csqa' # 'piqa'
origin_train_path = './data/%s/official/train.jsonl'%origin_dataset_name
origin_dev_path = './data/%s/official/dev.jsonl'%origin_dataset_name

dataset_name = 'csqa1_6' # 'piqa'
train_path = './data/%s/in_house/train.jsonl'%dataset_name
dev_path = './data/%s/in_house/dev.jsonl'%dataset_name
test_path = './data/%s/in_house/test.jsonl'%dataset_name

with open(origin_dev_path, 'r', encoding='utf-8') as f:
    all_lines = f.readlines()
    test_cnt = len(all_lines)

# Copy origin dev set to test set
if not os.path.isdir(os.path.dirname(test_path)):
    os.makedirs(os.path.dirname(test_path))
shutil.copy(origin_dev_path,test_path)

with open(origin_train_path, 'r') as fin, open(train_path, 'w') as fout1, open(dev_path, 'w') as fout2:
    all_lines = fin.readlines()
    train_cnt = len(all_lines)
    dev_id_lst = list(range(train_cnt))
    random.shuffle(dev_id_lst)
    dev_id_lst = dev_id_lst[:test_cnt]
    for i, one_line in enumerate(all_lines):
        one_line = json.loads(one_line)
        if i in dev_id_lst:
            fout2.write(json.dumps(one_line))
            fout2.write('\n')
        else:
            fout1.write(json.dumps(one_line))
            fout1.write('\n')

# with open('data/siqa/in_house/test.jsonl','r') as f_in, open('data/siqa/in_house/test1.jsonl','w') as f_out:
#     for line in f_in.readlines():
#         res = json.loads(line)
#         res['question']['stem'] = res['question']['context']+' '+res['question']['stem']
#         del res['question']['context']
#         f_out.write(json.dumps(res))
#         f_out.write('\n')