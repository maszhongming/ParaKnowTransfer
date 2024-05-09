import json
import random
from os.path import join

n_sample = 32 # for each dataset

datasets = ['mmlu']
data_path = '../data'

def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data

def get_content(instance, use_chat_prompt=True):
    if use_chat_prompt:
        message_text = ''
        messages = instance['messages']
        for message in messages:
            if message["role"] == "system":
                message_text += "<|system|>\n" + message["content"].strip() + "\n\n"
            elif message["role"] == "user":
                message_text += "<|user|>\n" + message["content"].strip() + "\n\n"
            elif message["role"] == "assistant":
                message_text += "<|assistant|>\n" + message["content"].strip() + "\n\n"
        return message_text
    else:
        assert instance['messages'][0]['role'] == 'user'
        return instance['messages'][0]['content'] + '\n' + instance['messages'][1]['content']

sampled_data = []
for dataset in datasets:
    cur_data = load_jsonl(join(data_path, dataset + f'/{dataset}_data_1000.jsonl'))
    random.shuffle(cur_data)
    for i in range(n_sample):
        cur_data[i]['messages'][1]['content'] = cur_data[i]['messages'][1]['content'].split()[-1] # for mmlu
        sampled_data.append(cur_data[i])

print(len(sampled_data))
with open(join(data_path, 'mmlu/sample_mmlu.jsonl'), 'w') as f:
    for i in range(len(sampled_data)):
        print(json.dumps(sampled_data[i], ensure_ascii=False), file=f)
