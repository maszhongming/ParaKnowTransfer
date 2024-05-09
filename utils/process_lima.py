import json

data = []
with open('../data/lima/train.jsonl') as f:
    for line in f:
        data.append(json.loads(line))

processed_data = []
# only keep data with no more than 2 turns, resulting in 1000 samples
for i in range(len(data)):
    if len(data[i]['conversations']) > 2:
        continue
    cur_data = {}
    cur_data['dataset'] = 'lima'
    cur_data['id'] = 'lima_' + str(i)
    cur_data['messages'] = []
    # user
    cur_message = {}
    cur_message['role'] = 'user'
    cur_message['content'] = data[i]['conversations'][0]
    cur_data['messages'].append(cur_message)
    # assistant
    cur_message = {}
    cur_message['role'] = 'assistant'
    cur_message['content'] = data[i]['conversations'][1]
    cur_data['messages'].append(cur_message)
    processed_data.append(cur_data)

with open('../data/lima/lima_data_1000.jsonl', 'w') as f:
    for i in range(len(processed_data)):
        print(json.dumps(processed_data[i], ensure_ascii=False), file=f)

