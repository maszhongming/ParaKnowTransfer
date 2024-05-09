import datasets
import json
import random
from tqdm import tqdm

categories = ['high_school_european_history', 'business_ethics', 'clinical_knowledge', 'medical_genetics', 'high_school_us_history', 'high_school_physics', 'high_school_world_history', 'virology', 'high_school_microeconomics', 'econometrics', 'college_computer_science', 'high_school_biology', 'abstract_algebra', 'professional_accounting', 'philosophy', 'professional_medicine', 'nutrition', 'global_facts', 'machine_learning', 'security_studies', 'public_relations', 'professional_psychology', 'prehistory', 'anatomy', 'human_sexuality', 'college_medicine', 'high_school_government_and_politics', 'college_chemistry', 'logical_fallacies', 'high_school_geography', 'elementary_mathematics', 'human_aging', 'college_mathematics', 'high_school_psychology', 'formal_logic', 'high_school_statistics', 'international_law', 'high_school_mathematics', 'high_school_computer_science', 'conceptual_physics', 'miscellaneous', 'high_school_chemistry', 'marketing', 'professional_law', 'management', 'college_physics', 'jurisprudence', 'world_religions', 'sociology', 'us_foreign_policy', 'high_school_macroeconomics', 'computer_security', 'moral_scenarios', 'moral_disputes', 'electrical_engineering', 'astronomy', 'college_biology']

def format_category(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def process(instance, category, index):
    cur = {}
    cur['dataset'] = 'mmlu'
    cur['category'] = category
    cur['id'] = category + '_' + str(index)
    messages = []
    # input
    cur_message = {}
    cur_message['role'] = 'user'
    prompt = "The following are multiple choice questions (with answers) about{}.\n\n".format(
        format_category(category)
    )
    prompt += instance['input'] + '\n'
    prompt += 'A. ' + instance['A'] + '\n'
    prompt += 'B. ' + instance['B'] + '\n'
    prompt += 'C. ' + instance['C'] + '\n'
    prompt += 'D. ' + instance['D'] + '\nAnswer:'
    cur_message['content'] = prompt
    messages.append(cur_message)
    # output
    cur_message = {}
    cur_message['role'] = 'assistant'
    cur_message['content'] = 'The answer is: ' + instance['target']
    messages.append(cur_message)

    cur['messages'] = messages

    return cur

processed_data = []

for category in tqdm(categories):
    data = datasets.load_dataset('lukaemon/mmlu', category)
    cur_cnt = 0
    for split in ['train', 'validation']:
        for i in range(len(data[split])):
            cur_data = process(data[split][i], category, cur_cnt)
            processed_data.append(cur_data)
            cur_cnt += 1
    print('{}: {}'.format(category, cur_cnt))

print('Total {} samples !!!'.format(len(processed_data)))
random.shuffle(processed_data)
with open('../data/mmlu/mmlu_data.jsonl', 'w') as f:
    for i in range(len(processed_data)):
        print(json.dumps(processed_data[i], ensure_ascii=False), file=f)

