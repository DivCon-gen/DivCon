import pickle
import csv
import openai
import json
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str,  default="HRS", help="input a dataset between HRS or NSR-1K")

args = parser.parse_args()


#Extract Ground Truth objects/layout
gt = []
if args.dataset == 'HRS':
    with open('../gt_benchmark/HRS/spatial_compositions_prompts.csv', 'r', newline='', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)
        for row in csv_reader:
            gt.append(row[0]) 
else:
    with open('../gt_benchmark/NSR-1K/spatial.val.json', 'rb') as f:
        NSR = json.load(f)
    for i, info in enumerate(NSR):
        gt.append(info['prompt'])


def load_json(path_file):
    with open(path_file) as f:
        data = json.load(f)
    return data
def text_list(text):
    text =  text.replace(' ','')
    text =  text.replace('\n','')
    text =  text.replace('\t','')
    digits = text[1:-1].split(',')
    # import pdb; pdb.set_trace()
    result = []
    for d in digits:
        result.append(int(d))
    return tuple(result)

pred_layout = {}

#llm_function
openai.api_key = '' #replace with your openai api key

error = 0

for i, prompt in tqdm(enumerate(gt), total=len(gt), desc="Processing"):
    if i>=0 and prompt not in pred_layout:
        #1st stage
        messages = load_json('example1_spatial.json')
        messages.append(
            {"role": "user", "content": prompt},
        )
        chat = openai.ChatCompletion.create(
            model="gpt-4", messages=messages
        )
        try: 
            obj_spatial = chat.choices[0].message.content
            print(obj_spatial)
        except: continue
        
        #2nd stage
        message = 'Provide box coordinates for an image with ' + obj_spatial
        messages = load_json('example2_spatial.json')
        messages.append(
            {"role": "user", "content": message},
        )
        chat = openai.ChatCompletion.create(
            model="gpt-4", messages=messages
        )
        try: 
            completed_text = chat.choices[0].message.content
            boxes = completed_text.split('\n')
            d = {}
            name_objects = []
            boxes_of_object = []
            for b in boxes:
                if b == '': continue
                if not '(' in b: continue 
                b_split = b.split(":")
                name_objects.append(b_split[0])
                boxes_of_object.append(text_list(b_split[1]))
            pred_layout[prompt] = [name_objects, boxes_of_object]
        except: continue

        if args.dataset == 'HRS':
            with open("HRS_spatial.p", "wb") as pickle_file:
                pickle.dump(pred_layout, pickle_file)
        else:
            with open("NSR1K_spatial.p", "wb") as pickle_file:
                pickle.dump(pred_layout, pickle_file)
print(len(pred_layout))