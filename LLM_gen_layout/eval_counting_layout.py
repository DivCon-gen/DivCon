import pickle
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--pred_layout", type=str,  default="./HRS_counting.p", help="input pred layout")

args = parser.parse_args()

#Extract Ground Truth objects/layout
gt = {}
with open('../gt_benchmark/HRS/counting_prompts.csv', 'r', newline='', encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    header = next(csv_reader)
    for row in csv_reader:
        pp = row[-1]
        gt[pp] = row[2:6]

#load pred box
with open(args.pred_layout, 'rb') as f:
    box = pickle.load(f)

#Set metrics
precision_list = []
recall_list = []
acc_list = []

#for each prompt
for prompt, items in box.items():
    ct_intersection = 0
    pred_total = 0
    precision = 0
    recall = 0
    gt_total = 0
    accuracy = 0
    f1 = 0

    #Extract predict objects/layout
    pred_objects = items[0]
    layout = items[1]
    count = {}
    for i in set(pred_objects):
        count[i] = pred_objects.count(i)


    #Calculate Accuracy
    gt_ct1, gt_obj1, gt_ct2, gt_obj2 = gt[prompt]


    #count intersection
    for pred_obj in count:
        #handle samge 2 gt
        if gt_obj1==gt_obj2:
            if gt_obj1 in pred_obj.lower() or gt_obj1.replace(" ", "_") in pred_obj.lower():
                pred_total += count[pred_obj]
            continue
        #handle both gt in pred and same 2 gt
        not_count_obj = 0
        if gt_obj1 in pred_obj.lower() and gt_obj2 in pred_obj.lower() and gt_obj2:
            # print(gt_obj1, gt_obj2, pred_obj)
            if pred_obj.lower().find(gt_obj1) < pred_obj.lower().find(gt_obj2):
                not_count_obj = 2
            else: not_count_obj = 1

        if gt_obj1 in pred_obj.lower() or gt_obj1.replace(" ", "_") in pred_obj.lower():
            if not_count_obj !=1: pred_total += count[pred_obj]
        #handle alter name
        if gt_obj1=='tv' and 'television' in pred_obj.lower(): pred_total += count[pred_obj]
        if gt_obj1=='skis' and 'ski' in pred_obj.lower(): pred_total += count[pred_obj]
        if gt_obj1=='knife' and 'knives' in pred_obj.lower(): pred_total += count[pred_obj]

        if gt_obj2:
            if gt_obj2 in pred_obj.lower() or gt_obj2.replace(" ", "_") in pred_obj.lower():
                if not_count_obj!=2:pred_total += count[pred_obj]
            #handle alter name
            if gt_obj2=='tv' and 'television' in pred_obj.lower(): pred_total += count[pred_obj]
            if gt_obj2=='skis' and 'ski' in pred_obj.lower(): pred_total += count[pred_obj]
            if gt_obj2=='knife' and 'knives' in pred_obj.lower(): pred_total += count[pred_obj]
    #ground truth total count
    gt_total += int(gt_ct1) + int(gt_ct2)
    ct_intersection += min(gt_total, pred_total)


    if pred_total != 0:
        precision = ct_intersection/pred_total
        recall = ct_intersection/gt_total
        f1 = 2*precision*recall/(precision+recall)
    if pred_total==gt_total:
        accuracy = 1
    # if accuracy == 0:
    #     print(gt[prompt])
    #     print(pred_objects)
    precision_list.append(precision)
    recall_list.append(recall)
    acc_list.append(accuracy)

precision = sum(precision_list)/len(precision_list)
recall = sum(recall_list)/len(recall_list)
f1 = 2*precision*recall/(precision+recall)
print('Precision:',f'{precision*100:.2f}','%')
print('Recall:',f'{recall*100:.2f}','%')
print('f1:',f'{f1*100:.2f}','%')
print('Accuracy:',f'{sum(acc_list)/len(acc_list)*100:.2f}','%')