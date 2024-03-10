import pickle
import json
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--pred_layout", type=str,  default="./NSR1K_counting.p", help="input pred layout")

args = parser.parse_args()

#Extract Ground Truth objects/layout

gt = json.load(open('../gt_benchmark/NSR-1K/counting.val.json'))

#load pred box
with open(args.pred_layout, 'rb') as f:
    box = pickle.load(f)

#Set metrics
precision_list = []
recall_list = []
acc_list = []

#for each prompt
for i, items in enumerate(gt):
    ct_intersection = 0
    pred_total = 0
    precision = 0
    recall = 0
    gt_total = 0
    accuracy = 0

    #Extract predict objects/layout
    prompt = items['prompt']
    try:
        pred_objects, layout = box[prompt]
    except:
        continue
    count = {}
    for i in set(pred_objects):
        count[i] = pred_objects.count(i)


    #Calculate Accuracy
    if items['sub-type'] == 'comparison':
        (gt_obj1, gt_ct1), (gt_obj2, gt_ct2) = items['num_object']
        if gt_obj1 in count and gt_obj2 in count:
            if gt_ct1==gt_ct2==count[gt_obj1]==count[gt_obj2]: acc_list.append(1)
            elif gt_ct1==count[gt_obj1] and (gt_ct1-gt_ct2)*(count[gt_obj1]-count[gt_obj2])>0: acc_list.append(1)
            else: acc_list.append(0)
    else:
        for gt_obj, gt_ct in items['num_object']:
                gt_total += gt_ct
                for pred_obj in count:
                    if gt_obj == re.sub(r"[\d\s]+$", "", pred_obj).lower() or gt_obj.replace(" ", "_") in pred_obj.lower():
                        pred_total += count[pred_obj]
                    if gt_obj=='tv' and 'television' in pred_obj.lower(): pred_total += count[pred_obj]
                    if gt_obj=='skis' and 'ski' in pred_obj.lower(): pred_total += count[pred_obj]
                    if gt_obj=='knife' and 'knives' in pred_obj.lower(): pred_total += count[pred_obj]
        ct_intersection += min(gt_total, pred_total)


        if pred_total != 0:
            precision = ct_intersection/pred_total
            recall = ct_intersection/gt_total
        if pred_total==gt_total:
            accuracy = 1
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