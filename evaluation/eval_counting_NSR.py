import pickle
import csv
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--in_result", type=str,  default="attly_attgen.p", help="input folder")

args = parser.parse_args()


gt = json.load(open('../gt_benchmark/NSR-1K/counting.val.json'))

with open(args.in_result, 'rb') as f:
    box = pickle.load(f)

#Set metrics
precision_list = []
recall_list = []
acc_list = []

#for each prompt
for id, pred in box.items():
    ct_intersection = 0
    pred_total = 0
    precision = 0
    recall = 0
    gt_total = 0
    accuracy = 0
    objs, _ = pred

    #Extract predict objects/layout
    count = {}
    for obj in objs:
        if obj in count:
            count[obj] += 1
        else: 
            count[obj] = 1


    #Calculate Accuracy
    if gt[int(id)]['sub-type'] == 'comparison':
        (gt_obj1, gt_ct1), (gt_obj2, gt_ct2) = gt[int(id)]['num_object']
        if gt_obj1 in count and gt_obj2 in count:
            if gt_ct1==gt_ct2==count[gt_obj1]==count[gt_obj2]: acc_list.append(1)
            elif gt_ct1==count[gt_obj1] and (gt_ct1-gt_ct2)*(count[gt_obj1]-count[gt_obj2])>0: acc_list.append(1)
            else: acc_list.append(0)
    else:        
        for gt_obj, gt_ct in gt[int(id)]['num_object']:
            gt_total += gt_ct
            if gt_obj in count: 
                ct_intersection += min(int(gt_ct), count[gt_obj])
                pred_total += count[gt_obj]

        if ct_intersection != 0:
            precision = ct_intersection/pred_total
            recall = ct_intersection/gt_total
            if precision==1 and recall==1:
                accuracy = 1
        precision_list.append(precision)
        recall_list.append(recall)
        acc_list.append(accuracy)


precision = sum(precision_list)/len(precision_list)
recall = sum(recall_list)/len(recall_list)
f1 = 2*precision*recall/(precision+recall)
acc = sum(acc_list)/len(acc_list)

print('Average All Result:')

print('Precision:',f'{precision*100:.2f}','%')
print('Recall:',f'{recall*100:.2f}','%')
print('f1:',f'{f1*100:.2f}','%')
print('Accuracy:',f'{acc*100:.2f}','%')

