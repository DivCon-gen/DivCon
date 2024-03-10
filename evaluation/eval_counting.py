import pickle
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--in_result", type=str,  default="attly_attgen.p", help="input folder")

args = parser.parse_args()


gt = {}
with open('../gt_benchmark/HRS/counting_prompts.csv', 'r', newline='', encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    header = next(csv_reader)
    for id, row in enumerate(csv_reader):
        gt[id] = row[2:6]

with open(args.in_result, 'rb') as f:
    box = pickle.load(f)

#Set metrics
pre_easy = []
pre_med = []
pre_hard = []
recall_easy = []
recall_med = []
recall_hard = []
acc_easy = []
acc_med = []
acc_hard = []

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
    gt_ct1, gt_obj1, gt_ct2, gt_obj2 = gt[int(id)]
    #count intersection
    if gt_obj1 in count:
        ct_intersection += min(int(gt_ct1), count[gt_obj1])
        pred_total += count[gt_obj1]
    if gt_obj2 in count:
        ct_intersection += min(int(gt_ct2), count[gt_obj2])
        pred_total += count[gt_obj2]

    #ground truth total count
    gt_total += int(gt_ct1) + int(gt_ct2)


    if ct_intersection != 0:
        precision = ct_intersection/pred_total
        recall = ct_intersection/gt_total
        if precision==1 and recall==1:
            accuracy = 1
    if int(id)<1000:
        pre_easy.append(precision)
        recall_easy.append(recall)
        acc_easy.append(accuracy)
    if 1000<=int(id)<2000:
        pre_med.append(precision)
        recall_med.append(recall)
        acc_med.append(accuracy)
    if 2000<=int(id):
        pre_hard.append(precision)
        recall_hard.append(recall)
        acc_hard.append(accuracy)

if pre_easy:
    print('-------------------')
    print('Easy Level Result:')
    precision = sum(pre_easy)/len(pre_easy)
    recall = sum(recall_easy)/len(recall_easy)
    f1 = 2*precision*recall/(precision+recall)
    acc = sum(acc_easy)/len(acc_easy)*100
    print('Precision:',f'{precision*100:.2f}','%')
    print('Recall:',f'{recall*100:.2f}','%')
    print('f1:',f'{f1*100:.2f}','%')
    print('Accuracy:',f'{acc:.2f}','%')
    print('-------------------')

if pre_med:
    print('Med Level Result:')
    precision = sum(pre_med)/len(pre_med)
    recall = sum(recall_med)/len(recall_med)
    f1 = 2*precision*recall/(precision+recall)
    acc = sum(acc_med)/len(acc_med)*100
    print('Precision:',f'{precision*100:.2f}','%')
    print('Recall:',f'{recall*100:.2f}','%')
    print('f1:',f'{f1*100:.2f}','%')
    print('Accuracy:',f'{acc:.2f}','%')
    print('-------------------')

if pre_hard:
    print('Hard Level Result')
    precision = sum(pre_hard)/len(pre_hard)
    recall = sum(recall_hard)/len(recall_hard)
    f1 = 2*precision*recall/(precision+recall)
    acc = sum(acc_hard)/len(acc_hard)*100
    print('Precision:',f'{precision*100:.2f}','%')
    print('Recall:',f'{recall*100:.2f}','%')
    print('f1:',f'{f1*100:.2f}','%')
    print('Accuracy:',f'{acc:.2f}','%')
    print('-------------------')




# final result
precision_list = pre_easy + pre_med + pre_hard
recall_list = recall_easy + recall_med + recall_hard
acc_list = acc_easy + acc_med + acc_hard

precision = sum(precision_list)/len(precision_list)
recall = sum(recall_list)/len(recall_list)
f1 = 2*precision*recall/(precision+recall)
acc = sum(acc_list)/len(acc_list)*100

print('Average All Result:')

print('Precision:',f'{precision*100:.2f}','%')
print('Recall:',f'{recall*100:.2f}','%')
print('f1:',f'{f1*100:.2f}','%')
print('Accuracy:',f'{acc:.2f}','%')

