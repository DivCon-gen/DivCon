import pickle
import csv
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--pred_layout", type=str,  default="./NSR1K_spatial.p", help="input pred layout")

args = parser.parse_args()

def extr_pred_boxes(obj1):
    result_boxes = []
    for i, obj in enumerate(pred_objs):
        if obj1 in obj.lower() or obj1.replace(" ", "_") in obj.lower(): result_boxes.append(pred_boxes[i])
    return result_boxes

def check_rel(obj1_boxes, obj2_boxes, rel):
    for box in obj1_boxes:
        left, top, right, bottom = box
        center1 = [(left+right)/2, (top+bottom)/2]
        # print(box)
        for box in obj2_boxes:
            left, top, right, bottom = box
            center2 = [(left+right)/2, (top+bottom)/2]
            h_dif = center1[0]-center2[0]
            v_dif = center1[1]-center2[1]
            # print(box, h_dif, v_dif)
            if h_dif<0 and abs(h_dif)>=abs(v_dif): pred_rel = 'left'
            if h_dif>0 and abs(h_dif)>=abs(v_dif): pred_rel = 'right' 
            if v_dif<0 and abs(v_dif)>=abs(h_dif): pred_rel = 'top' 
            if v_dif>0 and abs(v_dif)>=abs(h_dif): pred_rel = 'bottom'
            if v_dif==h_dif==0: continue
            if pred_rel == rel: return 1
            if rel=='next to' and (pred_rel=='left' or pred_rel=='right'): return 1
    return 0



gt = json.load(open('../gt_benchmark/NSR-1K/spatial.val.json'))

with open(args.pred_layout, 'rb') as f:
    box = pickle.load(f)

acc_list = []
acc = 0

#for each prompt
for i, items in enumerate(gt):
    prompt = items['prompt']
    acc = 0
    pred_objs, pred_boxes = box[prompt]

    relation = items['relation']
    obj1 = items['obj1'][0]
    obj2 = items['obj2'][0]
    obj1_pred_boxes = extr_pred_boxes(obj1)
    obj2_pred_boxes = extr_pred_boxes(obj2)
    if obj1_pred_boxes and obj2_pred_boxes: 
        acc = check_rel(obj1_pred_boxes, obj2_pred_boxes, relation)
    acc_list.append(acc)


acc = sum(acc_list)/len(acc_list)*100
print('Average Result:')
print('Accuracy:',f'{acc:.2f}','%')