import pickle
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--pred_layout", type=str,  default="./HRS_spatial.p", help="input pred layout")

args = parser.parse_args()

def rel_type(rel):
    if rel in above_spatial_words: return 'above'
    if rel in below_spatial_words: return 'below'
    if rel in relative_relations: return 'between'
    if rel == 'on the left of': return 'left'
    if rel == 'on the right of': return 'right'

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
            if v_dif<0 and abs(v_dif)>=abs(h_dif): pred_rel = 'above' 
            if v_dif>0 and abs(v_dif)>=abs(h_dif): pred_rel = 'below'
            if v_dif==h_dif==0: continue
            # print(pred_rel, rel)
            if pred_rel == rel: return 1
    return 0

def check_between(obj1_boxes, obj2_boxes, obj3_boxes):
    if check_rel(obj1_boxes, obj2_boxes, 'left') and check_rel(obj1_boxes, obj3_boxes, 'right'): return 1
    if check_rel(obj1_boxes, obj2_boxes, 'right') and check_rel(obj1_boxes, obj3_boxes, 'left'): return 1
    if check_rel(obj1_boxes, obj2_boxes, 'above') and check_rel(obj1_boxes, obj3_boxes, 'below'): return 1
    if check_rel(obj1_boxes, obj2_boxes, 'below') and check_rel(obj1_boxes, obj3_boxes, 'above'): return 1

    return 0




#Extract Ground Truth objects/layout
gt = {}
with open('../gt_benchmark/HRS/spatial_compositions_prompts.csv', 'r', newline='', encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    header = next(csv_reader)
    for id, row in enumerate(csv_reader):
        gt[id] = row[0:7]

#load pred box
with open(args.pred_layout, 'rb') as f:
    box = pickle.load(f)
# print(box)
above_spatial_words = ["on", "above", "over"]
below_spatial_words = ["below", "beneath", "under"]
relative_relations = ["between", "among", "in the middle of"]
acc_ls = []
acc = 0
#for each prompt
for id, gt_info in gt.items():
    # if id != '33': continue
    prompt = gt_info[0]
    acc = 0

    try:
        pred_objs, pred_boxes = box[prompt]
    except:
        acc_ls.append(acc)
        continue
    rel1, rel2 = gt_info[5:]
    obj1, obj2, obj3, obj4 = gt[int(id)][1:5]
    rel1 = rel_type(rel1)
    if rel2: rel2 = rel_type(rel2)
    
    #easy level - 2 objs
    # print(obj3)
    if obj3=='':
        obj1_boxes = extr_pred_boxes(obj1)
        obj2_boxes = extr_pred_boxes(obj2)
        #if both objs generated
        if obj1_boxes and obj2_boxes: 
            acc = check_rel(obj1_boxes, obj2_boxes, rel1)
            acc_ls.append(acc)
        else:
            acc_ls.append(acc)

    #med level - 3 objs
    elif obj4=='':
        obj1_boxes = extr_pred_boxes(obj1)
        obj2_boxes = extr_pred_boxes(obj2)
        obj3_boxes = extr_pred_boxes(obj3)
        #if all 3 objs generated
        if obj1_boxes and obj2_boxes and obj3_boxes:
            if rel2:
                acc = (check_rel(obj1_boxes, obj2_boxes, rel1) + check_rel(obj1_boxes, obj3_boxes, rel2))/2
            else:
                acc = check_between(obj1_boxes, obj2_boxes, obj3_boxes)
            acc_ls.append(acc)
        else:
            acc_ls.append(acc)
    
    #hard level - 4 objs
    else:
        obj1_boxes = extr_pred_boxes(obj1)
        obj2_boxes = extr_pred_boxes(obj2)
        obj3_boxes = extr_pred_boxes(obj3)
        obj4_boxes = extr_pred_boxes(obj4)
        #if all 4 objs generated
        if obj1_boxes and obj2_boxes and obj3_boxes and obj4_boxes:
            if rel2:
                obj1_acc = check_rel(obj1_boxes, obj3_boxes, rel1) + check_rel(obj1_boxes, obj4_boxes, rel2)
                obj2_acc = check_rel(obj2_boxes, obj3_boxes, rel1) + check_rel(obj2_boxes, obj4_boxes, rel2)
                acc = (obj1_acc+obj2_acc)/4
            else:
                acc = (check_between(obj1_boxes, obj3_boxes, obj4_boxes) + check_between(obj2_boxes, obj3_boxes, obj4_boxes))/2
            acc_ls.append(acc)        
        elif obj1_boxes and obj3_boxes and obj4_boxes:
            if rel2:
                acc = (check_rel(obj1_boxes, obj3_boxes, rel1) + check_rel(obj1_boxes, obj4_boxes, rel2))/2
            else:
                acc = check_between(obj1_boxes, obj3_boxes, obj4_boxes)
            acc_ls.append(acc)        
        elif obj2_boxes and obj3_boxes and obj4_boxes:
            if rel2:
                acc = (check_rel(obj2_boxes, obj3_boxes, rel1) + check_rel(obj2_boxes, obj4_boxes, rel2))/2
            else:
                acc = check_between(obj2_boxes, obj3_boxes, obj4_boxes)
            acc_ls.append(acc)
        else:
            acc_ls.append(acc)
acc = sum(acc_ls)/len(acc_ls)*100
print('Average Result:')
print('Accuracy:',f'{acc:.2f}','%')