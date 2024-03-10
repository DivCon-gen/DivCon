from ultralytics import YOLO
import os
import pickle
from tqdm import tqdm
import argparse

#input info
parser = argparse.ArgumentParser()
parser.add_argument("--out_file", type=str,  default="result.p", help="file name for output")
parser.add_argument("--in_folder", type=str,  default="../attention-refocusing/visual/mylyv6_attgen_img", help="input folder")

args = parser.parse_args()

# Load a model
model = YOLO("yolov8x.pt")  # load a pretrained model (recommended for training)


#read img
folder_path = args.in_folder
jpg_files = []  
for root, dirs, files in os.walk(folder_path):
    for filename in files:
        if filename.endswith(".jpg"):
             jpg_files.append(os.path.join(root, filename)) 


# Use the model    
pred = {}
for img in tqdm(jpg_files):
    id = img.split('/')[-1].split('_')[0]
    results = model(img)  # predict on an image
    names = model.names
    print(names)
    
    idx = list(results[0].boxes.cls)
    boxes = list(results[0].boxes.xyxy)

    pred_obj = []
    for img_id in idx:
        pred_obj.append(names[int(img_id)])
    pred[id] = [pred_obj, boxes]

with open('detection_result/' + args.out_file,'wb') as fp:
    pickle.dump(pred, fp)
