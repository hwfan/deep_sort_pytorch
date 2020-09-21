import csv
import os
import json
import numpy as np
from tqdm import tqdm
csvFile_train = open("ava-predicted-box-csv/ava_train_predicted_boxes.csv", "r")
reader_train = csv.reader(csvFile_train)

csvFile_val = open("ava-predicted-box-csv/ava_val_predicted_boxes.csv", "r")
reader_val = csv.reader(csvFile_val)
vt_box_dict = {}

for set_idx, reader in enumerate([reader_train, reader_val]):
  if set_idx == 0:
    set_name = 'train'
  elif set_idx == 1:
    set_name = 'val'
  for idx, item in enumerate(tqdm(reader)):
    video_id = item[0]
    time = int(item[1])
    h_box = [float(item[2]), float(item[3]), float(item[4]), float(item[5])]
    score = float(item[-1])
    vt = video_id+'.'+str(time)
    if vt not in vt_box_dict:
       vt_box_dict[vt] = []
    hit = False
    for exist_box in vt_box_dict[vt]:
        if np.sum(np.abs(np.array(h_box)-np.array(exist_box[0]))) <= 1e-5:
            hit = True
    if not hit and score >= 0.8:
        vt_box_dict[vt].append([h_box, score])

json.dump(vt_box_dict, open('vt_box_dict.json','w'))