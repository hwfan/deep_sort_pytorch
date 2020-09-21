import csv
import os
import json
import numpy as np
from tqdm import tqdm
import pickle
csvFile_train = open("ava-gt-box-csv/ava_train_v2.2.csv", "r")
reader_train = csv.reader(csvFile_train)

csvFile_val = open("ava-gt-box-csv/ava_val_v2.2.csv", "r")
reader_val = csv.reader(csvFile_val)
vt_box_dict = {}

video_shape_dict = json.load(open('video_shape_dict.json','r'))
for set_idx, reader in enumerate([reader_train, reader_val]):
  if set_idx == 0:
    set_name = 'train'
  elif set_idx == 1:
    set_name = 'val'
  for idx, item in enumerate(tqdm(reader)):
    video_id = item[0]
    im_height, im_width, _ = video_shape_dict[video_id]
    time = int(item[1])
    h_box = [int(float(item[2])*im_width), int(float(item[3])*im_height), int(float(item[4])*im_width), int(float(item[5])*im_height)]
    human_id = int(item[-1])
    vt = video_id+'.'+str(time)
    if vt not in vt_box_dict:
      vt_box_dict[vt] = dict()
    if human_id not in vt_box_dict[vt]:
      vt_box_dict[vt][human_id] = h_box

pickle.dump(vt_box_dict, open('vt_box_dict_gt.pkl','wb'))