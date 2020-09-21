import cv2
import json
import os
from tqdm import tqdm
set_names = ['train', 'val']
fps_dict = dict()
for set_name in set_names:
    dirname = '/Disk4/yonglu/ava_data/videos_15min_' + set_name
    for video_filename in tqdm(os.listdir(dirname)):
        video_name = os.path.splitext(video_filename)[0]
        vdo = cv2.VideoCapture()
        vdo.open(os.path.join(dirname, video_filename))
        fps = int(vdo.get(cv2.CAP_PROP_FPS))
        fps_dict[video_name] = fps
json.dump(fps_dict, open('fps_dict.json','w'))