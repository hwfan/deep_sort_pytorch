import os
import pickle
parent_dir = '/Disk4/yonglu/ava_data'
set_names = ['train', 'val']
path_list = []
for each_set_name in set_names:
    video_dir = os.path.join(parent_dir, 'videos_15min_'+each_set_name)
    for filename in os.listdir(video_dir):
        filepath = os.path.join(video_dir, filename)
        path_list.append(filepath)
pickle.dump(path_list, open('ava_path_list.pkl','wb'))