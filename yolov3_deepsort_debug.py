import os
import cv2
import time
import argparse
import torch
import warnings
import numpy as np
import pickle
from custom_multiprocessing import process_pool
from tqdm import tqdm
import ipdb
import json

from detector import build_detector
from deep_sort import build_tracker
from utils.draw import draw_boxes
from utils.parser import get_config
from utils.log import get_logger
from utils.io import write_results


class VideoTracker(object):
    def __init__(self, cfg, args, video_path, video_bar, vt_box_dict):
        self.cfg = cfg
        self.args = args
        self.video_path = video_path
        self.video_name = os.path.splitext(os.path.split(video_path)[1])[0]
        self.logger = get_logger("root")
        self.video_bar = video_bar
        self.vt_box_dict = vt_box_dict
        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        if args.cam != -1:
            print("Using webcam " + str(args.cam))
            self.vdo = cv2.VideoCapture(args.cam)
        else:
            self.vdo = cv2.VideoCapture()
        
        self.detector = build_detector(cfg, use_cuda=use_cuda)
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
        self.class_names = self.detector.class_names

    def __enter__(self):
        if self.args.cam != -1:
            ret, frame = self.vdo.read()
            assert ret, "Error: Camera error"
            self.im_width = frame.shape[0]
            self.im_height = frame.shape[1]

        else:
            assert os.path.isfile(self.video_path), "Path error"
            self.vdo.open(self.video_path)
            self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = int(self.vdo.get(cv2.CAP_PROP_FPS))
            self.frames_num = int(self.vdo.get(cv2.CAP_PROP_FRAME_COUNT))
            assert self.vdo.isOpened()

        if self.args.save_path:
            os.makedirs(self.args.save_path, exist_ok=True)
            os.makedirs(os.path.join(self.args.save_path, 'vis'), exist_ok=True)
            # path of saved video and results
            self.save_video_path = os.path.join(self.args.save_path, 'vis', self.video_name+'.avi')
            self.save_results_path = os.path.join(self.args.save_path, self.video_name+'.txt')

            # create video writer
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.writer = cv2.VideoWriter(self.save_video_path, fourcc, 20, (self.im_width, self.im_height))

            # logging
            self.logger.info("Save results to {}".format(self.args.save_path))

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def run(self):
        results = []
        idx_frame = 0
        bar = tqdm(total=self.frames_num)
        while self.vdo.grab():
            
            idx_frame += 1
            if idx_frame % self.args.frame_interval:
                continue

            start = time.time()
            _, ori_im = self.vdo.retrieve()
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)
            sec = (idx_frame-1) // self.fps
            sec_key = self.video_name+'.'+str(sec + 900)
            if self.args.with_detection and (idx_frame-1) % self.fps == 0 and sec_key in self.vt_box_dict:
                
                try:
                    hboxes = self.vt_box_dict[sec_key]
                    bbox_xyxy = np.empty((0, 4), dtype=float)
                    cls_conf = np.empty((0,), dtype=float)
                    for box_info in hboxes:
                        box, conf = box_info
                        bbox_xyxy = np.concatenate((bbox_xyxy, np.array([box])), axis=0)
                        cls_conf = np.concatenate((cls_conf, np.array([conf])), axis=0)
                    bbox_xyxy[:, 0] *= float(self.im_width)
                    bbox_xyxy[:, 1] *= float(self.im_height)
                    bbox_xyxy[:, 2] *= float(self.im_width)
                    bbox_xyxy[:, 3] *= float(self.im_height)
                    bbox_xywh = bbox_xyxy.copy()
                    bbox_xywh[:, 2] = bbox_xywh[:, 2] - bbox_xywh[:, 0]
                    bbox_xywh[:, 3] = bbox_xywh[:, 3] - bbox_xywh[:, 1]
                    ipdb.set_trace()
                except:
                    ipdb.set_trace()
            # do detection
            bbox_xywh, cls_conf, cls_ids = self.detector(im)
            # select person class
            mask = cls_ids == 0

            bbox_xywh = bbox_xywh[mask]
            # bbox dilation just in case bbox too small, delete this line if using a better pedestrian detector
            # bbox_xywh[:, 3:] *= 1.2
            cls_conf = cls_conf[mask]
            if self.args.with_detection and (idx_frame-1) % self.fps == 0 and sec_key in self.vt_box_dict:
                ipdb.set_trace()
            # do tracking
            outputs = self.deepsort.update(bbox_xywh, cls_conf, im)

            # draw boxes for visualization
            if len(outputs) > 0:
                bbox_tlwh = []
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                ori_im = draw_boxes(ori_im, bbox_xyxy, identities)

                for bb_xyxy in bbox_xyxy:
                    bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))

                results.append((idx_frame - 1, bbox_tlwh, identities))

            end = time.time()

            if self.args.display:
                cv2.imshow("test", ori_im)
                cv2.waitKey(1)

            if self.args.save_path:
                self.writer.write(ori_im)

            # save results
            write_results(self.save_results_path, results, 'mot')
            bar.update(1)
            self.video_bar.update(0)
            # logging
            # self.logger.info("time: {:.03f}s, fps: {:.03f}, detection numbers: {}, tracking numbers: {}" \
            #                  .format(end - start, 1 / (end - start), bbox_xywh.shape[0], len(outputs)))

def split_range(num_parts, start_in, end_in):
    a = np.arange(start_in, end_in)
    res = np.array_split(a, num_parts)
    end = list(np.add.accumulate([len(x) for x in res]))
    start = [0] + end[:-1]
    ix = list(zip(start, end))
    return ix

def multiproc(args, gpu_list, data_length):
    cmd = ('CUDA_VISIBLE_DEVICES={gpu} python -u {binary} '
            '--gpus 0 --path-pkl {path_pkl} --save_path {save_path} '
            '--range {start} {end}' )
    # print(args.range)
    range_list = split_range(len(gpu_list), args.range[0], args.range[1])
    cmd_cwd_list = [(cmd.format(binary='yolov3_deepsort_ava.py', gpu=gpu, save_path=args.save_path, path_pkl=args.path_pkl, start=range_list[gpu_idx][0], end=range_list[gpu_idx][1]), '.') for gpu_idx, gpu in enumerate(gpu_list)]

    print('processes num: {:d}, data length: {:d}...'.format(len(cmd_cwd_list), data_length))

    pool = process_pool()
    pool.apply(cmd_cwd_list)
    pool.wait()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', default='0', type=str)
    parser.add_argument("--path-pkl", default='./ava_path_list.pkl',type=str)
    parser.add_argument("--predicted-box-json", default='./vt_box_dict.json',type=str)
    parser.add_argument("--config_detection", type=str, default="./configs/yolov3.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    # parser.add_argument("--ignore_display", dest="display", action="store_false", default=True)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--with_detection", action="store_true")
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="./ava_results/")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    parser.add_argument("--range",
                        nargs=2,
                        type=int,
                        default=[0, -1],
                        help="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_deepsort)
    full_paths = pickle.load(open(args.path_pkl,'rb'))
    vt_box_dict = json.load(open('vt_box_dict.json','r'))
    gpu_list = args.gpus.split(',')
    if args.range[1] == -1:
        args.range[1] = len(full_paths)
    if len(gpu_list) > 1:
        multiproc(args, gpu_list, len(full_paths))
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_list[0])
        video_bar = tqdm(total=args.range[1]-args.range[0])
        for path in full_paths[args.range[0]:args.range[1]]:
            with VideoTracker(cfg, args, video_path=path, video_bar=video_bar, vt_box_dict=vt_box_dict) as vdo_trk:
                vdo_trk.run()
            video_bar.update(1)
