#!/usr/bin/python3
# -*- conding:utf-8 -*-
# Author: Wangmei
# @Time:2021/7/20 21:33
# encoding=UTF-8
import socket
import sys
import threading
from io import StringIO
from six import StringIO
import cv2
from PIL import Image
import pandas as pd
import numpy as np
import time

import argparse
from utils.datasets import *
from utils.utils import *
from parse_data import parse_conf, parse_argument
from deep_sort.util import draw_bboxes
import torch
from data_sender import *
from deep_sort import DeepSort
from lidar_camera_utils import pnp_object_location
from math import sqrt
import base64

AA = []

def distDTW(ts1,ts2):
    DTW = {}
    for i in range(len(ts1)):
        DTW[(i, -1)] = np.inf
    for i in range(len(ts2)):
        DTW[(-1, i)] = np.inf
    DTW[(-1, -1)] = 0

    for i in range(len(ts1)):
        for j in range(len(ts2)):
            dist = (ts1[i] - ts2[j]) ** 2
            DTW[(i, j)] = dist + min(DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)])
    return sqrt(DTW[len(ts1) - 1, len(ts2) - 1])

def Reliability_calculation(locdata):
    filname1 = 'centroids.txt'
    with open(filname1, "r") as f:
        for line in f.readlines():
            line = line.strip('\n').replace('[[', '').replace(']]', '').replace('[', '').replace(']', '').split(
                ',')  # 去掉列表中每一个元素的换行符
    line = np.array(line).astype(float).reshape(3, -1)
    min_dis = []
    for iidx, i in enumerate(locdata):
        dist = np.empty(len(line), dtype=float)
        for jidx, j in enumerate(line):
            dist[jidx] = distDTW(i, j)
        min_dis.append(np.min(dist))
    return min_dis

def k_means_clust(data, num_clust, num_iter, w=3):
    ## 步骤一: 初始化均值点
    centroids = random.sample(list(data), num_clust)
    counter = 0
    for n in range(num_iter):
        counter += 1
        assignments = {}  #存储类别0，1，2等类号和所包含的类的号码
        # 遍历每一个样本点 i ,因为本题与之前有所不同，多了ind的编码
        for ind, i in enumerate(data):
            min_dist = float('inf')   #最近距离，初始定一个较大的值
            closest_clust = None     # closest_clust：最近的均值点编号
            ## 步骤二: 寻找最近的均值点
            for c_ind, j in enumerate(centroids):  #每个点和中心点的距离，共有num_clust个值
                # if LB_Keogh(i, j, 3) < min_dist:    #循环去找最小的那个
                    cur_dist = distDTW(i, j)
                    if cur_dist < min_dist:         #找到了ind点距离c_ind最近
                        min_dist = cur_dist
                        closest_clust = c_ind
            ## 步骤三: 更新 ind 所属簇
            if closest_clust in assignments:
                assignments[closest_clust].append(ind)
            else:
                assignments[closest_clust] = []
                assignments[closest_clust].append(ind)
        # recalculate centroids of clusters  ## 步骤四: 更新簇的均值点
        for key in assignments:
            clust_sum = 0
            data_len = 0
            for k in assignments[key]:
                data_len = data_len + len(data[k])
            data_len = int(data_len / len(assignments[key]))
            for k in assignments[key]:
                if len(data[k]) > data_len:
                    d = data[k]
                    data[k] = d[0:data_len]
                if len(data[k]) < data_len:
                    gap = data_len - len(data[k])
                    d = data[k]
                    add = [d[-1]] * int(gap)
                    data[k] = np.hstack((d,add))
            for k in assignments[key]:
                clust_sum = clust_sum + data[k]
            centroids[key] = [m / len(assignments[key]) for m in clust_sum]
    with open('centroids2', "w") as f:
        f.write(str(centroids))

def coord2loc():
    filnam = 'tracetxt2.txt'
    centers = np.loadtxt('center.txt', delimiter=',')
    location = np.array([2 * 0, 2 ** 1, 2 ** 2])
    data = []
    Numb = []
    with open(filnam, "r") as f:
        for line in f.readlines():
            line = line.strip('\n').split(',')  # 去掉列表中每一个元素的换行符
            raw = np.empty(len(line), dtype=int)
            loc = []
            for idx, val in enumerate(line):
                raw[idx] = val
            Numb.append(raw[0])
            for i in range(1, len(raw), 2):
                distence = (raw[i] - centers[:, 0]) ** 2 + (raw[i] - centers[:, 0]) ** 2
                distence = distence.tolist()
                j = distence.index(min(distence))
                loc.append(location[j])
            data.append(loc)
    num_clust = 3
    num_iter = 800
    k_means_clust(data, num_clust, num_iter, 3)

def trace_read():
    filname = '.txt'
    with open(filname, "r") as f:
        line = f.read().replace('[', '').replace(']], ', '\n').replace(']', '')  # 去掉列表中每一个元素的换行符
    filname2 = 'tracetxt2.txt'
    with open(filname2, "w") as f:
        f.write(line)
    coord2loc()

def write_txt(root_dir,data):
    with open(root_dir,"w",encoding='utf-8') as f:
        f.write(data)

def detect(dataset,device,model,h_inv,trace2txt,persionid):
    # Get names and colors
    names = model.names if hasattr(model, 'names') else model.modules.names
    # print('names = ', names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

    out__ = cv2.VideoWriter('./output.avi', fourcc, 20, (1280, 720))
    filnam2 = "tracetxt.txt"
    # Run inference
    imgsz = opt.img_size
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half())
    if dataset.ndimension() == 3:
        dataset = dataset.unsqueeze(0)
    for im0s in dataset:
        bbox_Tracking = []  # 矩形框
        cls_ids_Tracking = []  # 类别下标
        cls_conf = []  # 置信度
        # print('img 1 = ',img.shape)
        # print('img0s 1= ', img.shape)
        img = im0s.copy()
        img = torch.from_numpy(img).to(device)
        half = True
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        outputs_tracking = []
        # Inference
        t1 = time.time()
        preds = []
        pred = model(img, augment=opt.augment)[0]
        # print('pred1 = ',pred.shape)
        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                   fast=True, classes=opt.classes, agnostic=opt.agnostic_nms)
        # print('pred2 = ', len(pred),pred)
        webcam = False
        # Process detections
        path = 'load image'
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                # p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                p, s, im0 = path, '%g: ' % i, im0s.copy()
            else:
                p, s, im0 = path, '', im0s
            # save_path = str(Path(out) / Path(p).name)
            # print('img0 1 = ',im0[i].shape)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0[i].shape)[[1, 0, 1, 0]]  #  normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0[i].shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string
                # Write results
                for *xyxy, conf, cls in det:
                    xyxy = np.asarray((torch.tensor(xyxy).view(1, 4))).astype(int)[0]
                    cxcy = list([int((xyxy[0] + xyxy[2]) / 2), int((xyxy[1] + xyxy[3]) / 2), int(xyxy[2] - xyxy[0]),
                                 int(xyxy[3] - xyxy[1])])
                    bbox_Tracking = list(bbox_Tracking)
                    bbox_Tracking.append(cxcy)
                    cls_ids_Tracking = list(cls_ids_Tracking)
                    cls_ids_Tracking.append(cls)
                    cls_conf = list(cls_conf)
                    cls_conf.append(conf)
                    # label = '%s %.2f' % (names[int(cls)], conf)
                    # plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
            bbox_Tracking = np.asarray(bbox_Tracking)
            cls_ids_Tracking = np.asarray(cls_ids_Tracking)
            cls_conf = np.asarray(cls_conf)
            global deepsort
            if len(bbox_Tracking) > 0:
                outputs_tracking = deepsort.update(bbox_Tracking, cls_conf, cls_ids_Tracking, im0[i])
            if outputs_tracking is not None and len(outputs_tracking) > 0:
                # if len(boxes) > 0:
                bbox_xyxy = outputs_tracking[:, :4]  # x1, y1, x2, y2
                identities = outputs_tracking[:, 5]  # track_id
                clsTracking = outputs_tracking[:, 4]  # classLabel index
                trace = outputs_tracking[:, -1]  # trace of object
                locs = []
                for idx, tra in enumerate(trace):
                    loc = []
                    for coord in tra:
                        distence = (coord[0] - centers[:, 0]) ** 2 + (coord[1] - centers[:, 0]) ** 2
                        distence = distence.tolist()
                        j = distence.index(min(distence))
                        loc.append(location[j])
                    locs.append(loc)
                min_dis = Reliability_calculation(locs)
                for cxcy2 in outputs_tracking:
                    # peopleid = str(cxcy2[5])
                    coords2 = [int((cxcy2[0] + cxcy2[2]) / 2), int((cxcy2[1] + cxcy2[3]) / 2), int(cxcy2[2] - xyxy[0]),
                               int(cxcy2[3] - cxcy2[1])]
                    gap = int((coords2[3] - coords2[1]) / 5 * 2)
                    coords2[1] += gap
                    if cxcy2[5] in persionid:
                        for idx, entry in enumerate(persionid):
                            if entry == cxcy2[5]:
                                cxy = list([coords2[0], coords2[1]])
                                trace2txt[idx].append(cxy)
                                # print('trace2txt2 = ', trace2txt)
                    else:
                        persionid.append(cxcy2[5])
                        idnum = len(persionid) - 1
                        cxy = list([coords2[0], coords2[1]])
                        trace2txt.append([cxcy2[5]])
                        trace2txt[idnum].append(cxy)
                        # print('trace2txt1 = ', trace2txt)
                # trace2txt[identities].append(coords2)
                # 打印追踪后的框bbox  ids
                # print(im0[i].shape)
                ori_im = draw_bboxes(im0[i], bbox_xyxy, identities, clsTracking, trace, h_inv, min_dis)

                # resultSender(outputs_tracking, h_inv, Sensor_ID, udpClient, addr, (time.time() - t1))
            t2 = time.time()
            if len(trace2txt) > 300:
                # trace2txt = np.array(trace2txt)
                with open(filnam2, "w") as f:
                    f.write(str(trace2txt))
                trace2txt = []
                persionid = []
                thread = Thread(target=trace_read, args=())
                thread.start()
                # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if True:
                cv2.imshow(p, im0[i])
                out__.write(im0[i])
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    with open(filnam2, "w") as f:
                        f.write(str(trace2txt))
                    raise StopIteration

class ServerThreading(threading.Thread):
    # words = text2vec.load_lexicon()
    def __init__(self, clientsocket, device,model,h_inv,recvsize=1024 * 1024, encoding="utf-8",):
        threading.Thread.__init__(self)
        self._socket = clientsocket
        self._recvsize = recvsize
        self._encoding = encoding
        self.device = device
        self.model = model
        self.h_inv = h_inv
        pass

    def run(self):
        print("thread start.....")
        trace2txt = []
        persionid = []
        BUFSIZ = 1024 * 20
        try:
            # 接受数据
            msg = ''
            while True:
                rec_d = bytes([])
                # 读取recvsize个字节
                # rec = self._socket.recv(self._recvsize)
                while True:
                    rec = self._socket.recv(BUFSIZ)
                    data = rec
                    if not data or len(data) == 0:
                        break
                    else:
                        rec_d = rec_d + data
                # 解码
                rec_d = base64.b64decode(rec_d)
                np_arr = np.fromstring(rec_d, np.uint8)
                image = cv2.imdecode(np_arr, 1)
                cv2.imshow('image', image)

                # rec1=rec.decode(self._encoding)

                # print(type(img))
                # img.show()

                #print("******rec1******",rec1)
                msg += rec.decode(self._encoding)



                # 文本接受是否完毕，因为python socket不能自己判断接收数据是否完毕，
                # 所以需要自定义协议标志数据接受完毕
                if msg.strip().endswith('over'):
                    msg = msg[:-4]

                    break



           # AA.append(msg)

            #print("accept msg:%s" % str(msg))
            sendmsg = msg
            # 发送数据
            self._socket.send(("%s" % sendmsg).encode(self._encoding))
            pass
        except Exception as identifier:
            self._socket.send("500".encode(self._encoding))
            print(identifier)
            pass
        finally:
            self._socket.close()
        print("*******执行****")

        # print(type(msg))
        # img = eval(msg)
        # print(len(img))
        # img=np.asarray(a=img,dtype=np.uint8).reshape(640,480,3)
        # torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        # dataset = np.asarray(img)
        # detect(image,self.device,self.model,self.h_inv,trace2txt,persionid)
        #img = Image.fromarray(np.uint8(img))


        # cv2.imwrite("aaa.png",img)
        # print("img shape",img.shape)
        # cv2.imshow("Image",img)
        time.sleep(3)

        # img = Image.open(StringIO.StringIO(msg))
        # print(type(img))
       # img.show()

        # AS11 = pd.DataFrame(msg)
        # AS11.to_csv(".\AA.csv")
        # print("执行完")
        # print("thread over.....")

        pass

    def __del__(self):
        pass

deepsort = DeepSort('./deep_sort/deep/checkpoint/ckpt.t7')
def main(Sensor_ID, udpClient, addr, jsonName, centers, location):
    # 创建服务器套接字
    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 获取本地主机名称
    host = socket.gethostname()
    # 设置一个端口
    port = 12345
    # 将套接字与本地主机和端口绑定
    serversocket.bind((host, port))
    # 设置监听最大连接数
    serversocket.listen(5)
    # 获取本地服务器的连接信息
    myaddr = serversocket.getsockname()
   # print("server adress:%s" % str(myaddr))
    # 循环等待接受客户端信息

    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    solve_homegraphy = pnp_object_location(jsonName)
    object_2d_points, object_3d_point = solve_homegraphy.com_cfg()
    h, h_inv = solve_homegraphy.solve_Hom(object_2d_points, object_3d_point)
    # Initialize
    device = torch_utils.select_device(opt.device)
    half = True  # half precision only supported on CUDA


    # Load model
    google_utils.attempt_download(weights)
    model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    while True:
        # 获取一个客户端连接
        clientsocket, addr = serversocket.accept()
        #print("socket adress:%s" % str(addr))
        try:
            t = ServerThreading(clientsocket,device,model,h_inv)  # 为每一个请求开启一个处理线程
            t.start()
            pass
        except Exception as identifier:
            print(identifier)
            pass
        pass
    serversocket.close()
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights-s/best.pt', help='model.pt path')
    # parser.add_argument('--source', type=str, default='inference/videos/getvideo_189_2019-07-20_09-17-48.avi', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default='D:/pycode/dataset/2DMOT2015/test/AVG-TownCentre/img1',
                        help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--idx', default='2', help='idx')
    opt = parser.parse_args()
    opt.img_size = check_img_size(opt.img_size)
    print(opt)

    with torch.no_grad():
        Sensor_ID, local_IP, local_port, heart_port, cameraAddr, cam_IP, pls_xy, real_xy, RSU_IP, RSU_port, jsonName, is_show = parse_conf(
            'NL_config', opt.idx)
        udpClient, addr = udp_client(RSU_IP, RSU_port)
        centers = np.loadtxt('center.txt', delimiter=',')
        location = np.array([2 * 0, 2 ** 1, 2 ** 2])
        main(Sensor_ID, udpClient, addr, jsonName, centers, location)