import argparse
from utils.datasets import *
from utils.utils import *
from parse_data import parse_conf, parse_argument
import cv2
from deep_sort.util import draw_bboxes
import torch
import time
from data_sender import *
from deep_sort import DeepSort
from lidar_camera_utils import pnp_object_location
from math import sqrt

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
    filname1 = 'D:/pycode/code/Mean_Shift/centroids.txt'
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
        # print
        # counter
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
    filname = 'D:/pycode/code2021/yolov5_deepsort/tracetxt.txt'
    with open(filname, "r") as f:
        line = f.read().replace('[', '').replace(']], ', '\n').replace(']', '')  # 去掉列表中每一个元素的换行符
    filname2 = 'D:/pycode/code2021/yolov5_deepsort/tracetxt2.txt'
    with open(filname2, "w") as f:
        f.write(line)
    coord2loc()

deepsort = DeepSort('./deep_sort/deep/checkpoint/ckpt.t7')
def detect(Sensor_ID, udpClient, addr,jsonName,centers,location):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    # sources = ["rtsp://admin:zhongxinhik123@172.16.27.140:554/ch2/main/av_stream",
    #  "rtsp://admin:zhongxinhik123@172.16.27.141:554/ch2/main/av_stream"]
    # source = sources[0]
    webcam =source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
    solve_homegraphy = pnp_object_location(jsonName)
    object_2d_points, object_3d_point = solve_homegraphy.com_cfg()
    h, h_inv = solve_homegraphy.solve_Hom(object_2d_points, object_3d_point)
    # Initialize
    device = torch_utils.select_device(opt.device)
    half =True  # half precision only supported on CUDA
    trace2txt = []
    persionid = []
   
    # Load model
    google_utils.attempt_download(weights)
    model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    t0 = time.time()
    if webcam:
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)
    # 所有类型的目标都跟踪

    # Get names and colors
    names = model.names if hasattr(model, 'names') else model.modules.names
    # print('names = ', names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out__ = cv2.VideoWriter('./output.mp4',fourcc,25,(1920,1080))
    filnam2 = "tracetxt.txt"
    # Run inference
    
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half())
    for path, img, im0s, vid_cap in dataset:
        bbox_Tracking = []      #矩形框
        cls_ids_Tracking = []   #类别下标
        cls_conf = []           #置信度
        # print('img 1 = ',img.shape)
        # print('img0s 1= ', img.shape)
        img = torch.from_numpy(img).to(device)
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

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                # p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            # print('img0 1 = ',im0[i].shape)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string
                # Write results
                for *xyxy, conf, cls in det:
                    xyxy=np.asarray((torch.tensor(xyxy).view(1, 4))).astype(int)[0]
                    cxcy=list([int((xyxy[0]+xyxy[2])/2),int((xyxy[1]+xyxy[3])/2),int(xyxy[2]-xyxy[0]),int(xyxy[3]-xyxy[1])])
                    # if type(bbox_Tracking) != list:
                    #     bbox_Tracking = list(bbox_Tracking)
                    bbox_Tracking.append(cxcy)
                    # if type(cls_conf) != list:
                    # cls_ids_Tracking = list(cls_ids_Tracking)
                    cls_ids_Tracking.append(cls)
                    # # if type(cls_conf) != list:
                    # cls_conf = list(cls_conf)
                    cls_conf.append(conf)
                    #label = '%s %.2f' % (names[int(cls)], conf)
                    #plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
            bbox_Tracking=np.asarray(bbox_Tracking)
            # cls_ids_Tracking=Tensor.cpu()
            cls_ids_Tracking = np.asarray(cls_ids_Tracking)
            cls_conf=np.asarray(cls_conf)
            global deepsort
            if len(bbox_Tracking) > 0:
                outputs_tracking = deepsort.update(bbox_Tracking, cls_conf, cls_ids_Tracking, im0)
            if outputs_tracking is not None and len(outputs_tracking) > 0:
            # if len(boxes) > 0:
                bbox_xyxy = outputs_tracking[:, :4]   #x1, y1, x2, y2
                identities = outputs_tracking[:, 5]  #track_id
                clsTracking = outputs_tracking[:, 4]  #classLabel index
                trace = outputs_tracking[:, -1]   # trace of object
                locs = []
                for idx,tra in enumerate(trace):
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
                    coords2 = [int((cxcy2[0]+cxcy2[2])/2),int((cxcy2[1]+cxcy2[3])/2),int(cxcy2[2]-xyxy[0]),int(cxcy2[3]-cxcy2[1])]
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
                        idnum = len(persionid)-1
                        cxy = list([coords2[0],coords2[1]])
                        trace2txt.append([cxcy2[5]])
                        trace2txt[idnum].append(cxy)
                        # print('trace2txt1 = ', trace2txt)
                # trace2txt[identities].append(coords2)
                #打印追踪后的框bbox  ids
                # print(im0[i].shape)
                ori_im = draw_bboxes(im0, bbox_xyxy, identities, clsTracking, trace, h_inv,min_dis)
                
                resultSender(outputs_tracking, h_inv, Sensor_ID, udpClient, addr, (time.time()-t1))
            t2=time.time()
            if len(trace2txt)>300:
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
                # cv2.imshow(p, im0)
                
                out__.write(im0)
                with open(filnam2, "w") as f:
                    f.write(str(trace2txt))
                # if cv2.waitKey(1) == ord('q'):  # q to quit
                #     with open(filnam2, "w") as f:
                #         f.write(str(trace2txt))
                #     raise StopIteration


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights-s/best.pt', help='model.pt path')
    # parser.add_argument('--source', type=str, default='inference/videos/getvideo_189_2019-07-20_09-17-48.avi', help='source')  # file/folder, 0 for webcam
    # parser.add_argument('--source', type=str, default='D:/pycode/dataset/2DMOT2015/test/AVG-TownCentre/img1', help='source')
    parser.add_argument('--source', type=str, default='inference/videos/IMG_0463.MP4', help='source')# 输入数据或置0
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
        detect(Sensor_ID, udpClient, addr,jsonName,centers,location)
