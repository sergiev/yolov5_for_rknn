import random
import onnxruntime
import torch
import os
import cv2
import numpy as np
import time
from utils.plots import plot_one_box
from utils.general import non_max_suppression
from utils.datasets import letterbox


class Detect:
    stride = [8, 16, 32]  # strides computed during build

    def __init__(self, size, nc=80, anchors=()):  # detection layer
        super(Detect, self).__init__()
        self.size = size
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.anchors = a  # shape(nl,na,2)
        self.anchor_grid = a.clone().view(self.nl, 1, -1, 1, 1, 2)  # shape(nl,1,na,1,1,2)

    def predict(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        print("# anchors", self.na)
        print("# detection layers", self.nl)
        print("# outputs", self.no)
        print("len x", len(x), type(x))
        w = self.size[0]
        for i in range(self.nl):
            x[i] = torch.from_numpy(x[i])
            print("xi shape:", x[i].shape)
            bs, _, ny, nx, _ = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            print("nx ny", nx, ny)
            print("bs", bs)
            y = x[i].sigmoid()
            yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
            self.grid[i] = torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
            print("grid shape>>", self.grid[i].shape)
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * (w / nx)  # == (h / ny)
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
            z.append(y.view(bs, -1, self.no))
        out = torch.cat(z, 1)
        pred_res = non_max_suppression(out, 0.6, )[0]
        pred_res[:, :4] = scale_coords(img.shape[2:], pred_res[:, :4], img_src.shape).round()
        return pred_res


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def processImg(img_mat, new_shape=(416, 416)):
    img = letterbox(img_mat, new_shape=new_shape, auto=False)[0]
    # img = letterbox(img_mat, new_shape=new_shape)[0]
    cv2.imshow("img", img)
    cv2.waitKey()
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    return img
    return np.ascontiguousarray(img)


CLASSES = ("person", "bicycle", "car", "motorbike ", "aeroplane ", "bus ", "train", "truck ", "boat", "traffic light",
           "fire hydrant", "stop sign ", "parking meter", "bench", "bird", "cat", "dog ", "horse ", "sheep", "cow",
           "elephant",
           "bear", "zebra ", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
           "snowboard", "sports ball", "kite",
           "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
           "fork", "knife ",
           "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza ", "donut",
           "cake", "chair", "sofa",
           "pottedplant", "bed", "diningtable", "toilet ", "tvmonitor", "laptop	", "mouse	", "remote ",
           "keyboard ", "cell phone", "microwave",
           "oven ", "toaster", "sink", "refrigerator ", "book", "clock", "vase", "scissors ", "teddy bear ",
           "hair drier", "toothbrush ")
SIZE = (416, 416)
anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
d = Detect(SIZE, 80, anchors)
dirname = r"D:\Workspace\test_space_01\yolov5\yolov5-3.1_train\inference\images"
fs = (os.path.join(p, name) for p, _, names in os.walk(dirname) for name in names)
session = onnxruntime.InferenceSession(r"D:\Workspace\test_space_01\yolov5\yolov5-4.0\yolov5-4.0\weights\yolov5m_416x416.onnx")
for i in session.get_inputs():
    print(i)
for i in session.get_outputs():
    print(i)
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(CLASSES))]
for img_path in fs:
    if os.path.splitext(img_path)[1].lower() not in (".jpg", ".png", ".jpeg"):
        continue

    print("-" * 70)
    input_names = list(map(lambda x: x.name, session.get_inputs()))
    output_names = list(map(lambda x: x.name, session.get_outputs()))
    img_src = cv2.imread(img_path)
    img = img_src.copy()
    img = processImg(img)
    # img = img.reshape((1,) + img.shape)
    img = img[None]
    img = np.concatenate((img[..., ::2, ::2], img[..., 1::2, ::2], img[..., ::2, 1::2], img[..., 1::2, 1::2]), 1)
    img = img.astype(np.float32)
    img /= 255.0
    t0 = time.time()
    print("img shape               \t:", img.shape)
    pred_onx = session.run(
        output_names, {input_names[0]: img})
    print("_"*30)
    [print(m.shape) for m in pred_onx]
    print("_"*30)
    # print(pred_onx[0][..., 1])
    print("*" * 30)
    res = d.predict(pred_onx)
    print(res.shape)
    # out = non_max_suppression(res, 0.7)[0]
    # out[:, :4] = scale_coords(img.shape[2:], out[:, :4], img_src.shape).round()
    for *xyxy, conf, cls in res:
        label = '%s %.2f' % (CLASSES[int(cls)], conf)
        print(label)
        plot_one_box(xyxy, img_src, label=label, color=colors[int(cls)], line_thickness=3)
    cv2.imshow("src", img_src)
    cv2.waitKey()
