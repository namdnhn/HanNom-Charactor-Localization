import numpy as np
import torch
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import check_img_size, non_max_suppression

from yolov5.utils.augmentations import letterbox

label_test = \
"""
0 0.873333 0.742175 0.022222 0.034596
0 0.903889 0.780066 0.023333 0.031301
0 0.126111 0.612850 0.021111 0.036244
0 0.643889 0.407743 0.018889 0.031301
0 0.126111 0.570840 0.021111 0.031301
0 0.436667 0.743822 0.022222 0.037891
0 0.927778 0.822076 0.022222 0.036244
0 0.126667 0.700165 0.024444 0.036244
0 0.437222 0.864909 0.023333 0.036244
0 0.127222 0.657331 0.023333 0.036244
0 0.927778 0.781713 0.022222 0.034596
0 0.462222 0.909390 0.022222 0.036244
0 0.399444 0.702636 0.036667 0.047776
0 0.461667 0.743822 0.021111 0.034596
0 0.411111 0.447282 0.022222 0.037891
0 0.502778 0.135914 0.047778 0.037891
0 0.461111 0.864086 0.022222 0.037891
0 0.127222 0.529654 0.021111 0.031301
0 0.849444 0.282537 0.021111 0.031301
0 0.502222 0.303954 0.044444 0.037891
0 0.927222 0.530478 0.023333 0.032949
0 0.668333 0.448929 0.023333 0.034596
0 0.850000 0.741351 0.022222 0.032949
0 0.153333 0.658155 0.024444 0.037891
0 0.437222 0.782537 0.023333 0.032949
0 0.152778 0.743822 0.023333 0.037891
0 0.127222 0.405272 0.021111 0.029654
0 0.850000 0.781713 0.022222 0.034596
0 0.770556 0.448105 0.023333 0.036244
0 0.448889 0.702636 0.037778 0.044481
0 0.126667 0.445634 0.022222 0.034596
0 0.410556 0.406919 0.023333 0.032949
0 0.281667 0.742175 0.021111 0.034596
0 0.152222 0.447282 0.022222 0.037891
0 0.151667 0.614498 0.025556 0.036244
0 0.256111 0.406096 0.021111 0.031301
0 0.255556 0.448929 0.022222 0.034596
0 0.873333 0.326194 0.024444 0.036244
0 0.900556 0.406919 0.021111 0.032949
0 0.230556 0.446458 0.021111 0.032949
0 0.502222 0.527183 0.031111 0.036244
0 0.501111 0.237232 0.044444 0.042834
0 0.127778 0.490115 0.024444 0.037891
0 0.087222 0.704283 0.036667 0.044481
0 0.398889 0.364909 0.035556 0.047776
0 0.139444 0.363262 0.034444 0.044481
0 0.902222 0.490115 0.022222 0.034596
0 0.643333 0.448105 0.020000 0.032949
0 0.242222 0.703460 0.037778 0.049423
0 0.926111 0.446458 0.021111 0.032949
0 0.823333 0.325371 0.020000 0.031301
0 0.190000 0.703460 0.033333 0.042834
0 0.151667 0.531301 0.023333 0.034596
0 0.912778 0.699341 0.036667 0.044481
0 0.242778 0.658979 0.034444 0.042834
0 0.086667 0.321252 0.033333 0.039539
0 0.798333 0.325371 0.021111 0.031301
0 0.770556 0.410214 0.023333 0.032949
0 0.927222 0.409390 0.021111 0.031301
0 0.822222 0.284185 0.022222 0.034596
0 0.190556 0.365733 0.034444 0.042834
0 0.707778 0.701812 0.035556 0.052718
0 0.760000 0.700165 0.035556 0.042834
0 0.192222 0.278418 0.035556 0.046129
0 0.294444 0.658979 0.033333 0.042834
0 0.913333 0.656507 0.035556 0.041186
0 0.448889 0.658979 0.035556 0.042834
0 0.900556 0.447282 0.025556 0.037891
0 0.294444 0.612026 0.035556 0.047776
0 0.604444 0.238880 0.033333 0.042834
0 0.138889 0.322076 0.033333 0.041186
0 0.913333 0.368204 0.033333 0.041186
0 0.927778 0.488468 0.024444 0.034596
0 0.861667 0.698517 0.034444 0.046129
0 0.346111 0.659802 0.034444 0.041186
0 0.191111 0.659802 0.033333 0.044481
0 0.231667 0.406919 0.023333 0.036244
0 0.139444 0.279242 0.036667 0.044481
0 0.655556 0.701812 0.035556 0.046129
0 0.086667 0.614498 0.037778 0.049423
0 0.668333 0.408567 0.023333 0.032949
0 0.398333 0.660626 0.034444 0.039539
0 0.086667 0.660626 0.035556 0.042834
0 0.191111 0.613674 0.033333 0.047776
0 0.759444 0.658155 0.034444 0.041186
0 0.243333 0.320428 0.033333 0.037891
0 0.552778 0.323723 0.034444 0.041186
0 0.449444 0.614498 0.036667 0.046129
0 0.087222 0.364909 0.038889 0.044481
0 0.707778 0.612850 0.035556 0.046129
0 0.655556 0.367381 0.035556 0.046129
0 0.810000 0.701812 0.035556 0.046129
0 0.295000 0.322900 0.034444 0.039539
0 0.398333 0.322900 0.034444 0.039539
0 0.397222 0.613674 0.034444 0.051071
0 0.242778 0.613674 0.036667 0.047776
0 0.243333 0.364086 0.037778 0.046129
0 0.346111 0.322076 0.034444 0.041186
0 0.553889 0.282537 0.032222 0.041186
0 0.345556 0.702636 0.035556 0.044481
0 0.605000 0.281713 0.034444 0.039539
0 0.086667 0.278418 0.037778 0.049423
0 0.656111 0.657331 0.034444 0.042834
0 0.191111 0.322076 0.035556 0.041186
0 0.810556 0.658979 0.034444 0.042834
0 0.656667 0.612850 0.035556 0.046129
0 0.243333 0.279242 0.035556 0.044481
0 0.294444 0.279242 0.033333 0.044481
0 0.861111 0.656507 0.033333 0.041186
0 0.398333 0.279242 0.034444 0.047776
0 0.912778 0.283361 0.036667 0.046129
0 0.913333 0.611203 0.035556 0.046129
0 0.860556 0.612026 0.034444 0.044481
0 0.605000 0.191928 0.032222 0.047776
0 0.346111 0.366557 0.034444 0.051071
0 0.758889 0.613674 0.035556 0.047776
0 0.758333 0.365733 0.036667 0.049423
0 0.758333 0.282537 0.034444 0.044481
0 0.502222 0.443163 0.042222 0.046129
0 0.450556 0.322076 0.034444 0.041186
0 0.346667 0.612850 0.035556 0.049423
0 0.656111 0.280890 0.034444 0.044481
0 0.912778 0.327018 0.034444 0.041186
0 0.345556 0.278418 0.035556 0.046129
0 0.655556 0.325371 0.033333 0.041186
0 0.450556 0.278418 0.034444 0.042834
0 0.811111 0.612850 0.033333 0.046129
0 0.706111 0.280890 0.034444 0.044481
0 0.605000 0.325371 0.034444 0.044481
0 0.706111 0.325371 0.034444 0.041186
0 0.706111 0.369028 0.034444 0.046129
0 0.554444 0.237232 0.033333 0.046129
0 0.502222 0.487644 0.040000 0.042834
0 0.502222 0.271829 0.042222 0.029654
0 0.097778 0.865733 0.028889 0.041186
0 0.513889 0.739703 0.021111 0.029654
0 0.437778 0.908567 0.022222 0.034596
0 0.902778 0.821252 0.023333 0.034596
0 0.437778 0.823723 0.024444 0.036244
0 0.098889 0.780066 0.026667 0.037891
0 0.127222 0.741351 0.021111 0.032949
0 0.851667 0.863262 0.023333 0.032949
0 0.903333 0.864086 0.024444 0.034596
0 0.927778 0.742998 0.022222 0.036244
0 0.098333 0.743822 0.023333 0.031301
0 0.410000 0.780890 0.024444 0.032949
0 0.873889 0.783361 0.021111 0.034596
0 0.873333 0.282537 0.022222 0.031301
0 0.462222 0.781713 0.022222 0.031301
0 0.822778 0.365733 0.023333 0.032949
0 0.553333 0.366557 0.035556 0.034596
0 0.153333 0.406919 0.020000 0.032949
0 0.295556 0.363262 0.035556 0.044481
0 0.851111 0.822076 0.024444 0.032949
0 0.151667 0.572488 0.023333 0.031301
0 0.904444 0.908567 0.024444 0.037891
0 0.281667 0.781713 0.021111 0.034596
0 0.294444 0.700165 0.035556 0.042834
0 0.876111 0.865733 0.021111 0.034596
0 0.502222 0.369028 0.044444 0.032949
0 0.308333 0.781713 0.021111 0.031301
0 0.928333 0.908567 0.023333 0.034596
0 0.407778 0.742998 0.026667 0.032949
0 0.502778 0.338550 0.043333 0.031301
0 0.501667 0.202636 0.047778 0.029654
0 0.456111 0.366557 0.023333 0.037891
0 0.850556 0.906919 0.023333 0.034596
0 0.796667 0.281713 0.022222 0.032949
0 0.928333 0.864909 0.023333 0.036244
0 0.308333 0.741351 0.021111 0.029654
0 0.903333 0.739703 0.024444 0.032949
0 0.152222 0.701812 0.022222 0.036244
0 0.502222 0.170511 0.044444 0.037891
"""

# def read_label(img, str_output):
#     gt = []
#     for line in str_output.strip().split("\n"):
#         tmp = line.strip().split(' ')

#         w, h = img.shape[1], img.shape[0]
#         x = [(float)(w.strip()) for w in tmp]

#         x1 = int(x[1] * w)
#         width = int(x[3] * w)

#         y1 = int(x[2] * h)
#         height = int(x[4] * h)

#         gt += [(x1, y1, width, height, 0, 0, 0)]

#     return gt

def read_label(img, str_output):
    gt = []
    for line in str_output.strip().split("\n"):
        tmp = line.strip().split(' ')

        w, h = img.shape[1], img.shape[0]
        x = [(float)(w.strip()) for w in tmp]

        x1 = int(x[1] * w)
        width = int(x[3] * w)

        y1 = int(x[2] * h)
        height = int(x[4] * h)

        gt += [(x1, y1, width, height, 0, 0, 0)]

    return gt

def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

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

class HanNomOCR:
    def __init__(self, noise=50):
        """
        You should hard fix all the requirement parameters
        """
        self.name = 'HanNomOCR'
        self.noise = noise
        
        model_path = './weight/best.pt'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        np.random.seed(1)

        # Load YOLOv5 model
        self.model = DetectMultiBackend(model_path, device=device, dnn=False)
        self.model.eval()
        
        # self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
        
    # def detect(self, img):
    #     base_outputs = read_label(img, label_test)
    #     noise = np.random.randint(0, self.noise, size=(len(base_outputs), 4)) - (self.noise // 2)
    #     preds = []

    #     for i in range(len(base_outputs)):
    #         confidence = np.sum(np.abs(noise[i, :]))
    #         confidence = 1 - 1.0*confidence/200
    #         preds += [(confidence, base_outputs[i][0] + noise[i][0],
    #                           base_outputs[i][1] + noise[i][1],
    #                           base_outputs[i][2] + noise[i][2],
    #                           base_outputs[i][3] + noise[i][3])]
    #     # List of confidence, xcenter, ycenter, width, height
    #     return np.array(preds)
    
    def detect(self, img):
        img_size = check_img_size(640, s=self.model.stride)  # Ensure image size is a multiple of stride
        img0 = img.copy()
        img = letterbox(img, img_size, stride=self.model.stride, auto=True)[0]  # Resize image
        img = img.transpose((2, 0, 1))[::-1]  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # Expand for batch dim

        # Inference
        pred = self.model(img, augment=False, visualize=False)
        pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)
        
        preds = []
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    x_center = (xyxy[0] + xyxy[2]) / 2
                    y_center = (xyxy[1] + xyxy[3]) / 2
                    width = xyxy[2] - xyxy[0]
                    height = xyxy[3] - xyxy[1]
                    preds.append([conf.item(), x_center.item(), y_center.item(), width.item(), height.item()])

        return np.array(preds)
