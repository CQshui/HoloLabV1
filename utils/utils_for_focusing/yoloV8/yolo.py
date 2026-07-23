import colorsys
import mmap
import os
import time

import cv2
import numpy as np
import torch
from PIL import ImageDraw, ImageFont

from utils.utils_for_focusing.yoloV8.nets.yolo import YoloBody
from utils.utils_for_focusing.yoloV8.utils.utils import (cvtColor, get_classes, preprocess_input,
                                                         resize_image, show_config)
from utils.utils_for_focusing.yoloV8.utils.utils_bbox import DecodeBox
# import onnx
import torch.nn.functional as F


class YOLO(object):
    _defaults = {
        '''
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        '''
        # "model_path": 'model_data/yolov8_s.pth',
        "model_path": r'E:\DongJiayao\yolov8-pytorch\logs\best_epoch_weights.pth',
        "classes_path": r'F:\dongjiayao\Pycharm\Holo-Track\yoloV8\class.txt',

        '''
        #   输入图片的大小，必须为32的倍数。
        '''
        "input_shape": [640, 640],

        '''
        #   所使用到的yolov8的版本：
        #   n : 对应yolov8_n
        #   s : 对应yolov8_s
        #   m : 对应yolov8_m
        #   l : 对应yolov8_l
        #   x : 对应yolov8_x
        '''
        "phi": 's',

        '''
        #   只有得分大于置信度的预测框会被保留下来
        '''
        "confidence": 0.5,
        '''
        
        #   非极大抑制所用到的nms_iou大小
        '''
        "nms_iou": 0.3,

        '''
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        '''
        "letterbox_image": True,

        '''
        #   是否使用Cuda
        #   没有GPU可以设置成False
        '''
        "cuda": True,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    '''
    #   初始化YOLO
    '''

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value

        '''
        #   获得种类和先验框的数量
        '''
        # self.class_names, self.num_classes = get_classes(self.classes_path)
        self.class_names, self.num_classes = '0', 1
        self.bbox_util = DecodeBox(self.num_classes, (self.input_shape[0], self.input_shape[1]))

        '''
        #   画框设置不同的颜色
        '''
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        # note 生成模型是yolo内的主要耗时步骤
        self.generate()

        # note 该行用于显示模型参数
        # show_config(**self._defaults)

    '''
    #   生成模型
    '''

    def generate(self, onnx=False):
        # print('self.device=', self.device)
        # print('CUDA可用：', torch.cuda.is_available())

        """
        #   建立yolo模型，载入yolo模型的权重
        """
        self.net = YoloBody(self.input_shape, self.num_classes, self.phi)  # 1.4s
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 内存映射加载，加速！这一步应该只有第一次加载比较耗时，6s左右
        with open(self.model_path, 'rb') as f:
            buffer = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            state_dict = torch.load(buffer, map_location=self.device)
        self.net.load_state_dict(state_dict)
        # self.net.load_state_dict(torch.load(self.model_path, map_location=device))  # 4.2s

        self.net = self.net.fuse().eval()  # 0.8s

        # print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                # self.net = nn.DataParallel(self.net)  todo
                self.net = self.net.to(self.device)

    '''
    #   检测图片
    '''

    def detect_image(self, image, crop=False, count=False, draw=True):
        # st0 = time.time()
        """
        #   计算输入图片的高和宽
        """
        image_shape = np.array(np.shape(image)[0:2])
        '''
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        '''
        image = cvtColor(image)
        '''
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        '''
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        '''
        #   添加上batch_size维度
        #   h, w, 3 => 3, h, w => 1, 3, h, w
        '''
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        # ed0 = time.time()
        # print('in yolo', ed0 - st0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.to(self.device)
            '''
            #   将图像输入网络当中进行预测！
            '''
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            '''
            #   将预测框进行堆叠，然后进行非极大抑制
            '''
            results = self.bbox_util.non_max_suppression(outputs, self.num_classes, self.input_shape,
                                                         image_shape, self.letterbox_image, conf_thres=self.confidence,
                                                         nms_thres=self.nms_iou)

            if results[0] is None:
                return image

            top_label = np.array(results[0][:, 5], dtype='int32')
            top_conf = results[0][:, 4]
            top_boxes = results[0][:, :4]

            # 绘制检测框的原始代码...
            # 转换boxes为xywh格式
            boxes_xywh = []
            for box in top_boxes:
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                x_center = x1 + w / 2
                y_center = y1 + h / 2
                boxes_xywh.append([x_center, y_center, w, h])
            boxes_xywh = np.array(boxes_xywh)

        '''
        #   设置字体与边框厚度
        '''
        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))
        '''
        #   计数
        '''
        if count:
            print("top_label:", top_label)
            classes_nums = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(top_label == i)
                if num > 0:
                    print(self.class_names[i], " : ", num)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)
        '''
        #   是否进行目标的裁剪
        '''
        if crop:
            for i, c in list(enumerate(top_boxes)):
                top, left, bottom, right = top_boxes[i]
                top = max(0, np.floor(top).astype('int32'))
                left = max(0, np.floor(left).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom).astype('int32'))
                right = min(image.size[0], np.floor(right).astype('int32'))

                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)
        '''
        #   图像绘制
        '''
        if draw:
            for i, c in list(enumerate(top_label)):
                predicted_class = self.class_names[int(c)]
                box = top_boxes[i]
                score = top_conf[i]

                top, left, bottom, right = box

                top = max(0, np.floor(top).astype('int32'))
                left = max(0, np.floor(left).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom).astype('int32'))
                right = min(image.size[0], np.floor(right).astype('int32'))

                label = '{} {:.2f}'.format(predicted_class, score)
                draw = ImageDraw.Draw(image)
                label_bbox = draw.textbbox((0, 0), label, font=font)
                label_size = (label_bbox[2] - label_bbox[0], label_bbox[3] - label_bbox[1])
                label = label.encode('utf-8')

                print(label, top, left, bottom, right)

                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                for i in range(thickness):
                    draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
                draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
                draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
                del draw

        return image, (boxes_xywh, top_conf, top_label)

    def detect_image_np(self, image_np: np.ndarray,
                        crop=False, count=False, draw=True, device='cuda'):
        """
        极度加速版：直接输入 np.float32 或 uint8 数组（H×W×3，BGR或RGB），
        前处理全部在 GPU 上，避免 CPU 大拷贝。
        返回：out_img (H×W×3 uint8), (boxes_xywh, scores, labels)
        """
        # --- 1. 原始数据直接送 GPU (零拷贝) ---
        st = time.time()

        # 保持原始数据格式直接送 GPU，最小化 CPU 处理
        img_t = torch.as_tensor(image_np, device=device)  # [H,W,C] 或 [H,W]

        # --- 2. GPU 端通道扩展 (比 CPU 快 10-20 倍) ---
        if img_t.ndim == 2:  # 处理灰度图
            # 使用 view + expand 实现零拷贝通道扩展
            img_t = img_t.view(1, *img_t.shape)  # [H,W] -> [1,H,W]
            img_t = img_t.expand(3, -1, -1)  # [3,H,W] (共享内存)
        elif img_t.shape[2] == 1:  # 单通道扩展为三通道
            img_t = img_t.expand(-1, -1, 3)  # [H,W,3]

        # --- 3. 维度变换 + 归一化 ---
        # [H,W,C] -> [B,C,H,W] 并归一化
        img_t = img_t.unsqueeze(0).contiguous()  # [1,C,H,W]
        img_t = img_t.to(torch.float32).mul_(1.0 / 255.0)  # 原地操作省内存

        # --- 4. GPU 端 letterbox (比 CPU 快 3-5 倍) ---
        H0, W0 = img_t.shape[2], img_t.shape[3]
        Ht, Wt = self.input_shape

        # 计算缩放比例
        scale = min(Wt / W0, Ht / H0)
        new_w, new_h = int(W0 * scale), int(H0 * scale)
        # st0 = time.time()

        # GPU 双线性插值 (使用 texture memory 加速)
        img_t = F.interpolate(img_t, size=(new_h, new_w),
                              mode='bilinear', align_corners=False)

        # 计算 padding (保持长宽比)
        pad_w = Wt - new_w
        pad_h = Ht - new_h
        img_t = F.pad(img_t,
                      [pad_w // 2, pad_w - pad_w // 2,
                       pad_h // 2, pad_h - pad_h // 2],
                      value=0.5)  # 中性灰填充

        # --- 3. 模型推理（GPU） ---  todo 第一次预测似乎会耗时较长
        with torch.no_grad():
            preds = self.net(img_t)

            preds = self.bbox_util.decode_box(preds)

            results = self.bbox_util.non_max_suppression(
                preds, self.class_names.__len__(),
                self.input_shape, np.array([H0, W0]),
                True,  # letterbox_image=True
                conf_thres=self.confidence,
                nms_thres=self.nms_iou
            )

        # ed = time.time()
        # delta_time = ed - st

        if results[0] is None:  # todo, 当没有识别出物体时，会返回整张图，这不合理
            return image_np, (np.zeros((0, 4)), np.zeros(0), np.zeros(0)), 0

        det = results[0]
        xyxy = det[:, :4]  # 已经是原图坐标
        scores = det[:, 4]
        labels = det[:, 5].astype(int)

        # 转 xywh
        w = xyxy[:, 2] - xyxy[:, 0]
        h = xyxy[:, 3] - xyxy[:, 1]
        xc = xyxy[:, 0] + w / 2
        yc = xyxy[:, 1] + h / 2
        boxes_xywh = np.stack([xc, yc, w, h], axis=1)

        # ed0 = time.time()
        # print('detect_image_np', ed0 - st0)

        # --- 4. 可选统计 & 裁剪 ---
        if count:
            for u, cnt in zip(*np.unique(labels, return_counts=True)):
                print(f"{self.class_names[u]}: {cnt}")

        # out_img = image_np.copy()

        # 裁剪
        if crop:
            for i, (x, y, w, h) in enumerate(boxes_xywh.astype(int)):
                x1, y1 = x - w // 2, y - h // 2
                x2, y2 = x1 + w, y1 + h
                sub = image_np[y1:y2, x1:x2]
                cv2.imwrite(f"img_crop/crop_{i}.png", sub[:, :, ::-1])

        # --- 5. 可选绘制（CPU 上 OpenCV） ---
        if draw:
            for (x, y, w, h), s, c in zip(boxes_xywh, scores, labels):
                x1, y1 = int(x - w / 2), int(y - h / 2)
                x2, y2 = x1 + int(w), y1 + int(h)
                color = tuple((self.colors[c] * 255).astype(int).tolist())
                cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image_np,
                            f"{self.class_names[c]} {s:.2f}",
                            (x1, max(y1 - 5, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            color, 1, cv2.LINE_AA)
        ed = time.time()
        delta_time = ed - st
        print('delta_time', delta_time)
        return image_np, (boxes_xywh, scores, labels), delta_time

    def get_FPS(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])
        '''
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        '''
        image = cvtColor(image)
        '''
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        '''
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        '''
        #   添加上batch_size维度
        '''
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.to(self.device)
            '''
            #   将图像输入网络当中进行预测！
            '''
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            '''
            #   将预测框进行堆叠，然后进行非极大抑制
            '''
            results = self.bbox_util.non_max_suppression(outputs, self.num_classes, self.input_shape,
                                                         image_shape, self.letterbox_image, conf_thres=self.confidence,
                                                         nms_thres=self.nms_iou)

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                '''
                #   将图像输入网络当中进行预测！
                '''
                outputs = self.net(images)
                outputs = self.bbox_util.decode_box(outputs)
                '''
                #   将预测框进行堆叠，然后进行非极大抑制
                '''
                results = self.bbox_util.non_max_suppression(outputs, self.num_classes, self.input_shape,
                                                             image_shape, self.letterbox_image,
                                                             conf_thres=self.confidence, nms_thres=self.nms_iou)

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def detect_heatmap(self, image, heatmap_save_path):
        import cv2
        import matplotlib.pyplot as plt

        def sigmoid(x):
            y = 1.0 / (1.0 + np.exp(-x))
            return y

        '''
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        '''
        image = cvtColor(image)
        '''
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        '''
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        '''
        #   添加上batch_size维度
        '''
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.to(self.device)
            '''
            #   将图像输入网络当中进行预测！
            '''
            dbox, cls, x, anchors, strides = self.net(images)
            outputs = [xi.split((xi.size()[1] - self.num_classes, self.num_classes), 1)[1] for xi in x]

        plt.imshow(image, alpha=1)
        plt.axis('off')
        mask = np.zeros((image.size[1], image.size[0]))
        for sub_output in outputs:
            sub_output = sub_output.cpu().numpy()
            b, c, h, w = np.shape(sub_output)
            sub_output = np.transpose(np.reshape(sub_output, [b, -1, h, w]), [0, 2, 3, 1])[0]
            score = np.max(sigmoid(sub_output[..., :]), -1)
            score = cv2.resize(score, (image.size[0], image.size[1]))
            normed_score = (score * 255).astype('uint8')
            mask = np.maximum(mask, normed_score)

        plt.imshow(mask, alpha=0.5, interpolation='nearest', cmap="jet")

        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(heatmap_save_path, dpi=200, bbox_inches='tight', pad_inches=-0.1)
        print("Save to the " + heatmap_save_path)
        plt.show()

    # def convert_to_onnx(self, simplify, model_path):
    #     self.generate(onnx=True)
    #
    #     im = torch.zeros(1, 3, *self.input_shape).to('cpu')  # image size(1, 3, 512, 512) BCHW
    #     input_layer_names = ["images"]
    #     output_layer_names = ["output"]
    #
    #     # Export the model
    #     print(f'Starting export with onnx {onnx.__version__}.')
    #     torch.onnx.export(self.net,
    #                       im,
    #                       f=model_path,
    #                       verbose=False,
    #                       opset_version=12,
    #                       training=torch.onnx.TrainingMode.EVAL,
    #                       do_constant_folding=True,
    #                       input_names=input_layer_names,
    #                       output_names=output_layer_names,
    #                       dynamic_axes=None)
    #
    #     # Checks
    #     model_onnx = onnx.load(model_path)  # load onnx model
    #     onnx.checker.check_model(model_onnx)  # check onnx model
    #
    #     # Simplify onnx
    #     if simplify:
    #         import onnxsim
    #         print(f'Simplifying with onnx-simplifier {onnxsim.__version__}.')
    #         model_onnx, check = onnxsim.simplify(
    #             model_onnx,
    #             dynamic_input_shape=False,
    #             input_shapes=None)
    #         assert check, 'assert check failed'
    #         onnx.save(model_onnx, model_path)
    #
    #     print('Onnx model save as {}'.format(model_path))

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/" + image_id + ".txt"), "w", encoding='utf-8')
        image_shape = np.array(np.shape(image)[0:2])
        '''
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        '''
        image = cvtColor(image)
        '''
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        '''
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        '''
        #   添加上batch_size维度
        '''
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.to(self.device)
            '''
            #   将图像输入网络当中进行预测！
            '''
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            '''
            #   将预测框进行堆叠，然后进行非极大抑制
            '''
            results = self.bbox_util.non_max_suppression(outputs, self.num_classes, self.input_shape,
                                                         image_shape, self.letterbox_image, conf_thres=self.confidence,
                                                         nms_thres=self.nms_iou)

            if results[0] is None:
                return

            top_label = np.array(results[0][:, 5], dtype='int32')
            top_conf = results[0][:, 4]
            top_boxes = results[0][:, :4]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (
                predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))

        f.close()
        return


if __name__ == '__main__':
    yolo = YOLO()
