import os
import time
import matplotlib
import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
from matplotlib import pyplot as plt
from natsort import natsorted
from scipy.optimize import linear_sum_assignment
from utils.utils_for_focusing.yoloV8.yolo import YOLO
from PIL import Image, ImageDraw, ImageFont

matplotlib.use("TkAgg")

def imshow_np(img, title=""):
    plt.figure()
    # 灰度或彩色自动识别
    if img.ndim == 2:
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    else:
        # 如果是 float32 且范围在 [0,1]，则传入 imshow 自动处理；若在 [0,255] 可先 /255
        if img.dtype == np.uint8:
            plt.imshow(img)
        else:
            plt.imshow(img.astype('float32') / 255.0)
    plt.title(title)
    plt.axis('off')   # 关闭坐标轴
    plt.show()


def linear_assignment(cost_matrix):
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return (o)


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])
        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        # self.kf.Q[4:, 4:] *= 0.01
        self.kf.Q[4:, 4:] *= 0.001
        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox, **kwargs):
        """
        Updates the state vector with observed bbox.
        """
        # 如果在brent优化，不更新frame_count和trk.hit_streak
        if kwargs['in_brent'] is True:
            self.time_since_update = 0
            pass
        else:
            self.time_since_update = 0
            self.history = []
            self.hits += 1
            self.hit_streak += 1
            self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if (self.kf.x[6] + self.kf.x[2] <= 0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


def center_distance_batch(detections, trackers, max_distance=20):
    """
    计算检测框与跟踪框中心点距离矩阵
    detections: [N,4] (x1,y1,x2,y2)
    trackers: [M,4] (x1,y1,x2,y2)
    max_distance: 最大允许匹配距离(像素)
    """
    # 计算中心点
    det_centers = np.array([[(d[0] + d[2]) / 2, (d[1] + d[3]) / 2] for d in detections])
    trk_centers = np.array([[(t[0] + t[2]) / 2, (t[1] + t[3]) / 2] for t in trackers])

    # 计算欧氏距离矩阵
    dist_matrix = np.sqrt(np.sum((det_centers[:, np.newaxis] - trk_centers) ** 2, axis=2))

    # 将超过阈值的距离设为无穷大
    dist_matrix[dist_matrix > max_distance] = np.inf
    return dist_matrix


def associate_detections_to_tracks(detections, trackers, iou_threshold=0.3, dist_threshold=30):
    """
    双重条件匹配：同时满足IOU > threshold且中心距离 < dist_threshold
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    # 同时计算IOU和中心距离矩阵
    iou_matrix = iou_batch(detections, trackers)
    dist_matrix = center_distance_batch(detections, trackers, max_distance=dist_threshold)

    # 创建复合条件矩阵（同时满足IOU和距离条件）
    combined_matrix = np.zeros_like(iou_matrix)
    valid_mask = (iou_matrix > iou_threshold) & (dist_matrix < dist_threshold)
    combined_matrix[valid_mask] = -(iou_matrix[valid_mask] + (1 - dist_matrix[valid_mask] / dist_threshold))  # 组合权重

    # 匈牙利算法匹配（最大化综合得分）
    if np.all(combined_matrix == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    matched_indices = linear_assignment(combined_matrix)

    # 筛选有效匹配
    matches = []
    for m in matched_indices:
        if valid_mask[m[0], m[1]]:
            matches.append([m[0], m[1]])

    # 处理未匹配项
    unmatched_detections = list(set(range(len(detections))) - set([m[0] for m in matches]))
    unmatched_trackers = list(set(range(len(trackers))) - set([m[1] for m in matches]))

    return np.array(matches), np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3, distance_threshold=30):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.distance_threshold = distance_threshold
        self.trackers = []
        self.frame_count = 0

        # ID映射
        self.id_map = {}  # 内部ID到显示ID的映射
        self.next_display_id = 1  # 下一个可分配的显示ID

    def update(self, bbox_xywh, confidences_list, **kwargs):
        """
        更新函数，适配新的输入格式 bbox_xywh 和 confidences_list
        """

        # 如果在brent优化，不更新frame_count和trk.hit_streak
        in_brent = kwargs['in_brent']
        if in_brent is True:
            pass
        else:
            self.frame_count += 1

        # 将 bbox_xywh 和 confidences_list 转换为 dets 格式
        if len(bbox_xywh) == 0 or len(confidences_list) == 0:
            dets = np.empty((0, 5))
        else:
            # 转换 bbox_xywh 到 [x1, y1, x2, y2] 格式
            x1 = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2
            y1 = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2
            x2 = bbox_xywh[:, 0] + bbox_xywh[:, 2] / 2
            y2 = bbox_xywh[:, 1] + bbox_xywh[:, 3] / 2
            dets = np.column_stack((x1, y1, x2, y2, confidences_list))

        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]

        # numpy.ma.masked_invalid屏蔽出现无效值的数组（NaN或inf；numpy.ma.compress_rows压缩包含掩码值的2-D 数组的整行。
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        matched, unmatched_dets, unmatched_trks = associate_detections_to_tracks(dets, trks,
                                                                                 iou_threshold=self.iou_threshold,
                                                                                 dist_threshold=self.distance_threshold)

        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :], in_brent=in_brent)

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)
        i = len(self.trackers)

        # 自后向前遍历，仅返回在当前帧出现且命中周期大于self.min_hits（除非跟踪刚开始）的跟踪结果；如果未命中时间大于self.max_age则删除跟踪器。
        ret = []
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):    # todo 做了修改，移除了前几帧直接输出颗粒的特性
            # if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= 1):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            i -= 1
            if (trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if len(ret) > 0:
            ret = np.concatenate(ret)
        else:
            ret = np.empty((0, 5))

        # trk.id可能会跳变，需要重定向以便于观察
        # 处理输出结果的ID映射
        final_outputs = []
        if len(ret) > 0:
            for output in ret:
                real_id = int(output[4])

                # 为新出现的ID分配显示ID
                if real_id not in self.id_map:
                    self.id_map[real_id] = self.next_display_id
                    self.next_display_id += 1

                # 替换为显示ID
                output[4] = self.id_map[real_id]
                final_outputs.append(output)
            ret = np.array(final_outputs)

        return ret


class Kalman_tracker:
    def __init__(self, image_folder, output_folder, yolo_model, sort, image_stack=None, device='cuda'):
        # 初始化YOLOv8检测模型
        self.yolo_model = yolo_model
        self.sort = sort

        # 图像处理参数
        self.image_folder = image_folder
        self.output_folder = output_folder
        if self.output_folder is not None:
            os.makedirs(self.output_folder, exist_ok=True)
        self.image_stack = image_stack

        # 轨迹记录字典（保存每个ID的历史位置）
        self.trajectories = {}

        self.device = device

    def safe_crop(self, image, left, top, right, bottom):
        """
        安全裁剪图像，确保裁剪区域不会超出图像边界。

        :param image: PIL.Image 对象
        :param left: 裁剪区域的左边界
        :param top: 裁剪区域的上边界
        :param right: 裁剪区域的右边界
        :param bottom: 裁剪区域的下边界
        :return: 裁剪后的图像
        """
        # 获取图像的宽度和高度
        width, height = image.size

        # 调整裁剪区域，确保不超出图像边界
        left = max(0, left)
        top = max(0, top)
        right = min(width, right)
        bottom = min(height, bottom)

        # 如果裁剪区域无效（宽度或高度为0），返回原始图像
        if right <= left or bottom <= top:
            return image

        # 裁剪图像
        cropped_image = image.crop((left, top, right, bottom))
        return cropped_image

    def safe_crop_np(self, image, left, top, right, bottom):
        """
        对 float32 格式的 NumPy 图像安全裁剪，越界部分自动裁掉。

        :param img: 输入图像，NumPy ndarray，dtype 应为 float32，形状 (H, W) 或 (H, W, C)
        :param left: 裁剪框左边界 x 坐标
        :param top: 裁剪框上边界 y 坐标
        :param right: 裁剪框右边界 x 坐标
        :param bottom: 裁剪框下边界 y 坐标
        :return: 裁剪后图像，dtype 仍为 float32
        """
        h, w = image.shape[:2]

        # 限定在 [0, w]×[0, h] 之内
        left = max(0, left)
        top = max(0, top)
        right = min(w, right)
        bottom = min(h, bottom)

        # 若无效区域，直接返回原图
        if right <= left or bottom <= top:
            return image

        # 裁剪并返回
        # 支持灰度 (H,W) 和多通道 (H,W,C)
        return image[top:bottom, left:right, ...]

    def get_stacks(self, get_position=False):
        stack_lst = []  # stack_lst为需要预测处理的图像，stack_ori_lst为展示的原图
        stack_ori_lst = []
        position_lst = []  # 用于存储颗粒的位置信息
        particle_info = {}  # dict, 存储当前颗粒信息
        particle_ori_info = {}  # dict, 记录颗粒位置的极值, 确保所有截面的颗粒都被完整记录
        image_files = None

        # 两种输入情况，一种是输入文件夹路径，另一种是直接输入stack（w, h, num)，格式为数组，需要转为Image
        if self.image_folder is not None:
            # 获取排序后的图像文件列表
            image_files = sorted(
                [f for f in os.listdir(self.image_folder) if f.endswith(('.jpg', '.png'))],
                key=lambda x: float(x.split('_')[1].rsplit('.', 1)[0])
            )  # 提取文件名中的序号部分
            image_num = len(image_files)
        else:
            image_num = self.image_stack.shape[-1]

        for idx in range(image_num):
            if image_files is not None:
                print('Start [{}/{}]'.format(idx + 1, image_num))
                img_path = os.path.join(self.image_folder, image_files[idx])
                frame = Image.open(img_path).convert("RGB")
            else:
                frame = self.image_stack[:, :, idx]  # 提取第 idx 个图像数据

            # st1 = time.time()
            # YOLOv8检测
            # try:
                result_img, results = self.yolo_model.detect_image_np(frame, crop=False, count=False, draw=False, device=self.device)
                # result_img_pil = Image.fromarray(frame, mode='L')
                # imshow_np(frame, '检测结果')
            # except:
            #     results = ([], [], [])
            # ed1 = time.time()
            # print('get_stacks', ed1 - st1)

            boxes_xywh, confidences, labels = results

            bbox_xywh = []
            confidences_list = []
            for box, conf in zip(boxes_xywh, confidences):
                bbox_xywh.append(box)
                confidences_list.append(conf)

            # SORT跟踪，花费时间很短，不考虑优化
            if len(bbox_xywh) > 0:
                outputs = self.sort.update(np.array(bbox_xywh), np.array(confidences_list), in_brent=False)
                outputs = outputs.astype(np.int64)

                # 更新轨迹记录
                for output in outputs:
                    y1, x1, y2, x2, track_id = output

                    # 边框延长，因为不加边框会导致边框也被检测为edge且无法去除
                    pad = 50
                    # pad = 500
                    # 绘制当前检测框
                    left = max(0, min(x1, x2))
                    right = max(x1, x2)
                    top = max(0, min(y1, y2))
                    bottom = max(y1, y2)

                    left_padded = left - pad
                    right_padded = right + pad
                    top_padded = top - pad
                    bottom_padded = bottom + pad

                    if track_id not in self.trajectories:
                        particle_info[track_id] = [left_padded, right_padded, top_padded, bottom_padded]  # dict
                        particle_ori_info[track_id] = [left, right, top, bottom]  # dict note：注释后即显示pad后的单颗粒图
                        # particle_ori_info[track_id] = [left_padded, right_padded, top_padded, bottom_padded]  # dict note：注释后即显示pad后的单颗粒图
                        self.trajectories[track_id] = []
                    else:
                        particle_info_tmp = particle_ori_info[track_id]  # dict, 单颗粒信息
                        # particle_info_tmp = particle_info[track_id-1]
                        # 需要显示的图就不加pad了
                        left_ori = max(0, min(left, particle_info_tmp[0]))
                        right_ori = max(right, particle_info_tmp[1])
                        top_ori = max(0, min(top, particle_info_tmp[2]))
                        bottom_ori = max(bottom, particle_info_tmp[3])

                        particle_info[track_id] = [left_padded, right_padded, top_padded, bottom_padded]
                        particle_ori_info[track_id] = [left_ori, right_ori, top_ori, bottom_ori]

        # for info in particle_info:
        for i, info in particle_info.items():  # dict
            stack = []
            stack_ori = []
            # try:  # 可能这张图中没有此颗粒
            if self.image_folder is not None:
                for idx, img_file in enumerate(image_files):
                    img_path = os.path.join(self.image_folder, img_file)
                    frame = Image.open(img_path).convert("RGB")
                    left_padded, right_padded, top_padded, bottom_padded = info
                    left_ori, right_ori, top_ori, bottom_ori = particle_ori_info[i]
                    # stack.append(self.safe_crop_np(frame, left_padded, top_padded, right_padded, bottom_padded))
                    stack.append(self.safe_crop(frame, left_padded, top_padded, right_padded, bottom_padded))
                    # stack_ori.append(self.safe_crop_np(frame, left_ori, top_ori, right_ori, bottom_ori))
                    stack_ori.append(self.safe_crop(frame, left_ori, top_ori, right_ori, bottom_ori))

                stack_lst.append(stack)
                stack_ori_lst.append(stack_ori)
            else:
                # print(self.image_stack.shape[-1])
                for idx in range(self.image_stack.shape[-1]):
                    # print('idx', idx)
                    frame = self.image_stack[:, :, idx]  # 提取第 idx 个图像数据

                    # # 转换为PIL Image对象，并强制转为RGB模式
                    # frame = Image.fromarray(frame, mode='L')
                    # frame = frame.astype(np.float32)  # note float32是rcf的格式，不是在前面转换就是在后面转换，但单独列出来会有耗时; 而yolo需要的格式是uint8

                    left_padded, right_padded, top_padded, bottom_padded = info
                    left_ori, right_ori, top_ori, bottom_ori = particle_ori_info[i]
                    stack.append(self.safe_crop_np(frame, left_padded, top_padded, right_padded, bottom_padded))
                    # stack.append(self.safe_crop(frame, left_padded, top_padded, right_padded, bottom_padded))
                    stack_ori.append(self.safe_crop_np(frame, left_ori, top_ori, right_ori, bottom_ori))
                    # stack_ori.append(self.safe_crop(frame, left_ori, top_ori, right_ori, bottom_ori))

                stack_lst.append(stack)
                stack_ori_lst.append(stack_ori)
                position_lst.append((left_ori, top_ori, right_ori, bottom_ori))

            # except Exception as e:
            #     print('Error,', e)

        if get_position:
            return stack_lst, stack_ori_lst, image_files, position_lst
        else:
            return stack_lst, stack_ori_lst, image_files

    def process_images(self):
        # 获取排序后的图像文件列表
        image_files = sorted([f for f in os.listdir(self.image_folder) if f.endswith(('.jpg', '.png'))],
                             key=lambda x: int(x.split('_')[1]))  # 提取文件名中的序号部分

        for idx, img_file in enumerate(image_files):
            img_path = os.path.join(self.image_folder, img_file)
            frame = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(frame)

            # YOLOv8检测
            try:
                result_img, results = self.yolo_model.detect_image(frame, crop=False, count=False, draw=False)
                boxes_xywh, confidences, labels = results
                # print("boxes: ", boxes_xywh)  # y_c, x_c, height, width
                # result_img.show()

                bbox_xywh = []
                confidences_list = []
                for box, conf in zip(boxes_xywh, confidences):
                    bbox_xywh.append(box)
                    confidences_list.append(conf)

                # SORT跟踪
                if len(bbox_xywh) > 0:
                    outputs = self.sort.update(np.array(bbox_xywh), np.array(confidences_list), in_brent=False)
                    outputs = outputs.astype(np.int64)

                    # 更新轨迹记录
                    for output in outputs:
                        y1, x1, y2, x2, track_id = output
                        center = ((x1 + x2) // 2, (y1 + y2) // 2)

                        if track_id not in self.trajectories:
                            self.trajectories[track_id] = []
                        self.trajectories[track_id].append(center)

                        # 绘制当前检测框
                        left = max(0, min(x1, x2))
                        right = max(x1, x2)
                        top = max(0, min(y1, y2))
                        bottom = max(y1, y2)

                        draw.rectangle([left, top, right, bottom], outline="yellow", width=10)
                        font = ImageFont.truetype("arial.ttf", size=100)  # 设置字体大小为 40
                        draw.text((left, bottom - 100), f"ID:{track_id}", fill="yellow", font=font)

                        # 绘制历史轨迹（最近20个位置）
                        if len(self.trajectories[track_id]) > 1:
                            trajectory = self.trajectories[track_id][-20:]
                            for i in range(1, len(trajectory)):
                                draw.line([trajectory[i - 1], trajectory[i]], fill="red", width=2)

                # 保存结果图像
                output_path = os.path.join(self.output_folder, f"track_{img_file}")
                frame.save(output_path)
                print(f"Processed frame {idx + 1}/{len(image_files)}")

            except:
                result_img = self.yolo_model.detect_image(frame, crop=False, count=False, draw=False)

        print(f"Processing complete. Results saved to: {self.output_folder}")


if __name__ == '__main__':
    mot_tracker = Sort(max_age=9, min_hits=5, iou_threshold=0.3, distance_threshold=100)  # create instance of the SORT tracker
    yolo_model = YOLO(input_shape=[640, 640],
                      phi='s',
                      model_path=r'F:\dongjiayao\Pycharm\Holo-Track\yoloV8\logs\result_1\best_epoch_weights.pth',
                      cuda=True,
                      letterbox_image=True,
                      confidence=0.5,
                      nms_iou=0.3, )

    input_folder = r"F:\dongjiayao\Pycharm\Holo-Track\img\4\input"
    output_folder = r"F:\dongjiayao\Pycharm\Holo-Track\img\4\trace"
    tracker = Kalman_tracker(image_folder=input_folder, output_folder=output_folder, yolo_model=yolo_model,
                             sort=mot_tracker)
    tracker.process_images()
