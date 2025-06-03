import os
import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
from natsort import natsorted
from scipy.optimize import linear_sum_assignment
from yoloV8.yolo import YOLO
from PIL import Image, ImageDraw, ImageFont


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

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
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


def associate_detections_to_tracks(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if (len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    # 计算两两间的交并比，调用linear_assignment进行匹配
    iou_matrix = iou_batch(detections, trackers)
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    # 记录未匹配的检测框及轨迹
    unmatched_detections = []
    for d, det in enumerate(detections):
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # 过滤掉IoU低的匹配
    matches = []
    for m in matched_indices:
        if (iou_matrix[m[0], m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, bbox_xywh, confidences_list, **kwargs):
        """
        更新函数，适配新的输入格式 bbox_xywh 和 confidences_list
        """
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
        matched, unmatched_dets, unmatched_trks = associate_detections_to_tracks(dets, trks, self.iou_threshold)

        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)
        i = len(self.trackers)

        # 自后向前遍历，仅返回在当前帧出现且命中周期大于self.min_hits（除非跟踪刚开始）的跟踪结果；如果未命中时间大于self.max_age则删除跟踪器。
        ret = []
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            i -= 1
            if (trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if len(ret) > 0:
            ret = np.concatenate(ret)
        else:
            ret = np.empty((0, 5))

        return ret


class Kalman_tracker:
    def __init__(self, image_folder, output_folder, yolo_model, sort):
        # 初始化YOLOv8检测模型
        self.yolo_model = yolo_model
        self.sort = sort

        # 图像处理参数
        self.image_folder = image_folder
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)

        # 轨迹记录字典（保存每个ID的历史位置）
        self.trajectories = {}

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

    def get_stacks(self):
        # 获取排序后的图像文件列表
        image_files = natsorted(
            [f for f in os.listdir(self.image_folder) if f.endswith(('.jpg', '.png'))]
        )
        stack_lst = []  # stack_lst为需要预测处理的图像，stack_ori_lst为展示的原图
        stack_ori_lst = []
        particle_info = {}  # dict
        particle_ori_info = {}  # dict

        for idx, img_file in enumerate(image_files):
            print('Start [{}/{}]'.format(idx + 1, len(image_files)))
            img_path = os.path.join(self.image_folder, img_file)
            frame = Image.open(img_path).convert("RGB")

            # YOLOv8检测
            result_img, results = self.yolo_model.detect_image(frame, crop=False, count=False, draw=False)
            boxes_xywh, confidences, labels = results

            bbox_xywh = []
            confidences_list = []
            for box, conf in zip(boxes_xywh, confidences):
                bbox_xywh.append(box)
                confidences_list.append(conf)

            # SORT跟踪
            if len(bbox_xywh) > 0:
                outputs = self.sort.update(np.array(bbox_xywh), np.array(confidences_list))
                outputs = outputs.astype(np.int64)

                # 更新轨迹记录
                for output in outputs:
                    y1, x1, y2, x2, track_id = output

                    # 边框延长，因为不加边框会导致边框也被检测为edge且无法去除
                    pad = 200
                    # 绘制当前检测框
                    left = max(0, min(x1, x2)) - pad
                    right = max(x1, x2) + pad
                    top = max(0, min(y1, y2)) - pad
                    bottom = max(y1, y2) + pad

                    if track_id not in self.trajectories:
                        particle_info[track_id] = [left, right, top, bottom]  # dict
                        self.trajectories[track_id] = []
                    else:
                        particle_info_tmp = particle_info[track_id]  # dict
                        # particle_info_tmp = particle_info[track_id-1]
                        left_ori = max(0, min(left, particle_info_tmp[0]))
                        right_ori = max(right, particle_info_tmp[1])
                        top_ori = max(0, min(top, particle_info_tmp[2]))
                        bottom_ori = max(bottom, particle_info_tmp[3])

                        particle_info[track_id] = [left, right, top, bottom]
                        particle_ori_info[track_id] = [left_ori, right_ori, top_ori, bottom_ori]

        # for info in particle_info:
        for i, info in particle_info.items():  # dict
            stack = []
            stack_ori = []
            for idx, img_file in enumerate(image_files):
                img_path = os.path.join(self.image_folder, img_file)
                frame = Image.open(img_path).convert("RGB")
                left, right, top, bottom = info
                left_ori, right_ori, top_ori, bottom_ori = particle_ori_info[i]
                stack.append(self.safe_crop(frame, left, top, right, bottom))
                stack_ori.append(self.safe_crop(frame, left_ori, top_ori, right_ori, bottom_ori))

            stack_lst.append(stack)
            stack_ori_lst.append(stack_ori)

        return stack_lst, stack_ori_lst, image_files

    def process_images(self):
        # 获取排序后的图像文件列表
        image_files = natsorted(
            [f for f in os.listdir(self.image_folder) if f.endswith(('.jpg', '.png'))]
        )

        for idx, img_file in enumerate(image_files):
            img_path = os.path.join(self.image_folder, img_file)
            frame = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(frame)

            # YOLOv8检测
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
                outputs = self.sort.update(np.array(bbox_xywh), np.array(confidences_list))
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
                    draw.text((left, bottom-100), f"ID:{track_id}", fill="yellow", font=font)

                    # 绘制历史轨迹（最近20个位置）
                    if len(self.trajectories[track_id]) > 1:
                        trajectory = self.trajectories[track_id][-20:]
                        for i in range(1, len(trajectory)):
                            draw.line([trajectory[i - 1], trajectory[i]], fill="red", width=2)

            # 保存结果图像
            output_path = os.path.join(self.output_folder, f"track_{img_file}")
            frame.save(output_path)
            print(f"Processed frame {idx + 1}/{len(image_files)}")

        print(f"Processing complete. Results saved to: {self.output_folder}")


if __name__ == '__main__':
    mot_tracker = Sort(max_age=15, min_hits=3, iou_threshold=0.3)  # create instance of the SORT tracker
    yolo_model = YOLO(input_shape=[640, 640],
                      phi='s',
                      model_path=r'F:\dongjiayao\Pycharm\Holo-Track\yoloV8\logs\result_1\best_epoch_weights.pth',
                      cuda=True,
                      letterbox_image=True,
                      confidence=0.5,
                      nms_iou=0.3, )

    input_folder = r"F:\dongjiayao\Pycharm\Holo-Track\img\3\input3"
    output_folder = r"F:\dongjiayao\Pycharm\Holo-Track\img\3\trace3"
    tracker = Kalman_tracker(image_folder=input_folder, output_folder=output_folder, yolo_model=yolo_model,
                             sort=mot_tracker)
    tracker.process_images()
