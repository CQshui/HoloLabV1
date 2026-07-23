"""
YOLO+SORT 颗粒检测与追踪可视化模块。

对重建振幅平面运行 YOLO 检测 + SORT (卡尔曼) 追踪，
生成带标注框、轨迹的可视化图像，供 UI 预览颗粒识别与追踪效果。
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple

from utils.utils_for_focusing.sort.kalman_trace_iou_center import Sort
from utils.utils_for_focusing.yoloV8.yolo import YOLO


# 颜色池: 不同 track_id 分配不同颜色 (BGR)
_TRACK_COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (0, 255, 255), (255, 0, 255), (128, 255, 0), (255, 128, 0),
    (0, 128, 255), (128, 0, 255), (255, 255, 255), (0, 128, 128),
    (128, 128, 0), (128, 0, 128), (128, 128, 255), (255, 128, 128),
]


def _get_color(track_id: int) -> tuple:
    return _TRACK_COLORS[track_id % len(_TRACK_COLORS)]


def run_yolo_sort_preview(
    plane_images: Dict[float, np.ndarray],
    yolo_model: YOLO,
    device,
    confidence: float = 0.3,
    max_planes_per_grid: int = 12,
) -> Dict:
    """
    对重建振幅平面运行 YOLO+SORT，生成可视化。

    Parameters
    ----------
    plane_images : {z: image}
        重建平面振幅图 (uint8 或 float)
    yolo_model : YOLO
        已加载的 YOLO 模型实例
    device : torch.device
        推理设备
    confidence : float
        YOLO 检测置信度阈值（覆盖模型默认值）
    max_planes_per_grid : int
        网格视图最多展示的平面数

    Returns
    -------
    dict
        'grid_image'       : 多平面检测网格图 (BGR uint8)，可直接显示
        'trajectory_image' : 颗粒轨迹汇总图 (BGR uint8)
        'per_plane'        : [(z, annotated_image)] 每个平面的标注图
        'particles'        : {track_id: {'bboxes': {z: (x1,y1,x2,y2)}, 'centers': [(z,cx,cy)]}}
        'z_list'           : 排序后的 z 列表
        'total_particles'  : 追踪到的颗粒总数
    """
    z_list = sorted(plane_images.keys())
    if not z_list:
        raise ValueError("plane_images 为空")

    # 保存原始置信度，结束后恢复
    original_conf = yolo_model.confidence
    yolo_model.confidence = confidence

    try:
        # 1. 准备图像列表 (uint8 BGR)
        image_stack_list = []
        for z in z_list:
            img_norm = _to_uint8(plane_images[z])
            if img_norm.ndim == 2:
                img_norm = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2BGR)
            image_stack_list.append(img_norm)

        # 2. 逐平面运行 YOLO+SORT，收集各帧检测与轨迹历史
        per_plane_detections, track_id_to_bbox_history = _collect_per_plane_detections(
            yolo_model, device, plane_images, z_list
        )

        # 3. 生成网格图
        grid_image = _build_detection_grid(
            image_stack_list, z_list, per_plane_detections, max_planes_per_grid
        )

        # 4. 生成轨迹汇总图
        trajectory_image = _build_trajectory_summary(
            image_stack_list[0], z_list, track_id_to_bbox_history
        )

        # 5. 整理颗粒信息
        particles_info = _build_particles_info(track_id_to_bbox_history)

    finally:
        yolo_model.confidence = original_conf

    return {
        'grid_image': grid_image,
        'trajectory_image': trajectory_image,
        'per_plane': [(z_list[i], _draw_boxes_on_image(
            image_stack_list[i].copy(), per_plane_detections.get(i, [])
        )) for i in range(len(z_list))],
        'particles': particles_info,
        'z_list': z_list,
        'total_particles': len(particles_info),
    }


def _to_uint8(img: np.ndarray) -> np.ndarray:
    """将任意图像归一化到 uint8 [0,255]"""
    if img.dtype == np.uint8:
        return img.copy()
    abs_v = np.abs(img)
    if abs_v.max() > abs_v.min():
        norm = (abs_v - abs_v.min()) / (abs_v.max() - abs_v.min())
    else:
        norm = abs_v
    return (norm * 255).astype(np.uint8)


def _collect_per_plane_detections(
    yolo_model, device, plane_images, z_list
) -> Tuple[Dict, Dict]:
    """
    逐平面运行 YOLO+SORT，返回每帧的检测框和按 track_id 的历史。
    YOLO 检测 → SORT 追踪 → 收集逐帧/逐颗粒数据。
    """
    per_plane = {}           # plane_idx -> [(x1,y1,x2,y2,track_id,conf), ...]
    track_history = {}       # track_id -> list of (z, x1,y1,x2,y2)

    mot_tracker = Sort(max_age=15, min_hits=3, iou_threshold=0.3, distance_threshold=30)

    for idx, z in enumerate(z_list):
        img = _to_uint8(plane_images[z])
        _, results, _ = yolo_model.detect_image_np(
            img, crop=False, count=False, draw=False, device=device
        )
        boxes_xywh, confidences, labels = results

        if len(boxes_xywh) > 0:
            outputs = mot_tracker.update(
                np.array(boxes_xywh), np.array(confidences), in_brent=False
            )
            outputs = outputs.astype(np.int64)

            frame_dets = []
            for det in outputs:
                y1, x1, y2, x2, track_id = det
                conf = 1.0  # SORT 不传 confidence（近似）
                frame_dets.append((x1, y1, x2, y2, int(track_id), float(conf)))
                track_history.setdefault(int(track_id), []).append((z, x1, y1, x2, y2))
            per_plane[idx] = frame_dets
        else:
            per_plane[idx] = []

    return per_plane, track_history


def _draw_boxes_on_image(img: np.ndarray, detections: List[Tuple]) -> np.ndarray:
    """在图像上绘制检测框和 track ID，线宽/字号按图像尺寸自适应缩放。
    detections: [(x1,y1,x2,y2,track_id,conf),...]"""
    img_out = img.copy()
    h, w = img_out.shape[:2]
    scale = max(1.0, max(h, w) / 800.0)  # 以 800px 为基准自适应

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6 * scale
    font_thickness = max(1, int(1.5 * scale))
    box_thickness = max(2, int(3 * scale))

    for det in detections:
        x1, y1, x2, y2, track_id, conf = det
        color = _get_color(track_id)
        cv2.rectangle(img_out, (x1, y1), (x2, y2), color, box_thickness)
        label = f"ID:{track_id}"
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        pad = max(4, int(4 * scale))
        cv2.rectangle(img_out, (x1, y1 - th - baseline - pad), (x1 + tw + pad, y1 + pad), color, -1)
        cv2.putText(img_out, label, (x1 + pad // 2, y1 - baseline),
                    font, font_scale, (255, 255, 255), font_thickness)
    return img_out


def _build_detection_grid(
    image_list: List[np.ndarray], z_list: List[float],
    per_plane_detections: Dict, max_planes: int
) -> np.ndarray:
    """
    构建 N×N 检测网格图：均匀选取 max_planes 个平面，画上检测框。
    """
    n_planes = len(z_list)
    if n_planes <= max_planes:
        selected_indices = list(range(n_planes))
    else:
        step = max(1, n_planes // max_planes)
        selected_indices = list(range(0, n_planes, step))[:max_planes]

    n = len(selected_indices)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols

    cell_h, cell_w = image_list[0].shape[:2]
    cell_h = min(cell_h, 300)
    cell_w = min(cell_w, 300)

    grid_h = rows * cell_h + (rows + 1) * 4
    grid_w = cols * cell_w + (cols + 1) * 4
    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 240

    for gi, idx in enumerate(selected_indices):
        r, c = gi // cols, gi % cols
        y0 = (r + 1) * 4 + r * cell_h
        x0 = (c + 1) * 4 + c * cell_w

        img = image_list[idx]
        img = cv2.resize(img, (cell_w, cell_h))
        img = _draw_boxes_on_image(img, per_plane_detections.get(idx, []))

        # 写 z 标签
        cv2.putText(img, f"z={z_list[idx]*1e6:.1f}um", (4, cell_h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        grid[y0:y0 + cell_h, x0:x0 + cell_w] = img

    return grid


def _build_trajectory_summary(
    reference_img: np.ndarray, z_list: List[float],
    track_history: Dict[int, List[Tuple]]
) -> np.ndarray:
    """
    生成轨迹汇总图：一张叠加图画出所有颗粒的中心点轨迹。
    线宽/字号按图像尺寸自适应缩放。
    """
    h, w = reference_img.shape[:2]
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 30  # 深色背景
    scale = max(1.0, max(h, w) / 800.0)  # 以 800px 为基准自适应

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5 * scale
    font_thickness = max(1, int(1.5 * scale))
    line_thickness = max(2, int(2.5 * scale))
    circle_radius = max(4, int(5 * scale))

    for track_id, entries in track_history.items():
        color = _get_color(track_id)
        if len(entries) < 2:
            # 单帧: 画一个圆圈
            _, x1, y1, x2, y2 = entries[0]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(canvas, (cx, cy), circle_radius, color, line_thickness)
            cv2.putText(canvas, f"ID:{track_id}", (cx + circle_radius + 2, cy),
                        font, font_scale, color, font_thickness)
            continue

        # 多帧: 画轨迹线
        for j in range(len(entries) - 1):
            _, x1_a, y1_a, x2_a, y2_a = entries[j]
            _, x1_b, y1_b, x2_b, y2_b = entries[j + 1]
            ca = ((x1_a + x2_a) // 2, (y1_a + y2_a) // 2)
            cb = ((x1_b + x2_b) // 2, (y1_b + y2_b) // 2)
            cv2.line(canvas, ca, cb, color, line_thickness)

        # ID 标注在最后一个位置
        _, x1, y1, x2, y2 = entries[-1]
        last_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        cv2.putText(canvas, f"ID:{track_id}",
                    (last_center[0] + 6, last_center[1] - 6),
                    font, font_scale, color, font_thickness)

    # 标题
    cv2.putText(canvas, f"Trajectories ({len(z_list)} planes, {len(track_history)} particles)",
                (10, h - 10), font, font_scale * 1.2, (200, 200, 200), font_thickness)

    return canvas


def _build_particles_info(track_history: Dict) -> Dict:
    """整理颗粒信息为结构化 dict。"""
    info = {}
    for track_id, entries in track_history.items():
        bboxes = {}
        centers = []
        for z, x1, y1, x2, y2 in entries:
            bboxes[z] = (x1, y1, x2, y2)
            centers.append((z, (x1 + x2) // 2, (y1 + y2) // 2))
        info[track_id] = {
            'bboxes': bboxes,
            'centers': centers,
            'num_planes_detected': len(entries),
        }
    return info


if __name__ == '__main__':
    print('YOLO+SORT Preview Module')
