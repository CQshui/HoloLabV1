import sys
import time
from pathlib import Path

# 将项目根目录添加到 sys.path
root_dir = Path(__file__).parent.parent.parent  # 根据实际情况调整层级
# print(root_dir)
sys.path.append(str(root_dir))

import cv2
import numpy as np
import torch

from PIL import Image
from torch.utils import data
import os

from torch.utils.data import DataLoader

from utils.utils_for_focusing.rcf.rcf_predict import calculate_brightness_concentration

# from utils.utils_for_focusing.sort.kalman_trace import Kalman_tracker, Sort
from utils.utils_for_focusing.sort.kalman_trace_iou_center import Kalman_tracker, Sort
from utils.utils_for_focusing.yoloV8.yolo import YOLO
from utils.utils_for_focusing.rcf.data_loader import convert_to_rgb, prepare_image_PIL
from utils.utils_for_focusing.rcf.models.RCF import RCF


def zero_edge_pixels(image, edge_size=5):
    """
    将图像边缘的指定像素归零。
    :param image: 输入的 PIL 图像对象
    :param edge_size: 边缘像素的大小，默认为 5
    :return: 处理后的 PIL 图像对象
    """
    # 获取图像的宽度和高度
    width, height = image.size

    # 创建一个与原图像相同大小的黑色图像
    black_image = Image.new('RGB', (width, height), (0, 0, 0))

    # 计算裁剪区域
    left = edge_size
    top = edge_size
    right = width - edge_size
    bottom = height - edge_size

    # 裁剪原图像的内部区域
    cropped_image = image.crop((left, top, right, bottom))

    # 将裁剪后的图像粘贴到黑色图像的对应位置
    black_image.paste(cropped_image, (left, top))

    return black_image


class RCFLoader(data.Dataset):
    def __init__(self, input_img_stack, origin_img_stack, k_size=8):
        self.input_img_stack = input_img_stack
        self.origin_img_stack = origin_img_stack
        self.k_size = k_size

    def __len__(self):
        return len(self.input_img_stack)

    def ranging_resize(self, input_img, ranging):
        # 输入为np.array, float32
        pass

    # input_img_stack的输入目前是Image格式，最好也改成torch.tensor
    def __getitem__(self, index):
        img = self.input_img_stack[index]
        # img = np.array(self.stack[index], dtype=np.float32)
        img_ori = self.origin_img_stack[index]
        # img_ori = np.array(self.stack_ori[index], dtype=np.float32)

        img = convert_to_rgb(img)  # 确保图像是RGB格式
        img = cv2.resize(img, (img.shape[1] // self.k_size, img.shape[0] // self.k_size),
                         interpolation=cv2.INTER_NEAREST)
        # img = cv2.resize(img, (224, 224),
        #                  interpolation=cv2.INTER_NEAREST)
        img = prepare_image_PIL(img)

        # 转 tensor: CxHxW, 归一化到 [0,1]
        img = torch.from_numpy(img).float()
        # 原始图直接转 tensor
        img_ori = torch.from_numpy(img_ori).float()

        return img, img_ori


def rcf_predict(model, predict_loader, save_dir='', device='cuda'):
    model.eval()

    if save_dir:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

    # 存储文件名和对应的熵值
    concentrations = []
    max_concentration = -np.inf
    best_focus_image = None
    best_focus_name = None
    best_index = None

    # for idx, (image, image_origin) in enumerate(predict_loader):
    for idx in range(predict_loader.length):
        # if idx == 0:
        # st0 = time.time()
        image, image_origin = predict_loader.get_data(idx)      # note: 此处的predict_loader是focusing.py中的get_dataset类

        image = image.to(device)
        _, _, H, W = image.shape
        # ed0 = time.time()
        # print('get data', ed0 - st0)

        st0 = time.time()
        results = model(image)
        ed0 = time.time()
        print('RCF预测', ed0 - st0)
        result = torch.squeeze(results[-1].detach()).cpu().numpy()

        # else
        # st0 = time.time()
        result_pil = Image.fromarray((result * 255).astype(np.uint8))
        # result.show()

        # 边缘归黑 todo
        black_pad = 0
        result_pil = zero_edge_pixels(result_pil, edge_size=black_pad)

        # 计算亮度密度
        concentration = calculate_brightness_concentration(np.array(result_pil))
        concentrations.append(concentration)

        # # 边缘检测图像保存
        # tmp_pth = os.path.join(output_folder, str(time.time()))
        # os.mkdir(tmp_pth)
        # result.save(os.path.join(tmp_pth, "edge_{:3f}.jpg".format(concentration)))
        # img_ori_tmp = torch.squeeze(image_origin.squeeze()).cpu().numpy()
        # cv2.imwrite(os.path.join(tmp_pth, "image.jpg"), img_ori_tmp)

        # 更新最大 concentration 的图像
        if concentration > max_concentration:
            max_concentration = concentration
            best_focus_image = image_origin
            best_index = idx
        # ed0 = time.time()
        # print('else', ed0 - st0)
        # # debug用，查看原图及其对应的预测结果
        # img_tmp = torch.squeeze(image_origin.squeeze()).cpu().numpy()
        # result_tmp = (result * 255).astype(np.uint8)
        # result_tmp = cv2.resize(result_tmp, (img_tmp.shape[1], img_tmp.shape[0]))
        # # 水平拼接
        # combined = np.hstack((img_tmp, result_tmp))
        # # 保存结果
        # cv2.imwrite(os.path.join(r'F:\dongjiayao\Data\HoloLab_testData\autofocus\tmp1', '{}.png'.format(concentration)), combined)

        # print("Running test [%d/%d]" % (idx + 1, len(predict_loader)))
        # print("Running test [%d/%d]" % (idx + 1, predict_loader.length))

    return best_focus_image, best_index


def rcf_predict_V1(model, predict_loader, save_dir='', device='cuda'):
    torch.backends.cudnn.benchmark = False

    model.eval()
    concentrations = []
    best_focus_image = None
    best_focus_name = None
    best_index = None
    max_concentration = -np.inf

    # 批量处理所有图像
    for batch_idx, batch_data in enumerate(predict_loader):
        # 解包批数据（图像+原始图像）
        images, images_origin = batch_data
        images = images.to(device)

        st = time.time()

        # 批量预测
        with torch.no_grad():
            results = model(images)

        ed = time.time()
        # g = torch.cuda.CUDAGraph()
        # with torch.cuda.graph(g):
        #     results = model(images)

        # 处理批量结果
        for i in range(len(images)):
            result = torch.squeeze(results[-1][i]).cpu().numpy()
            result_pil = Image.fromarray((result * 255).astype(np.uint8))
            black_pad = 0
            result_pil = zero_edge_pixels(result_pil, edge_size=black_pad)

            concentration = calculate_brightness_concentration(np.array(result_pil))
            concentrations.append(concentration)

            # 边缘检测图像保存
            # tmp_pth = os.path.join(r'E:\Projects\HoloLabV1\tmp', str(time.time()))
            # os.mkdir(tmp_pth)
            # result_pil.save(os.path.join(tmp_pth, "edge_{:3f}.jpg".format(concentration)))
            # img_ori_tmp = torch.squeeze(images_origin[i].squeeze()).cpu().numpy()
            # cv2.imwrite(os.path.join(tmp_pth, "image.jpg"), img_ori_tmp)

            # 更新最佳聚焦图像
            if concentration > max_concentration:
                max_concentration = concentration
                # best_focus_image = images_origin[i].cpu().numpy()  # 使用原始图像
                best_focus_image = images_origin[i] # 使用原始图像
                best_index = i

        ed = time.time()

    return best_focus_image, best_index


def rcf_predict_V2(model,
                   predict_loader,
                   particle_sizes,
                   device='cuda'):
    r"""
    使用一次 forward 处理所有颗粒，并返回每颗粒最佳聚焦的 **原尺寸** 图像

    DataLoader 中一条样本由 pad_collate 组织为
        imgs_batch      : (B, 3, H_max, W_max)   ──> 进模型
        img_oris_batch  : (B, H_max, W_max)      ──> 只展示，不进模型
        orig_sizes      : [(H_img, W_img, H_ori, W_ori), ...]
    """
    model.eval()

    # -------------- 1. 收集所有 batch ----------------
    all_imgs      = []          # (N, 3, H_max, W_max) → concat 后送模型
    all_oris      = []          # [(H_ori,W_ori) 裁剪后 np.ndarray]
    img_hw_list   = []          # [(H_img, W_img)] 与 all_imgs 对齐


    for imgs_batch, img_oris_batch, orig_sizes in predict_loader:
        B = imgs_batch.size(0)

        # ① 收集 imgs → 后面一次 cat
        all_imgs.append(imgs_batch)            # 仍在 CPU，稍后一起 .to(device)

        # ② 把 img_ori 裁回原尺寸，存入 list
        for j in range(B):
            h_img, w_img, h_ori, w_ori = orig_sizes[j]

            # 裁剪 ori
            ori_np = (img_oris_batch[j, :h_ori, :w_ori]    # (H_ori,W_ori)
                      .cpu()
                      .numpy())

            all_oris.append(ori_np)
            img_hw_list.append((h_img, w_img))             # 与 ori 同 index

    # -------------- 2. 一次性 forward -----------------
    imgs_tensor = torch.cat(all_imgs, dim=0).to(device)    # (N,3,H_max,W_max)

    st = time.time()
    with torch.no_grad():
        results = model(imgs_tensor)                       # list/tuple
        last_feat = results[-1]                            # (N,1,H_max,W_max) 或类似
    batch_results = last_feat.squeeze(1).cpu().numpy()     # → (N,H_max,W_max)
    ed = time.time()
    # -------------- 3. 每颗粒选最佳聚焦 ----------------
    focused_images = []
    start_idx = 0

    for p_size in particle_sizes:
        end_idx = start_idx + p_size

        # 当前颗粒所有帧
        particle_results = batch_results[start_idx:end_idx]
        particle_oris    = all_oris[start_idx:end_idx]
        particle_hw      = img_hw_list[start_idx:end_idx]

        # 选“亮度集中度”最大的一帧
        best_focus_img   = None
        max_concentration = -np.inf

        for k in range(p_size):
            h_img, w_img = particle_hw[k]

            # 裁掉 padding 区域再评估
            cropped_res = particle_results[k][:h_img, :w_img]
            res_pil     = Image.fromarray((cropped_res * 255).astype(np.uint8))

            res_pil     = zero_edge_pixels(res_pil, edge_size=0)  # 你的函数
            conc        = calculate_brightness_concentration(
                             np.array(res_pil))                   # 你的函数

            if conc > max_concentration:
                max_concentration = conc
                best_focus_img    = particle_oris[k]               # 已是原尺寸

        focused_images.append(best_focus_img)
        start_idx = end_idx

    print("rcf_predict_V2 运行时长：", ed - st, " s")
    return focused_images

def main():
    yolo_model = YOLO(input_shape=[640, 640],
                      phi='s',
                      model_path=r'F:\dongjiayao\Pycharm\Holo-Track\yoloV8\logs\result_1\best_epoch_weights.pth',
                      cuda=True,
                      letterbox_image=True,
                      confidence=0.5,
                      nms_iou=0.3, )

    # input_folder = r"E:\DongJiayao\Data\DS-RCF\input"
    # input_folder = r"F:\dongjiayao\Pycharm\Holo-Track\img\3\input3"
    input_folder = r"F:\dongjiayao\Data\HoloLab_testData\autofocus\reconstruction"
    # input_folder = r"F:\dongjiayao\Pycharm\Holo-Track\img\3\temp"
    global output_folder
    output_folder = r"E:\Projects\HoloLabV1\tmp"
    final_folder = r"F:\dongjiayao\Data\HoloLab_testData\autofocus\ai_output"

    # 使用示例
    mot_tracker = Sort(max_age=15, min_hits=3, iou_threshold=0.3, distance_threshold=30)
    tracker = Kalman_tracker(image_folder=input_folder,
                             output_folder=output_folder,
                             yolo_model=yolo_model,
                             sort=mot_tracker)
    stacks, stacks_ori, names = tracker.get_stacks()

    rcf_model = RCF()
    rcf_model.cuda()
    checkpoint = torch.load(r'F:\dongjiayao\Pycharm\Holo-Track\rcf\trained_model\checkpoint_epoch29.pth')
    # checkpoint = torch.load(r'F:\dongjiayao\Pycharm\Holo-Track\rcf\trained_model\so3_checkpoint_epoch29.pth')
    rcf_model.load_state_dict(checkpoint['state_dict'])

    for i, stack in enumerate(stacks):
        stack_ori = stacks_ori[i]
        dataset = RCFLoader(stack, stack_ori, k_size=6)
        loader = DataLoader(dataset, batch_size=1, num_workers=1, drop_last=True, shuffle=True)

        img_tmp, name_tmp = rcf_predict(rcf_model, loader)
        img_tmp = torch.squeeze(img_tmp.squeeze()).cpu().numpy()
        cv2.imwrite(os.path.join(final_folder, "{}.jpg".format(i)), img_tmp)


if __name__ == "__main__":
    main()
