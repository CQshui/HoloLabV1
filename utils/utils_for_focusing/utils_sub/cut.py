from PIL import Image
import os

# 定义输入和输出文件夹
input_folder = r'F:\dongjiayao\Pycharm\Holo-Track\img\input2'  # 包含原始图片的文件夹路径
output_folder = r'F:\dongjiayao\Pycharm\Holo-Track\img\cut2'  # 保存裁剪后图片的文件夹路径

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 需要裁剪的区域坐标（左上角和右下角）
x1 = 2508  # 左上角的x坐标
y1 = 1986  # 左上角的y坐标
x2 = 3666  # 右下角的x坐标
y2 = 3366  # 右下角的y坐标

# 遍历输入文件夹中的所有图片
for filename in os.listdir(input_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):  # 检查是否是图片文件
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        try:
            # 打开图片
            with Image.open(input_path) as img:
                # 获取图片的宽度和高度
                width, height = img.size

                # 确保裁剪区域不会超出图片范围
                crop_left = max(0, x1)
                crop_upper = max(0, y1)
                crop_right = min(x2, width)
                crop_lower = min(y2, height)

                # 如果裁剪区域有效（不为空）
                if crop_right > crop_left and crop_lower > crop_upper:
                    # 裁剪图片
                    cropped_img = img.crop((crop_left, crop_upper, crop_right, crop_lower))

                    # 保存裁剪后的图片
                    cropped_img.save(output_path)
                    print(f"裁剪并保存: {filename}")
                else:
                    print(f"裁剪区域无效，跳过: {filename}")
        except IOError:
            print(f"无法打开或处理图片: {filename}")