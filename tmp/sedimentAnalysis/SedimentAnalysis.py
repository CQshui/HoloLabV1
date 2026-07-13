import cv2
import numpy as np
import GetLiquidLevelCurve
from PIL import Image  # 用于图像显示

# 使用 OpenCV 读取图像 (BGR 格式)
# Im = cv2.imread("test/Image.png")
# Lb = cv2.imread("test/VesselMask.png", cv2.IMREAD_GRAYSCALE)  # 以灰度模式读取掩码
#
Im = cv2.imread("test/Image__2025-08-17__17-02-08.bmp")
Im_gray = cv2.imread("test/Image__2025-08-17__17-07-49.bmp", cv2.IMREAD_GRAYSCALE)
Lb = np.zeros_like(Im_gray)  # 以灰度模式读取掩码

# 将图像转换为 RGB 格式用于处理
Im_rgb = cv2.cvtColor(Im, cv2.COLOR_BGR2RGB)

# 调用处理函数
[BinaryCurve, OverLay] = GetLiquidLevelCurve.GetLiquidLevelCurve(Im_rgb, Lb)

# 显示结果 (使用 PIL)
Image.fromarray(OverLay).show()

# 保存结果 (使用 OpenCV)
cv2.imwrite("Output.png", cv2.cvtColor(OverLay, cv2.COLOR_RGB2BGR))