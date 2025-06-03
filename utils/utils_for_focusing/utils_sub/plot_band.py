"""
绘制带状误差图
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置图形样式（修正部分）
sns.set_theme(style="whitegrid")  # 使用seaborn的主题设置
plt.rcParams.update({'font.size': 12})  # 统一字体设置

# 读取数据
df = pd.read_csv(r"F:\dongjiayao\Pycharm\Holo-Track\utils\tmp\RCF\concentration_z_1.csv", index_col='z')

# 计算统计量
mean_values = df.mean(axis=1)
std_values = df.std(axis=1)
median_values = df.median(axis=1)
q25 = df.quantile(0.25, axis=1)
q75 = df.quantile(0.75, axis=1)

# 创建画布
plt.figure(figsize=(15, 8), dpi=100)

# 绘制带状区域（优化颜色和透明度）
plt.fill_between(mean_values.index,
                mean_values - 1.96*std_values,
                mean_values + 1.96*std_values,
                color='#1f77b4',  # 使用标准蓝色
                alpha=0.2,
                label='95% CI')

plt.fill_between(median_values.index,
                q25,
                q75,
                color='#ff7f0e',  # 使用标准橙色
                alpha=0.3,
                label='IQR')

# 绘制均值线和中位线（优化线型）
plt.plot(mean_values.index, mean_values,
        color='#2ca02c',  # 绿色
        linewidth=2.5,
        label='Mean')

plt.plot(median_values.index, median_values,
        color='#d62728',  # 红色
        linestyle=(0, (5, 2)),  # 自定义虚线样式
        linewidth=2.5,
        label='Median')

# 图形装饰（优化标签和标题）
plt.xlabel('Z Position (μm)', fontsize=14, fontweight='bold')
plt.ylabel('Concentration (a.u.)', fontsize=14, fontweight='bold')
plt.title('Concentration Distribution Along Z-axis',
         fontsize=16, pad=20, fontweight='bold')

# 优化图例位置和样式
plt.legend(loc='upper right', frameon=True, shadow=True)

# 优化坐标轴
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlim(df.index.min(), df.index.max())
plt.xticks(np.arange(df.index.min(), df.index.max()+1, 5))  # 设置x轴刻度间隔

# 设置y轴范围（新增代码）
# plt.ylim(0, 255)

# 调整布局并显示
plt.tight_layout()
plt.show()
