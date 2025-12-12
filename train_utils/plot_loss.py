import matplotlib.pyplot as plt
import re
import os

# ================= 配置区域 =================
# 1. 替换为你的日志文件名 (确保文件就在同级目录下)
log_file_path = "results20240801-164016.txt"

# 2. 图片保存名称
save_name = "400lun_U2Net++_Training_Loss_Curve.png"


# ===========================================

def plot_single_curve():
    # 1. 设置字体为 Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12

    # 2. 读取数据
    epochs = []
    losses = []

    if not os.path.exists(log_file_path):
        print(f"错误：找不到文件 {log_file_path}")
        return

    # 【修复点】在这里加了 encoding='utf-8'
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        # 如果UTF-8还不行，尝试忽略错误读取
        print("UTF-8 读取失败，尝试忽略错误读取...")
        with open(log_file_path, 'r', encoding='gbk', errors='ignore') as f:
            lines = f.readlines()

    for line in lines:
        # 正则提取数据
        epoch_match = re.search(r'epoch:\s*(\d+)', line)
        loss_match = re.search(r'train_loss:\s*([\d\.]+)', line)

        if epoch_match and loss_match:
            epochs.append(int(epoch_match.group(1)))
            losses.append(float(loss_match.group(1)))

    if not epochs:
        print("未提取到数据，请检查日志格式或文件路径。")
        return

    # 3. 创建画布
    plt.figure(figsize=(8, 6), dpi=300)

    # 4. 绘制单条曲线 (蓝色)
    plt.plot(epochs, losses,
             color='blue',  # 线条颜色
             linestyle='-',  # 实线
             linewidth=1.5)  # 线宽

    # 5. 设置标签和标题
    plt.title('Training Loss Curve', fontsize=16, fontweight='bold', pad=12)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Training Loss', fontsize=14)

    # 6. 设置网格和刻度
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tick_params(labelsize=12)

    # 7. 保存
    plt.tight_layout()
    plt.savefig(save_name, bbox_inches='tight')

    print(f"成功！图片已保存为: {save_name}")
    # plt.show() # 如果在服务器上跑不需要显示，可以注释掉


if __name__ == "__main__":
    plot_single_curve()