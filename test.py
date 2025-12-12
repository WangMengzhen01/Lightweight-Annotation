import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

# from tqdm import tqdm  # 如果没安装tqdm，这行注释掉

# 导入你的模型定义
try:
    from u2net_pp import u2net_improved
except ImportError:
    print("找不到 u2net_pp.py，请检查路径")
    exit()


# --- 1. 配置区域 ---
class TestConfig:
    # 路径设置
    image_dir = r'C:\Users\Downloads\DUTS-TE\DUTS-TE-Image'
    mask_dir = r'C:\Users\Downloads\DUTS-TE\DUTS-TE-Mask'

    # 模型权重路径 (baseline/se_only/cbam_only/ours)
    model_path = r'save_weights_ours/model_best.pth'
    # 结果保存路径
    save_dir = r'test_results/exp4_ours'

    image_size = 320
    threshold = 0.5


# 创建保存文件夹
os.makedirs(TestConfig.save_dir, exist_ok=True)


# --- 2. 指标计算函数 ---
def calculate_metrics_numpy(pred_map, gt_mask, threshold=0.5):
    mae = np.mean(np.abs(pred_map - gt_mask))
    pred_binary = (pred_map > threshold).astype(np.float32)
    gt_binary = (gt_mask > 0.5).astype(np.float32)

    TP = np.sum((pred_binary == 1) & (gt_binary == 1))
    TN = np.sum((pred_binary == 0) & (gt_binary == 0))
    FP = np.sum((pred_binary == 1) & (gt_binary == 0))
    FN = np.sum((pred_binary == 0) & (gt_binary == 1))

    eps = 1e-6
    accuracy = (TP + TN) / (TP + TN + FP + FN + eps)
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)
    return accuracy, mae, precision, recall, f1


# --- 3. 主测试逻辑 ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device}")

    print(f"正在加载模型: {TestConfig.model_path}")
    net = u2net_improved(out_ch=1)

    if os.path.exists(TestConfig.model_path):
        checkpoint = torch.load(TestConfig.model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            print("检测到 Checkpoint 格式，正在提取模型权重...")
            net.load_state_dict(checkpoint['model'])
        else:
            print("检测到纯权重格式，直接加载...")
            net.load_state_dict(checkpoint)
    else:
        print(f"错误：找不到权重文件 {TestConfig.model_path}")
        return

    net.to(device)
    net.eval()

    transform = transforms.Compose([
        transforms.Resize((TestConfig.image_size, TestConfig.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    img_list = [f for f in os.listdir(TestConfig.image_dir) if f.endswith('.jpg') or f.endswith('.png')]
    img_list.sort()

    total_acc, total_mae, total_pre, total_rec, total_f1 = 0, 0, 0, 0, 0
    count = 0

    print(f"开始测试，共 {len(img_list)} 张图片...")

    with torch.no_grad():
        for i, img_name in enumerate(img_list):
            if i % 100 == 0: print(f"正在处理: {i}/{len(img_list)}")

            img_path = os.path.join(TestConfig.image_dir, img_name)
            mask_name = os.path.splitext(img_name)[0] + '.png'
            mask_path = os.path.join(TestConfig.mask_dir, mask_name)

            if not os.path.exists(mask_path):
                continue

            image = Image.open(img_path).convert('RGB')
            w_orig, h_orig = image.size
            gt_mask = Image.open(mask_path).convert('L')
            gt_mask_np = np.array(gt_mask).astype(np.float32) / 255.0

            inputs = transform(image).unsqueeze(0).to(device)

            # --- 【核心修改点】 ---
            # 1. 直接获取输出，不进行解包
            d0 = net(inputs)

            # 2. 如果万一返回了列表（防止版本不一致），取第一个；否则直接用
            if isinstance(d0, (list, tuple)):
                d0 = d0[0]

            # 3. 直接插值恢复尺寸 (d0是 [1,1,320,320])
            # mode='bilinear' 要求输入是 4D，正好符合
            pred = F.interpolate(d0, size=(h_orig, w_orig), mode='bilinear', align_corners=False)

            # 4. 转回 CPU 并去掉多余维度 -> [H, W]
            pred = pred.squeeze().cpu().numpy()

            # 5. 归一化 (防溢出)
            pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
            # ---------------------

            pred_save = (pred * 255).astype(np.uint8)
            save_path = os.path.join(TestConfig.save_dir, os.path.splitext(img_name)[0] + '.png')
            Image.fromarray(pred_save).save(save_path)

            acc, mae, pre, rec, f1 = calculate_metrics_numpy(pred, gt_mask_np, TestConfig.threshold)
            total_acc += acc
            total_mae += mae
            total_pre += pre
            total_rec += rec
            total_f1 += f1
            count += 1

    if count > 0:
        print("\n" + "=" * 40)
        print(f" 测试完成！(共 {count} 张)")
        print(f" 模型权重: {TestConfig.model_path}")
        print("-" * 40)
        print(f" Accuracy  : {total_acc / count:.4f}")
        print(f" MAE       : {total_mae / count:.4f}")
        print(f" Precision : {total_pre / count:.4f}")
        print(f" Recall    : {total_rec / count:.4f}")
        print(f" F1-Score  : {total_f1 / count:.4f}")
        print("=" * 40)
    else:
        print("未处理任何图片。")


if __name__ == '__main__':
    main()