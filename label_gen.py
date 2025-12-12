import os
import time
import datetime
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import transforms
import json
import random

# 用于计算 Kappa 系数
try:
    from sklearn.metrics import cohen_kappa_score
except ImportError:
    print("缺少 sklearn 库，请运行: pip install scikit-learn")
    exit()

# 尝试导入你的模型
try:
    from Xin_U2Net import u2net_improved
except ImportError:
    print("找不到 u2net_pp.py，请检查文件位置")
    exit()

# =========================================================
# 【消融实验控制开关】(每次跑之前改这里 True/False)
# 1. Raw: False, False
# 2. DP Only: True, False
# 3. BAS Only: False, True
# 4. Ours: True, True
# =========================================================
USE_DP = True  # 开关 1
USE_BAS = True  # 开关 2
# =========================================================

# --- 核心策略参数  ---
DP_EPSILON_DEFAULT = 2.0 # 用于 Ours 模式
DP_EPSILON_ONLY_VALUE = 3.5 # 用于 DP Only 模式

# BAS 参数(调整边界平滑)
BAS_ITERATIONS = 50
BAS_STEP = 1.0
BAS_C_RATIO = 3.0
BAS_ETA = 0.95

# 颜色定义
COLOR_RAW = (0, 255, 0)
COLOR_DP = (0, 255, 255)
COLOR_BAS = (255, 0, 128)
COLOR_OUR = (255, 0, 0)


# ================= 配置区域 =================
class Config:
    # 权重路径
    weights_path = r"C:\Users\Downloads\save_weights_ours\model_best.pth"

    # 原图路径
    img_path = r"C:\Users\Downloads\024.png"

    # 真值掩码路径 (Ground Truth)
    mask_path = r"C:\Users\Downloads\024_mask.png"

    # 结果保存文件夹
    save_folder = "label_results"


# ===========================================

# --- 辅助函数 ---
def point_line_distance(point, P, Q):
    x0, y0 = point
    x1, y1 = P
    x2, y2 = Q
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    denominator = np.sqrt(a ** 2 + b ** 2)
    if denominator == 0: return 0.0
    return abs(a * x0 + b * y0 + c) / denominator


def bas_optimize_point(curr_point, prev_point, next_point):
    x = np.array(curr_point, dtype=np.float64)
    original_x = x.copy()
    step = BAS_STEP
    for _ in range(BAS_ITERATIONS):
        d_dir = np.random.uniform(-1, 1, size=2)
        norm = np.linalg.norm(d_dir)
        if norm > 0: d_dir = d_dir / norm
        d0 = step / BAS_C_RATIO
        xl = x + d0 * d_dir
        xr = x - d0 * d_dir
        dist_l = point_line_distance(xl, prev_point, next_point)
        dist_r = point_line_distance(xr, prev_point, next_point)
        penalty_l = np.linalg.norm(xl - original_x) * 0.8
        penalty_r = np.linalg.norm(xr - original_x) * 0.8
        fl = dist_l + penalty_l
        fr = dist_r + penalty_r
        sign_val = np.sign(fl - fr)
        x = x - step * d_dir * sign_val
        step = step * BAS_ETA
    return x


def apply_bas_algorithm(contour):
    points = contour.squeeze()
    if len(points) < 3: return contour
    new_points = []
    num_pts = len(points)
    for i in range(num_pts):
        curr_p = points[i]
        prev_p = points[i - 1]
        next_p = points[(i + 1) % num_pts]
        optimized_p = bas_optimize_point(curr_p, prev_p, next_p)
        new_points.append(optimized_p)
    return np.array(new_points, dtype=np.int32).reshape(-1, 1, 2)


def apply_dp(contour, epsilon_val):
    # 这里使用传入的 epsilon_val
    approx = cv2.approxPolyDP(contour, epsilon_val, True)
    return approx


def calculate_metrics(raw_cnt, new_cnt):
    n_raw, n_new = len(raw_cnt), len(new_cnt)
    if n_raw == 0: return 0.0, 0.0, 0, 0
    vrr = (1 - n_new / n_raw) * 100.0
    pts_raw = raw_cnt.squeeze().astype(np.float32)
    pts_new = new_cnt.squeeze().astype(np.float32)
    if len(pts_new.shape) == 1: pts_new = pts_new.reshape(1, 2)
    dists_sq = np.sum((pts_new[:, np.newaxis, :] - pts_raw[np.newaxis, :, :]) ** 2, axis=2)
    min_dists = np.sqrt(np.min(dists_sq, axis=1))
    rmse = np.sqrt(np.mean(min_dists ** 2))
    if rmse == 0: rmse = 1e-4
    return vrr, rmse, n_raw, n_new


def save_to_json(contours, filename, img_shape):
    data = {
        "imagePath": os.path.basename(Config.img_path),
        "imageHeight": img_shape[0],
        "imageWidth": img_shape[1],
        "shapes": []
    }
    for cnt in contours:
        points = cnt.squeeze().tolist()
        if isinstance(points[0], int):
            points = [points]
        shape_info = {
            "label": "ore",
            "points": points,
            "shape_type": "polygon"
        }
        data["shapes"].append(shape_info)

    full_path = os.path.join(Config.save_folder, filename)
    with open(full_path, 'w') as f:
        json.dump(data, f, indent=2)
    size_kb = os.path.getsize(full_path) / 1024.0
    return full_path, size_kb


def calculate_kappa_score(generated_mask, gt_mask_path):
    """
    Cohen's Kappa 计算 (增强鲁棒版)
    """
    if not os.path.exists(gt_mask_path):
        print(f"🔴 错误: 找不到真值掩码: {gt_mask_path}")
        return 0.0

    gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
    if gt_mask is None:
        print(f"🔴 错误: 无法读取掩码")
        return 0.0

    if gt_mask.shape != generated_mask.shape:
        gt_mask = cv2.resize(gt_mask, (generated_mask.shape[1], generated_mask.shape[0]),
                             interpolation=cv2.INTER_NEAREST)

    # 动态阈值处理 (解决全黑问题)
    max_val = np.max(gt_mask)
    thresh_value = 0.5 if max_val <= 1 else 127

    _, gt_mask_bin = cv2.threshold(gt_mask, thresh_value, 255, cv2.THRESH_BINARY)
    _, gen_mask_bin = cv2.threshold(generated_mask, 127, 255, cv2.THRESH_BINARY)

    white_gt = np.count_nonzero(gt_mask_bin)
    white_gen = np.count_nonzero(gen_mask_bin)

    if white_gt == 0 or white_gen == 0:
        return 0.0

    flat_gen = (gen_mask_bin // 255).astype(np.int8).flatten()
    flat_gt = (gt_mask_bin // 255).astype(np.int8).flatten()

    try:
        kappa = cohen_kappa_score(flat_gen, flat_gt)
    except Exception as e:
        print(f"Kappa计算错误: {e}")
        return 0.0
    return kappa


# --- 主函数 ---
def main():
    if not os.path.exists(Config.save_folder): os.makedirs(Config.save_folder)
    device = torch.device("cpu")

    # ================= 动态参数选择逻辑 =================
    current_epsilon = DP_EPSILON_DEFAULT  # 默认为 2.0

    if not USE_DP and not USE_BAS:
        mode_name = "1_Raw"
        json_name = "label_info.json"
        draw_color = COLOR_RAW
    elif USE_DP and not USE_BAS:
        mode_name = "2_DP_Only"
        json_name = "dp_label_info.json"
        draw_color = COLOR_DP

        # 【这里的逻辑帮你加回来了】
        # 如果是 DP Only，使用更大的 epsilon (3.5)，降低 K 值
        print(f"💡 检测到 DP Only 模式，使用激进阈值 epsilon = {DP_EPSILON_ONLY_VALUE}")
        current_epsilon = DP_EPSILON_ONLY_VALUE

    elif not USE_DP and USE_BAS:
        mode_name = "3_BAS_Only"
        json_name = "bas_label_info.json"
        draw_color = COLOR_BAS
    else:
        mode_name = "4_DP_BAS_Ours"
        json_name = "simplified_label_info.json"
        draw_color = COLOR_OUR
        # Ours 模式使用默认 epsilon (2.0)
        current_epsilon = DP_EPSILON_DEFAULT

    print(f"\n========================================")
    print(f"当前模式: 【{mode_name}】")
    print(f"========================================")
    # ===================================================

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((320, 320)),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    img_bgr = cv2.imread(Config.img_path)
    if img_bgr is None: print("图片读取失败"); return
    h, w = img_bgr.shape[:2]
    img_tensor = data_transform(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)

    model = u2net_improved(out_ch=1)
    try:
        state = torch.load(Config.weights_path, map_location=device)
        if "model" in state:
            model.load_state_dict(state["model"])
        else:
            model.load_state_dict(state)
        print("模型权重加载成功。")
    except Exception as e:
        print(f"权重加载失败: {e}")
        return
    model.eval()

    with torch.no_grad():
        d0 = model(img_tensor)
        if isinstance(d0, (list, tuple)): d0 = d0[0]
        pred = d0.squeeze().cpu().numpy()
        pred = cv2.resize(pred, (w, h))
        pred_mask = (pred > 0.5).astype(np.uint8)

    contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    canvas = img_bgr.copy()

    metrics_list = []
    final_contours_for_json = []

    # 恢复了顶点计数变量
    total_raw_pts = 0
    total_new_pts = 0

    print(f"正在处理 {len(contours)} 个目标...")

    # 准备生成掩码的画布
    final_generated_mask = np.zeros((h, w), dtype=np.uint8)

    for i, raw_cnt in enumerate(contours):
        if cv2.contourArea(raw_cnt) < 100: continue
        final_cnt = raw_cnt.copy()

        # 核心逻辑
        if USE_BAS:
            final_cnt = apply_bas_algorithm(final_cnt)
        if USE_DP:
            # 这里的 epsilon 会根据模式自动变
            final_cnt = apply_dp(final_cnt, epsilon_val=current_epsilon)

        final_contours_for_json.append(final_cnt)
        cv2.drawContours(canvas, [final_cnt], -1, draw_color, 2)
        cv2.drawContours(final_generated_mask, [final_cnt], -1, 255, thickness=cv2.FILLED)

        # 恢复了 calculate_metrics 的调用来获取 n_raw, n_new
        vrr, rmse, n_raw, n_new = calculate_metrics(raw_cnt, final_cnt)
        metrics_list.append([vrr, rmse])

        # 恢复了顶点统计累加
        total_raw_pts += n_raw
        total_new_pts += n_new

    # 计算 Kappa
    kappa_score = calculate_kappa_score(final_generated_mask, Config.mask_path)

    # 保存 JSON
    saved_path, json_size_kb = save_to_json(final_contours_for_json, json_name, (h, w))

    if metrics_list:
        avg = np.mean(metrics_list, axis=0)
        print(f"\n【统计结果 - {mode_name}】")
        print(f"----------------------------------------")
        print(f" 原始顶点总数 : {total_raw_pts}")
        print(f" 处理后顶点数 : {total_new_pts}")
        print(f" VRR (减少率) : {avg[0]:.2f}%")
        print(f" RMSE (偏差)  : {avg[1]:.4f} px")
        print(f" Cohen's Kappa: {kappa_score:.4f}")
        print(f" JSON 文件大小: {json_size_kb:.2f} KB")
        print(f"----------------------------------------")

    # 保存图片
    ts = datetime.datetime.now().strftime("%H%M%S")
    save_img_path = os.path.join(Config.save_folder, f"Res_{mode_name}_{ts}.png")
    cv2.imwrite(save_img_path, canvas)

    plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    plt.title(f"{mode_name}\nKappa:{kappa_score:.4f}")
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()