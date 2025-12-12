# 基于U2-Net的轻量化矿石图像标注模型 (U2Net++)

## 📢 重要说明 (Update for Reproducibility)
为了响应同行评审关于**可复现性**的建议，我们对本仓库进行了重大更新：
- **统一训练框架**：通过配置开关，一套代码即可复现 Baseline (U2Net) 和改进模型 (U2Net++)。
- **完整预处理流程**：公开了从 Labelme 标注到训练集生成的完整处理细节。
- **轻量化标签生成**：新增独立的 `label_gen.py` 脚本，支持通过配置开关（DP/BAS）来**独立观测 Douglas-Peucker 简化与 BAS 边界优化的具体作用**。

## 简介
针对矿石图像人工标注耗时耗力、质量不稳定等问题，本研究在 U2Net 网络基础上提出改进的轻量化自动标注矿石图像模型。

## 主要改进
- **网络结构优化**：
  - 引入 **CBAM** 特征融合模块（Backbone），提升碎矿石特征提取能力。
  - 引入 **SE** 跳跃连接方式，减少特征信息丢失。
- **轻量化标注**：
  - 结合 **BAS (Boundary Adjustment Strategy)** 算法进行参数优化。
  - 引入 **Douglas-Peucker** 算法进行轮廓简化。

## 环境要求
- Python 3.8+
- Pytorch 1.13.1
- CUDA 11.6+ (建议使用 GPU 训练)
- 详细环境配置见 `requirements.txt`

## 项目结构
```text
├── u2net_pp.py            # [核心] 改进的模型定义 (含消融实验开关)
├── train.py               # [核心] 统一训练脚本
├── test.py                # [推理] 模型预测脚本 (生成显著性分割图)
├── label_gen.py           # [标注] 轻量化标签生成脚本 (含DP/BAS开关)
├── my_dataset.py          # 数据集读取
├── transforms.py          # 数据增强
└── requirements.txt       # 环境依赖
```
## 🛠️ 数据准备流程 (Data Preparation Pipeline)
为了确保实验的一致性和鲁棒性，我们执行了以下标准化的预处理流程：
### 1. 标注与格式转换
   - **工具**：使用 Labelme 软件进行细粒度多边形标注。
   - **转换**：将生成的 JSON 文件转换为二进制掩码 (Binary Masks)。
### 2. 尺寸调整与离线增强
   原始图像可经过以下步骤进行扩充：
   - 随机缩放、随机裁剪和水平翻转。
   - 按 7:2:1 的比例随机划分为训练集、验证集和测试集。
### 3. 数据集存放结构
请将处理好的数据按 DUTS 格式存放：
```text
DUTS-TR/
├── DUTS-TR-Image/        # 训练集图片
└── DUTS-TR-Mask/         # 训练集标注 (png格式)

DUTS-TE/
├── DUTS-TE-Image/        # 测试集图片  
└── DUTS-TE-Mask/         # 测试集标注 (png格式)
```
(注意：训练时请将 --data-path 指向包含 DUTS-TR 的根目录)

## 🚀 训练与消融实验 (Training & Ablation)
我们设计了统一的模型文件 u2net_pp.py，通过修改文件顶部的开关即可复现论文中的所有实验。
### 1. 配置模型开关
打开**u2net_pp.py**，修改以下变量：
实验目标	**USE_CBAM**	**USE_SE_SKIP**	说明
- **Baseline** (原始 U2Net)	False	False	原始基准模型
- **Only CBAM**	True	False	仅加入 CBAM 模块
- **Only SE**	False	True	仅加入 SE 跳跃连接
- **U2Net++** (Ours)	True	True	本文提出的最终模型
### 2. 启动训练
修改好开关后，运行以下命令（每次实验建议修改代码中的 EXP_NAME 以区分保存路径）：
```text
python train.py --data-path ./DUTS-TR --batch-size 4 --epochs 400
```
### 预训练权重
- Baseline 权重：可通过上述脚本复现，或访问 U2Net官方GitHub。
- 改进模型权重：可通过联系作者获取学术授权。

## 🎯 推理与轻量化标签生成 (Inference & Label Generation)
该部分分为两步：首先生成分割掩码，然后将其转换为轻量化多边形标签。
### 1. 模型推理 (生成分割图)
使用 test.py 加载训练好的权重，生成像素级显著性分割图（Mask）。
```text
python test.py --model saved_models/model_best.pth --input test_images/ --output results/masks/
```
### 2. 生成轻量化标签 (DP & BAS)
使用 label_gen.py 将分割图转换为 JSON 标注文件。
- **消融研究**：该脚本包含 USE_DP 和 USE_BAS 开关，可独立验证不同后处理策略的效果。
```text
python label_gen.py --input results/masks/ --output results/jsons/
```
通过调整脚本中的参数，可以复现论文表 3 中关于“仅DP”、“仅BAS”和“BAS-DP”的对比实验。

## 实验结果
  **Method**	          MAE↓  maxF1↑
- **U2Net (Baseline)**	0.045	0.868
- **U2Net++ (Ours)**	  0.038 0.892
## 许可证与引用
- 本项目仅限学术研究使用。
- 如果本项目对你的工作有帮助，请引用相关论文

## 致谢
- 本项目基于 U^2-Net 开发，感谢原作者的杰出工作。
