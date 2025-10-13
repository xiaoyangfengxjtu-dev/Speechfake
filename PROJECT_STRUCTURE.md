# 项目文件结构说明

## 核心训练脚本

### `main.py` ⭐⭐⭐⭐⭐
**作用**: 主训练脚本，整个项目的核心
- 训练、验证和评估模型
- 加载配置文件
- 数据加载和预处理
- 优化器和学习率调度
- 模型保存和检查点管理
- TensorBoard 日志记录
```bash
python main.py --config experiments/table3_aasist_bd.conf
```

## 模型文件 (`models/`)

### `models/AASIST.py` ⭐⭐⭐⭐⭐
**作用**: AASIST 基线模型实现
- 图注意力网络层 (GraphAttentionLayer)
- 异构图注意力层 (HtrgGraphAttentionLayer)
- 图池化层 (GraphPool)
- SincConv 层 (CONV)
- 残差块 (Residual_block)
- 完整的 AASIST 模型

### `models/W2VAASIST.py` ⭐⭐⭐⭐⭐
**作用**: W2V+AASIST 基线模型实现
- Wav2Vec2 特征提取器集成
- 特征投影层
- AASIST 图注意力网络
- Wav2Vec2 层冻结策略 (95%)
- 梯度检查点支持

### `models/RawNet2Spoof.py`
**作用**: RawNet2 基线模型 (原项目遗留)
- 未用于当前 SpeechFake 实验
- 可作为额外基线参考

### `models/RawGATST.py`
**作用**: RawGATST 基线模型 (原项目遗留)
- 未用于当前 SpeechFake 实验

## 数据加载脚本

### `speechfake_dataloader.py` ⭐⭐⭐⭐⭐
**作用**: SpeechFake 多数据集统一加载器 (新实现)
- 支持 SpeechFake BD, ASVspoof2019, ITW, FOR
- 语言过滤 (EN, CN)
- 生成方式过滤 (TTS, VC, NV)
- 音频预处理 (4s 对齐，zero-padding，裁剪)
- CSV 格式元数据支持
```python
from speechfake_dataloader import get_dataloader
train_loader = get_dataloader(...)
```

### `data_utils.py`
**作用**: ASVspoof2019 原始数据加载工具
- 生成 ASVspoof2019 文件列表
- Padding 函数
- 原项目遗留，已被 `speechfake_dataloader.py` 部分替代

### `download_dataset.py`
**作用**: 数据集下载辅助脚本
- 简单的数据集下载工具
- 可根据需要扩展

## 评估脚本

### `evaluation.py` ⭐⭐⭐⭐⭐
**作用**: 核心评估指标计算
- 计算 EER (Equal Error Rate)
- 计算 t-DCF (tandem Detection Cost Function)
- DET 曲线计算
- 与 ASVspoof 官方评估一致
```python
from evaluation import calculate_tDCF_EER
eer, tdcf = calculate_tDCF_EER(...)
```

### `speechfake_evaluation.py` ⭐⭐⭐⭐
**作用**: 多数据集批量评估脚本
- 在多个测试集上评估模型
- 自动生成评估结果
- JSON 格式结果保存
```bash
python speechfake_evaluation.py --model_path xxx.pth
```

### `table3_evaluation.py` ⭐⭐⭐⭐⭐
**作用**: Table 3 结果表格生成
- 汇总所有实验结果
- 生成性能对比表格 (CSV, LaTeX, Markdown)
- 保存原始数据 (JSON)
```bash
python table3_evaluation.py --exp_root ./exp_result
```

## 实验配置管理

### `run_table3_experiments.py` ⭐⭐⭐⭐⭐
**作用**: 批量运行 Table 3 所有实验
- 自动运行 8 个实验 (4个 AASIST + 4个 W2V+AASIST)
- 进度跟踪
- 错误处理
```bash
python run_table3_experiments.py
```

### `prepare_speechfake_data.py` ⭐⭐⭐⭐
**作用**: SpeechFake 数据集准备工具
- 分析数据集结构
- 生成 CSV 元数据文件
- 验证数据集完整性
```bash
python prepare_speechfake_data.py --data_dir /path/to/data --output_csv train.csv
```

## 服务器部署工具

### `update_server_paths.py` ⭐⭐⭐⭐
**作用**: 批量更新配置文件中的数据集路径
- 将本地路径替换为服务器路径
- 支持干运行模式 (预览更改)
- 批量处理所有配置文件
```bash
python update_server_paths.py --new_path /server/datasets/ --dry_run
```

### `check_server_config.py` ⭐⭐⭐⭐
**作用**: 检查服务器配置和数据集
- 验证数据集路径是否存在
- 检查数据集完整性
- 配置文件验证
```bash
python check_server_config.py
```

### `upload_to_github.sh` ⭐⭐⭐
**作用**: GitHub 上传自动化脚本
- 自动设置远程仓库
- 推送代码到 GitHub
- 错误处理和提示
```bash
./upload_to_github.sh username repo-name
```

## 工具函数

### `utils.py` ⭐⭐⭐⭐
**作用**: 通用工具函数
- 学习率调度器 (cosine, Keras decay, SGDR)
- 优化器创建
- 随机种子设置
- 字符串转布尔值

## 配置文件

### `config/AASIST.conf` ⭐⭐⭐⭐⭐
**作用**: AASIST 模型配置
- 论文对齐的训练参数
- 模型架构参数
- 数据集路径

### `config/SpeechFake_W2V_AASIST.conf` ⭐⭐⭐⭐⭐
**作用**: W2V+AASIST 模型配置
- Wav2Vec2 检查点路径
- AASIST 图参数
- 训练超参数

### `experiments/*.conf` ⭐⭐⭐⭐⭐
**作用**: Table 3 实验配置文件 (8个)
- `table3_aasist_*.conf` - AASIST 实验 (4个)
- `table3_w2v_*.conf` - W2V+AASIST 实验 (4个)
- 每个配置对应一个训练数据集

## 文档文件

### `README.md` ⭐⭐⭐⭐⭐
**作用**: 项目主文档
- 项目概述
- 安装和设置
- 使用方法
- 两个基线模型介绍

### `TABLE3_EXPERIMENTS.md` ⭐⭐⭐⭐
**作用**: Table 3 实验详细指南
- 实验目标和预期结果
- 配置文件说明
- 运行步骤
- 参数设置

### `SERVER_DEPLOYMENT.md` ⭐⭐⭐⭐
**作用**: 服务器部署完整指南
- 目录结构设置
- 路径配置修改
- 部署步骤
- 故障排除

### `GITHUB_UPLOAD.md` ⭐⭐⭐
**作用**: GitHub 上传指南
- 上传步骤
- 文件清单
- 服务器克隆流程

### `QUICK_START.txt`
**作用**: 快速开始指南
- 简单的入门步骤

### `ALIGNMENT_PENDING_ITEMS.md`
**作用**: 论文对齐检查清单 (临时文件)
- 记录参数对齐情况
- 可以删除

## 依赖文件

### `requirements.txt` ⭐⭐⭐⭐⭐
**作用**: Python 依赖包列表
- PyTorch, torchaudio
- transformers (Wav2Vec2)
- pandas, numpy
- 评估工具包
```bash
pip install -r requirements.txt
```

## 文件重要性图例
- ⭐⭐⭐⭐⭐ 核心文件，必须了解
- ⭐⭐⭐⭐ 重要文件，经常使用
- ⭐⭐⭐ 辅助文件，有用但非必需
- 无星号：可选或临时文件

## 常用工作流程

### 1. 训练单个模型
```bash
python main.py --config experiments/table3_aasist_bd.conf
```

### 2. 批量运行所有实验
```bash
python run_table3_experiments.py
```

### 3. 生成结果表格
```bash
python table3_evaluation.py --exp_root ./exp_result
```

### 4. 服务器部署
```bash
python update_server_paths.py --new_path /server/datasets/
python check_server_config.py
```

## 关键文件依赖关系

```
main.py
├── models/AASIST.py 或 models/W2VAASIST.py
├── speechfake_dataloader.py
├── evaluation.py
├── utils.py
└── config/*.conf

run_table3_experiments.py
└── main.py (调用 8 次)

table3_evaluation.py
└── evaluation.py (读取结果)
```
