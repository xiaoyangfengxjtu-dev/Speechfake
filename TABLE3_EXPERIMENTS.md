# Table 3 Overall Performance Experiments

复现论文 Table 3 的完整实验指南 - **两个基线模型对比**

## 实验目标

复现论文 Table 3 中 **AASIST** 和 **W2V+AASIST** 两个基线模型的性能对比表格：

| Training Dataset | Model | BD | BD-EN | BD-CN | ASV19 | ITW | FOR |
|------------------|-------|----|----|----|----|----|----|
| ASV19 | AASIST | 39.36 | 41.05 | 39.07 | 1.88 | 45.27 | 36.08 |
| BD | AASIST | 3.48 | 3.98 | 2.68 | 23.62 | 7.53 | 23.35 |
| BD-EN | AASIST | 9.02 | 6.17 | 12.00 | 30.65 | 6.96 | 28.99 |
| BD-CN | AASIST | 16.58 | 24.59 | 5.43 | 16.56 | 8.54 | 25.48 |
| ASV19 | W2V+AASIST | 23.78 | 20.15 | 24.93 | 0.89 | 10.07 | 6.18 |
| BD | W2V+AASIST | 3.54 | 3.55 | 2.83 | 2.91 | 2.01 | 6.00 |
| BD-EN | W2V+AASIST | 8.65 | 4.58 | 10.44 | 5.28 | 2.62 | 8.33 |
| BD-CN | W2V+AASIST | 8.99 | 11.40 | 4.51 | 0.99 | 3.34 | 4.88 |

## 实验配置文件

### 基线模型 1: AASIST
- **模型实现**: `models/AASIST.py` - 完整的图注意力网络
- **配置**: `config/AASIST.conf` - 论文对齐参数
- **实验配置**:
  - `experiments/table3_aasist_asv19.conf` - ASVspoof2019 训练
  - `experiments/table3_aasist_bd.conf` - BD 训练
  - `experiments/table3_aasist_bd_en.conf` - BD-EN 训练
  - `experiments/table3_aasist_bd_cn.conf` - BD-CN 训练

### 基线模型 2: W2V+AASIST
- **模型实现**: `models/W2VAASIST.py` - Wav2Vec2 + AASIST 集成
- **配置**: `config/SpeechFake_W2V_AASIST.conf` - 论文对齐参数
- **实验配置**:
  - `experiments/table3_w2v_asv19.conf` - ASVspoof2019 训练
  - `experiments/table3_w2v_bd.conf` - BD 训练
  - `experiments/table3_w2v_bd_en.conf` - BD-EN 训练
  - `experiments/table3_w2v_bd_cn.conf` - BD-CN 训练

## 运行实验

### 方式 1: 批量运行所有实验
```bash
python run_table3_experiments.py
```

### 方式 2: 单独运行实验
```bash
# AASIST 实验
python main.py --config experiments/table3_aasist_asv19.conf --seed 1234 --comment "AASIST_ASV19"
python main.py --config experiments/table3_aasist_bd.conf --seed 1234 --comment "AASIST_BD"
python main.py --config experiments/table3_aasist_bd_en.conf --seed 1234 --comment "AASIST_BD_EN"
python main.py --config experiments/table3_aasist_bd_cn.conf --seed 1234 --comment "AASIST_BD_CN"

# W2V+AASIST 实验
python main.py --config experiments/table3_w2v_asv19.conf --seed 1234 --comment "W2V_ASV19"
python main.py --config experiments/table3_w2v_bd.conf --seed 1234 --comment "W2V_BD"
python main.py --config experiments/table3_w2v_bd_en.conf --seed 1234 --comment "W2V_BD_EN"
python main.py --config experiments/table3_w2v_bd_cn.conf --seed 1234 --comment "W2V_BD_CN"
```

## 生成结果表格

训练完成后，运行评估脚本生成表格：

```bash
python table3_evaluation.py --exp_root ./exp_result --output table3_results
```

这将生成：
- `table3_results.csv` - CSV 格式
- `table3_results.tex` - LaTeX 格式
- `table3_results.md` - Markdown 格式

## 实验参数

所有实验使用论文 Table 9 的精确参数：

### AASIST
- Batch Size: 1024
- Learning Rate: 1e-4
- Epochs: 50
- Weight Decay: 1e-4
- Optimizer: Adam
- Loss: Weighted CE (0.9 real, 0.1 fake)

### W2V+AASIST
- Batch Size: 512
- Learning Rate: 1e-6
- Epochs: 50
- Weight Decay: 1e-4
- Optimizer: Adam
- Loss: Weighted CE (0.9 real, 0.1 fake)
- Wav2Vec2: XLS-R-300M (frozen 95% layers)

## 预期结果

根据论文 Table 3，你应该得到类似的 EER 结果。主要观察：

1. **ASV19 训练**: 在 ASV19 测试集上表现最好，但在其他数据集上泛化较差
2. **BD 训练**: 在 SpeechFake 数据集上表现最好，泛化能力强
3. **W2V+AASIST**: 通常比 AASIST 表现更好，特别是在跨数据集泛化上
4. **语言特定训练**: BD-EN 和 BD-CN 在对应语言上表现更好

## 注意事项

1. **数据准备**: 确保所有数据集 (ASV19, BD, ITW, FOR) 都已正确准备
2. **GPU 内存**: AASIST 需要 24GB+ VRAM (batch=1024)
3. **训练时间**: 每个实验大约需要 2-4 小时 (取决于硬件)
4. **存储空间**: 确保有足够的磁盘空间存储模型和结果

## 故障排除

如果实验失败，检查：
1. 数据路径是否正确
2. GPU 内存是否足够
3. 配置文件语法是否正确
4. 依赖包是否完整安装
