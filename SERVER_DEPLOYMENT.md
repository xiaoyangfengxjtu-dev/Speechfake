# 服务器部署配置指南

在服务器上部署和运行 SpeechFake 实验的完整指南

## 1. 目录结构设置

### 推荐的服务器目录结构
```
/your/workspace/
├── speechfake-baselines/          # 项目代码
│   ├── config/
│   ├── experiments/
│   ├── models/
│   └── ...
├── datasets/                      # 数据集目录
│   ├── ASVspoof2019_LA/          # ASVspoof2019 LA 数据集
│   ├── SPEECHFAKE/               # SpeechFake 数据集
│   ├── ITW/                      # In-the-Wild 数据集
│   └── FOR/                      # FakeOrReal 数据集
└── exp_result/                   # 实验结果目录
```

## 2. 需要修改的配置文件

### 2.1 主要配置文件修改

**所有 `.conf` 文件中的 `database_path` 需要修改为服务器实际路径：**

#### 当前设置 (本地开发)
```json
"database_path": "./datasets/"
```

#### 服务器设置 (示例)
```json
"database_path": "/path/to/your/datasets/"
```

### 2.2 具体需要修改的文件

#### Table 3 实验配置文件 (8个)
- `experiments/table3_aasist_asv19.conf`
- `experiments/table3_aasist_bd.conf`
- `experiments/table3_aasist_bd_en.conf`
- `experiments/table3_aasist_bd_cn.conf`
- `experiments/table3_w2v_asv19.conf`
- `experiments/table3_w2v_bd.conf`
- `experiments/table3_w2v_bd_en.conf`
- `experiments/table3_w2v_bd_cn.conf`

#### 其他配置文件 (4个)
- `experiments/paper_exact_reproduction.conf`
- `experiments/exp_bd_w2v_aasist.conf`
- `config/AASIST.conf`
- `config/SpeechFake_W2V_AASIST.conf`

## 3. 数据集路径配置

### 3.1 ASVspoof2019 LA 数据集
```
/path/to/your/datasets/ASVspoof2019_LA/
├── ASVspoof2019_LA_train/
│   ├── flac/
│   └── ...
├── ASVspoof2019_LA_dev/
│   ├── flac/
│   └── ...
├── ASVspoof2019_LA_eval/
│   ├── flac/
│   └── ...
└── ASVspoof2019_LA_cm_protocols/
    ├── ASVspoof2019.LA.cm.train.trn.txt
    ├── ASVspoof2019.LA.cm.dev.trl.txt
    └── ASVspoof2019.LA.cm.eval.trl.txt
```

### 3.2 SpeechFake 数据集
```
/path/to/your/datasets/SPEECHFAKE/
├── train/
│   ├── audio/
│   └── train.csv
├── dev/
│   ├── audio/
│   └── dev.csv
└── test/
    ├── audio/
    └── test.csv
```

### 3.3 其他数据集
```
/path/to/your/datasets/
├── ITW/
│   ├── audio/
│   └── metadata.csv
└── FOR/
    ├── audio/
    └── metadata.csv
```

## 4. 批量修改脚本

### 4.1 创建路径更新脚本

创建 `update_server_paths.py` 脚本：

```python
#!/usr/bin/env python3
"""
Update database paths for server deployment
"""

import json
import os
from pathlib import Path

def update_config_paths(config_dir: str, old_path: str, new_path: str):
    """
    Update database_path in all .conf files
    """
    config_path = Path(config_dir)
    conf_files = list(config_path.rglob("*.conf"))
    
    updated_files = []
    
    for conf_file in conf_files:
        try:
            with open(conf_file, 'r') as f:
                config = json.load(f)
            
            if 'database_path' in config and config['database_path'] == old_path:
                config['database_path'] = new_path
                
                with open(conf_file, 'w') as f:
                    json.dump(config, f, indent=4)
                
                updated_files.append(str(conf_file))
                print(f"Updated: {conf_file}")
                
        except Exception as e:
            print(f"Error updating {conf_file}: {e}")
    
    return updated_files

if __name__ == "__main__":
    # 修改这些路径为你的服务器实际路径
    OLD_PATH = "./datasets/"
    NEW_PATH = "/path/to/your/datasets/"  # 修改为实际路径
    
    # 更新配置文件
    updated = update_config_paths(".", OLD_PATH, NEW_PATH)
    
    print(f"\nUpdated {len(updated)} configuration files:")
    for file in updated:
        print(f"  - {file}")
```

### 4.2 使用方法

```bash
# 1. 修改脚本中的 NEW_PATH 为你的实际数据集路径
# 2. 运行脚本
python update_server_paths.py
```

## 5. 服务器部署步骤

### 5.1 环境准备
```bash
# 1. 克隆项目到服务器
git clone <your-repo> speechfake-baselines
cd speechfake-baselines

# 2. 安装依赖
pip install -r requirements.txt

# 3. 创建数据集目录
mkdir -p /path/to/your/datasets
```

### 5.2 数据集准备
```bash
# 1. 下载并解压所有数据集到指定目录
# 2. 确保目录结构符合要求
# 3. 生成 SpeechFake 数据集的 CSV 文件
python prepare_speechfake_data.py \
    --data_dir /path/to/your/datasets/SPEECHFAKE/train/ \
    --output_csv train.csv \
    --audio_dir audio
```

### 5.3 更新配置文件
```bash
# 1. 修改 update_server_paths.py 中的路径
# 2. 运行路径更新脚本
python update_server_paths.py
```

### 5.4 验证配置
```bash
# 检查配置文件是否正确更新
grep -r "database_path" experiments/ config/
```

## 6. 运行实验

### 6.1 单个实验
```bash
python main.py \
    --config experiments/table3_aasist_bd.conf \
    --seed 1234 \
    --output_dir ./exp_result \
    --comment "AASIST_BD"
```

### 6.2 批量实验
```bash
python run_table3_experiments.py
```

### 6.3 生成结果
```bash
python table3_evaluation.py \
    --exp_root ./exp_result \
    --output table3_results
```

## 7. 常见问题

### 7.1 路径问题
- 确保所有数据集路径使用绝对路径
- 检查路径权限
- 验证数据集完整性

### 7.2 GPU 内存问题
- AASIST 需要 24GB+ VRAM
- 可以减小 batch_size 或使用梯度累积
- 监控 GPU 使用情况

### 7.3 存储空间
- 确保有足够空间存储实验结果
- 定期清理临时文件
- 考虑使用符号链接优化存储

## 8. 监控和日志

### 8.1 训练监控
```bash
# 使用 tensorboard 监控训练
tensorboard --logdir ./exp_result --port 6006
```

### 8.2 系统监控
```bash
# 监控 GPU 使用
nvidia-smi -l 1

# 监控磁盘空间
df -h
```

## 9. 示例服务器配置

假设服务器数据集路径为 `/data/speechfake/`：

```bash
# 更新所有配置文件
sed -i 's|"database_path": "./datasets/"|"database_path": "/data/speechfake/"|g' experiments/*.conf
sed -i 's|"database_path": "./datasets/"|"database_path": "/data/speechfake/"|g' config/*.conf
```

这样就能快速将所有配置文件更新为服务器路径。
