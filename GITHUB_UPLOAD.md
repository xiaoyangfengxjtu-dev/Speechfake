# GitHub 上传指南

## 当前状态

所有代码已经准备就绪，已提交到本地 Git 仓库：

- **提交信息**: Complete SpeechFake baseline implementation with Table 3 experiments
- **文件更改**: 34个文件，3345行新增代码
- **主要功能**: 完整的 SpeechFake 基线实现，包含 Table 3 实验复现

## 上传步骤

### 1. 在 GitHub 上创建新仓库

1. 访问 https://github.com/new
2. 仓库名称建议: `speechfake-baselines` 或 `speechfake-table3-reproduction`
3. 描述: "SpeechFake Baseline Implementation - AASIST & W2V+AASIST with Table 3 Experiments"
4. 选择 Public 或 Private
5. **不要**初始化 README, .gitignore, 或 license (我们已经有了)
6. 点击 "Create repository"

### 2. 更新远程仓库地址

```bash
# 方法1: 更改现有远程地址
git remote set-url origin https://github.com/YOUR_USERNAME/REPOSITORY_NAME.git

# 方法2: 添加新的远程地址
git remote add new-origin https://github.com/YOUR_USERNAME/REPOSITORY_NAME.git
```

### 3. 推送代码到 GitHub

```bash
# 推送到主分支
git push -u origin main

# 或者如果使用新远程地址
git push -u new-origin main
```

### 4. 验证上传

1. 访问你的 GitHub 仓库页面
2. 确认所有文件都已上传
3. 检查 README.md 是否正确显示
4. 验证项目结构完整

## 项目包含的内容

### 两个完整的基线模型实现

#### 1. AASIST 基线
- **模型文件**: `models/AASIST.py` - 完整的图注意力网络实现
- **配置**: `config/AASIST.conf` - 论文对齐的参数配置
- **实验**: 4个 Table 3 实验配置 (ASV19, BD, BD-EN, BD-CN)
- **架构**: Heterogeneous stacking graph attention network with spectro-temporal attention

#### 2. W2V+AASIST 基线  
- **模型文件**: `models/W2VAASIST.py` - Wav2Vec2 + AASIST 集成实现
- **配置**: `config/SpeechFake_W2V_AASIST.conf` - 论文对齐的参数配置
- **实验**: 4个 Table 3 实验配置 (ASV19, BD, BD-EN, BD-CN)
- **架构**: Wav2Vec2 XLS-R-300M 前端 + AASIST 后端分类器
- **特点**: 冻结 95% Wav2Vec2 层，只微调最后 transformer block

### 实验配置 (8个)
- `experiments/table3_aasist_*.conf` - AASIST 实验 (4个)
- `experiments/table3_w2v_*.conf` - W2V+AASIST 实验 (4个)

### 工具脚本
- `run_table3_experiments.py` - 批量运行所有实验
- `table3_evaluation.py` - 生成结果表格和原始数据
- `update_server_paths.py` - 服务器路径更新工具
- `check_server_config.py` - 配置检查工具

### 文档
- `README.md` - 项目主文档
- `TABLE3_EXPERIMENTS.md` - Table 3 实验指南
- `SERVER_DEPLOYMENT.md` - 服务器部署指南
- `experiments/README.md` - 实验配置说明

## 服务器部署流程

上传到 GitHub 后，在服务器上的部署流程：

```bash
# 1. 克隆仓库
git clone https://github.com/YOUR_USERNAME/REPOSITORY_NAME.git
cd REPOSITORY_NAME

# 2. 安装依赖
pip install -r requirements.txt

# 3. 准备数据集
# 下载所有数据集到指定目录

# 4. 更新配置路径
python update_server_paths.py --new_path /path/to/your/datasets/

# 5. 检查配置
python check_server_config.py

# 6. 运行实验
python run_table3_experiments.py
```

## 注意事项

1. **仓库大小**: 项目不包含大文件（模型权重、数据集），上传很快
2. **敏感信息**: 确认没有包含 API 密钥或其他敏感信息
3. **许可证**: 项目使用 MIT 许可证
4. **依赖**: 确保 requirements.txt 包含所有必要依赖

## 故障排除

### 如果推送失败
```bash
# 检查远程地址
git remote -v

# 重新设置远程地址
git remote set-url origin https://github.com/YOUR_USERNAME/REPOSITORY_NAME.git

# 强制推送 (谨慎使用)
git push -f origin main
```

### 如果文件太大
项目已经通过 .gitignore 排除了大文件：
- 模型权重文件 (*.pth)
- 实验结果目录 (exp_result/)
- 数据集目录
- Python 缓存文件

## 完成后的下一步

1. **测试克隆**: 在另一台机器上测试克隆和运行
2. **文档完善**: 根据需要更新 README
3. **Issue 跟踪**: 设置 GitHub Issues 用于问题跟踪
4. **CI/CD**: 考虑设置 GitHub Actions 进行自动化测试

上传完成后，你就可以在服务器上克隆并开始运行 Table 3 实验了！
