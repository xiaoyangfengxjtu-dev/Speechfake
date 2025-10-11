# SpeechFake Experiments Configuration

This directory contains configuration files for reproducing experiments from the SpeechFake paper.

## Paper Parameters (Table 9)

Our configurations strictly follow the training parameters from the paper:

### W2V+AASIST
- **Batch Size**: 512
- **Learning Rate**: 1e-6 (very small due to pretrained Wav2Vec2)
- **Epochs**: 50
- **Weight Decay**: 1e-4
- **Loss**: Weighted Cross Entropy (0.9 real, 0.1 fake)

### AASIST
- **Batch Size**: 1024 (requires large GPU memory)
- **Learning Rate**: 1e-4
- **Epochs**: 50
- **Weight Decay**: 1e-4
- **Loss**: Weighted Cross Entropy (0.9 real, 0.1 fake)

### Key Configuration Files
- `paper_reproduction.conf` - W2V+AASIST with exact paper parameters
- `exp_bd_w2v_aasist.conf` - Example BD training configuration
- `../config/AASIST.conf` - AASIST with paper parameters

### Table 3 Overall Performance Experiments
- `table3_aasist_*.conf` - AASIST experiments (ASV19, BD, BD-EN, BD-CN)
- `table3_w2v_*.conf` - W2V+AASIST experiments (ASV19, BD, BD-EN, BD-CN)
- See `../TABLE3_EXPERIMENTS.md` for complete experiment guide

## Experiment Structure

### 4.2 Overall Performance (Table 3)
Trains models on different datasets and evaluates across multiple test sets.

**Training sets:**
- `exp_asv19_aasist.conf` - AASIST trained on ASVspoof2019
- `exp_asv19_w2v.conf` - W2V+AASIST trained on ASVspoof2019
- `exp_bd_aasist.conf` - AASIST trained on full BD
- `exp_bd_w2v.conf` - W2V+AASIST trained on full BD
- `exp_bd_en_aasist.conf` - AASIST trained on BD-EN
- `exp_bd_en_w2v.conf` - W2V+AASIST trained on BD-EN
- `exp_bd_cn_aasist.conf` - AASIST trained on BD-CN
- `exp_bd_cn_w2v.conf` - W2V+AASIST trained on BD-CN

**Test sets:**
- BD, BD-EN, BD-CN, ASV19, ITW, FOR

### 4.3 Cross-Generator Performance (Table 4)
Evaluates generalization across different generator types.

**Training sets:**
- `exp_bd_tts_aasist.conf` - Trained on BD-TTS
- `exp_bd_vc_aasist.conf` - Trained on BD-VC
- `exp_bd_nv_aasist.conf` - Trained on BD-NV
- `exp_bd_tts_w2v.conf` - W2V+AASIST trained on BD-TTS
- `exp_bd_vc_w2v.conf` - W2V+AASIST trained on BD-VC
- `exp_bd_nv_w2v.conf` - W2V+AASIST trained on BD-NV

**Test sets:**
- BD-TTS, BD-VC, BD-NV, BD, BD-UT

### 4.4 Cross-Lingual Performance (Table 5)
Evaluates performance across different languages.

**Training:**
- Train on English + Chinese multilingual dataset

**Test sets:**
- en, zh, es, fr, hi, ja, ko, fa, it, others

### 4.5 Cross-Speaker Performance (Table 6)
Evaluates speaker generalization using TorToiSe TTS.

**Training:**
- Subset with 100 real + 10 fake speakers

**Test trials:**
1. Seen real & fake speakers
2. Unseen real & fake speakers
3. Unseen real & seen fake speakers
4. Seen real & unseen fake speakers
5. Mixed seen/unseen speakers

## Usage

### Training
```bash
python main.py --config experiments/exp_bd_w2v.conf
```

### Evaluation
```bash
python speechfake_evaluation.py \
    --model_path exp_result/model_weights.pth \
    --config experiments/exp_bd_w2v.conf \
    --data_root /path/to/datasets \
    --test_sets BD_test BD_EN_test BD_CN_test ASV19_eval ITW_test FOR_test \
    --output results_bd_w2v.json
```

### Batch Evaluation Script
```bash
bash run_all_experiments.sh
```

## Results Format

Results are saved in JSON format:
```json
{
  "BD_test": 3.54,
  "BD_EN_test": 3.55,
  "BD_CN_test": 2.83,
  "ASV19_eval": 2.91,
  "ITW_test": 2.01,
  "FOR_test": 6.00
}
```

## Directory Structure

```
experiments/
├── README.md                    # This file
├── overall_performance/         # Configs for Section 4.2
├── cross_generator/             # Configs for Section 4.3
├── cross_lingual/               # Configs for Section 4.4
├── cross_speaker/               # Configs for Section 4.5
└── run_all_experiments.sh       # Batch run script
```

