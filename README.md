# SpeechFake Baseline Implementation

Implementation of baseline models for the **SpeechFake** bilingual audio deepfake detection dataset, including AASIST and W2V+AASIST architectures.

## Overview

This repository provides:
- **AASIST**: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks
- **W2V+AASIST**: Wav2Vec2 XLS-R as frontend + AASIST as backend classifier
- Unified data loader for multiple datasets
- Multi-dataset evaluation scripts
- Experiment configurations for reproducing paper results

## Supported Datasets

- **SpeechFake Bilingual Dataset (BD)** - Primary dataset ([ModelScope](https://www.modelscope.cn/datasets/inclusionAI/SPEECHFAKE))
- **ASVspoof2019 LA** - Benchmark comparison
- **In-the-Wild (ITW)** - Cross-dataset evaluation
- **FakeOrReal (FOR)** - Cross-dataset evaluation

## Quick Start

### Installation

```bash
git clone https://github.com/yourusername/speechfake-baselines.git
cd speechfake-baselines
pip install -r requirements.txt
```

### Data Preparation

```bash
# Analyze dataset structure
python prepare_speechfake_data.py --data_dir /path/to/SPEECHFAKE/ --analyze

# Create CSV files
python prepare_speechfake_data.py \
    --data_dir /path/to/SPEECHFAKE/train/ \
    --output_csv train.csv \
    --audio_dir audio
```

### Training

```bash
# Train W2V+AASIST with exact paper parameters (Table 9)
python main.py --config experiments/paper_reproduction.conf

# Train AASIST with exact paper parameters (Table 9)
python main.py --config config/AASIST.conf

# Train W2V+AASIST on full Bilingual Dataset (example)
python main.py --config experiments/exp_bd_w2v_aasist.conf
```

**Paper Parameters (Table 9):**
- **W2V+AASIST**: Batch=512, LR=1e-6, Epochs=50, Weight Decay=1e-4
- **AASIST**: Batch=1024, LR=1e-4, Epochs=50, Weight Decay=1e-4
- **Loss**: Weighted CE (0.9 real, 0.1 fake)
- **Data Split**: 6:1:3 (train:dev:test) for SpeechFake BD
- **Audio**: 4s@16kHz, zero-pad + random/center crop
- **Wav2Vec2**: XLS-R-300M, freeze 95% layers, fine-tune last layers

### Evaluation

```bash
python speechfake_evaluation.py \
    --model_path exp_result/model/weights/best.pth \
    --config experiments/exp_bd_w2v_aasist.conf \
    --data_root /path/to/datasets/ \
    --test_sets BD_test BD_EN_test BD_CN_test ASV19_eval \
    --output results.json
```

## Project Structure

```
speechfake-baselines/
├── models/                      # Model implementations
│   ├── AASIST.py               # AASIST model
│   ├── W2VAASIST.py            # W2V+AASIST model
│   └── ...
├── config/                      # Model configurations
├── experiments/                 # Experiment configs
├── main.py                      # Training script
├── speechfake_dataloader.py     # Data loader
├── speechfake_evaluation.py     # Evaluation script
├── prepare_speechfake_data.py   # Data preprocessing
└── requirements.txt             # Dependencies
```

## 🔬 Model Architectures

### AASIST
- Heterogeneous stacking graph attention network
- Captures spoofing artifacts in temporal and spectral domains
- ~500K trainable parameters

### W2V+AASIST
- **Frontend**: Wav2Vec2 XLS-R-300M (frozen)
- **Backend**: AASIST graph attention network
- ~300.5M total parameters (~500K trainable)

## 💻 Hardware Requirements

- **GPU**: NVIDIA A800 or equivalent (24GB+ VRAM recommended)
- **RAM**: 32GB+
- **Storage**: 500GB+ for datasets

## 📖 Citation

```bibtex
@article{speechfake2024,
  title={SpeechFake: A Large-Scale Bilingual Audio Deepfake Detection Dataset},
  author={...},
  journal={ICLR},
  year={2025}
}

@inproceedings{jung2022aasist,
  title={AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks},
  author={Jung, Jee-weon and Heo, Hee-Soo and Tak, Hemlata and Shim, Hye-jin and Chung, Joon Son and Lee, Bong-Jin and Yu, Ha-Jin and Evans, Nicholas},
  booktitle={ICASSP},
  year={2022}
}
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

Built upon:
- [AASIST Official Implementation](https://github.com/clovaai/aasist)
- [ASVspoof 2021 Baseline](https://github.com/asvspoof-challenge/2021)
- Hugging Face Transformers
