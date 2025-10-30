"""
Unified DataLoader for SpeechFake Experiments
Supports multiple datasets: ASVspoof2019, SpeechFake BD, ITW, FakeOrReal
"""

import os
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import numpy as np
from scipy.io.wavfile import read as scipy_wavread


def _load_wav_via_scipy(path):
    """Load WAV file using scipy (more robust for various formats)"""
    sr, data = scipy_wavread(path)
    # Convert to float32
    if data.dtype in [np.int16, np.int32]:
        data = data.astype(np.float32) / float(np.iinfo(data.dtype).max)
    elif data.dtype == np.float32:
        pass
    else:
        data = data.astype(np.float32)
    # Convert stereo to mono
    if data.ndim == 2:
        data = data.mean(axis=1)
    # Return torch tensor and sample rate
    return torch.from_numpy(data), int(sr)


class SpeechFakeDataset(Dataset):
    """
    Unified dataset class for speech deepfake detection
    Supports multiple dataset formats and filtering
    """
    def __init__(
        self,
        csv_file: str,
        audio_dir: str,
        sample_rate: int = 16000,
        max_length: int = 64600,  # ~4 seconds at 16kHz
        subset_filter: Optional[Dict[str, List[str]]] = None,
        transform: Optional[callable] = None
    ):
        """
        Args:
            csv_file: Path to CSV file with columns [file, label, ...]
            audio_dir: Root directory containing audio files
            sample_rate: Target sample rate
            max_length: Maximum audio length in samples
            subset_filter: Dict to filter data, e.g., {'language': ['en', 'zh']}
            transform: Optional transform to apply to waveform
        """
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.transform = transform
        
        # Load CSV
        self.data = pd.read_csv(csv_file)
        
        # Apply subset filter if provided
        if subset_filter:
            for column, values in subset_filter.items():
                if column in self.data.columns:
                    self.data = self.data[self.data[column].isin(values)]
        
        # Label mapping
        self.label_map = {"bonafide": 1, "spoof": 0}
        
        print(f"Loaded dataset from {csv_file}")
        print(f"  Total samples: {len(self.data)}")
        if subset_filter:
            print(f"  Applied filter: {subset_filter}")
        
        # Print label distribution
        if 'label' in self.data.columns:
            label_counts = self.data['label'].value_counts()
            print(f"  Label distribution: {dict(label_counts)}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            waveform: (audio_length,) tensor
            label: 0 for spoof, 1 for bonafide
        """
        row = self.data.iloc[idx]
        
        # Get audio file path
        file_path = self.audio_dir / row['file']
        
        # Get label
        label_str = row['label']
        label = self.label_map.get(label_str, 0)
        
        # Load audio
        waveform, sr = _load_wav_via_scipy(str(file_path))
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # waveform is already 1D from scipy loader, no need to convert to mono or squeeze
        
        # Pad or truncate to fixed length (as per paper: "Chunk or pad to 4s")
        current_length = waveform.shape[0]
        if current_length < self.max_length:
            # Zero padding for short audio
            padding = self.max_length - current_length
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        else:
            # Chunk strategy: random crop for training, center crop for evaluation
            if self.transform is not None:  # Training mode
                start = torch.randint(0, current_length - self.max_length + 1, (1,)).item()
            else:  # Evaluation mode
                start = (current_length - self.max_length) // 2
            waveform = waveform[start:start + self.max_length]
        
        # Apply transform if provided
        if self.transform:
            waveform = self.transform(waveform)
        
        return waveform, label


class SpeechFakeEvalDataset(Dataset):
    """
    Evaluation dataset that returns file IDs for score generation
    Compatible with existing evaluation pipeline
    """
    def __init__(
        self,
        csv_file: str,
        audio_dir: str,
        sample_rate: int = 16000,
        max_length: int = 64600,
        subset_filter: Optional[Dict[str, List[str]]] = None
    ):
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.max_length = max_length
        
        # Load CSV
        self.data = pd.read_csv(csv_file)
        
        # Apply subset filter
        if subset_filter:
            for column, values in subset_filter.items():
                if column in self.data.columns:
                    self.data = self.data[self.data[column].isin(values)]
        
        print(f"Loaded eval dataset from {csv_file}")
        print(f"  Total samples: {len(self.data)}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Returns:
            waveform: (audio_length,) tensor
            file_id: string identifier for the file
        """
        row = self.data.iloc[idx]
        file_path = self.audio_dir / row['file']
        file_id = Path(row['file']).stem  # File name without extension
        
        # Load and process audio (same as training)
        waveform, sr = _load_wav_via_scipy(str(file_path))
        
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # waveform is already 1D from scipy loader
        
        # Center crop/pad for evaluation
        current_length = waveform.shape[0]
        if current_length < self.max_length:
            padding = self.max_length - current_length
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        else:
            start = (current_length - self.max_length) // 2
            waveform = waveform[start:start + self.max_length]
        
        return waveform, file_id


def create_speechfake_scores_file(
    csv_file: str,
    output_file: str
):
    """
    Create a scores file compatible with evaluation.py from CSV
    Format: file_id source label score
    """
    df = pd.read_csv(csv_file)
    
    with open(output_file, 'w') as f:
        for _, row in df.iterrows():
            file_id = Path(row['file']).stem
            # Source could be generator type or '-'
            source = row.get('generator', '-')
            label = row['label']
            # Placeholder score, will be replaced during evaluation
            score = 0.0
            f.write(f"{file_id} {source} {label} {score}\n")
    
    print(f"Created trial file: {output_file}")


def get_dataloader(
    csv_file: str,
    audio_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    sample_rate: int = 16000,
    max_length: int = 64600,
    subset_filter: Optional[Dict[str, List[str]]] = None,
    is_eval: bool = False
) -> DataLoader:
    """
    Create a DataLoader for training or evaluation
    
    Args:
        csv_file: Path to CSV file
        audio_dir: Root directory for audio files
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle: Whether to shuffle data
        sample_rate: Target sample rate
        max_length: Maximum audio length
        subset_filter: Filter for dataset subsets
        is_eval: If True, return file IDs instead of labels
    
    Returns:
        DataLoader
    """
    if is_eval:
        dataset = SpeechFakeEvalDataset(
            csv_file=csv_file,
            audio_dir=audio_dir,
            sample_rate=sample_rate,
            max_length=max_length,
            subset_filter=subset_filter
        )
    else:
        dataset = SpeechFakeDataset(
            csv_file=csv_file,
            audio_dir=audio_dir,
            sample_rate=sample_rate,
            max_length=max_length,
            subset_filter=subset_filter
        )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False if is_eval else True
    )
    
    return dataloader


# Dataset configuration templates for experiments in the paper
DATASET_CONFIGS = {
    # Training sets
    "ASV19_train": {
        "csv": "ASVspoof2019_LA_train.csv",
        "audio_dir": "ASVspoof2019_LA_train/",
        "filter": None
    },
    "BD_train": {
        "csv": "SpeechFake_train.csv",
        "audio_dir": "SpeechFake/audio/",
        "filter": None
    },
    "BD_EN_train": {
        "csv": "SpeechFake_train.csv",
        "audio_dir": "SpeechFake/audio/",
        "filter": {"language": ["en"]}
    },
    "BD_CN_train": {
        "csv": "SpeechFake_train.csv",
        "audio_dir": "SpeechFake/audio/",
        "filter": {"language": ["zh"]}
    },
    "BD_TTS_train": {
        "csv": "SpeechFake_train.csv",
        "audio_dir": "SpeechFake/audio/",
        "filter": {"generator_type": ["TTS"]}
    },
    "BD_VC_train": {
        "csv": "SpeechFake_train.csv",
        "audio_dir": "SpeechFake/audio/",
        "filter": {"generator_type": ["VC"]}
    },
    "BD_NV_train": {
        "csv": "SpeechFake_train.csv",
        "audio_dir": "SpeechFake/audio/",
        "filter": {"generator_type": ["NV"]}
    },
    
    # Test sets
    "ASV19_eval": {
        "csv": "ASVspoof2019_LA_eval.csv",
        "audio_dir": "ASVspoof2019_LA_eval/",
        "filter": None
    },
    "BD_test": {
        "csv": "SpeechFake_test.csv",
        "audio_dir": "SpeechFake/audio/",
        "filter": None
    },
    "BD_EN_test": {
        "csv": "SpeechFake_test.csv",
        "audio_dir": "SpeechFake/audio/",
        "filter": {"language": ["en"]}
    },
    "BD_CN_test": {
        "csv": "SpeechFake_test.csv",
        "audio_dir": "SpeechFake/audio/",
        "filter": {"language": ["zh"]}
    },
    "ITW_test": {
        "csv": "InTheWild_test.csv",
        "audio_dir": "InTheWild/audio/",
        "filter": None
    },
    "FOR_test": {
        "csv": "FakeOrReal_test.csv",
        "audio_dir": "FakeOrReal/audio/",
        "filter": None
    }
}


if __name__ == "__main__":
    # Test the dataloader
    print("\n" + "="*60)
    print("Testing SpeechFake DataLoader")
    print("="*60 + "\n")
    
    # Example usage
    csv_path = "path/to/your/train.csv"
    audio_dir = "path/to/your/audio/"
    
    # Test training dataloader
    print("Creating training dataloader...")
    train_loader = get_dataloader(
        csv_file=csv_path,
        audio_dir=audio_dir,
        batch_size=4,
        num_workers=0,
        shuffle=True,
        is_eval=False
    )
    
    print(f"\nDataLoader created successfully!")
    print(f"Number of batches: {len(train_loader)}")
    
    # Test one batch
    for waveforms, labels in train_loader:
        print(f"\nBatch shapes:")
        print(f"  Waveforms: {waveforms.shape}")
        print(f"  Labels: {labels.shape}")
        print(f"  Label values: {labels}")
        break
    
    print("\n" + "="*60)
    print("DataLoader test completed!")
    print("="*60 + "\n")

