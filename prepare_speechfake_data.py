"""
Data preparation script for SpeechFake dataset
Handles different possible formats and creates unified CSV files
"""

import os
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict
import json


def detect_dataset_format(data_dir: str) -> str:
    """
    Auto-detect the format of the dataset
    
    Returns:
        Format type: 'csv', 'json', 'protocol', or 'unknown'
    """
    data_path = Path(data_dir)
    
    # Check for CSV files
    csv_files = list(data_path.glob("*.csv")) + list(data_path.glob("**/*.csv"))
    if csv_files:
        return 'csv'
    
    # Check for JSON files
    json_files = list(data_path.glob("*.json")) + list(data_path.glob("**/*.json"))
    if json_files:
        return 'json'
    
    # Check for protocol files (.txt)
    txt_files = list(data_path.glob("*.txt")) + list(data_path.glob("**/*.txt"))
    if txt_files:
        return 'protocol'
    
    return 'unknown'


def load_csv_format(csv_file: str) -> pd.DataFrame:
    """
    Load CSV format dataset
    Expected columns: file, label, [language], [generator], [generator_type], etc.
    """
    df = pd.read_csv(csv_file)
    
    # Check required columns
    required_cols = ['file', 'label']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in CSV. Columns: {df.columns.tolist()}")
    
    print(f"Loaded CSV with columns: {df.columns.tolist()}")
    return df


def load_json_format(json_file: str) -> pd.DataFrame:
    """
    Load JSON format dataset
    Converts to DataFrame
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle different JSON structures
    if isinstance(data, list):
        df = pd.DataFrame(data)
    elif isinstance(data, dict):
        # Might be {file_id: {metadata}} format
        records = []
        for file_id, metadata in data.items():
            record = {'file': file_id}
            record.update(metadata)
            records.append(record)
        df = pd.DataFrame(records)
    else:
        raise ValueError(f"Unsupported JSON structure: {type(data)}")
    
    print(f"Loaded JSON with columns: {df.columns.tolist()}")
    return df


def load_protocol_format(protocol_file: str) -> pd.DataFrame:
    """
    Load ASVspoof protocol format
    Format: speaker_id file_id system_id label
    or: file_id label
    """
    records = []
    
    with open(protocol_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            
            if len(parts) == 4:
                # ASVspoof format: speaker_id file_id system_id label
                speaker_id, file_id, system_id, label = parts
                records.append({
                    'file': file_id,
                    'label': label,
                    'speaker_id': speaker_id,
                    'system_id': system_id
                })
            elif len(parts) == 2:
                # Simple format: file_id label
                file_id, label = parts
                records.append({
                    'file': file_id,
                    'label': label
                })
            else:
                print(f"Warning: Unexpected line format: {line.strip()}")
    
    df = pd.DataFrame(records)
    print(f"Loaded protocol file with columns: {df.columns.tolist()}")
    return df


def create_unified_csv(
    data_dir: str,
    output_csv: str,
    audio_dir: str = "audio",
    add_file_extension: str = ".wav"
) -> pd.DataFrame:
    """
    Create a unified CSV file from various formats
    
    Args:
        data_dir: Directory containing metadata files
        output_csv: Output CSV file path
        audio_dir: Subdirectory containing audio files (relative to data_dir)
        add_file_extension: File extension to add if not present
    
    Returns:
        DataFrame with unified format
    """
    print(f"\nProcessing dataset in: {data_dir}")
    
    # Detect format
    format_type = detect_dataset_format(data_dir)
    print(f"Detected format: {format_type}")
    
    # Load data based on format
    if format_type == 'csv':
        csv_files = list(Path(data_dir).glob("*.csv"))
        if not csv_files:
            csv_files = list(Path(data_dir).glob("**/*.csv"))
        print(f"Found CSV files: {[f.name for f in csv_files]}")
        
        # Use the first CSV file (or you can specify which one)
        df = load_csv_format(str(csv_files[0]))
        
    elif format_type == 'json':
        json_files = list(Path(data_dir).glob("*.json"))
        if not json_files:
            json_files = list(Path(data_dir).glob("**/*.json"))
        print(f"Found JSON files: {[f.name for f in json_files]}")
        
        df = load_json_format(str(json_files[0]))
        
    elif format_type == 'protocol':
        txt_files = list(Path(data_dir).glob("*.txt"))
        if not txt_files:
            txt_files = list(Path(data_dir).glob("**/*.txt"))
        print(f"Found protocol files: {[f.name for f in txt_files]}")
        
        df = load_protocol_format(str(txt_files[0]))
        
    else:
        raise ValueError(f"Unknown dataset format in {data_dir}")
    
    # Normalize file paths
    if 'file' in df.columns:
        # Add audio directory prefix if not present
        df['file'] = df['file'].apply(lambda x: 
            str(Path(audio_dir) / x) if not str(x).startswith(audio_dir) else x
        )
        
        # Add file extension if not present
        if add_file_extension:
            df['file'] = df['file'].apply(lambda x:
                x if x.endswith(tuple(['.wav', '.flac', '.mp3'])) else x + add_file_extension
            )
    
    # Normalize labels
    if 'label' in df.columns:
        # Convert to lowercase
        df['label'] = df['label'].str.lower()
        
        # Standardize label names
        label_map = {
            'bonafide': 'bonafide',
            'bona-fide': 'bonafide',
            'real': 'bonafide',
            'genuine': 'bonafide',
            'spoof': 'spoof',
            'fake': 'spoof',
            'deepfake': 'spoof'
        }
        df['label'] = df['label'].map(lambda x: label_map.get(x, x))
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"\nSaved unified CSV to: {output_csv}")
    print(f"Total samples: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    if 'label' in df.columns:
        print(f"Label distribution:\n{df['label'].value_counts()}")
    
    return df


def analyze_dataset(data_dir: str):
    """
    Analyze dataset structure and print information
    Useful for understanding the dataset before creating loader
    """
    data_path = Path(data_dir)
    
    print("\n" + "="*80)
    print(f"Dataset Analysis: {data_dir}")
    print("="*80)
    
    # List all files
    print("\nDirectory Structure:")
    for root, dirs, files in os.walk(data_path):
        level = root.replace(str(data_path), '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        
        # Show first few files in each directory
        subindent = ' ' * 2 * (level + 1)
        for file in files[:3]:  # Show first 3 files
            print(f'{subindent}{file}')
        if len(files) > 3:
            print(f'{subindent}... and {len(files) - 3} more files')
    
    # Count audio files
    audio_extensions = ['.wav', '.flac', '.mp3', '.m4a']
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(list(data_path.rglob(f"*{ext}")))
    
    print(f"\n🎵 Audio Files: {len(audio_files)} files found")
    if audio_files:
        print(f"   Extensions: {set(f.suffix for f in audio_files)}")
    
    # Find metadata files
    metadata_files = {
        'CSV': list(data_path.rglob("*.csv")),
        'JSON': list(data_path.rglob("*.json")),
        'TXT': list(data_path.rglob("*.txt"))
    }
    
    print(f"\nMetadata Files:")
    for file_type, files in metadata_files.items():
        if files:
            print(f"   {file_type}: {len(files)} files")
            for f in files:
                print(f"      - {f.relative_to(data_path)}")
    
    # Try to read first metadata file
    for file_type, files in metadata_files.items():
        if files and file_type == 'CSV':
            print(f"\nSample from {files[0].name}:")
            try:
                df = pd.read_csv(files[0], nrows=5)
                print(df.to_string())
                print(f"\nColumns: {df.columns.tolist()}")
            except Exception as e:
                print(f"Error reading CSV: {e}")
            break
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare SpeechFake dataset")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing the dataset")
    parser.add_argument("--analyze", action="store_true",
                        help="Only analyze dataset structure without creating CSV")
    parser.add_argument("--output_csv", type=str, default="unified_dataset.csv",
                        help="Output CSV file path")
    parser.add_argument("--audio_dir", type=str, default="audio",
                        help="Subdirectory containing audio files")
    
    args = parser.parse_args()
    
    if args.analyze:
        # Just analyze the dataset
        analyze_dataset(args.data_dir)
    else:
        # Create unified CSV
        df = create_unified_csv(
            data_dir=args.data_dir,
            output_csv=args.output_csv,
            audio_dir=args.audio_dir
        )
        
        print("\nDataset preparation complete!")
        print(f"You can now use this CSV with the dataloader:")
        print(f"  python -c 'from speechfake_dataloader import get_dataloader; ...")

