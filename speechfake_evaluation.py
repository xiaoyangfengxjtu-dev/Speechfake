"""
Evaluation script for SpeechFake experiments
Computes EER across multiple test sets as described in the paper
"""

import os
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader

from speechfake_dataloader import get_dataloader, DATASET_CONFIGS
from evaluation import compute_eer


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str = "cuda"
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Evaluate model on a dataset and return scores
    
    Args:
        model: Trained model
        dataloader: DataLoader for evaluation
        device: Device to use
    
    Returns:
        bonafide_scores: Scores for bonafide samples
        spoof_scores: Scores for spoof samples
        file_ids: List of file IDs
    """
    model.eval()
    
    all_scores = []
    all_labels = []
    all_file_ids = []
    
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 2:
                waveforms, labels_or_ids = batch
                
                # Check if labels_or_ids are strings (eval mode) or integers (train mode)
                if isinstance(labels_or_ids[0], str):
                    # Evaluation mode: labels_or_ids are file IDs
                    file_ids = labels_or_ids
                    labels = None
                else:
                    # Training mode: labels_or_ids are actual labels
                    labels = labels_or_ids.numpy()
                    file_ids = [f"file_{i}" for i in range(len(labels))]
                
                waveforms = waveforms.to(device)
                
                # Forward pass
                _, outputs = model(waveforms)
                
                # Get scores for bonafide class (index 1)
                scores = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                
                all_scores.extend(scores)
                if labels is not None:
                    all_labels.extend(labels)
                all_file_ids.extend(file_ids)
    
    all_scores = np.array(all_scores)
    
    # If we have labels, separate scores
    if all_labels:
        all_labels = np.array(all_labels)
        bonafide_scores = all_scores[all_labels == 1]
        spoof_scores = all_scores[all_labels == 0]
    else:
        # No labels available (pure evaluation mode)
        bonafide_scores = None
        spoof_scores = None
    
    return bonafide_scores, spoof_scores, all_file_ids


def compute_eer_from_dataloader(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str = "cuda"
) -> float:
    """
    Compute EER from a dataloader
    
    Returns:
        EER in percentage
    """
    bonafide_scores, spoof_scores, _ = evaluate_model(model, dataloader, device)
    
    if bonafide_scores is None or spoof_scores is None:
        raise ValueError("Cannot compute EER without labels")
    
    eer, _ = compute_eer(bonafide_scores, spoof_scores)
    return eer * 100  # Convert to percentage


def run_multi_dataset_evaluation(
    model: torch.nn.Module,
    data_root: str,
    test_configs: List[str],
    batch_size: int = 32,
    num_workers: int = 4,
    device: str = "cuda",
    output_file: str = None
) -> Dict[str, float]:
    """
    Evaluate model on multiple test sets
    
    Args:
        model: Trained model
        data_root: Root directory containing all datasets
        test_configs: List of test dataset names (keys in DATASET_CONFIGS)
        batch_size: Batch size for evaluation
        num_workers: Number of workers
        device: Device to use
        output_file: Optional file to save results
    
    Returns:
        Dictionary of test set names to EER values
    """
    results = {}
    
    print("\n" + "="*80)
    print("Multi-Dataset Evaluation")
    print("="*80 + "\n")
    
    for config_name in test_configs:
        if config_name not in DATASET_CONFIGS:
            print(f"Warning: Config '{config_name}' not found, skipping...")
            continue
        
        config = DATASET_CONFIGS[config_name]
        csv_file = os.path.join(data_root, config["csv"])
        audio_dir = os.path.join(data_root, config["audio_dir"])
        
        if not os.path.exists(csv_file):
            print(f"Warning: CSV file not found: {csv_file}, skipping...")
            continue
        
        print(f"\nEvaluating on: {config_name}")
        print(f"  CSV: {csv_file}")
        print(f"  Audio dir: {audio_dir}")
        
        try:
            # Create dataloader
            dataloader = get_dataloader(
                csv_file=csv_file,
                audio_dir=audio_dir,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=False,
                subset_filter=config["filter"],
                is_eval=False  # Need labels for EER computation
            )
            
            # Compute EER
            eer = compute_eer_from_dataloader(model, dataloader, device)
            results[config_name] = eer
            
            print(f"  EER: {eer:.2f}%")
            
        except Exception as e:
            print(f"  Error: {e}")
            results[config_name] = None
    
    # Save results if output file specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("Evaluation Summary")
    print("="*80)
    for test_name, eer in results.items():
        if eer is not None:
            print(f"{test_name:30s}: {eer:6.2f}%")
        else:
            print(f"{test_name:30s}: FAILED")
    print("="*80 + "\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="SpeechFake Multi-Dataset Evaluation")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model weights")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to model configuration file")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory containing all datasets")
    parser.add_argument("--test_sets", type=str, nargs='+',
                        default=["BD_test", "BD_EN_test", "BD_CN_test", "ASV19_eval", "ITW_test", "FOR_test"],
                        help="List of test sets to evaluate")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--output", type=str, default="evaluation_results.json",
                        help="Output file for results")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Load model configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    model_config = config["model_config"]
    
    # Import and create model
    from importlib import import_module
    module = import_module(f"models.{model_config['architecture']}")
    Model = getattr(module, "Model")
    
    # Create and load model
    device = args.device if torch.cuda.is_available() else "cpu"
    model = Model(model_config).to(device)
    
    # Load weights
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"Model loaded from: {args.model_path}")
    
    # Run evaluation
    results = run_multi_dataset_evaluation(
        model=model,
        data_root=args.data_root,
        test_configs=args.test_sets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        output_file=args.output
    )
    
    return results


if __name__ == "__main__":
    main()

