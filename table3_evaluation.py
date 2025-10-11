#!/usr/bin/env python3
"""
Table 3 Overall Performance Evaluation Script
Generates performance comparison table and raw data similar to the paper's Table 3
"""

import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

def load_model_results(exp_dir: Path, config_name: str) -> Dict[str, Dict[str, float]]:
    """
    Load detailed results (EER, t-DCF) from experiment directory
    Returns: {test_set: {'eer': value, 'tdcf': value}}
    """
    results = {}
    
    # Look for evaluation results in the experiment directory
    metric_files = list(exp_dir.glob("**/t-DCF_EER.txt"))
    
    for metric_file in metric_files:
        # Extract test set name from path
        test_set = metric_file.parent.name
        
        try:
            with open(metric_file, 'r') as f:
                content = f.read()
                # Parse EER and t-DCF from the file
                eer = None
                tdcf = None
                
                for line in content.split('\n'):
                    if 'EER:' in line:
                        eer = float(line.split('EER:')[1].strip())
                    elif 'min t-DCF:' in line or 't-DCF:' in line:
                        tdcf = float(line.split(':')[1].strip())
                
                if eer is not None:
                    results[test_set] = {
                        'eer': eer,
                        'tdcf': tdcf if tdcf is not None else None
                    }
                    
        except Exception as e:
            print(f"Warning: Could not load {metric_file}: {e}")
    
    return results

def create_performance_table(exp_root: Path) -> pd.DataFrame:
    """
    Create Table 3 performance comparison
    """
    # Define the table structure
    training_datasets = ['ASV19', 'BD', 'BD-EN', 'BD-CN']
    models = ['AASIST', 'W2V+AASIST']
    test_datasets_speechfake = ['BD', 'BD-EN', 'BD-CN']
    test_datasets_others = ['ASV19', 'ITW', 'FOR']
    
    # Initialize results dictionary
    results = []
    
    # Process each model and training dataset combination
    for model in models:
        for train_dataset in training_datasets:
            # Find experiment directory
            exp_pattern = f"*{train_dataset}_{model.lower().replace('+', '_')}_ep50*"
            exp_dirs = list(exp_root.glob(exp_pattern))
            
            if not exp_dirs:
                print(f"Warning: No experiment found for {model} trained on {train_dataset}")
                continue
                
            exp_dir = exp_dirs[0]  # Take the first match
            
            # Load results
            model_results = load_model_results(exp_dir, f"{train_dataset}_{model}")
            
            # Create row data
            row = {
                'Training Dataset': train_dataset,
                'Model': model
            }
            
            # Add SpeechFake test results (EER only for table)
            for test_set in test_datasets_speechfake:
                if test_set in model_results:
                    row[f'BD_{test_set}'] = model_results[test_set]['eer']
                else:
                    row[f'BD_{test_set}'] = 'N/A'
            
            # Add Others test results (EER only for table)
            for test_set in test_datasets_others:
                if test_set in model_results:
                    row[f'Others_{test_set}'] = model_results[test_set]['eer']
                else:
                    row[f'Others_{test_set}'] = 'N/A'
            
            results.append(row)
    
    # Create DataFrame
    columns = ['Training Dataset', 'Model']
    for test_set in test_datasets_speechfake:
        columns.append(f'BD_{test_set}')
    for test_set in test_datasets_others:
        columns.append(f'Others_{test_set}')
    
    df = pd.DataFrame(results, columns=columns)
    return df

def save_raw_data(exp_root: Path, output_file: str):
    """
    Save all raw experimental data (EER and t-DCF) to JSON file
    """
    training_datasets = ['ASV19', 'BD', 'BD-EN', 'BD-CN']
    models = ['AASIST', 'W2V+AASIST']
    
    raw_data = {}
    
    for model in models:
        raw_data[model] = {}
        
        for train_dataset in training_datasets:
            # Find experiment directory
            exp_pattern = f"*{train_dataset}_{model.lower().replace('+', '_')}_ep50*"
            exp_dirs = list(exp_root.glob(exp_pattern))
            
            if not exp_dirs:
                print(f"Warning: No experiment found for {model} trained on {train_dataset}")
                continue
                
            exp_dir = exp_dirs[0]
            
            # Load detailed results
            model_results = load_model_results(exp_dir, f"{train_dataset}_{model}")
            
            # Store raw data
            experiment_key = f"{train_dataset}_{model}"
            raw_data[model][experiment_key] = {
                'training_dataset': train_dataset,
                'model': model,
                'experiment_dir': str(exp_dir),
                'results': model_results
            }
    
    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(raw_data, f, indent=2)
    
    print(f"Raw experimental data saved to: {output_file}")
    return raw_data

def format_table_latex(df: pd.DataFrame) -> str:
    """
    Format the table for LaTeX output
    """
    latex = "\\begin{table}[h]\n"
    latex += "\\centering\n"
    latex += "\\caption{Performance evaluation (EER\\%) of different models trained on ASVspoof2019 (ASV19) or Bilingual Dataset (BD) across multiple test sets}\n"
    latex += "\\label{tab:overall_performance}\n"
    latex += "\\begin{tabular}{|c|c|ccc|ccc|}\n"
    latex += "\\hline\n"
    latex += "\\multirow{2}{*}{Training Dataset} & \\multirow{2}{*}{Model} & \\multicolumn{3}{c|}{Testing Dataset (SpeechFake)} & \\multicolumn{3}{c|}{Testing Dataset (Others)} \\\\\n"
    latex += "\\cline{3-8}\n"
    latex += "& & BD & BD-EN & BD-CN & ASV19 & ITW & FOR \\\\\n"
    latex += "\\hline\n"
    
    for _, row in df.iterrows():
        latex += f"{row['Training Dataset']} & {row['Model']} & "
        latex += f"{row['BD_BD']} & {row['BD_BD-EN']} & {row['BD_BD-CN']} & "
        latex += f"{row['Others_ASV19']} & {row['Others_ITW']} & {row['Others_FOR']} \\\\\n"
    
    latex += "\\hline\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"
    
    return latex

def format_table_markdown(df: pd.DataFrame) -> str:
    """
    Format the table for Markdown output
    """
    md = "## Table 3: Overall Performance Evaluation (EER%)\n\n"
    md += "| Training Dataset | Model | BD | BD-EN | BD-CN | ASV19 | ITW | FOR |\n"
    md += "|------------------|-------|----|----|----|----|----|----|\n"
    
    for _, row in df.iterrows():
        md += f"| {row['Training Dataset']} | {row['Model']} | "
        md += f"{row['BD_BD']} | {row['BD_BD-EN']} | {row['BD_BD-CN']} | "
        md += f"{row['Others_ASV19']} | {row['Others_ITW']} | {row['Others_FOR']} |\n"
    
    return md

def main():
    parser = argparse.ArgumentParser(description="Generate Table 3 performance comparison")
    parser.add_argument("--exp_root", type=str, default="./exp_result", 
                       help="Root directory containing experiment results")
    parser.add_argument("--output", type=str, default="table3_results", 
                       help="Output file prefix")
    parser.add_argument("--format", type=str, choices=["csv", "latex", "markdown", "all"], 
                       default="all", help="Output format")
    
    args = parser.parse_args()
    
    exp_root = Path(args.exp_root)
    if not exp_root.exists():
        print(f"Error: Experiment root directory {exp_root} does not exist")
        return
    
    # Save raw experimental data
    print("Saving raw experimental data...")
    raw_data_file = f"{args.output}_raw_data.json"
    raw_data = save_raw_data(exp_root, raw_data_file)
    
    # Create performance table
    print("Generating Table 3 performance comparison...")
    df = create_performance_table(exp_root)
    
    if df.empty:
        print("No results found. Make sure experiments have been run and results are available.")
        return
    
    # Save results
    if args.format in ["csv", "all"]:
        csv_file = f"{args.output}.csv"
        df.to_csv(csv_file, index=False)
        print(f"CSV table saved to: {csv_file}")
    
    if args.format in ["latex", "all"]:
        latex_file = f"{args.output}.tex"
        latex_content = format_table_latex(df)
        with open(latex_file, 'w') as f:
            f.write(latex_content)
        print(f"LaTeX table saved to: {latex_file}")
    
    if args.format in ["markdown", "all"]:
        md_file = f"{args.output}.md"
        md_content = format_table_markdown(df)
        with open(md_file, 'w') as f:
            f.write(md_content)
        print(f"Markdown table saved to: {md_file}")
    
    print("\nTable 3 generation complete!")
    print(f"Found {len(df)} experiment results")
    print(f"Raw data available in: {raw_data_file}")

if __name__ == "__main__":
    main()
