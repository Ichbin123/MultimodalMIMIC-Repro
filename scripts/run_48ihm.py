#!/usr/bin/env python3
"""
Script to run 48-hour in-hospital mortality prediction experiments
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
from experiments.train import run_experiment


def run_multiple_seeds(args, seeds=[42, 123, 456]):
    """Run experiment with multiple seeds and report statistics"""
    all_results = []
    
    for seed in seeds:
        print(f"\n{'='*50}")
        print(f"Running experiment with seed {seed}")
        print(f"{'='*50}")
        
        args.seed = seed
        test_metrics = run_experiment(args)
        all_results.append(test_metrics)
    
    # Compute statistics
    print(f"\n{'='*50}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*50}")
    
    if args.task == '48ihm':
        f1_scores = [r['f1'] for r in all_results]
        aupr_scores = [r['aupr'] for r in all_results]
        
        print(f"F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
        print(f"AUPR: {np.mean(aupr_scores):.4f} ± {np.std(aupr_scores):.4f}")
        
        # Save summary
        summary = {
            'f1_mean': float(np.mean(f1_scores)),
            'f1_std': float(np.std(f1_scores)),
            'aupr_mean': float(np.mean(aupr_scores)),
            'aupr_std': float(np.std(aupr_scores)),
            'individual_results': all_results
        }
    
    else:  # 24phe
        f1_scores = [r['f1_macro'] for r in all_results]
        auroc_scores = [r['auroc_macro'] for r in all_results]
        
        print(f"F1 (Macro): {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
        print(f"AUROC (Macro): {np.mean(auroc_scores):.4f} ± {np.std(auroc_scores):.4f}")
        
        summary = {
            'f1_macro_mean': float(np.mean(f1_scores)),
            'f1_macro_std': float(np.std(f1_scores)),
            'auroc_macro_mean': float(np.mean(auroc_scores)),
            'auroc_macro_std': float(np.std(auroc_scores)),
            'individual_results': all_results
        }
    
    # Save summary results
    summary_path = os.path.join(args.save_dir, f'summary_{args.model_type}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='48-hour mortality prediction experiments')
    parser.add_argument('--model_type', 
                       choices=['time_series', 'clinical_notes', 'multimodal'],
                       default='multimodal',
                       help='Model type to train')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to processed MIMIC-III data')
    parser.add_argument('--save_dir', type=str, default='./results/48ihm',
                       help='Directory to save results')
    parser.add_argument('--single_seed', type=int, default=None,
                       help='Run with single seed instead of multiple')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456],
                       help='List of seeds for multiple runs')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    
    args = parser.parse_args()
    args.task = '48ihm'  # Fixed for this script
    
    if args.single_seed is not None:
        # Single seed run
        args.seed = args.single_seed
        test_metrics = run_experiment(args)
        print(f"Results: {test_metrics}")
    else:
        # Multiple seed runs
        summary = run_multiple_seeds(args, args.seeds)
        print(f"Summary saved to: {os.path.join(args.save_dir, f'summary_{args.model_type}.json')}")


if __name__ == '__main__':
    main()
