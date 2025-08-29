#!/usr/bin/env python3
"""
Evaluation script for trained models and baseline comparisons
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
import pandas as pd
import torch
from experiments.train import Trainer
from experiments.config import get_config


class ModelEvaluator:
    """Evaluation class for comparing different models"""
    
    def __init__(self, config, task='48ihm'):
        self.config = config
        self.task = task
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def evaluate_model(self, model_type, checkpoint_path):
        """Evaluate a single model"""
        print(f"Evaluating {model_type} model...")
        
        # Initialize trainer
        trainer = Trainer(self.config, model_type, self.task)
        
        # Load checkpoint
        trainer.load_checkpoint(os.path.basename(checkpoint_path))
        
        # Evaluate on test set
        test_metrics = trainer.evaluate(trainer.test_loader, 'test')
        
        return test_metrics
    
    def run_baseline_comparison(self, checkpoint_dir):
        """Run comparison with all baseline models"""
        model_types = ['time_series', 'clinical_notes', 'multimodal']
        results = {}
        
        for model_type in model_types:
            checkpoint_path = os.path.join(checkpoint_dir, model_type, 'best_model.pth')
            
            if os.path.exists(checkpoint_path):
                metrics = self.evaluate_model(model_type, checkpoint_path)
                results[model_type] = metrics
                print(f"{model_type}: {metrics}")
            else:
                print(f"Warning: No checkpoint found for {model_type}")
        
        return results
    
    def generate_results_table(self, results, save_path=None):
        """Generate results table in paper format"""
        if self.task == '48ihm':
            columns = ['Model', 'F1', 'AUPR']
            data = []
            
            for model_type, metrics in results.items():
                data.append([
                    model_type.replace('_', ' ').title(),
                    f"{metrics['f1']:.2f}",
                    f"{metrics['aupr']:.2f}"
                ])
        
        else:  # 24phe
            columns = ['Model', 'F1 (Macro)', 'AUROC']
            data = []
            
            for model_type, metrics in results.items():
                data.append([
                    model_type.replace('_', ' ').title(), 
                    f"{metrics['f1_macro']:.2f}",
                    f"{metrics['auroc_macro']:.2f}"
                ])
        
        df = pd.DataFrame(data, columns=columns)
        
        if save_path:
            df.to_csv(save_path, index=False)
            print(f"Results table saved to: {save_path}")
        
        print("\nResults Table:")
        print(df.to_string(index=False))
        
        return df


def run_multi_seed_evaluation(args):
    """Run evaluation across multiple seeds and compute statistics"""
    seeds = [42, 123, 456]
    all_results = {model_type: [] for model_type in ['time_series', 'clinical_notes', 'multimodal']}
    
    config = get_config(args.task)
    config['data_path'] = args.data_path
    
    evaluator = ModelEvaluator(config, args.task)
    
    for seed in seeds:
        print(f"\nEvaluating models with seed {seed}...")
        
        checkpoint_dir = os.path.join(args.checkpoint_dir, f'seed_{seed}')
        results = evaluator.run_baseline_comparison(checkpoint_dir)
        
        for model_type, metrics in results.items():
            all_results[model_type].append(metrics)
    
    # Compute statistics
    final_results = {}
    
    for model_type, results_list in all_results.items():
        if not results_list:
            continue
            
        if args.task == '48ihm':
            f1_scores = [r['f1'] for r in results_list]
            aupr_scores = [r['aupr'] for r in results_list]
            
            final_results[model_type] = {
                'f1_mean': np.mean(f1_scores),
                'f1_std': np.std(f1_scores),
                'aupr_mean': np.mean(aupr_scores),
                'aupr_std': np.std(aupr_scores)
            }
        
        else:  # 24phe
            f1_scores = [r['f1_macro'] for r in results_list]
            auroc_scores = [r['auroc_macro'] for r in results_list]
            
            final_results[model_type] = {
                'f1_macro_mean': np.mean(f1_scores),
                'f1_macro_std': np.std(f1_scores),
                'auroc_macro_mean': np.mean(auroc_scores),
                'auroc_macro_std': np.std(auroc_scores)
            }
    
    # Print final results
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS SUMMARY FOR {args.task.upper()}")
    print(f"{'='*60}")
    
    for model_type, stats in final_results.items():
        print(f"\n{model_type.replace('_', ' ').title()}:")
        for metric, value in stats.items():
            print(f"  {metric}: {value:.4f}")
    
    # Save summary
    summary_path = os.path.join(args.output_dir, f'final_results_{args.task}.json')
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(summary_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nSummary saved to: {summary_path}")
    
    return final_results


def main():
    parser = argparse.ArgumentParser(description='Evaluate MultimodalMIMIC models')
    parser.add_argument('--task', choices=['48ihm', '24phe'], required=True,
                       help='Evaluation task')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='Directory containing model checkpoints')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to processed MIMIC-III data')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--single_model', type=str, default=None,
                       choices=['time_series', 'clinical_notes', 'multimodal'],
                       help='Evaluate single model type only')
    
    args = parser.parse_args()
    
    if args.single_model:
        # Evaluate single model
        config = get_config(args.task)
        config['data_path'] = args.data_path
        
        evaluator = ModelEvaluator(config, args.task)
        checkpoint_path = os.path.join(args.checkpoint_dir, args.single_model, 'best_model.pth')
        
        metrics = evaluator.evaluate_model(args.single_model, checkpoint_path)
        print(f"Results for {args.single_model}: {metrics}")
    
    else:
        # Multi-seed evaluation
        final_results = run_multi_seed_evaluation(args)


if __name__ == '__main__':
    main()
