import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
import wandb
import os
import json
from tqdm import tqdm
import argparse

from data.data_loader import MIMICDataModule
from models.utde import TimeSeriesModel
from models.text_encoder import IrregularClinicalNotesModel
from models.multimodal_fusion import MultimodalModel
from experiments.config import get_config


class Trainer:
    """Training class for multimodal medical prediction models"""
    
    def __init__(self, config, model_type='multimodal', task='48ihm'):
        self.config = config
        self.model_type = model_type
        self.task = task
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize data module
        self.data_module = MIMICDataModule(
            data_path=config['data_path'],
            task=task,
            batch_size=config['batch_size'],
            max_notes=config.get('max_notes', 5)
        )
        
        # Get data loaders
        self.train_loader = self.data_module.get_dataloader('train', shuffle=True)
        self.val_loader = self.data_module.get_dataloader('val', shuffle=False)
        self.test_loader = self.data_module.get_dataloader('test', shuffle=False)
        
        # Get global means
        self.global_means = self.data_module.get_global_means().to(self.device)
        
        # Initialize model
        self.model = self._build_model()
        self.model.to(self.device)
        
        # Initialize optimizer and loss
        self.optimizer = self._build_optimizer()
        self.criterion = self._build_criterion()
        
        # Metrics tracking
        self.best_val_score = 0.0
        self.train_losses = []
        self.val_scores = []
        
    def _build_model(self):
        """Build model based on type and task"""
        d_m = self.config['d_m']  # Number of time series features
        d_h = self.config['hidden_dim']
        alpha = self.config['alpha']
        num_classes = 1 if self.task == '48ihm' else 25  # Binary vs multi-label
        
        if self.model_type == 'time_series':
            model = TimeSeriesModel(
                d_m=d_m, d_h=d_h, alpha=alpha, 
                gate_level=self.config['gate_level'],
                num_heads=self.config['num_heads'],
                num_layers=self.config['num_layers'],
                num_classes=num_classes
            )
        elif self.model_type == 'clinical_notes':
            model = IrregularClinicalNotesModel(
                d_h=d_h, alpha=alpha,
                model_name=self.config['text_model_name'],
                max_length=self.config['max_text_length'],
                num_heads=self.config['num_heads'],
                num_layers=self.config['num_layers'],
                num_classes=num_classes
            )
        elif self.model_type == 'multimodal':
            model = MultimodalModel(
                d_m=d_m, d_h=d_h, alpha=alpha,
                num_classes=num_classes,
                gate_level=self.config['gate_level'],
                model_name=self.config['text_model_name'],
                max_length=self.config['max_text_length'],
                num_heads=self.config['num_heads'],
                num_layers=self.config['num_layers'],
                fusion_layers=self.config['fusion_layers']
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return model
    
    def _build_optimizer(self):
        """Build optimizer with different learning rates for different components"""
        param_groups = []
        
        # Different learning rates for pretrained vs new parameters
        plm_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if 'text_encoder' in name and 'projection' not in name:
                plm_params.append(param)
            else:
                other_params.append(param)
        
        if plm_params:
            param_groups.append({
                'params': plm_params,
                'lr': self.config['plm_learning_rate']
            })
        
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': self.config['learning_rate']
            })
        
        optimizer = optim.Adam(param_groups)
        return optimizer
    
    def _build_criterion(self):
        """Build loss function based on task"""
        if self.task == '48ihm':
            # Binary classification with class imbalance (1:7 ratio)
            pos_weight = torch.tensor([7.0]).to(self.device)
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:  # 24phe
            # Multi-label classification
            return nn.BCEWithLogitsLoss()
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Fine-tune PLM only in first 3 epochs for clinical notes
        if hasattr(self.model, 'txt_model') or hasattr(self.model, 'text_encoder'):
            freeze_plm = epoch >= 3
            self._freeze_plm(freeze_plm)
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')
        
        for batch in pbar:
            self.optimizer.zero_grad()
            
            # Forward pass
            if self.model_type == 'time_series':
                logits = self.model(
                    batch['x_ts'].to(self.device),
                    batch['t_ts'].to(self.device),
                    self.global_means
                )
            elif self.model_type == 'clinical_notes':
                logits = self.model(
                    batch['clinical_notes'],
                    batch['note_times']
                )
            else:  # multimodal
                logits = self.model(
                    batch['x_ts'].to(self.device),
                    batch['t_ts'].to(self.device),
                    self.global_means,
                    batch['clinical_notes'],
                    batch['note_times']
                )
            
            # Compute loss
            labels = batch['labels'].to(self.device)
            loss = self.criterion(logits.squeeze(), labels.squeeze())
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def _freeze_plm(self, freeze):
        """Freeze or unfreeze pretrained language model parameters"""
        if hasattr(self.model, 'txt_model'):
            for param in self.model.txt_model.text_encoder.text_encoder.parameters():
                param.requires_grad = not freeze
        elif hasattr(self.model, 'text_encoder'):
            for param in self.model.text_encoder.text_encoder.parameters():
                param.requires_grad = not freeze
    
    def evaluate(self, dataloader, split_name='val'):
        """Evaluate model on given dataloader"""
        self.model.eval()
        all_logits = []
        all_labels = []
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f'Evaluating {split_name}'):
                # Forward pass
                if self.model_type == 'time_series':
                    logits = self.model(
                        batch['x_ts'].to(self.device),
                        batch['t_ts'].to(self.device),
                        self.global_means
                    )
                elif self.model_type == 'clinical_notes':
                    logits = self.model(
                        batch['clinical_notes'],
                        batch['note_times']
                    )
                else:  # multimodal
                    logits = self.model(
                        batch['x_ts'].to(self.device),
                        batch['t_ts'].to(self.device),
                        self.global_means,
                        batch['clinical_notes'],
                        batch['note_times']
                    )
                
                labels = batch['labels'].to(self.device)
                loss = self.criterion(logits.squeeze(), labels.squeeze())
                
                total_loss += loss.item()
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())
        
        # Concatenate all predictions
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Compute metrics
        metrics = self._compute_metrics(all_logits, all_labels)
        metrics['loss'] = total_loss / len(dataloader)
        
        return metrics
    
    def _compute_metrics(self, logits, labels):
        """Compute evaluation metrics based on task"""
        if self.task == '48ihm':
            # Binary classification metrics
            probs = torch.sigmoid(logits).numpy()
            preds = (probs > 0.5).astype(int)
            labels_np = labels.numpy().astype(int)
            
            f1 = f1_score(labels_np, preds)
            aupr = average_precision_score(labels_np, probs)
            
            return {'f1': f1, 'aupr': aupr}
        
        else:  # 24phe
            # Multi-label classification metrics
            probs = torch.sigmoid(logits).numpy()
            preds = (probs > 0.5).astype(int)
            labels_np = labels.numpy().astype(int)
            
            # Macro F1 and AUROC
            f1_macro = f1_score(labels_np, preds, average='macro', zero_division=0)
            auroc_macro = roc_auc_score(labels_np, probs, average='macro')
            
            return {'f1_macro': f1_macro, 'auroc_macro': auroc_macro}
    
    def train(self):
        """Main training loop"""
        print(f"Training {self.model_type} model for {self.task}")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Training loop
        epochs = self.config['epochs']
        if self.model_type == 'clinical_notes' or self.model_type == 'multimodal':
            epochs = min(epochs, 6)  # Clinical notes converge faster
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.evaluate(self.val_loader, 'val')
            
            # Track best model
            if self.task == '48ihm':
                val_score = val_metrics['f1']
            else:
                val_score = val_metrics['f1_macro']
            
            if val_score > self.best_val_score:
                self.best_val_score = val_score
                self.save_checkpoint('best_model.pth')
            
            # Log metrics
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            for metric, value in val_metrics.items():
                print(f"  Val {metric.upper()}: {value:.4f}")
            
            # Log to wandb if available
            if wandb.run:
                log_dict = {'epoch': epoch, 'train_loss': train_loss}
                log_dict.update({f'val_{k}': v for k, v in val_metrics.items()})
                wandb.log(log_dict)
        
        # Final test evaluation
        print("\nEvaluating on test set...")
        test_metrics = self.evaluate(self.test_loader, 'test')
        
        print("Final Test Results:")
        for metric, value in test_metrics.items():
            print(f"  Test {metric.upper()}: {value:.4f}")
        
        return test_metrics
    
    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        os.makedirs(self.config['save_dir'], exist_ok=True)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_val_score': self.best_val_score
        }
        torch.save(checkpoint, os.path.join(self.config['save_dir'], filename))
    
    def load_checkpoint(self, filename):
        """Load model checkpoint"""
        checkpoint_path = os.path.join(self.config['save_dir'], filename)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_score = checkpoint['best_val_score']


def run_experiment(args):
    """Run complete experiment"""
    # Get configuration
    config = get_config(args.task)
    config.update(vars(args))
    
    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(
            project=f"MultimodalMIMIC-{args.task}",
            name=f"{args.model_type}_{args.seed}",
            config=config
        )
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Run training
    trainer = Trainer(config, args.model_type, args.task)
    test_metrics = trainer.train()
    
    # Save results
    results = {
        'task': args.task,
        'model_type': args.model_type,
        'seed': args.seed,
        'test_metrics': test_metrics,
        'config': config
    }
    
    results_dir = os.path.join(config['save_dir'], 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, f'{args.model_type}_{args.seed}.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return test_metrics


def main():
    parser = argparse.ArgumentParser(description='Train MultimodalMIMIC models')
    parser.add_argument('--task', choices=['48ihm', '24phe'], default='48ihm',
                       help='Prediction task')
    parser.add_argument('--model_type', 
                       choices=['time_series', 'clinical_notes', 'multimodal'],
                       default='multimodal', help='Model type to train')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to processed MIMIC-III data')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    
    args = parser.parse_args()
    
    # Run experiment
    test_metrics = run_experiment(args)
    print(f"\nFinal results: {test_metrics}")


if __name__ == '__main__':
    main()
