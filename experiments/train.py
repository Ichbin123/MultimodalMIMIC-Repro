import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

class MIMICDataset(Dataset):
    """Dataset class for MIMIC-III multimodal data"""
    def __init__(self, data_path, split='train', task='mortality', max_seq_len=48, max_notes=5):
        self.data_path = data_path
        self.split = split
        self.task = task
        self.max_seq_len = max_seq_len
        self.max_notes = max_notes
        
        # Load data
        self.load_data()
        
    def load_data(self):
        """Load preprocessed MIMIC-III data"""
        # Load time series data
        ts_file = os.path.join(self.data_path, f'timeseries_{self.split}.csv')
        self.ts_data = pd.read_csv(ts_file)
        
        # Load clinical notes
        notes_file = os.path.join(self.data_path, f'notes_{self.split}.csv')
        self.notes_data = pd.read_csv(notes_file)
        
        # Load labels
        labels_file = os.path.join(self.data_path, f'labels_{self.split}.csv')
        self.labels = pd.read_csv(labels_file)
        
        # Get unique patient IDs
        self.patient_ids = self.labels['patient_id'].unique()
        
        # Time series features (assuming MIMIC-III benchmark features)
        self.ts_features = [
            'Capillary refill rate', 'Diastolic blood pressure', 'Fraction inspired oxygen',
            'Glascow coma scale eye opening', 'Glascow coma scale motor response',
            'Glascow coma scale total', 'Glascow coma scale verbal response',
            'Glucose', 'Heart Rate', 'Height', 'Mean blood pressure', 'Oxygen saturation',
            'Respiratory rate', 'Systolic blood pressure', 'Temperature', 'Weight', 'pH'
        ]
        
    def __len__(self):
        return len(self.patient_ids)
    
    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        
        # Get time series data for this patient
        patient_ts = self.ts_data[self.ts_data['patient_id'] == patient_id].copy()
        patient_ts = patient_ts.sort_values('time')
        
        # Extract time series values and times
        ts_values = patient_ts[self.ts_features].fillna(0).values  # Simple zero fill for missing
        ts_times = patient_ts['time'].values
        
        # Pad or truncate to max_seq_len
        if len(ts_values) > self.max_seq_len:
            ts_values = ts_values[:self.max_seq_len]
            ts_times = ts_times[:self.max_seq_len]
        else:
            # Pad with zeros
            padding_len = self.max_seq_len - len(ts_values)
            ts_values = np.pad(ts_values, ((0, padding_len), (0, 0)), 'constant')
            ts_times = np.pad(ts_times, (0, padding_len), 'constant', constant_values=ts_times[-1] if len(ts_times) > 0 else 0)
        
        # Get clinical notes for this patient
        patient_notes = self.notes_data[self.notes_data['patient_id'] == patient_id].copy()
        patient_notes = patient_notes.sort_values('time')
        
        # Extract notes and times
        notes_text = patient_notes['text'].tolist()[:self.max_notes]
        notes_times = patient_notes['time'].values[:self.max_notes]
        
        # Pad notes if necessary
        while len(notes_text) < self.max_notes:
            notes_text.append("")
            notes_times = np.append(notes_times, notes_times[-1] if len(notes_times) > 0 else 0)
        
        # Get label
        label = self.labels[self.labels['patient_id'] == patient_id][self.task].iloc[0]
        
        # Create query times (regular intervals)
        query_times = np.linspace(0, self.max_seq_len, self.max_seq_len)
        
        return {
            'patient_id': patient_id,
            'ts_data': torch.FloatTensor(ts_values),
            'ts_times': torch.FloatTensor(ts_times),
            'notes_text': notes_text,
            'notes_times': torch.FloatTensor(notes_times),
            'query_times': torch.FloatTensor(query_times),
            'label': torch.LongTensor([label]) if self.task in ['mortality', 'phenotype'] else torch.FloatTensor([label])
        }

def collate_fn(batch):
    """Custom collate function for batching"""
    patient_ids = [item['patient_id'] for item in batch]
    ts_data = torch.stack([item['ts_data'] for item in batch])
    ts_times = torch.stack([item['ts_times'] for item in batch])
    query_times = torch.stack([item['query_times'] for item in batch])
    notes_times = torch.stack([item['notes_times'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    
    # Handle notes text (list of lists)
    notes_text = [item['notes_text'] for item in batch]
    
    return {
        'patient_ids': patient_ids,
        'ts_data': ts_data,
        'ts_times': ts_times,
        'notes_text': notes_text,
        'notes_times': notes_times,
        'query_times': query_times,
        'labels': labels
    }

class EHRTrainer:
    """Training class for multimodal EHR model"""
    def __init__(self, model, train_loader, val_loader, test_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Set up optimizer and loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        if config['task'] == 'mortality':
            self.criterion = nn.BCEWithLogitsLoss()
        elif config['task'] == 'phenotype':
            self.criterion = nn.BCEWithLogitsLoss()  # Multi-label classification
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_score = 0
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            self.optimizer.zero_grad()
            
            # Move data to device
            ts_data = batch['ts_data'].to(self.device)
            ts_times = batch['ts_times'].to(self.device)
            notes_times = batch['notes_times'].to(self.device)
            query_times = batch['query_times'].to(self.device)
            labels = batch['labels'].to(self.device).squeeze()
            notes_text = batch['notes_text']
            
            # Forward pass
            try:
                logits = self.model(ts_data, ts_times, notes_text, notes_times, query_times)
                logits = logits.squeeze()
                
                # Compute loss
                loss = self.criterion(logits, labels.float())
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'loss': loss.item()})
                
            except Exception as e:
                print(f"Error in training batch: {e}")
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def evaluate(self, data_loader, split_name="Val"):
        """Evaluate model on given data loader"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(data_loader, desc=f"Evaluating {split_name}")
            for batch in pbar:
                # Move data to device
                ts_data = batch['ts_data'].to(self.device)
                ts_times = batch['ts_times'].to(self.device)
                notes_times = batch['notes_times'].to(self.device)
                query_times = batch['query_times'].to(self.device)
                labels = batch['labels'].to(self.device).squeeze()
                notes_text = batch['notes_text']
                
                try:
                    # Forward pass
                    logits = self.model(ts_data, ts_times, notes_text, notes_times, query_times)
                    logits = logits.squeeze()
                    
                    # Compute loss
                    loss = self.criterion(logits, labels.float())
                    total_loss += loss.item()
                    num_batches += 1
                    
                    # Store predictions and labels
                    if self.config['task'] == 'mortality':
                        predictions = torch.sigmoid(logits).cpu().numpy()
                    else:  # phenotype
                        predictions = torch.sigmoid(logits).cpu().numpy()
                    
                    all_predictions.extend(predictions)
                    all_labels.extend(labels.cpu().numpy())
                    
                except Exception as e:
                    print(f"Error in evaluation batch: {e}")
                    continue
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # Compute metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        if self.config['task'] == 'mortality':
            # Binary classification metrics
            auc = roc_auc_score(all_labels, all_predictions)
            aupr = average_precision_score(all_labels, all_predictions)
            f1 = f1_score(all_labels, all_predictions > 0.5)
            
            metrics = {'loss': avg_loss, 'auc': auc, 'aupr': aupr, 'f1': f1}
            print(f"{split_name} - Loss: {avg_loss:.4f}, AUC: {auc:.4f}, AUPR: {aupr:.4f}, F1: {f1:.4f}")
            
        else:  # phenotype - multi-label
            # Compute macro-averaged metrics
            auc = roc_auc_score(all_labels, all_predictions, average='macro')
            f1 = f1_score(all_labels, all_predictions > 0.5, average='macro')
            
            metrics = {'loss': avg_loss, 'auc': auc, 'f1': f1}
            print(f"{split_name} - Loss: {avg_loss:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}")
        
        return metrics
    
    def train(self, num_epochs):
        """Main training loop"""
        print(f"Training model for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch()
            
            # Evaluate
            val_metrics = self.evaluate(self.val_loader, "Validation")
            
            # Save best model
            current_score = val_metrics['f1']  # Use F1 as main metric
            if current_score > self.best_val_score:
                self.best_val_score = current_score
                torch.save(self.model.state_dict(), 'best_model.pt')
                print(f"New best model saved with F1: {current_score:.4f}")
        
        # Load best model and evaluate on test set
        self.model.load_state_dict(torch.load('best_model.pt'))
        test_metrics = self.evaluate(self.test_loader, "Test")
        
        return test_metrics
    
    def plot_training_curves(self):
        """Plot training and validation loss curves"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_curves.png')
        plt.show()

def main():
    """Main training function"""
    # Configuration
    config = {
        'data_path': 'data/processed/',
        'task': 'mortality',  # or 'phenotype'
        'batch_size': 32,
        'learning_rate': 0.0004,
        'num_epochs': 20,
        'hidden_dim': 128,
        'num_fusion_layers': 3,
        'num_heads': 8,
        'ff_dim': 512,
        'max_seq_len': 48,
        'max_notes': 5,
        'ts_input_dim': 17  # Number of time series features
    }
    
    # Create datasets
    train_dataset = MIMICDataset(
        config['data_path'], 
        split='train', 
        task=config['task'],
        max_seq_len=config['max_seq_len'],
        max_notes=config['max_notes']
    )
    
    val_dataset = MIMICDataset(
        config['data_path'], 
        split='val', 
        task=config['task'],
        max_seq_len=config['max_seq_len'],
        max_notes=config['max_notes']
    )
    
    test_dataset = MIMICDataset(
        config['data_path'], 
        split='test', 
        task=config['task'],
        max_seq_len=config['max_seq_len'],
        max_notes=config['max_notes']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    # Create model
    num_classes = 1 if config['task'] == 'mortality' else 25  # 25 phenotypes
    model = MultimodalEHRModel(
        ts_input_dim=config['ts_input_dim'],
        hidden_dim=config['hidden_dim'],
        num_fusion_layers=config['num_fusion_layers'],
        num_heads=config['num_heads'],
        ff_dim=config['ff_dim'],
        max_seq_len=config['max_seq_len'],
        num_classes=num_classes
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create trainer and train
    trainer = EHRTrainer(model, train_loader, val_loader, test_loader, config)
    test_metrics = trainer.train(config['num_epochs'])
    
    print(f"\nFinal test results: {test_metrics}")
    
    # Plot training curves
    trainer.plot_training_curves()

if __name__ == "__main__":
    main()
