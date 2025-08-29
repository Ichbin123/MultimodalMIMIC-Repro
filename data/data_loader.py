import pandas as pd
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Any
import os


class MIMICDataset(Dataset):
    """MIMIC-III dataset for irregular multimodal EHR modeling"""
    
    def __init__(self, data_path: str, task: str, split: str, max_notes: int = 5):
        """
        Args:
            data_path: Path to processed MIMIC-III data
            task: '48ihm' or '24phe'
            split: 'train', 'val', or 'test'
            max_notes: Maximum number of clinical notes per patient
        """
        self.data_path = data_path
        self.task = task
        self.split = split
        self.max_notes = max_notes
        
        # Load data
        self.load_data()
        
    def load_data(self):
        """Load time series, clinical notes, and labels"""
        task_path = os.path.join(self.data_path, self.task)
        
        # Load time series data
        with open(os.path.join(task_path, f'{self.split}_ts.pkl'), 'rb') as f:
            ts_data = pickle.load(f)
        
        self.x_ts = ts_data['observations']  # [N, d_m, max_obs]
        self.t_ts = ts_data['timestamps']    # [N, d_m, max_obs] 
        self.global_means = ts_data['global_means']  # [d_m]
        
        # Load clinical notes
        with open(os.path.join(task_path, f'{self.split}_notes.pkl'), 'rb') as f:
            notes_data = pickle.load(f)
        
        self.clinical_notes = notes_data['notes']  # List of [List of note texts]
        self.note_times = notes_data['timestamps']  # List of [List of note times]
        
        # Load labels
        with open(os.path.join(task_path, f'{self.split}_labels.pkl'), 'rb') as f:
            self.labels = pickle.load(f)
        
        # Ensure consistent patient count
        assert len(self.x_ts) == len(self.clinical_notes) == len(self.labels)
        self.n_patients = len(self.x_ts)
        
        print(f"Loaded {self.n_patients} patients for {self.task} {self.split}")
        
    def __len__(self):
        return self.n_patients
    
    def __getitem__(self, idx):
        """Get a single patient's data"""
        # Time series data
        x_ts = torch.tensor(self.x_ts[idx], dtype=torch.float32)
        t_ts = torch.tensor(self.t_ts[idx], dtype=torch.float32)
        
        # Clinical notes (limit to max_notes recent notes)
        patient_notes = self.clinical_notes[idx]
        patient_note_times = self.note_times[idx]
        
        if len(patient_notes) > self.max_notes:
            # Take the most recent notes
            patient_notes = patient_notes[-self.max_notes:]
            patient_note_times = patient_note_times[-self.max_notes:]
        
        # Labels
        if self.task == '48ihm':
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
        else:  # 24phe - multi-label classification
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return {
            'x_ts': x_ts,
            't_ts': t_ts,
            'clinical_notes': patient_notes,
            'note_times': patient_note_times,
            'label': label,
            'patient_id': idx
        }


def collate_multimodal(batch):
    """Custom collate function for multimodal data"""
    # Extract components
    x_ts_batch = torch.stack([item['x_ts'] for item in batch])
    t_ts_batch = torch.stack([item['t_ts'] for item in batch])
    labels_batch = torch.stack([item['label'] for item in batch])
    
    # Clinical notes and times (variable length)
    clinical_notes_batch = [item['clinical_notes'] for item in batch]
    note_times_batch = [item['note_times'] for item in batch]
    patient_ids = [item['patient_id'] for item in batch]
    
    return {
        'x_ts': x_ts_batch,
        't_ts': t_ts_batch,
        'clinical_notes': clinical_notes_batch,
        'note_times': note_times_batch,
        'labels': labels_batch,
        'patient_ids': patient_ids
    }


class MIMICDataModule:
    """Data module for MIMIC-III experiments"""
    
    def __init__(self, data_path: str, task: str, batch_size: int = 32, 
                 max_notes: int = 5, num_workers: int = 4):
        self.data_path = data_path
        self.task = task
        self.batch_size = batch_size
        self.max_notes = max_notes
        self.num_workers = num_workers
        
    def get_dataloader(self, split: str, shuffle: bool = None):
        """Get dataloader for specified split"""
        if shuffle is None:
            shuffle = (split == 'train')
            
        dataset = MIMICDataset(
            data_path=self.data_path,
            task=self.task,
            split=split,
            max_notes=self.max_notes
        )
        
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=collate_multimodal,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return dataloader
    
    def get_global_means(self):
        """Get global means from training set"""
        train_dataset = MIMICDataset(self.data_path, self.task, 'train', self.max_notes)
        return torch.tensor(train_dataset.global_means, dtype=torch.float32)
