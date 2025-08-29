"""Configuration file for MultimodalMIMIC experiments"""

CONFIG_48IHM = {
    # Data parameters
    'd_m': 17,  # Number of time series features (following Harutyunyan et al.)
    'alpha': 48,  # Prediction window in hours
    'max_notes': 5,  # Maximum clinical notes per patient
    
    # Model architecture
    'hidden_dim': 128,
    'num_heads': 8,
    'num_layers': 3,
    'fusion_layers': 3,
    'gate_level': 'hidden_space',  # 'patient', 'temporal', 'hidden_space'
    
    # Text model parameters
    'text_model_name': 'yikuan8/Clinical-Longformer',
    'max_text_length': 1024,
    
    # Training parameters
    'batch_size': 32,
    'learning_rate': 0.0004,
    'plm_learning_rate': 2e-5,
    'epochs': 20,
    'dropout': 0.1,
    
    # Paths
    'data_path': './data/processed',
    'save_dir': './checkpoints/48ihm',
    
    # Other
    'num_workers': 4,
}

CONFIG_24PHE = {
    # Data parameters
    'd_m': 17,
    'alpha': 24,  # 24-hour prediction window
    'max_notes': 5,
    
    # Model architecture  
    'hidden_dim': 128,
    'num_heads': 8,
    'num_layers': 3,
    'fusion_layers': 3,
    'gate_level': 'hidden_space',
    
    # Text model parameters
    'text_model_name': 'yikuan8/Clinical-Longformer',
    'max_text_length': 1024,
    
    # Training parameters
    'batch_size': 32,
    'learning_rate': 0.0004,
    'plm_learning_rate': 2e-5,
    'epochs': 20,
    'dropout': 0.1,
    
    # Paths
    'data_path': './data/processed',
    'save_dir': './checkpoints/24phe',
    
    # Other
    'num_workers': 4,
}


def get_config(task):
    """Get configuration for specific task"""
    if task == '48ihm':
        return CONFIG_48IHM.copy()
    elif task == '24phe':
        return CONFIG_24PHE.copy()
    else:
        raise ValueError(f"Unknown task: {task}")


# Hyperparameter search ranges for optimization
SEARCH_RANGES = {
    'hidden_dim': [64, 128],
    'num_heads': [4, 8],
    'gate_level': ['patient', 'temporal', 'hidden_space'],
    'learning_rate': [1e-4, 4e-4, 1e-3],
    'batch_size': [16, 32, 64]
}
