# MultimodalMIMIC: Irregular Multimodal EHR Modeling

This repository contains a reproduction of the paper "Improving Medical Predictions by Irregular Multimodal Electronic Health Records Modeling" by Xinlu Zhang et al. (ICML 2023).

## Overview

This work addresses the challenge of modeling irregular multimodal Electronic Health Records (EHRs) for medical predictions in Intensive Care Units (ICUs). The key innovations include:

1. **UTDE (Unified TDE)**: A gating mechanism that dynamically combines hand-crafted imputation and learned interpolation embeddings for irregular time series
2. **mTAND for Clinical Notes**: Casting clinical note representations as irregular time series and applying time attention mechanisms
3. **Interleaved Multimodal Fusion**: Cross-modal attention across temporal steps to integrate irregularity into multimodal representations

## Key Results

- **6.5% relative improvement** in F1 score for time series modeling
- **3.6% relative improvement** in F1 score for clinical notes modeling  
- **4.3% relative improvement** in F1 score for multimodal fusion
- Evaluated on MIMIC-III dataset for 48-hour mortality prediction and 24-hour phenotype classification

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.12+
- CUDA 11.6+ (for GPU support)

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/MultimodalMIMIC-Reproduction.git
cd MultimodalMIMIC-Reproduction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Data Preparation

### MIMIC-III Dataset

1. **Access MIMIC-III**: 
   - Complete the required training at https://physionet.org/
   - Request access to MIMIC-III Clinical Database
   - Download the dataset files

2. **Data Preprocessing**:
   ```bash
   # Extract time series features (following Harutyunyan et al., 2019)
   python scripts/preprocess_data.py --data_path /path/to/mimic-iii --output_path ./data/processed

   # This will create:
   # - Time series data for 48-hour mortality prediction
   # - Time series data for 24-hour phenotype classification
   # - Clinical notes extracted within prediction windows
   ```

### Data Structure

```
data/
├── processed/
│   ├── 48ihm/
│   │   ├── train_ts.pkl      # Training time series
│   │   ├── train_notes.pkl   # Training clinical notes
│   │   ├── val_ts.pkl        # Validation data
│   │   ├── test_ts.pkl       # Test data
│   │   └── labels.pkl        # Target labels
│   └── 24phe/
│       └── ...               # Similar structure for phenotype classification
```

## Usage

### Training Models

#### 1. Time Series Only (UTDE)
```bash
python scripts/run_48ihm.py --mode time_series --model utde
python scripts/run_24phe.py --mode time_series --model utde
```

#### 2. Clinical Notes Only (mTAND_txt)
```bash
python scripts/run_48ihm.py --mode clinical_notes --model mtand_txt
python scripts/run_24phe.py --mode clinical_notes --model mtand_txt
```

#### 3. Multimodal Fusion
```bash
python scripts/run_48ihm.py --mode multimodal --model interleaved_fusion
python scripts/run_24phe.py --mode multimodal --model interleaved_fusion
```

### Configuration

Modify hyperparameters in `experiments/config.py`:

```python
# Example configuration for 48-hour mortality prediction
CONFIG_48IHM = {
    'batch_size': 32,
    'learning_rate': 0.0004,
    'plm_learning_rate': 2e-5,
    'epochs': 20,
    'hidden_dim': 128,
    'num_heads': 8,
    'num_layers': 3,
    'gate_level': 'hidden_space',  # 'patient', 'temporal', 'hidden_space'
}
```

## Model Architecture

### 1. UTDE Module
- Combines imputation and mTAND embeddings via gating mechanism
- Three levels of gating: patient, temporal, and hidden space
- Dynamic integration based on patient-specific patterns

### 2. Clinical Notes Processing
- Clinical-Longformer encoding with max length 1024
- mTAND_txt for handling irregular note-taking times
- Time attention mechanism across note representations

### 3. Multimodal Fusion
- Interleaved self-attention and cross-attention layers
- Temporal integration across modalities
- J=3 fusion layers for optimal performance

## Evaluation

### Metrics

- **48-hour Mortality Prediction**: F1 Score, AUPR
- **24-hour Phenotype Classification**: Macro F1, AUROC

### Baseline Comparisons

**Time Series Baselines**:
- Imputation, IP-Net, mTAND, GRU-D, SeFT, RAINDROP, DGM²-O, MTGNN

**Clinical Notes Baselines**:
- Flat, HierTrans, T-LSTM, FT-LSTM, GRU-D

**Fusion Baselines**:
- Concatenation, Tensor Fusion, MAG, MulT

### Results

| Task | Method | F1 | AUPR/AUROC |
|------|--------|----|----|
| 48-IHM | UTDE | 45.26±0.70 | 49.64±1.00 |
| 48-IHM | mTAND_txt | 52.57±1.30 | 56.05±1.09 |
| 48-IHM | Interleaved Fusion | **56.45±1.30** | **60.23±1.54** |
| 24-PHE | UTDE | 24.89±0.43 | 75.56±0.17 |
| 24-PHE | mTAND_txt | 52.95±0.06 | 85.43±0.07 |
| 24-PHE | Interleaved Fusion | **54.84±0.31** | **86.06±0.06** |

## Repository Structure

```
├── data/                    # Data processing utilities
├── models/                  # Core model implementations
├── experiments/             # Training and evaluation scripts
├── scripts/                 # Executable scripts
├── notebooks/              # Analysis notebooks
├── tests/                  # Unit tests
└── docs/                   # Documentation
```

## Citation

```bibtex
@inproceedings{zhang2023improving,
  title={Improving Medical Predictions by Irregular Multimodal Electronic Health Records Modeling},
  author={Zhang, Xinlu and Li, Shiyang and Chen, Zhiyu and Yan, Xifeng and Petzold, Linda},
  booktitle={Proceedings of the 40th International Conference on Machine Learning},
  year={2023},
  publisher={PMLR}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original paper authors: Xinlu Zhang, Shiyang Li, Zhiyu Chen, Xifeng Yan, Linda Petzold
- MIMIC-III dataset from MIT Lab for Computational Physiology
- Financial support by NIH grant NIH 7R01HL149670

## Contact

For questions about this reproduction, please open an issue or contact [xujili1105@gmail.com].

## Important Notes

### Data Privacy
- MIMIC-III contains sensitive medical data
- Requires completed CITI training and signed data use agreement
- Follow all ethical guidelines for medical data research

### Computational Requirements
- GPU recommended (experiments conducted on RTX-3090)
- ~30 minutes training time for UTDE module
- ~6 epochs convergence for clinical notes models

### Reproducibility
- All experiments use 3 random seeds
- Results reported with mean ± standard deviation
- Fixed data splits following Harutyunyan et al. (2019)
