import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from .mtand import mTANDModule


class ClinicalNotesEncoder(nn.Module):
    """Clinical notes encoder using Clinical-Longformer"""
    
    def __init__(self, model_name='yikuan8/Clinical-Longformer', 
                 max_length=1024, d_h=128):
        super(ClinicalNotesEncoder, self).__init__()
        
        # Load pretrained Clinical-Longformer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_encoder = AutoModel.from_pretrained(model_name)
        
        self.max_length = max_length
        self.d_t = self.text_encoder.config.hidden_size  # Usually 768
        self.d_h = d_h
        
        # Project to common hidden dimension
        self.projection = nn.Linear(self.d_t, d_h)
        
    def encode_notes(self, clinical_notes):
        """
        Encode a batch of clinical notes
        Args:
            clinical_notes: List of note texts [l_txt]
        Returns:
            Note embeddings [l_txt, d_h]
        """
        if len(clinical_notes) == 0:
            return torch.zeros(0, self.d_h)
        
        # Tokenize notes
        inputs = self.tokenizer(
            clinical_notes,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Get [CLS] token representations
        with torch.no_grad():
            outputs = self.text_encoder(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [l_txt, d_t]
        
        # Project to common dimension
        note_embeddings = self.projection(cls_embeddings)  # [l_txt, d_h]
        
        return note_embeddings


class IrregularClinicalNotesModel(nn.Module):
    """Complete clinical notes model with mTAND for irregularity"""
    
    def __init__(self, d_h=128, alpha=48, model_name='yikuan8/Clinical-Longformer',
                 max_length=1024, d_v=64, num_heads=8, num_layers=3, num_classes=1):
        super(IrregularClinicalNotesModel, self).__init__()
        
        self.d_h = d_h
        self.alpha = alpha
        
        # Text encoder
        self.text_encoder = ClinicalNotesEncoder(model_name, max_length, d_h)
        
        # mTAND for clinical notes irregularity
        self.mtand_txt = mTANDModule(d_h, d_v, num_heads)
        
        # Transformer backbone
        self.transformer = TransformerEncoder(d_h, num_heads, num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_h, d_h // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_h // 2, num_classes)
        )
    
    def forward(self, clinical_notes_batch, note_times_batch):
        """
        Args:
            clinical_notes_batch: List of [clinical_notes] for each patient
            note_times_batch: List of [note_times] for each patient
        Returns:
            Predictions [batch_size, num_classes]
        """
        batch_embeddings = []
        alpha_points = torch.arange(self.alpha, dtype=torch.float32)
        
        for clinical_notes, note_times in zip(clinical_notes_batch, note_times_batch):
            if len(clinical_notes) == 0:
                # No notes, use zero embeddings
                patient_emb = torch.zeros(self.alpha, self.d_h)
            else:
                # Encode clinical notes
                note_embeddings = self.text_encoder.encode_notes(clinical_notes)  # [l_txt, d_h]
                note_times_tensor = torch.tensor(note_times, dtype=torch.float32)
                
                # Apply mTAND to handle irregularity
                # Treat each dimension of note embeddings as irregular time series
                interpolated_list = []
                for dim in range(self.d_h):
                    dim_values = note_embeddings[:, dim]  # [l_txt]
                    dim_interp = self.mtand_txt(alpha_points, note_times_tensor, dim_values)
                    interpolated_list.append(dim_interp.unsqueeze(-1))
                
                # Combine all dimensions
                patient_emb = torch.cat(interpolated_list, dim=-1)  # [alpha, d_h]
            
            batch_embeddings.append(patient_emb)
        
        # Stack batch
        batch_embeddings = torch.stack(batch_embeddings, dim=0)  # [batch_size, alpha, d_h]
        
        # Apply Transformer
        encoded = self.transformer(batch_embeddings)  # [batch_size, alpha, d_h]
        
        # Classification
        last_hidden = encoded[:, -1, :]  # [batch_size, d_h]
        logits = self.classifier(last_hidden)
        
        return logits
    
    def get_embeddings(self, clinical_notes_batch, note_times_batch):
        """Get embeddings for multimodal fusion"""
        batch_embeddings = []
        alpha_points = torch.arange(self.alpha, dtype=torch.float32)
        
        for clinical_notes, note_times in zip(clinical_notes_batch, note_times_batch):
            if len(clinical_notes) == 0:
                patient_emb = torch.zeros(self.alpha, self.d_h)
            else:
                note_embeddings = self.text_encoder.encode_notes(clinical_notes)
                note_times_tensor = torch.tensor(note_times, dtype=torch.float32)
                
                interpolated_list = []
                for dim in range(self.d_h):
                    dim_values = note_embeddings[:, dim]
                    dim_interp = self.mtand_txt(alpha_points, note_times_tensor, dim_values)
                    interpolated_list.append(dim_interp.unsqueeze(-1))
                
                patient_emb = torch.cat(interpolated_list, dim=-1)
            
            batch_embeddings.append(patient_emb)
        
        batch_embeddings = torch.stack(batch_embeddings, dim=0)
        encoded = self.transformer(batch_embeddings)
        
        return encoded


# Import TransformerEncoder from utde.py
from .utde import TransformerEncoder
