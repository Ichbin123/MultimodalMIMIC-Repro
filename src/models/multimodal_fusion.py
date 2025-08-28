import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism"""
    def __init__(self, d_model, num_heads):
        super(CrossModalAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        
    def forward(self, query_modality, key_value_modality):
        """
        Args:
            query_modality: [batch, seq_len, d_model] - modality making queries
            key_value_modality: [batch, seq_len, d_model] - modality providing keys/values
        """
        batch_size, seq_len, _ = query_modality.shape
        
        # Project to multiple heads
        Q = self.query_proj(query_modality).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key_proj(key_value_modality).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_proj(key_value_modality).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        
        # Concatenate heads and project
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        return self.output_proj(attention_output)

class InterleavedFusionLayer(nn.Module):
    """Single layer of interleaved self-attention and cross-attention"""
    def __init__(self, d_model, num_heads, ff_dim):
        super(InterleavedFusionLayer, self).__init__()
        self.d_model = d_model
        
        # Self-attention layers
        self.ts_self_attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.txt_self_attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        
        # Cross-attention layers
        self.ts_cross_attention = CrossModalAttention(d_model, num_heads)
        self.txt_cross_attention = CrossModalAttention(d_model, num_heads)
        
        # Layer normalization
        self.ts_norm1 = nn.LayerNorm(d_model)
        self.ts_norm2 = nn.LayerNorm(d_model)
        self.ts_norm3 = nn.LayerNorm(d_model)
        
        self.txt_norm1 = nn.LayerNorm(d_model)
        self.txt_norm2 = nn.LayerNorm(d_model)
        self.txt_norm3 = nn.LayerNorm(d_model)
        
        # Feed-forward networks
        self.ts_ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model)
        )
        self.txt_ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model)
        )
        
    def forward(self, ts_input, txt_input):
        """
        Args:
            ts_input: [batch, seq_len, d_model] - time series representations
            txt_input: [batch, seq_len, d_model] - text representations
        """
        # Self-attention for both modalities
        ts_self_out, _ = self.ts_self_attention(ts_input, ts_input, ts_input)
        ts_self_out = self.ts_norm1(ts_input + ts_self_out)
        
        txt_self_out, _ = self.txt_self_attention(txt_input, txt_input, txt_input)
        txt_self_out = self.txt_norm1(txt_input + txt_self_out)
        
        # Cross-attention
        ts_cross_out = self.ts_cross_attention(ts_self_out, txt_self_out)
        ts_cross_out = self.ts_norm2(ts_self_out + ts_cross_out)
        
        txt_cross_out = self.txt_cross_attention(txt_self_out, ts_self_out)
        txt_cross_out = self.txt_norm2(txt_self_out + txt_cross_out)
        
        # Feed-forward
        ts_ff_out = self.ts_ff(ts_cross_out)
        ts_output = self.ts_norm3(ts_cross_out + ts_ff_out)
        
        txt_ff_out = self.txt_ff(txt_cross_out)
        txt_output = self.txt_norm3(txt_cross_out + txt_ff_out)
        
        return ts_output, txt_output

class ClinicalNoteEncoder(nn.Module):
    """Clinical note encoder using pretrained language model"""
    def __init__(self, model_name="yikuan8/Clinical-Longformer", max_length=1024, hidden_dim=128):
        super(ClinicalNoteEncoder, self).__init__()
        self.max_length = max_length
        self.hidden_dim = hidden_dim
        
        # Load pretrained model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Projection layer to match time series hidden dimension
        self.projection = nn.Linear(self.encoder.config.hidden_size, hidden_dim)
        
    def forward(self, clinical_notes):
        """
        Args:
            clinical_notes: List of strings or batch of tokenized inputs
        Returns:
            note_embeddings: [batch, num_notes, hidden_dim]
        """
        if isinstance(clinical_notes[0], str):
            # Tokenize if input is raw text
            inputs = self.tokenizer(
                clinical_notes, 
                padding=True, 
                truncation=True, 
                max_length=self.max_length,
                return_tensors="pt"
            )
        else:
            inputs = clinical_notes
            
        # Get embeddings
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            # Use [CLS] token embedding
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [batch, hidden_size]
        
        # Project to desired dimension
        note_embeddings = self.projection(cls_embeddings)  # [batch, hidden_dim]
        
        return note_embeddings

class MultimodalEHRModel(nn.Module):
    """Complete multimodal EHR prediction model"""
    def __init__(self, 
                 ts_input_dim,
                 hidden_dim=128,
                 num_fusion_layers=3,
                 num_heads=8,
                 ff_dim=512,
                 max_seq_len=48,
                 num_classes=1,
                 text_model_name="yikuan8/Clinical-Longformer"):
        super(MultimodalEHRModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        
        # Time series encoder (UTDE)
        self.ts_encoder = UTDE(
            input_dim=ts_input_dim,
            hidden_dim=hidden_dim,
            max_len=max_seq_len
        )
        
        # Clinical note encoder
        self.note_encoder = ClinicalNoteEncoder(
            model_name=text_model_name,
            hidden_dim=hidden_dim
        )
        
        # mTAND for clinical notes
        self.note_mtand = mTAND(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim
        )
        
        # Multimodal fusion layers
        self.fusion_layers = nn.ModuleList([
            InterleavedFusionLayer(hidden_dim, num_heads, ff_dim)
            for _ in range(num_fusion_layers)
        ])
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, ts_data, ts_times, clinical_notes, note_times, query_times):
        """
        Args:
            ts_data: [batch, ts_seq_len, ts_features] - irregular time series data
            ts_times: [batch, ts_seq_len] - time series observation times
            clinical_notes: List of clinical note strings for each patient
            note_times: [batch, num_notes] - note taking times
            query_times: [batch, query_seq_len] - regular query times for interpolation
        """
        batch_size = ts_data.shape[0]
        
        # Encode time series using UTDE
        ts_embeddings = self.ts_encoder(ts_data, ts_times, query_times)  # [batch, query_len, hidden_dim]
        
        # Encode clinical notes
        note_embeddings_list = []
        for i in range(batch_size):
            # Get embeddings for each note of this patient
            patient_notes = clinical_notes[i] if isinstance(clinical_notes[i], list) else [clinical_notes[i]]
            note_embs = []
            for note in patient_notes:
                note_emb = self.note_encoder([note])  # [1, hidden_dim]
                note_embs.append(note_emb)
            
            if note_embs:
                patient_note_embs = torch.stack(note_embs, dim=1)  # [1, num_notes, hidden_dim]
            else:
                # Handle case with no notes
                patient_note_embs = torch.zeros(1, 1, self.hidden_dim, device=ts_data.device)
            
            note_embeddings_list.append(patient_note_embs)
        
        # Stack note embeddings - pad to same length
        max_notes = max(emb.shape[1] for emb in note_embeddings_list)
        padded_note_embeddings = []
        for emb in note_embeddings_list:
            if emb.shape[1] < max_notes:
                padding = torch.zeros(1, max_notes - emb.shape[1], self.hidden_dim, device=emb.device)
                emb = torch.cat([emb, padding], dim=1)
            padded_note_embeddings.append(emb)
        
        note_embeddings = torch.cat(padded_note_embeddings, dim=0)  # [batch, max_notes, hidden_dim]
        
        # Apply mTAND to clinical note embeddings
        # Pad note_times to match max_notes
        padded_note_times = []
        for i in range(batch_size):
            patient_times = note_times[i]
            if len(patient_times) < max_notes:
                padding = torch.full((max_notes - len(patient_times),), patient_times[-1] if len(patient_times) > 0 else 0.0, device=note_times.device)
                patient_times = torch.cat([patient_times, padding])
            padded_note_times.append(patient_times)
        
        padded_note_times = torch.stack(padded_note_times)  # [batch, max_notes]
        
        txt_embeddings = self.note_mtand(
            note_embeddings, 
            padded_note_times, 
            query_times
        )  # [batch, query_len, hidden_dim]
        
        # Apply fusion layers
        current_ts = ts_embeddings
        current_txt = txt_embeddings
        
        for fusion_layer in self.fusion_layers:
            current_ts, current_txt = fusion_layer(current_ts, current_txt)
        
        # Get final representations (last time step)
        final_ts = current_ts[:, -1, :]  # [batch, hidden_dim]
        final_txt = current_txt[:, -1, :]  # [batch, hidden_dim]
        
        # Concatenate and classify
        final_repr = torch.cat([final_ts, final_txt], dim=-1)  # [batch, 2*hidden_dim]
        logits = self.classifier(final_repr)  # [batch, num_classes]
        
        return logits

# Import the UTDE and mTAND classes (assuming they're defined above or imported)
try:
    from .utde import UTDE, mTAND
except ImportError:
    # If running as standalone, the classes should be defined in the same file or imported appropriately
    pass
