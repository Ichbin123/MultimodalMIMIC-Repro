import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Final linear projection
        output = self.w_o(attn_output)
        
        return output, attn_weights
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, V)
        
        return output, attn_weights


class CrossModalAttention(nn.Module):
    """Cross-modal attention between different modalities"""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(CrossModalAttention, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        
    def forward(self, query_modality, key_value_modality, mask=None):
        """
        Args:
            query_modality: Query modality [batch_size, seq_len, d_model]
            key_value_modality: Key/Value modality [batch_size, seq_len, d_model]
        Returns:
            Cross-attended output [batch_size, seq_len, d_model]
        """
        output, attn_weights = self.attention(
            query=query_modality,
            key=key_value_modality,
            value=key_value_modality,
            mask=mask
        )
        return output, attn_weights


class InterleavedFusionLayer(nn.Module):
    """Single interleaved fusion layer with self-attention and cross-attention"""
    
    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.1):
        super(InterleavedFusionLayer, self).__init__()
        
        d_ff = d_ff or 4 * d_model
        
        # Self-attention for each modality
        self.self_attn_ts = MultiHeadAttention(d_model, num_heads, dropout)
        self.self_attn_txt = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Cross-attention between modalities
        self.cross_attn_ts = CrossModalAttention(d_model, num_heads, dropout)
        self.cross_attn_txt = CrossModalAttention(d_model, num_heads, dropout)
        
        # Feedforward networks
        self.ffn_ts = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.ffn_txt = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # Layer normalization
        self.norm1_ts = nn.LayerNorm(d_model)
        self.norm1_txt = nn.LayerNorm(d_model)
        self.norm2_ts = nn.LayerNorm(d_model)
        self.norm2_txt = nn.LayerNorm(d_model)
        self.norm3_ts = nn.LayerNorm(d_model)
        self.norm3_txt = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, z_ts, z_txt):
        """
        Args:
            z_ts: Time series representations [batch_size, alpha, d_model]
            z_txt: Text representations [batch_size, alpha, d_model]
        Returns:
            Updated representations (z_ts_new, z_txt_new)
        """
        # Self-attention for each modality
        z_ts_self, _ = self.self_attn_ts(z_ts, z_ts, z_ts)
        z_txt_self, _ = self.self_attn_txt(z_txt, z_txt, z_txt)
        
        # Residual connection and normalization
        z_ts_hat = self.norm1_ts(z_ts + self.dropout(z_ts_self))
        z_txt_hat = self.norm1_txt(z_txt + self.dropout(z_txt_self))
        
        # Cross-attention
        z_ts_cross, _ = self.cross_attn_ts(z_ts_hat, z_txt_hat)
        z_txt_cross, _ = self.cross_attn_txt(z_txt_hat, z_ts_hat)
        
        # Residual connection and normalization
        z_ts_cross = self.norm2_ts(z_ts_hat + self.dropout(z_ts_cross))
        z_txt_cross = self.norm2_txt(z_txt_hat + self.dropout(z_txt_cross))
        
        # Feedforward networks
        z_ts_ffn = self.ffn_ts(z_ts_cross)
        z_txt_ffn = self.ffn_txt(z_txt_cross)
        
        # Final residual connection and normalization
        z_ts_out = self.norm3_ts(z_ts_cross + self.dropout(z_ts_ffn))
        z_txt_out = self.norm3_txt(z_txt_cross + self.dropout(z_txt_ffn))
        
        return z_ts_out, z_txt_out


class InterleavedMultimodalFusion(nn.Module):
    """Complete interleaved multimodal fusion module"""
    
    def __init__(self, d_model, num_heads=8, num_layers=3, d_ff=None, dropout=0.1):
        super(InterleavedMultimodalFusion, self).__init__()
        
        self.num_layers = num_layers
        
        # Stack of interleaved fusion layers
        self.fusion_layers = nn.ModuleList([
            InterleavedFusionLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, z_ts, z_txt):
        """
        Args:
            z_ts: Time series embeddings [batch_size, alpha, d_model]
            z_txt: Text embeddings [batch_size, alpha, d_model]
        Returns:
            Fused representations (z_ts_final, z_txt_final)
        """
        # Apply J layers of interleaved fusion
        for layer in self.fusion_layers:
            z_ts, z_txt = layer(z_ts, z_txt)
        
        return z_ts, z_txt


class MultimodalModel(nn.Module):
    """Complete multimodal model combining UTDE and mTAND_txt with interleaved fusion"""
    
    def __init__(self, d_m, d_h=128, alpha=48, num_classes=1, 
                 gate_level='hidden_space', model_name='yikuan8/Clinical-Longformer',
                 max_length=1024, num_heads=8, num_layers=3, fusion_layers=3):
        super(MultimodalModel, self).__init__()
        
        self.d_h = d_h
        self.alpha = alpha
        
        # Import models
        from .utde import TimeSeriesModel
        from .text_encoder import IrregularClinicalNotesModel
        
        # Individual modality models (without final classifier)
        self.ts_model = TimeSeriesModel(
            d_m=d_m, d_h=d_h, alpha=alpha, gate_level=gate_level,
            num_heads=num_heads, num_layers=num_layers, num_classes=num_classes
        )
        
        self.txt_model = IrregularClinicalNotesModel(
            d_h=d_h, alpha=alpha, model_name=model_name, max_length=max_length,
            num_heads=num_heads, num_layers=num_layers, num_classes=num_classes
        )
        
        # Multimodal fusion
        self.fusion = InterleavedMultimodalFusion(
            d_model=d_h, num_heads=num_heads, num_layers=fusion_layers
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(2 * d_h, d_h),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_h, d_h // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_h // 2, num_classes)
        )
    
    def forward(self, x_ts, t_ts, global_means, clinical_notes_batch, note_times_batch):
        """
        Args:
            x_ts: Time series data [batch_size, d_m, max_obs]
            t_ts: Time series timestamps [batch_size, d_m, max_obs]
            global_means: Global feature means [d_m]
            clinical_notes_batch: Batch of clinical notes
            note_times_batch: Batch of note timestamps
        Returns:
            Multimodal predictions [batch_size, num_classes]
        """
        # Get embeddings from each modality
        z_ts = self.ts_model.get_embeddings(x_ts, t_ts, global_means)  # [batch_size, alpha, d_h]
        z_txt = self.txt_model.get_embeddings(clinical_notes_batch, note_times_batch)  # [batch_size, alpha, d_h]
        
        # Apply multimodal fusion
        z_ts_fused, z_txt_fused = self.fusion(z_ts, z_txt)
        
        # Extract final representations (last hidden states)
        ts_final = z_ts_fused[:, -1, :]  # [batch_size, d_h]
        txt_final = z_txt_fused[:, -1, :]  # [batch_size, d_h]
        
        # Concatenate modalities
        multimodal_repr = torch.cat([ts_final, txt_final], dim=-1)  # [batch_size, 2*d_h]
        
        # Final prediction
        logits = self.classifier(multimodal_repr)  # [batch_size, num_classes]
        
        return logits


# Baseline fusion methods for comparison
class ConcatenationFusion(nn.Module):
    """Simple concatenation fusion baseline"""
    
    def __init__(self, d_h, num_classes):
        super(ConcatenationFusion, self).__init__()
        self.classifier = nn.Linear(2 * d_h, num_classes)
    
    def forward(self, z_ts, z_txt):
        # Use last hidden states
        ts_repr = z_ts[:, -1, :]
        txt_repr = z_txt[:, -1, :]
        concat_repr = torch.cat([ts_repr, txt_repr], dim=-1)
        return self.classifier(concat_repr)


class TensorFusion(nn.Module):
    """Tensor fusion baseline"""
    
    def __init__(self, d_h, num_classes, dropout=0.1):
        super(TensorFusion, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # Tensor fusion creates d_h * d_h dimensional output
        self.classifier = nn.Sequential(
            nn.Linear(d_h * d_h, d_h),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_h, num_classes)
        )
    
    def forward(self, z_ts, z_txt):
        ts_repr = z_ts[:, -1, :]  # [batch_size, d_h]
        txt_repr = z_txt[:, -1, :]  # [batch_size, d_h]
        
        # Outer product for tensor fusion
        fused = torch.bmm(ts_repr.unsqueeze(2), txt_repr.unsqueeze(1))  # [batch_size, d_h, d_h]
        fused = fused.view(fused.size(0), -1)  # [batch_size, d_h*d_h]
        
        fused = self.dropout(fused)
        return self.classifier(fused)


class MAGFusion(nn.Module):
    """Multimodal Adaptation Gate fusion baseline"""
    
    def __init__(self, d_h, num_classes, dropout=0.1):
        super(MAGFusion, self).__init__()
        # Adaptation gates
        self.gate_ts = nn.Sequential(
            nn.Linear(d_h, d_h),
            nn.Sigmoid()
        )
        self.gate_txt = nn.Sequential(
            nn.Linear(d_h, d_h),
            nn.Sigmoid()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(2 * d_h, d_h),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_h, num_classes)
        )
    
    def forward(self, z_ts, z_txt):
        ts_repr = z_ts[:, -1, :]  # [batch_size, d_h]
        txt_repr = z_txt[:, -1, :]  # [batch_size, d_h]
        
        # Apply adaptation gates
        gate_ts = self.gate_ts(txt_repr)
        gate_txt = self.gate_txt(ts_repr)
        
        # Modulate representations
        ts_adapted = ts_repr * gate_ts
        txt_adapted = txt_repr * gate_txt
        
        # Concatenate and classify
        fused = torch.cat([ts_adapted, txt_adapted], dim=-1)
        return self.classifier(fused)
