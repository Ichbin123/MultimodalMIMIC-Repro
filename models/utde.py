import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class Time2Vec(nn.Module):
    """Time2Vec encoding for temporal information"""
    
    def __init__(self, d_v):
        super(Time2Vec, self).__init__()
        self.d_v = d_v
        self.w = nn.Parameter(torch.randn(d_v))
        self.phi = nn.Parameter(torch.randn(d_v))
    
    def forward(self, tau):
        """
        Args:
            tau: Time points tensor of shape [batch_size, seq_len] or [seq_len]
        Returns:
            Time embeddings of shape [..., d_v]
        """
        # Expand dimensions for broadcasting
        original_shape = tau.shape
        tau = tau.unsqueeze(-1)  # [..., 1]
        
        # Linear term for first dimension
        linear_term = self.w[0] * tau + self.phi[0]
        
        # Sine terms for remaining dimensions
        sine_terms = torch.sin(self.w[1:] * tau + self.phi[1:])
        
        # Combine linear and sine terms
        time_embedding = torch.cat([linear_term, sine_terms], dim=-1)
        
        return time_embedding


class mTANDModule(nn.Module):
    """Multi-Time Attention Network for Discretized time series"""
    
    def __init__(self, d_h, d_v=64, num_heads=8):
        super(mTANDModule, self).__init__()
        self.d_h = d_h
        self.d_v = d_v
        self.num_heads = num_heads
        
        # Multiple Time2Vec encodings
        self.time_encoders = nn.ModuleList([
            Time2Vec(d_v) for _ in range(num_heads)
        ])
        
        # Attention parameters for each head
        self.w_q = nn.ModuleList([
            nn.Linear(d_v, d_v) for _ in range(num_heads)
        ])
        self.w_k = nn.ModuleList([
            nn.Linear(d_v, d_v) for _ in range(num_heads)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(num_heads, d_h)
        
    def forward(self, alpha, t_obs, x_obs):
        """
        Args:
            alpha: Regular time points for interpolation [alpha]
            t_obs: Irregular observation times [l_obs]
            x_obs: Observation values [l_obs]
        Returns:
            Interpolated embeddings at alpha [alpha, d_h]
        """
        batch_outputs = []
        
        for v in range(self.num_heads):
            # Encode time points
            theta_alpha = self.time_encoders[v](alpha)  # [alpha, d_v]
            theta_t_obs = self.time_encoders[v](t_obs)  # [l_obs, d_v]
            
            # Compute attention weights
            queries = self.w_q[v](theta_alpha)  # [alpha, d_v]
            keys = self.w_k[v](theta_t_obs)     # [l_obs, d_v]
            
            # Attention mechanism
            attention_scores = torch.matmul(queries, keys.transpose(-2, -1))  # [alpha, l_obs]
            attention_scores = attention_scores / math.sqrt(self.d_v)
            attention_weights = F.softmax(attention_scores, dim=-1)
            
            # Apply attention to values
            if len(x_obs.shape) == 1:
                x_obs = x_obs.unsqueeze(-1)
            interpolated = torch.matmul(attention_weights, x_obs)  # [alpha, 1]
            
            batch_outputs.append(interpolated)
        
        # Concatenate and project
        concatenated = torch.cat(batch_outputs, dim=-1)  # [alpha, num_heads]
        output = self.output_proj(concatenated)  # [alpha, d_h]
        
        return output


class ImputationModule(nn.Module):
    """Imputation-based TDE module"""
    
    def __init__(self, d_m, d_h, alpha):
        super(ImputationModule, self).__init__()
        self.d_m = d_m
        self.d_h = d_h
        self.alpha = alpha
        
        # 1D Convolutional layer
        self.conv1d = nn.Conv1d(d_m, d_h, kernel_size=1, stride=1)
        
    def forward(self, x_ts, t_ts, global_means):
        """
        Args:
            x_ts: Irregular time series observations [d_m, l_obs_per_feature]
            t_ts: Observation times [d_m, l_obs_per_feature]  
            global_means: Global mean for each feature [d_m]
        Returns:
            Imputation embeddings [alpha, d_h]
        """
        # Discretize and impute for each feature
        regular_series = torch.zeros(self.d_m, self.alpha)
        
        for j in range(self.d_m):
            feature_obs = x_ts[j]
            feature_times = t_ts[j]
            
            # Remove invalid observations (NaN, negative times, etc.)
            valid_mask = ~torch.isnan(feature_obs) & (feature_times >= 0)
            valid_obs = feature_obs[valid_mask]
            valid_times = feature_times[valid_mask]
            
            if len(valid_obs) == 0:
                # No valid observations, use global mean
                regular_series[j] = global_means[j]
                continue
            
            # Discretize observations to hourly intervals
            discretized = torch.full((self.alpha,), float('nan'))
            
            for i, obs_time in enumerate(valid_times):
                time_idx = int(obs_time.item())
                if 0 <= time_idx < self.alpha:
                    # If multiple observations in same interval, use the last one
                    discretized[time_idx] = valid_obs[i]
            
            # Forward fill missing values
            last_obs = global_means[j]
            for t in range(self.alpha):
                if torch.isnan(discretized[t]):
                    discretized[t] = last_obs
                else:
                    last_obs = discretized[t]
            
            regular_series[j] = discretized
        
        # Apply 1D convolution
        # regular_series: [d_m, alpha] -> [alpha, d_h]
        embeddings = self.conv1d(regular_series.unsqueeze(0)).squeeze(0).transpose(0, 1)
        
        return embeddings


class UTDEModule(nn.Module):
    """Unified TDE module combining imputation and mTAND"""
    
    def __init__(self, d_m, d_h, alpha, gate_level='hidden_space', d_v=64, num_heads=8):
        super(UTDEModule, self).__init__()
        self.gate_level = gate_level
        self.d_h = d_h
        self.alpha = alpha
        
        # TDE submodules
        self.imputation = ImputationModule(d_m, d_h, alpha)
        self.mtand = mTANDModule(d_h, d_v, num_heads)
        
        # Gating mechanism
        if gate_level == 'patient':
            self.gate_mlp = nn.Sequential(
                nn.Linear(2 * d_h * alpha, d_h),
                nn.ReLU(),
                nn.Linear(d_h, 1),
                nn.Sigmoid()
            )
        elif gate_level == 'temporal':
            self.gate_mlp = nn.Sequential(
                nn.Linear(2 * d_h, d_h),
                nn.ReLU(),
                nn.Linear(d_h, alpha),
                nn.Sigmoid()
            )
        elif gate_level == 'hidden_space':
            self.gate_mlp = nn.Sequential(
                nn.Linear(2 * d_h, d_h),
                nn.ReLU(),
                nn.Linear(d_h, d_h),
                nn.Sigmoid()
            )
    
    def forward(self, x_ts, t_ts, global_means):
        """
        Args:
            x_ts: Irregular time series [d_m, max_obs]
            t_ts: Observation times [d_m, max_obs]
            global_means: Global means [d_m]
        Returns:
            Unified embeddings [alpha, d_h]
        """
        # Get imputation embeddings
        e_imp = self.imputation(x_ts, t_ts, global_means)  # [alpha, d_h]
        
        # Get mTAND embeddings for each feature
        alpha_points = torch.arange(self.alpha, dtype=torch.float32)
        e_attn_list = []
        
        for j in range(x_ts.shape[0]):  # For each feature
            feature_obs = x_ts[j]
            feature_times = t_ts[j]
            
            # Remove invalid observations
            valid_mask = ~torch.isnan(feature_obs) & (feature_times >= 0)
            valid_obs = feature_obs[valid_mask]
            valid_times = feature_times[valid_mask]
            
            if len(valid_obs) > 0:
                # Apply mTAND for this feature
                feature_interp = self.mtand(alpha_points, valid_times, valid_obs)
                e_attn_list.append(feature_interp.unsqueeze(-1))
            else:
                # No valid observations, use zeros
                e_attn_list.append(torch.zeros(self.alpha, self.d_h, 1))
        
        # Combine all features
        e_attn = torch.cat(e_attn_list, dim=-1).mean(dim=-1)  # [alpha, d_h]
        
        # Apply gating mechanism
        concat_emb = torch.cat([e_imp, e_attn], dim=-1)  # [alpha, 2*d_h]
        
        if self.gate_level == 'patient':
            # Single gate value for entire patient
            gate_input = concat_emb.flatten()
            g = self.gate_mlp(gate_input)  # [1]
            z_ts = g * e_imp + (1 - g) * e_attn
            
        elif self.gate_level == 'temporal':
            # Different gate for each time step
            gate_input = concat_emb.mean(dim=-1)  # [alpha]
            g = self.gate_mlp(gate_input.mean(keepdim=True).expand(concat_emb.shape[0], -1))  # [alpha]
            g = g.unsqueeze(-1)  # [alpha, 1]
            z_ts = g * e_imp + (1 - g) * e_attn
            
        elif self.gate_level == 'hidden_space':
            # Different gate for each hidden dimension
            g = self.gate_mlp(concat_emb)  # [alpha, d_h]
            z_ts = g * e_imp + (1 - g) * e_attn
        
        return z_ts


class TransformerEncoder(nn.Module):
    """Transformer encoder for time series backbone"""
    
    def __init__(self, d_h, num_heads=8, num_layers=3, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_h,
            nhead=num_heads,
            dim_feedforward=4 * d_h,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        self.pos_encoding = PositionalEncoding(d_h, dropout)
    
    def forward(self, x):
        """
        Args:
            x: Input embeddings [batch_size, seq_len, d_h]
        Returns:
            Encoded representations [batch_size, seq_len, d_h]
        """
        x = self.pos_encoding(x)
        output = self.transformer(x)
        return output


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class TimeSeriesModel(nn.Module):
    """Complete time series model with UTDE and Transformer backbone"""
    
    def __init__(self, d_m, d_h=128, alpha=48, gate_level='hidden_space', 
                 num_heads=8, num_layers=3, num_classes=1):
        super(TimeSeriesModel, self).__init__()
        
        self.utde = UTDEModule(d_m, d_h, alpha, gate_level)
        self.transformer = TransformerEncoder(d_h, num_heads, num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_h, d_h // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_h // 2, num_classes)
        )
    
    def forward(self, x_ts, t_ts, global_means):
        """
        Args:
            x_ts: Irregular time series [batch_size, d_m, max_obs]
            t_ts: Observation times [batch_size, d_m, max_obs]
            global_means: Global feature means [d_m]
        Returns:
            Predictions [batch_size, num_classes]
        """
        batch_size = x_ts.shape[0]
        embeddings_list = []
        
        # Process each patient
        for i in range(batch_size):
            patient_emb = self.utde(x_ts[i], t_ts[i], global_means)  # [alpha, d_h]
            embeddings_list.append(patient_emb)
        
        # Stack embeddings
        embeddings = torch.stack(embeddings_list, dim=0)  # [batch_size, alpha, d_h]
        
        # Apply Transformer
        encoded = self.transformer(embeddings)  # [batch_size, alpha, d_h]
        
        # Extract last hidden state for classification
        last_hidden = encoded[:, -1, :]  # [batch_size, d_h]
        
        # Classify
        logits = self.classifier(last_hidden)  # [batch_size, num_classes]
        
        return logits
        
    def get_embeddings(self, x_ts, t_ts, global_means):
        """Get intermediate embeddings for multimodal fusion"""
        batch_size = x_ts.shape[0]
        embeddings_list = []
        
        # Process each patient
        for i in range(batch_size):
            patient_emb = self.utde(x_ts[i], t_ts[i], global_means)  # [alpha, d_h]
            embeddings_list.append(patient_emb)
        
        # Stack embeddings
        embeddings = torch.stack(embeddings_list, dim=0)  # [batch_size, alpha, d_h]
        
        # Apply Transformer
        encoded = self.transformer(embeddings)  # [batch_size, alpha, d_h]
        
        return encoded
