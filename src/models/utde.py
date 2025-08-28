import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Time2Vec(nn.Module):
    """Time2Vec implementation for encoding temporal information"""
    def __init__(self, time_dim):
        super(Time2Vec, self).__init__()
        self.time_dim = time_dim
        self.linear_weights = nn.Parameter(torch.randn(1))
        self.linear_bias = nn.Parameter(torch.randn(1))
        self.periodic_weights = nn.Parameter(torch.randn(time_dim - 1))
        self.periodic_bias = nn.Parameter(torch.randn(time_dim - 1))
        
    def forward(self, time_points):
        # time_points: [batch_size, seq_len] or [seq_len]
        if len(time_points.shape) == 1:
            time_points = time_points.unsqueeze(0)
        
        # Linear component
        linear_part = self.linear_weights * time_points + self.linear_bias
        
        # Periodic components
        periodic_parts = torch.sin(
            self.periodic_weights.unsqueeze(0).unsqueeze(0) * time_points.unsqueeze(-1) + 
            self.periodic_bias.unsqueeze(0).unsqueeze(0)
        )
        
        # Concatenate linear and periodic parts
        time_embedding = torch.cat([linear_part.unsqueeze(-1), periodic_parts], dim=-1)
        return time_embedding

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len = query.shape[:2]
        
        # Project to multiple heads
        Q = self.query_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        
        # Concatenate heads and project
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        return self.output_proj(attention_output)

class mTAND(nn.Module):
    """Multi-time Attention Networks for irregular time series"""
    def __init__(self, input_dim, hidden_dim, num_time_embeddings=8, time_dim=64):
        super(mTAND, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_time_embeddings = num_time_embeddings
        self.time_dim = time_dim
        
        # Multiple Time2Vec embeddings
        self.time_embeddings = nn.ModuleList([
            Time2Vec(time_dim) for _ in range(num_time_embeddings)
        ])
        
        # Attention mechanisms for each time embedding
        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(time_dim, num_heads=4) for _ in range(num_time_embeddings)
        ])
        
        # Final projection layer
        self.output_proj = nn.Linear(num_time_embeddings * input_dim, hidden_dim)
        
    def forward(self, values, time_points, query_times):
        """
        Args:
            values: [batch_size, seq_len, input_dim] - irregular observations
            time_points: [batch_size, seq_len] - observation times
            query_times: [batch_size, query_len] - regular query times
        """
        batch_size, query_len = query_times.shape
        outputs = []
        
        for i, (time_emb, attn_layer) in enumerate(zip(self.time_embeddings, self.attention_layers)):
            # Encode time points
            time_encoded_keys = time_emb(time_points)  # [batch_size, seq_len, time_dim]
            time_encoded_queries = time_emb(query_times)  # [batch_size, query_len, time_dim]
            
            # Apply attention for each feature dimension
            feature_outputs = []
            for j in range(self.input_dim):
                feature_values = values[:, :, j:j+1]  # [batch_size, seq_len, 1]
                
                # Expand values to match time embedding dimension
                expanded_values = feature_values.expand(-1, -1, self.time_dim)
                
                # Apply attention
                interpolated = attn_layer(
                    time_encoded_queries, 
                    time_encoded_keys, 
                    expanded_values
                )
                
                # Take mean across time dimension to get single value per query time
                interpolated_feature = interpolated.mean(dim=-1, keepdim=True)  # [batch_size, query_len, 1]
                feature_outputs.append(interpolated_feature)
            
            # Concatenate features
            output = torch.cat(feature_outputs, dim=-1)  # [batch_size, query_len, input_dim]
            outputs.append(output)
        
        # Concatenate all time embedding outputs and project
        combined_output = torch.cat(outputs, dim=-1)  # [batch_size, query_len, num_time_embeddings * input_dim]
        final_output = self.output_proj(combined_output)  # [batch_size, query_len, hidden_dim]
        
        return final_output

class UTDE(nn.Module):
    """Unified TDE module combining imputation and mTAND"""
    def __init__(self, input_dim, hidden_dim, max_len, gate_level='hidden'):
        super(UTDE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.gate_level = gate_level
        
        # Imputation branch
        self.imputation_conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        
        # mTAND branch
        self.mtand = mTAND(input_dim, hidden_dim)
        
        # Gating mechanism
        if gate_level == 'patient':
            self.gate_net = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
        elif gate_level == 'temporal':
            self.gate_net = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, max_len),
                nn.Sigmoid()
            )
        else:  # hidden level
            self.gate_net = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid()
            )
    
    def imputation_forward(self, irregular_data, time_points, query_times):
        """Simple imputation approach"""
        batch_size, query_len = query_times.shape
        imputed_data = torch.zeros(batch_size, query_len, self.input_dim, device=irregular_data.device)
        
        # Simple forward fill imputation
        for b in range(batch_size):
            last_values = torch.zeros(self.input_dim, device=irregular_data.device)
            for q, query_time in enumerate(query_times[b]):
                # Find observations before this query time
                valid_mask = time_points[b] <= query_time
                if valid_mask.any():
                    # Get the last observation before query time
                    valid_indices = torch.where(valid_mask)[0]
                    last_idx = valid_indices[-1]
                    last_values = irregular_data[b, last_idx]
                imputed_data[b, q] = last_values
        
        # Apply 1D convolution
        imputed_data = imputed_data.transpose(1, 2)  # [batch, features, time]
        imputed_embeddings = self.imputation_conv(imputed_data)
        imputed_embeddings = imputed_embeddings.transpose(1, 2)  # [batch, time, hidden]
        
        return imputed_embeddings
    
    def forward(self, irregular_data, time_points, query_times):
        """
        Args:
            irregular_data: [batch_size, seq_len, input_dim]
            time_points: [batch_size, seq_len]
            query_times: [batch_size, query_len]
        """
        # Get imputation embeddings
        imputation_embeddings = self.imputation_forward(irregular_data, time_points, query_times)
        
        # Get mTAND embeddings
        mtand_embeddings = self.mtand(irregular_data, time_points, query_times)
        
        # Compute gate values
        combined_features = torch.cat([imputation_embeddings, mtand_embeddings], dim=-1)
        
        if self.gate_level == 'patient':
            # Average over time dimension for patient-level gate
            patient_features = combined_features.mean(dim=1)  # [batch, 2*hidden]
            gate = self.gate_net(patient_features).unsqueeze(1).unsqueeze(1)  # [batch, 1, 1]
        elif self.gate_level == 'temporal':
            # Average over hidden dimension for temporal-level gate
            temporal_features = combined_features.mean(dim=-1)  # [batch, time]
            gate_logits = self.gate_net(combined_features.view(-1, 2 * self.hidden_dim))  # [batch*time, max_len]
            gate = gate_logits.view(combined_features.shape[0], combined_features.shape[1], -1)[:, :, :1]  # [batch, time, 1]
        else:  # hidden level
            gate = self.gate_net(combined_features.view(-1, 2 * self.hidden_dim))
            gate = gate.view(combined_features.shape[0], combined_features.shape[1], self.hidden_dim)
        
        # Combine embeddings using gate
        final_embeddings = gate * imputation_embeddings + (1 - gate) * mtand_embeddings
        
        return final_embeddings
