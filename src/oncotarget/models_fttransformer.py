"""FT-Transformer model for continuous tabular data with self-supervised pretraining support."""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional


class FeatureTokenizer(nn.Module):
    """Tokenize continuous features into transformer tokens."""
    
    def __init__(
        self, 
        n_features: int, 
        d_model: int, 
        shared_linear: bool = True
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        
        if shared_linear:
            # Shared linear projection + per-feature bias/scale
            self.shared_proj = nn.Linear(1, d_model)
            self.feature_bias = nn.Parameter(torch.zeros(n_features, d_model))
            self.feature_scale = nn.Parameter(torch.ones(n_features, d_model))
        else:
            # Per-feature linear projection
            self.feature_projs = nn.ModuleList([
                nn.Linear(1, d_model) for _ in range(n_features)
            ])
            
        self.shared_linear = shared_linear
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, n_features] continuous features
        Returns:
            tokens: [batch_size, n_features, d_model]
        """
        batch_size, n_features = x.shape
        assert n_features == self.n_features
        
        if self.shared_linear:
            # Apply shared projection to each feature
            tokens = self.shared_proj(x.unsqueeze(-1))  # [B, F, d_model]
            # Apply per-feature bias and scale
            tokens = tokens * self.feature_scale + self.feature_bias
        else:
            # Apply per-feature projections
            tokens = torch.stack([
                self.feature_projs[i](x[:, i:i+1]) 
                for i in range(n_features)
            ], dim=1)  # [B, F, d_model]
            
        return self.layer_norm(tokens)


class FTTransformer(nn.Module):
    """Feature Tokenizer + Transformer for continuous tabular data."""
    
    def __init__(
        self,
        n_features: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        ff_mult: int = 4,
        dropout: float = 0.1,
        cls_token: bool = True,
        task: str = "binary"  # "binary", "regression", or "reconstruction"
    ) -> None:
        super().__init__()
        
        self.n_features = n_features
        self.d_model = d_model
        self.cls_token = cls_token
        self.task = task
        
        # Feature tokenizer
        self.tokenizer = FeatureTokenizer(n_features, d_model)
        
        # Learnable CLS token
        if cls_token:
            self.cls_embedding = nn.Parameter(torch.randn(1, 1, d_model))
            
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            activation="relu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Task-specific heads
        if task == "binary":
            self.head = nn.Linear(d_model, 1)
        elif task == "regression":
            self.head = nn.Linear(d_model, 1) 
        elif task == "reconstruction":
            self.head = nn.Linear(d_model, 1)  # Each token predicts its own feature value
        else:
            raise ValueError(f"Unknown task: {task}")
            
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, n_features] continuous features
            mask: [batch_size, n_features] optional mask for reconstruction (1=masked)
        Returns:
            output: task-dependent shape
        """
        batch_size = x.shape[0]
        
        # Apply masking if provided (for reconstruction)
        if mask is not None:
            x_masked = x.clone()
            x_masked[mask.bool()] = 0.0  # Zero out masked positions
        else:
            x_masked = x
            
        # Tokenize features
        tokens = self.tokenizer(x_masked)  # [B, F, d_model]
        
        # Add CLS token if enabled
        if self.cls_token:
            cls_tokens = self.cls_embedding.expand(batch_size, -1, -1)
            tokens = torch.cat([cls_tokens, tokens], dim=1)  # [B, 1+F, d_model]
            
        # Apply transformer
        encoded = self.transformer(tokens)  # [B, seq_len, d_model]
        
        # Task-specific output
        if self.task in ["binary", "regression"]:
            # Use CLS token for classification/regression
            if self.cls_token:
                cls_output = encoded[:, 0]  # [B, d_model]
            else:
                # Use mean pooling if no CLS token
                cls_output = encoded.mean(dim=1)  # [B, d_model]
            output = self.head(cls_output).squeeze(-1)  # [B]
            
        elif self.task == "reconstruction":
            # Use feature tokens for reconstruction
            if self.cls_token:
                feature_outputs = encoded[:, 1:]  # [B, F, d_model]
            else:
                feature_outputs = encoded  # [B, F, d_model]
            # Apply head to each feature token to get per-feature reconstruction
            output = self.head(feature_outputs)  # [B, F, 1] since head outputs 1 value per feature
            output = output.squeeze(-1)  # [B, F]
            
        return output
    
    def get_encoder_state_dict(self) -> dict[str, torch.Tensor]:
        """Get state dict for encoder components (tokenizer + transformer)."""
        encoder_state = {}
        for name, param in self.named_parameters():
            if name.startswith(('tokenizer.', 'transformer.', 'cls_embedding')):
                encoder_state[name] = param.clone()
        return encoder_state
    
    def load_encoder_state_dict(
        self, 
        state_dict: dict[str, torch.Tensor], 
        strict: bool = False
    ) -> None:
        """Load encoder weights from state dict."""
        # Filter to only encoder parameters
        encoder_keys = {name for name in self.state_dict().keys() 
                       if name.startswith(('tokenizer.', 'transformer.', 'cls_embedding'))}
        
        filtered_state = {k: v for k, v in state_dict.items() if k in encoder_keys}
        
        if strict and len(filtered_state) != len(encoder_keys):
            missing = encoder_keys - set(filtered_state.keys())
            raise RuntimeError(f"Missing encoder keys: {missing}")
            
        self.load_state_dict(filtered_state, strict=False)


def create_fttransformer_for_task(
    n_features: int,
    task: str = "binary",
    **kwargs
) -> FTTransformer:
    """Factory function to create FT-Transformer for specific task."""
    return FTTransformer(n_features=n_features, task=task, **kwargs)


def create_reconstruction_model(
    n_features: int,
    **kwargs
) -> FTTransformer:
    """Create FT-Transformer for masked gene modeling (reconstruction)."""
    return FTTransformer(
        n_features=n_features, 
        task="reconstruction",
        cls_token=False,  # Don't need CLS for reconstruction
        **kwargs
    )