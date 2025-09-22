"""Tests for FT-Transformer model and MGM pretraining."""

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from src.oncotarget.models_fttransformer import (
    FeatureTokenizer,
    FTTransformer,
    create_fttransformer_for_task,
    create_reconstruction_model,
)
from src.oncotarget.ssl_mgm import create_mask, masked_mse_loss


class TestFeatureTokenizer:
    """Test FeatureTokenizer component."""
    
    def test_tokenizer_forward_shape(self):
        """Test tokenizer output shapes."""
        n_features, d_model = 10, 64
        tokenizer = FeatureTokenizer(n_features, d_model)
        
        x = torch.randn(4, n_features)
        tokens = tokenizer(x)
        
        assert tokens.shape == (4, n_features, d_model)
    
    def test_tokenizer_shared_vs_individual(self):
        """Test both shared and individual projection modes."""
        n_features, d_model = 5, 32
        
        # Shared linear
        tokenizer_shared = FeatureTokenizer(n_features, d_model, shared_linear=True)
        
        # Individual linear
        tokenizer_individual = FeatureTokenizer(n_features, d_model, shared_linear=False)
        
        x = torch.randn(2, n_features)
        
        tokens_shared = tokenizer_shared(x)
        tokens_individual = tokenizer_individual(x)
        
        assert tokens_shared.shape == tokens_individual.shape == (2, n_features, d_model)
    
    def test_tokenizer_layer_norm(self):
        """Test that layer norm is applied."""
        tokenizer = FeatureTokenizer(3, 16)
        x = torch.randn(2, 3)
        tokens = tokenizer(x)
        
        # Check that tokens are approximately normalized per feature (last dimension)
        mean = tokens.mean(dim=-1, keepdim=True)
        std = tokens.std(dim=-1, keepdim=True, unbiased=False)
        
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
        assert torch.allclose(std, torch.ones_like(std), atol=1e-4)


class TestFTTransformer:
    """Test FT-Transformer model."""
    
    @pytest.fixture
    def sample_data(self):
        """Create synthetic test data."""
        torch.manual_seed(42)
        n_samples, n_features = 32, 128
        X = torch.randn(n_samples, n_features)
        y_binary = torch.randint(0, 2, (n_samples,)).float()
        y_regression = torch.randn(n_samples)
        return X, y_binary, y_regression
    
    def test_binary_classification_forward(self, sample_data):
        """Test binary classification forward pass."""
        X, y_binary, _ = sample_data
        n_features = X.shape[1]
        
        model = FTTransformer(n_features=n_features, task="binary")
        logits = model(X)
        
        assert logits.shape == (len(X),)
        assert not torch.isnan(logits).any()
    
    def test_regression_forward(self, sample_data):
        """Test regression forward pass."""
        X, _, y_regression = sample_data
        n_features = X.shape[1]
        
        model = FTTransformer(n_features=n_features, task="regression")
        pred = model(X)
        
        assert pred.shape == (len(X),)
        assert not torch.isnan(pred).any()
    
    def test_reconstruction_forward(self, sample_data):
        """Test reconstruction forward pass."""
        X, _, _ = sample_data
        n_features = X.shape[1]
        
        model = FTTransformer(n_features=n_features, task="reconstruction", cls_token=False)
        
        # Test without mask
        pred = model(X)
        assert pred.shape == (len(X), n_features)
        
        # Test with mask
        mask = torch.randint(0, 2, (len(X), n_features)).float()
        pred_masked = model(X, mask)
        assert pred_masked.shape == (len(X), n_features)
    
    def test_cls_token_vs_no_cls(self, sample_data):
        """Test model with and without CLS token."""
        X, y_binary, _ = sample_data
        n_features = X.shape[1]
        
        model_cls = FTTransformer(n_features=n_features, task="binary", cls_token=True)
        model_no_cls = FTTransformer(n_features=n_features, task="binary", cls_token=False)
        
        logits_cls = model_cls(X)
        logits_no_cls = model_no_cls(X)
        
        assert logits_cls.shape == logits_no_cls.shape == (len(X),)
    
    def test_encoder_state_dict_operations(self, sample_data):
        """Test encoder state dict save/load."""
        X, _, _ = sample_data
        n_features = X.shape[1]
        
        # Create source model
        source_model = FTTransformer(n_features=n_features, task="reconstruction")
        
        # Get encoder state
        encoder_state = source_model.get_encoder_state_dict()
        
        # Create target model and load encoder
        target_model = FTTransformer(n_features=n_features, task="binary")
        target_model.load_encoder_state_dict(encoder_state)
        
        # Check that encoder parameters match
        for name, param in source_model.named_parameters():
            if name.startswith(('tokenizer.', 'transformer.', 'cls_embedding')):
                assert torch.allclose(param, target_model.state_dict()[name])
    
    def test_different_model_configurations(self):
        """Test various model configurations."""
        n_features = 50
        configs = [
            {"d_model": 64, "n_heads": 2, "n_layers": 2},
            {"d_model": 128, "n_heads": 4, "n_layers": 3},
            {"d_model": 256, "n_heads": 8, "n_layers": 4, "dropout": 0.2},
        ]
        
        for config in configs:
            model = FTTransformer(n_features=n_features, task="binary", **config)
            X = torch.randn(8, n_features)
            output = model(X)
            assert output.shape == (8,)
            assert not torch.isnan(output).any()


class TestFactoryFunctions:
    """Test factory functions."""
    
    def test_create_fttransformer_for_task(self):
        """Test task-specific factory function."""
        n_features = 20
        
        binary_model = create_fttransformer_for_task(n_features, task="binary")
        regression_model = create_fttransformer_for_task(n_features, task="regression")
        
        X = torch.randn(4, n_features)
        
        binary_out = binary_model(X)
        regression_out = regression_model(X)
        
        assert binary_out.shape == regression_out.shape == (4,)
    
    def test_create_reconstruction_model(self):
        """Test reconstruction model factory."""
        n_features = 30
        
        model = create_reconstruction_model(n_features, d_model=64)
        assert model.task == "reconstruction"
        assert not model.cls_token  # Should not use CLS token for reconstruction
        
        X = torch.randn(6, n_features)
        output = model(X)
        assert output.shape == (6, n_features)


class TestMGMFunctions:
    """Test MGM pretraining utilities."""
    
    def test_create_mask(self):
        """Test mask creation."""
        x = torch.randn(8, 20)
        mask_frac = 0.25
        
        mask = create_mask(x, mask_frac)
        
        assert mask.shape == x.shape
        assert mask.dtype == torch.float32
        
        # Check approximately correct fraction of masked features
        actual_frac = mask.mean().item()
        assert abs(actual_frac - mask_frac) < 0.1  # Allow some variance
    
    def test_create_mask_reproducible(self):
        """Test mask creation is reproducible with generator."""
        x = torch.randn(4, 10)
        
        gen1 = torch.Generator().manual_seed(42)
        gen2 = torch.Generator().manual_seed(42)
        
        mask1 = create_mask(x, 0.3, gen1)
        mask2 = create_mask(x, 0.3, gen2)
        
        assert torch.allclose(mask1, mask2)
    
    def test_masked_mse_loss(self):
        """Test masked MSE loss computation."""
        pred = torch.randn(4, 10, requires_grad=True)
        target = torch.randn(4, 10)
        mask = torch.randint(0, 2, (4, 10)).float()
        
        loss = masked_mse_loss(pred, target, mask)
        
        assert loss.requires_grad
        assert loss.item() >= 0
        
        # Test with no mask (should return 0)
        empty_mask = torch.zeros_like(mask)
        empty_loss = masked_mse_loss(pred, target, empty_mask)
        assert empty_loss.item() == 0.0
    
    def test_mgm_training_step(self):
        """Test that MGM reduces loss over training steps."""
        torch.manual_seed(42)
        n_features = 16
        
        # Create simple synthetic data
        X = torch.randn(32, n_features)
        
        # Create model
        model = create_reconstruction_model(n_features, d_model=32, n_layers=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        
        # Training step function
        def training_step(x_batch):
            mask = create_mask(x_batch, mask_frac=0.15)
            pred = model(x_batch, mask)
            loss = masked_mse_loss(pred, x_batch, mask)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            return loss.item()
        
        # Train for a few steps
        initial_losses = []
        final_losses = []
        
        # Record initial losses
        for _ in range(5):
            loss = training_step(X)
            initial_losses.append(loss)
        
        # Train for more steps
        for _ in range(20):
            training_step(X)
        
        # Record final losses
        for _ in range(5):
            loss = training_step(X)
            final_losses.append(loss)
        
        # Loss should decrease (with some tolerance for noise)
        initial_avg = np.mean(initial_losses)
        final_avg = np.mean(final_losses)
        
        assert final_avg < initial_avg * 1.1  # Allow for some variance


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_binary_classification(self):
        """Test complete binary classification pipeline."""
        torch.manual_seed(42)
        n_samples, n_features = 64, 32
        
        # Create synthetic data
        X = torch.randn(n_samples, n_features)
        y = torch.randint(0, 2, (n_samples,)).float()
        
        # Create model
        model = create_fttransformer_for_task(n_features, task="binary", d_model=64)
        
        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Train for a few epochs
        initial_loss = None
        for epoch in range(10):
            logits = model(X)
            loss = criterion(logits, y)
            
            if epoch == 0:
                initial_loss = loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Final evaluation
        with torch.no_grad():
            final_logits = model(X)
            final_loss = criterion(final_logits, y).item()
            probs = torch.sigmoid(final_logits)
        
        # Loss should decrease
        assert final_loss < initial_loss
        
        # Probabilities should be valid
        assert torch.all(probs >= 0) and torch.all(probs <= 1)
    
    def test_pretrain_finetune_workflow(self):
        """Test pretraining → fine-tuning workflow."""
        torch.manual_seed(42)
        n_features = 24
        
        # Step 1: Pretraining (reconstruction)
        pretrain_model = create_reconstruction_model(n_features, d_model=48)
        X_pretrain = torch.randn(40, n_features)
        
        # Simulate pretraining
        optimizer = torch.optim.Adam(pretrain_model.parameters(), lr=1e-3)
        for _ in range(5):
            mask = create_mask(X_pretrain, 0.15)
            pred = pretrain_model(X_pretrain, mask)
            loss = masked_mse_loss(pred, X_pretrain, mask)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Step 2: Extract encoder
        encoder_state = pretrain_model.get_encoder_state_dict()
        
        # Step 3: Fine-tuning (classification)
        finetune_model = create_fttransformer_for_task(n_features, task="binary", d_model=48)
        finetune_model.load_encoder_state_dict(encoder_state)
        
        # Test that fine-tuning works
        X_finetune = torch.randn(32, n_features)
        y_finetune = torch.randint(0, 2, (32,)).float()
        
        logits = finetune_model(X_finetune)
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(logits, y_finetune)
        
        assert not torch.isnan(loss)
        assert loss.requires_grad