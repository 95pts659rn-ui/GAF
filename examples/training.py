import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.geometric_mlp import GeometricMLP
from operators.harmonic_perturb import HarmonicPerturbOperator
from propagations.feedback_hooks import LoggingHook, AccumulationHook
from propagations.integration_strategies import BackreactionAwareLoss, BackreactionMonitor

def generate_synthetic_data(num_samples=1000, input_dim=10, noise_level=0.1, seed: int=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    X = torch.randn(num_samples, input_dim)
    # Define target function: combination of linear and nonlinear terms
    y = torch.sin(X[:, 0:1]) + torch.cos(X[:, 1:2]) + 0.1 * X.sum(dim=1, keepdim=True)
    # Add noise
    y += noise_level * torch.randn_like(y)
    # Split into train and test
    split = int(0.8 * num_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    return X_train, y_train, X_test, y_test

class TrainingRunner:
    def __init__(self, model, learning_rate=0.001, backreaction_weight=0.01, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Loss function with backreaction awareness
        base_loss = nn.MSELoss()
        self.loss_fn = BackreactionAwareLoss(
            base_loss,
            backreaction_weight=backreaction_weight,
            backreaction_mode='l2'
        )
        
        # Monitoring
        self.backreaction_monitor = BackreactionMonitor()
        
        # Training history
        self.train_losses = []
        self.test_losses = []
        self.backreaction_history = []
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Forward pass
            y_pred, backreaction_list = self.model(X_batch, return_backreaction=True)
            
            # Accumulate backreaction for monitoring
            if backreaction_list:
                combined_backreaction = torch.cat(backreaction_list, dim=-1)
                self.backreaction_monitor.update(combined_backreaction)
            
            # Compute loss
            total_batch_loss, loss_components = self.loss_fn(
                y_pred,
                y_batch,
                backreaction=combined_backreaction if backreaction_list else None
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            total_batch_loss.backward()
            self.optimizer.step()
            
            total_loss += total_batch_loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def evaluate(self, test_loader):
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                y_pred = self.model(X_batch)
                loss = nn.MSELoss()(y_pred, y_batch)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.test_losses.append(avg_loss)
        
        return avg_loss
    
    def train(self, train_loader, test_loader, num_epochs=100, log_interval=10):
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            test_loss = self.evaluate(test_loader)
            
            if (epoch + 1) % log_interval == 0:
                print(f"Epoch {epoch + 1} / {num_epochs}")
                print(f"Train Loss: {train_loss:.6f}")
                print(f"Test Loss: {test_loss:.6f}")
                
                # Print backreaction statistics
                br_metrics = self.backreaction_monitor.get_latest()
                if br_metrics:
                    print(f"Backreaction Mean: {br_metrics['mean']:.6f}")
                    print(f"Backreaction Max: {br_metrics['max']:.6f}")


def run():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    print("\nGenerating synthetic data")
    X_train, y_train, X_test, y_test = generate_synthetic_data(
        num_samples=2000,
        input_dim=10,
        output_dim=1,
        noise_level=0.1
    )
    
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Create model
    print("\nCreating geometric MLP")

    operator = HarmonicPerturbOperator(
        max_degree=2,
        amplitude=0.1,
        trainable=True
    )
    
    model = GeometricMLP(
        input_dim=10,
        hidden_dims=[64, 32, 16],
        output_dim=1,
        geometric_operator=operator,
        backreaction_mode='norm',
        num_geometric_layers=2,
        use_batch_norm=True,
        dropout_rate=0.1,
        track_backreaction=True
    )

    print(f"Model architecture:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f".. {name}: {param.shape}")

    print("\nRegistering backreaction hooks")
    logging_hook = LoggingHook(log_interval=50, name="training_monitor")
    accumulation_hook = AccumulationHook(decay_factor=0.95, name="accumulation")
    
    model.register_hook(logging_hook)
    model.register_hook(accumulation_hook)

    print("\nStarting training")
    runner = TrainingRunner(
        model,
        learning_rate=0.001,
        backreaction_weight=0.01,
        device=device
    )
    
    runner.train(
        train_loader,
        test_loader,
        num_epochs=100,
        log_interval=10
    )

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nFinal Train Loss: {runner.train_losses[-1]:.6f}")
    print(f"Final Test Loss: {runner.test_losses[-1]:.6f}")
    print(f"Best Test Loss: {min(runner.test_losses):.6f}")

    br_stats = runner.backreaction_monitor.get_metrics()
    if br_stats['mean']:
        print(f"\nBackreaction Statistics (final):")
        print(f"Mean: {br_stats['mean'][-1]:.6f}")
        print(f"Max: {br_stats['max'][-1]:.6f}")
        print(f"Std: {br_stats['std'][-1]:.6f}")

        print("\nGenerating plots...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss curves
    ax = axes[0, 0]
    ax.plot(runner.train_losses, label='Train Loss', linewidth=2)
    ax.plot(runner.test_losses, label='Test Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Test Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Backreaction trends
    ax = axes[0, 1]
    br_metrics = runner.backreaction_monitor.get_metrics()
    if br_metrics['mean']:
        ax.plot(br_metrics['mean'], label='Mean', linewidth=2)
        ax.fill_between(
            range(len(br_metrics['mean'])),
            np.array(br_metrics['mean']) - np.array(br_metrics['std']),
            np.array(br_metrics['mean']) + np.array(br_metrics['std']),
            alpha=0.3
        )
        ax.set_xlabel('Batch')
        ax.set_ylabel('Backreaction Magnitude')
        ax.set_title('Backreaction Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Model predictions vs targets (test set)
    ax = axes[1, 0]
    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test.to(device)).cpu().numpy()
    
    ax.scatter(y_test.numpy(), y_pred_test, alpha=0.5, s=20)
    
    # Add diagonal
    min_val = min(y_test.min(), y_pred_test.min())
    max_val = max(y_test.max(), y_pred_test.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax.set_xlabel('Target')
    ax.set_ylabel('Prediction')
    ax.set_title('Model Predictions vs Targets')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Residuals
    ax = axes[1, 1]
    residuals = y_test.numpy() - y_pred_test
    ax.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Residual')
    ax.set_ylabel('Frequency')
    ax.set_title('Prediction Residuals Distribution')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
    print("Saved: training_results.png")
    plt.close()
    
    # Save model
    print("\nSaving trained model...")
    torch.save(model.state_dict(), 'geometric_model.pth')
    print("Saved: geometric_model.pth")
    
    print("\nExample complete!")

if __name__ == '__main__':
    run()
