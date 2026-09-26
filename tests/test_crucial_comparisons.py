import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from activations.geometric_activation import GeometricActivationLayer
from models.geometric_mlp import GeometricMLP
from operators.harmonic_perturb import HarmonicPerturbOperator
from operators.radial_warp import RadialWarpOperator
from operators.rotator import RotatorOperator


def format_model_label(name):
    label = name.replace('_', ' ')
    label = label.replace('RELU', 'ReLU')
    label = label.replace('TANH', 'Tanh')
    label = label.replace('SIGMOID', 'Sigmoid')
    label = label.replace('Blockwise', 'Blockwise')
    label = label.replace('Rotator', 'Rotator')
    label = label.replace('Radialwarp', 'Radial Warp')
    label = label.replace('RadialWarp', 'Radial Warp')
    label = label.replace('Harmonic', 'Harmonic')
    label = label.replace('Geometric', 'Geometric')
    label = ' '.join(part.capitalize() if part.lower() not in {'and', 'of', 'with'} else part for part in label.split())
    label = label.replace('Relu', 'ReLU')
    label = label.replace('Tanh', 'Tanh')
    return label.strip()


class StandardMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation_type='relu', use_batch_norm=False, dropout_rate=0.0):
        super().__init__()
        if activation_type == 'relu':
            activation = nn.ReLU
        elif activation_type == 'sigmoid':
            activation = nn.Sigmoid
        elif activation_type == 'tanh':
            activation = nn.Tanh
        else:
            raise ValueError(f'Unknown activation: {activation_type}')

        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(activation())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class GeometricHybridMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, geometric_operator, backreaction_mode='norm', activation_type='tanh', use_batch_norm=False, dropout_rate=0.0):
        super().__init__()
        if activation_type == 'relu':
            activation = nn.ReLU
        elif activation_type == 'sigmoid':
            activation = nn.Sigmoid
        elif activation_type == 'tanh':
            activation = nn.Tanh
        else:
            activation = nn.ReLU

        self.layers = nn.ModuleList()
        current_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            self.layers.append(nn.Linear(current_dim, hidden_dim))
            if use_batch_norm:
                self.layers.append(nn.BatchNorm1d(hidden_dim))

            if i == 0:
                self.layers.append(nn.Linear(hidden_dim, 3))
                self.layers.append(GeometricActivationLayer(operator=geometric_operator, backreaction_mode=backreaction_mode))
                self.layers.append(nn.Linear(3, hidden_dim))
            else:
                self.layers.append(activation())
            if dropout_rate > 0:
                self.layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim

        self.layers.append(nn.Linear(current_dim, output_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ComparisonRunner:
    @staticmethod
    def format_model_label(name):
        return format_model_label(name)

    def __init__(self, device='cpu'):
        self.device = device
        self.results = {}

    def train_and_evaluate(self, model, train_loader, test_loader, num_epochs=5, learning_rate=1e-2, model_name='model'):
        model = model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        train_losses = []
        test_losses = []
        backreaction_magnitudes = []
        final_predictions = []
        final_targets = []

        for _ in range(num_epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()

                if isinstance(model, GeometricMLP):
                    y_pred, backreaction_list = model(X_batch, return_backreaction=True)
                    if backreaction_list:
                        br = torch.cat(backreaction_list, dim=-1)
                        mag = torch.norm(br, dim=-1).mean().item()
                        backreaction_magnitudes.append(mag)
                else:
                    y_pred = model(X_batch)

                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()

            model.eval()
            train_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in train_loader:
                    if isinstance(model, GeometricMLP):
                        y_pred, _ = model(X_batch.to(self.device), return_backreaction=True)
                    else:
                        y_pred = model(X_batch.to(self.device))
                    train_loss += criterion(y_pred, y_batch.to(self.device)).item()
                train_loss /= len(train_loader)
                train_losses.append(train_loss)

                test_loss = 0.0
                for X_batch, y_batch in test_loader:
                    if isinstance(model, GeometricMLP):
                        y_pred, _ = model(X_batch.to(self.device), return_backreaction=True)
                    else:
                        y_pred = model(X_batch.to(self.device))
                    test_loss += criterion(y_pred, y_batch.to(self.device)).item()
                test_loss /= len(test_loader)
                test_losses.append(test_loss)

        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                if isinstance(model, GeometricMLP):
                    preds, _ = model(X_batch.to(self.device), return_backreaction=True)
                else:
                    preds = model(X_batch.to(self.device))
                final_predictions.append(preds.cpu())
                final_targets.append(y_batch.cpu())

        predictions = torch.cat(final_predictions)
        targets = torch.cat(final_targets)
        final_mse = torch.mean((predictions - targets) ** 2).item()
        final_mae = torch.mean(torch.abs(predictions - targets)).item()

        result = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'final_mse': final_mse,
            'final_mae': final_mae,
            'best_test_loss': min(test_losses),
            'backreaction_magnitudes': backreaction_magnitudes if backreaction_magnitudes else None,
            'predictions': predictions.numpy(),
            'targets': targets.numpy(),
        }
        self.results[model_name] = result
        return result

    def compare_convergence(self):
        metrics = {}
        for name, result in self.results.items():
            losses = np.asarray(result['test_losses'])
            final_loss = losses[-1]
            initial_loss = losses[0]
            target = final_loss + 0.1 * (initial_loss - final_loss)
            epochs_to_90 = int(np.argmax(losses <= target) or len(losses))
            metrics[name] = {
                'final_loss': final_loss,
                'best_loss': float(np.min(losses)),
                'epochs_to_90pct': epochs_to_90,
                'total_loss_reduction': float(initial_loss - final_loss),
                'mse': result['final_mse'],
                'mae': result['final_mae'],
            }
        return metrics

    def print_comparison_table(self):
        metrics = self.compare_convergence()
        print('\n' + '=' * 100)
        print('MODEL COMPARISON RESULTS')
        print('=' * 100)
        print(f"{'Model':<25} {'Final Loss':<12} {'Best Loss':<12} {'Epochs@90%':<12} {'MSE':<12} {'MAE':<12}")
        print('-' * 100)
        for name in sorted(metrics.keys()):
            m = metrics[name]
            print(f"{name:<25} {m['final_loss']:<12.6f} {m['best_loss']:<12.6f} {m['epochs_to_90pct']:<12d} {m['mse']:<12.6f} {m['mae']:<12.6f}")
        print('=' * 100)

    def plot_comparison(self, output_file='comparison_results.png'):
        if not HAS_MATPLOTLIB:
            print('(Matplotlib not available - skipping plots)')
            return

        metrics = self.compare_convergence()
        names = list(metrics.keys())
        labels = [self.format_model_label(name) for name in names]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        for name, result in self.results.items():
            axes[0, 0].plot(result['train_losses'], label=f"{self.format_model_label(name)} (train)", alpha=0.7)
            axes[0, 1].plot(result['test_losses'], label=self.format_model_label(name), alpha=0.7, linewidth=2)

        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Training Loss')
        axes[0, 0].set_title('Training Loss Curves')
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Test Loss')
        axes[0, 1].set_title('Test Loss Curves (Convergence Comparison)')
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].grid(True, alpha=0.3)

        mses = [metrics[n]['mse'] for n in names]
        colors = ['green' if 'geometric' in n.lower() else 'blue' for n in names]
        axes[1, 0].bar(range(len(names)), mses, color=colors, alpha=0.7)
        axes[1, 0].set_xticks(range(len(names)))
        axes[1, 0].set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        axes[1, 0].set_ylabel('MSE')
        axes[1, 0].set_title('Final MSE Comparison')
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        epochs_90 = [max(0, metrics[n]['epochs_to_90pct']) for n in names]
        axes[1, 1].bar(range(len(names)), epochs_90, color=colors, alpha=0.7)
        axes[1, 1].set_xticks(range(len(names)))
        axes[1, 1].set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        axes[1, 1].set_ylabel('Epochs to 90% Convergence')
        axes[1, 1].set_title('Convergence Speed')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        axes[1, 1].set_ylim(0, max(1, max(epochs_90)) * 1.2 if epochs_90 else 1)

        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f'  Plot saved: {output_file}')
        plt.close()


def generate_task_1_sine_regression(num_samples=128):
    torch.manual_seed(42)
    np.random.seed(42)
    X = torch.linspace(-2.0, 2.0, num_samples).unsqueeze(-1)
    y = torch.sin(2 * np.pi * X) + 0.1 * torch.randn_like(X)
    idx = torch.randperm(num_samples)
    X, y = X[idx], y[idx]
    split = int(0.8 * num_samples)
    return X[:split], y[:split], X[split:], y[split:], 'Sine Regression'


def generate_task_2_spiral_classification(num_samples=200):
    torch.manual_seed(42)
    np.random.seed(42)
    num_per_class = num_samples // 2
    theta = np.linspace(0, 4 * np.pi, num_per_class)
    r = np.linspace(0, 1, num_per_class)
    x1 = np.vstack([r * np.cos(theta), r * np.sin(theta)])
    x2 = np.vstack([(1.1 - r) * np.cos(theta + np.pi), (1.1 - r) * np.sin(theta + np.pi)])
    X = np.hstack([x1, x2]).T
    y = np.hstack([np.zeros(num_per_class), np.ones(num_per_class)])[:, None]
    X = torch.tensor(X, dtype=torch.float32) + 0.1 * torch.randn_like(torch.tensor(X, dtype=torch.float32))
    y = torch.tensor(y, dtype=torch.float32)
    split = int(0.8 * num_samples)
    return X[:split], y[:split], X[split:], y[split:], 'Spiral Classification'


def generate_task_3_highdim_polynomial(num_samples=200, input_dim=20):
    torch.manual_seed(42)
    np.random.seed(42)
    X = torch.randn(num_samples, input_dim)
    y = (
        X[:, 0] * X[:, 1]
        + torch.sin(X[:, 2])
        + X[:, 3] * X[:, 4]
        + torch.cos(X[:, 5])
    ).unsqueeze(-1) + 0.1 * torch.randn(num_samples, 1)
    split = int(0.8 * num_samples)
    return X[:split], y[:split], X[split:], y[split:], 'High-Dim Polynomial'


def test_format_model_label_english_text():
    assert format_model_label('Geometric_Harmonic') == 'Geometric Harmonic'
    assert format_model_label('Geometric_Harmonic_Blockwise') == 'Geometric Harmonic Blockwise'
    assert format_model_label('Standard_RELU') == 'Standard ReLU'
    assert format_model_label('Geometric_Rotator') == 'Geometric Rotator'


def test_generate_task_functions_return_expected_shapes():
    X_train, y_train, X_test, y_test, name = generate_task_1_sine_regression()
    assert name == 'Sine Regression'
    assert X_train.shape[1] == 1 and y_train.shape[1] == 1
    assert X_test.shape[1] == 1 and y_test.shape[1] == 1

    X_train, y_train, X_test, y_test, name = generate_task_2_spiral_classification()
    assert name == 'Spiral Classification'
    assert X_train.shape[1] == 2 and y_train.shape[1] == 1
    assert X_test.shape[1] == 2 and y_test.shape[1] == 1

    X_train, y_train, X_test, y_test, name = generate_task_3_highdim_polynomial()
    assert name == 'High-Dim Polynomial'
    assert X_train.shape[1] == 20 and y_train.shape[1] == 1
    assert X_test.shape[1] == 20 and y_test.shape[1] == 1


def test_standard_and_geometric_models_run_smoke():
    X_train, y_train, X_test, y_test, _ = generate_task_1_sine_regression(num_samples=64)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=16)

    standard_model = StandardMLP(1, [8, 4], 1, activation_type='tanh')
    geometric_model = GeometricHybridMLP(
        1,
        [8, 4],
        1,
        geometric_operator=RotatorOperator(axis='z', angle=0.2, trainable=True),
        backreaction_mode='norm',
    )

    runner = ComparisonRunner()
    standard_result = runner.train_and_evaluate(standard_model, train_loader, test_loader, num_epochs=2, model_name='Standard_Tanh')
    geometric_result = runner.train_and_evaluate(geometric_model, train_loader, test_loader, num_epochs=2, model_name='Geometric_Rotator')

    assert standard_result['final_mse'] >= 0.0
    assert geometric_result['final_mse'] >= 0.0
    assert runner.compare_convergence()['Geometric_Rotator']['final_loss'] >= 0.0


def test_geometric_mlp_supports_crucial_operator_stack():
    model = GeometricMLP(
        20,
        [16, 8],
        1,
        geometric_operator=HarmonicPerturbOperator(max_degree=2, trainable=True),
        backreaction_mode='norm',
        num_geometric_layers=1,
        use_blockwise=False,
    )
    x = torch.randn(4, 20)
    y, history = model(x, return_backreaction=True)
    assert y.shape == (4, 1)
    assert isinstance(history, list)


def test_radial_warp_default_signature_supports_crucial_comparison():
    operator = RadialWarpOperator(warp_type='polynomial')
    x = torch.tensor([[1.0, 0.5, 0.0]], dtype=torch.float32)
    result = operator(x)
    assert result.shape == x.shape
    assert result[0, 0] > 0


def run_task_1():
    X_train, y_train, X_test, y_test, _ = generate_task_1_sine_regression(num_samples=500)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

    runner = ComparisonRunner()
    for act_type in ['relu', 'tanh', 'sigmoid']:
        model = StandardMLP(1, [32, 16, 8], 1, activation_type=act_type, use_batch_norm=True)
        runner.train_and_evaluate(model, train_loader, test_loader, num_epochs=150, model_name=f'Standard_{act_type.upper()}')

    model = GeometricHybridMLP(1, [32, 16, 8], 1, geometric_operator=RotatorOperator(axis='z', trainable=True), backreaction_mode='norm')
    runner.train_and_evaluate(model, train_loader, test_loader, num_epochs=150, model_name='Geometric_Rotator')

    model = GeometricHybridMLP(1, [32, 16, 8], 1, geometric_operator=RadialWarpOperator(warp_type='polynomial'), backreaction_mode='norm')
    runner.train_and_evaluate(model, train_loader, test_loader, num_epochs=150, model_name='Geometric_RadialWarp')

    model = GeometricHybridMLP(1, [32, 16, 8], 1, geometric_operator=HarmonicPerturbOperator(max_degree=2, trainable=True), backreaction_mode='norm')
    runner.train_and_evaluate(model, train_loader, test_loader, num_epochs=150, model_name='Geometric_Harmonic')

    runner.print_comparison_table()
    runner.plot_comparison('task1_sine_comparison.png')
    return runner


def run_task_2():
    X_train, y_train, X_test, y_test, _ = generate_task_2_spiral_classification(num_samples=600)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

    runner = ComparisonRunner()
    for act_type in ['relu', 'tanh']:
        model = StandardMLP(2, [64, 32, 16], 1, activation_type=act_type, use_batch_norm=True)
        runner.train_and_evaluate(model, train_loader, test_loader, num_epochs=150, model_name=f'Standard_{act_type.upper()}')

    model = GeometricHybridMLP(2, [64, 32, 16], 1, geometric_operator=RotatorOperator(axis='z', trainable=True), backreaction_mode='norm')
    runner.train_and_evaluate(model, train_loader, test_loader, num_epochs=150, model_name='Geometric_Rotator')

    model = GeometricHybridMLP(2, [64, 32, 16], 1, geometric_operator=RadialWarpOperator(warp_type='polynomial'), backreaction_mode='norm')
    runner.train_and_evaluate(model, train_loader, test_loader, num_epochs=150, model_name='Geometric_RadialWarp')

    model = GeometricHybridMLP(2, [64, 32, 16], 1, geometric_operator=HarmonicPerturbOperator(max_degree=2, trainable=True), backreaction_mode='norm')
    runner.train_and_evaluate(model, train_loader, test_loader, num_epochs=150, model_name='Geometric_Harmonic')

    runner.print_comparison_table()
    runner.plot_comparison('task2_spiral_comparison.png')
    return runner


def run_task_3():
    X_train, y_train, X_test, y_test, _ = generate_task_3_highdim_polynomial(num_samples=800, input_dim=20)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64)

    runner = ComparisonRunner()
    for act_type in ['relu', 'tanh']:
        model = StandardMLP(20, [128, 64, 32], 1, activation_type=act_type, use_batch_norm=True, dropout_rate=0.1)
        runner.train_and_evaluate(model, train_loader, test_loader, num_epochs=150, model_name=f'Standard_{act_type.upper()}')

    model = GeometricMLP(20, [128, 64, 32], 1, geometric_operator=HarmonicPerturbOperator(max_degree=2, trainable=True), backreaction_mode='norm', num_geometric_layers=1, dropout_rate=0.1, use_blockwise=True, blockwise_threshold=30)
    runner.train_and_evaluate(model, train_loader, test_loader, num_epochs=150, model_name='Geometric_Harmonic_Blockwise')

    model = GeometricMLP(20, [128, 64, 32], 1, geometric_operator=RotatorOperator(trainable=True), backreaction_mode='norm', num_geometric_layers=1, use_blockwise=True, blockwise_threshold=30)
    runner.train_and_evaluate(model, train_loader, test_loader, num_epochs=150, model_name='Geometric_Rotator_Blockwise')

    runner.print_comparison_table()
    runner.plot_comparison('task3_highdim_comparison.png')
    return runner


def main():
    print('\nCRUCIAL COMPARATIVE TESTS: GEOMETRIC vs STANDARD ACTIVATIONS\n')
    run_task_1()
    run_task_2()
    run_task_3()


if __name__ == '__main__':
    main()
