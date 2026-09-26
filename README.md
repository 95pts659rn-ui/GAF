# Geometric Activation Framework (GAF)

GAF is a PyTorch-based framework for building neural networks with geometric structure. It combines spherical-coordinate transformations, geometric activation layers, and backreaction-aware training to create models that operate on geometry-aware representations rather than only raw Cartesian features.

## What this project does

The framework is designed to:

- transform activations through spherical geometry
- compose different geometric operators
- track how transformations alter the original signal via backreaction metrics
- build geometric MLP-style models with trainable operators
- monitor and visualize activation behavior during training

This repository is organized around a simple idea: make certain layers respect geometric structure while still fitting naturally into a PyTorch training workflow.

## Repository structure

- `activations/` — geometric activation layers
- `geometry/` — coordinate conversion and geometric metrics
- `operators/` — spherical operators such as rotations, radial warps, and perturbations
- `models/` — model definitions such as `GeometricMLP`
- `propagations/` — feedback hooks, loss functions, and monitoring utilities
- `tests/` — validation tests and comparison plots
- `examples/` — training example and generated visual results

## Core concepts

### Geometric activation

A geometric activation layer converts inputs from Cartesian coordinates into spherical coordinates, applies a geometric operator, and converts back. This makes the activation process sensitive to angular and radial structure.

### Spherical operators

Operators can be used to apply structured transformations, such as:

- rotations around axes
- radial warping
- harmonic perturbations
- composite operator chains

### Backreaction

Backreaction measures how much a transformation changes the original geometry of the signal. GAF exposes several modes for measuring this effect:

- `vector`: elementwise difference vector
- `norm`: magnitude of the difference
- `energy`: squared-distance style metric
- `divergence`: divergence-based geometric metric

## Quick example

```python
import torch
from models.geometric_mlp import GeometricMLP
from operators.harmonic_perturb import HarmonicPerturbOperator

operator = HarmonicPerturbOperator(
    max_degree=2,
    amplitude=0.1,
    trainable=True,
)

model = GeometricMLP(
    input_dim=10,
    hidden_dims=[64, 32, 16],
    output_dim=1,
    geometric_operator=operator,
    backreaction_mode='norm',
    num_geometric_layers=2,
    track_backreaction=True,
)

x = torch.randn(32, 10)
y_pred, backreaction = model(x, return_backreaction=True)
print(y_pred.shape)
```

## Example training

The script in `examples/training.py` demonstrates a complete training flow using synthetic data, a geometric MLP, backreaction-aware loss, and plot generation.

```bash
python examples/training.py
```

This example saves:

- `training_results.png`
- `geometric_model.pth`

### Training results

![Training results](examples/training_results.png)

## Test comparisons

The repository includes benchmark-style plots in the `tests/` directory. These are useful for inspecting how the geometric approach behaves on different tasks.

### Sine comparison

![Sine comparison](tests/task1_sine_comparison.png)

### Spiral comparison

![Spiral comparison](tests/task2_spiral_comparison.png)

### High-dimensional comparison

![High-dimensional comparison](tests/task3_highdim_comparison.png)

## Testing

Run the built-in test suite with:

```bash
pytest tests/
```

The tests cover components such as:

- backreaction calculations
- coordinate conversion
- operator behavior
- geometric comparison scripts

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/95pts659rn-ui/GAF.git
cd GAF
pip install torch numpy matplotlib pytest
```

## Requirements

- Python 3.9+
- PyTorch
- NumPy
- Matplotlib
- pytest

## Notes

This project is intentionally lightweight and easy to extend. The main idea is to provide a clean geometric layer abstraction that can be plugged into standard PyTorch models without a large dependency footprint.

If you want to experiment further, the simplest next step is to modify the operator, replace the training objective, or add a new geometric transformation in `operators/`.
