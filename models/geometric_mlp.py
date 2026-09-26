import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from activations.geometric_activation import GeometricActivationLayer
from activations.blockwise_geometric import BlockwiseGeometricLayer
from operators.base import SphericalOperator, IdentityOperator
from operators.composite import CompositeOperator
from operators.rotator import RotatorOperator
from operators.radial_warp import RadialWarpOperator
from propagations.feedback_hooks import HookManager, BackreactionHook

class GeometricMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        geometric_operator=None,
        backreaction_mode='norm',
        num_geometric_layers=None,
        use_batch_norm=False,
        dropout_rate=0.0,
        final_activation=None,
        track_backreaction=True,
        use_blockwise=True,
        blockwise_threshold=30
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.backreaction_mode = backreaction_mode
        self.track_backreaction = track_backreaction
        self.use_blockwise = use_blockwise
        self.blockwise_threshold = blockwise_threshold
        
        # Initialize hook manager for backreaction handling
        self.hook_manager = HookManager()
        
        # Store backreaction history
        self.backreaction_history = []
        
        # Use identity operator as default
        if geometric_operator is None:
            geometric_operator = IdentityOperator()
            
        self.geometric_operator = geometric_operator
        
        # Determine how many geometric layers to use
        total_layers = len(hidden_dims) + 1  # +1 for output layer
        if num_geometric_layers is None:
            num_geometric_layers = total_layers
        else:
            num_geometric_layers = min(num_geometric_layers, total_layers)
        
        # Build the network
        self.layers = nn.ModuleList()
        self.geometric_activations = {}  # Track which layers are geometric
        
        # Ensure input dim is > 3 for the first transformation, or use projection
        if input_dim < 3:
            # If input is < 3 dims, project it to 3 dimensions for geometric processing
            self.input_projection = nn.Linear(input_dim, 3)
            current_dim = 3
        else:
            self.input_projection = None
            current_dim = input_dim
        
        current_dim = input_dim
        geometric_layer_count = 0
        
        # Add hidden layers
        for i, hidden_dim in enumerate(hidden_dims):
            # Add linear layer
            linear = nn.Linear(current_dim, hidden_dim)
            self.layers.append(linear)
            
            # Add batch norm if requested
            if use_batch_norm:
                self.layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Add geometric activation if we should
            is_geometric = geometric_layer_count < num_geometric_layers
            if is_geometric:
                # Decide whether to use blockwise based on hidden dimension
                use_blockwise_here = self.use_blockwise and hidden_dim >= self.blockwise_threshold
                
                if use_blockwise_here:
                    # Use blockwise geometric layer for high dimensions
                    geom_activation = BlockwiseGeometricLayer(
                        input_dim=hidden_dim,
                        operator=self.geometric_operator if i == 0 else self._clone_operator(self.geometric_operator),
                        backreaction_mode=backreaction_mode,
                        store_backreaction=track_backreaction
                    )
                    self.layers.append(geom_activation)
                    self.geometric_activations[len(self.layers) - 1] = geom_activation
                else:
                    # Use standard 3D projection for low dimensions
                    # Project to 3D for geometric transformation
                    self.layers.append(nn.Linear(hidden_dim, 3))
                    
                    geom_activation = GeometricActivationLayer(
                        operator=self.geometric_operator if i == 0 else self._clone_operator(self.geometric_operator),
                        backreaction_mode=backreaction_mode,
                        store_backreaction=track_backreaction
                    )
                    self.layers.append(geom_activation)
                    self.geometric_activations[len(self.layers) - 1] = geom_activation
                    
                    # Unproject back to hidden_dim
                    self.layers.append(nn.Linear(3, hidden_dim))
                
                geometric_layer_count += 1
            else:
                # Add standard activation if not geometric
                self.layers.append(nn.ReLU())
            
            # Add dropout if requested
            if dropout_rate > 0:
                self.layers.append(nn.Dropout(dropout_rate))
            
            current_dim = hidden_dim
        
        # Add output layer
        output_layer = nn.Linear(current_dim, output_dim)
        self.layers.append(output_layer)
        
        # Add final activation if provided
        if final_activation is not None:
            self.layers.append(final_activation)
    
    def _clone_operator(self, operator):
        if isinstance(operator, IdentityOperator):
            return IdentityOperator()
        elif isinstance(operator, RotatorOperator):
            return RotatorOperator(
                axis=operator.axis,
                angle=operator.angle.item() if hasattr(operator, 'angle') else None,
                trainable=operator.is_trainable()
            )
        elif isinstance(operator, RadialWarpOperator):
            return RadialWarpOperator(
                warp_type=operator.warp_type,
                trainable=operator.is_trainable()
            )
        else:
            # For complex operators, just return a copy
            return operator
    
    def register_hook(self, hook):
        self.hook_manager.register(hook)
    
    def remove_hook(self, hook):
        self.hook_manager.unregister(hook)
    
    def forward(self, x, return_backreaction):
        # Project input to 3D if necessary
        if self.input_projection is not None:
            x = self.input_projection(x)
        
        # Clear backreaction history
        if self.track_backreaction:
            self.backreaction_history = []
        
        # Apply layers
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx in self.geometric_activations:
                # Geometric activation layer
                geom_layer = self.geometric_activations[layer_idx]
                x, backreaction = geom_layer(x, return_backreaction=True)
                
                if self.track_backreaction and backreaction is not None:
                    self.backreaction_history.append(backreaction)
                    
                    # Apply hooks
                    self.hook_manager.apply_hooks(
                        geom_layer, 
                        backreaction,
                        layer_id=f"geometric_{layer_idx}"
                    )
            else:
                # Regular layer
                x = layer(x)
        
        if return_backreaction:
            return x, self.backreaction_history
        else:
            return x
    
    def get_backreaction_history(self):
        return self.backreaction_history
    
    def extra_repr(self) -> str:
        operator_name = self.geometric_operator.__class__.__name__
        return (f"input_dim={self.input_dim}, hidden_dims={self.hidden_dims}, output_dim={self.output_dim}, operator={operator_name}, backreaction_mode={self.backreaction_mode}")
