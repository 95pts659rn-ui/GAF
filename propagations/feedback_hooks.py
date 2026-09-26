import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from activations.geometric_activation import GeometricActivationLayer

class BackreactionHook:    
    def __init__(self, name="unnamed_hook"):
        self.name = name
        self.enabled = True
    
    def __call__(self, layer, backreaction, **kwargs):
        if self.enabled:
            self.process(layer, backreaction, **kwargs)
    
    def process(self, layer, backreaction, **kwargs):
        raise NotImplementedError("Subclasses must implement process()")
    
    def enable(self):
        self.enabled = True
    
    def disable(self):
        self.enabled = False

class HookManager:
    def __init__(self):
        self.hooks = []
    
    def register(self, hook):
        self.hooks.append(hook)
    
    def unregister(self, hook):
        if hook in self.hooks:
            self.hooks.remove(hook)
    
    def apply_hooks(self, layer, backreaction, **kwargs):
        for hook in self.hooks:
            hook(layer, backreaction, **kwargs)
    
    def get_hook_by_name(self, name):
        for hook in self.hooks:
            if hook.name == name:
                return hook
        return None
    
    def clear_hooks(self):
        self.hooks.clear()

class LoggingHook(BackreactionHook):
    def __init__(self, log_interval=10, name="logging_hook"):
        super().__init__(name)
        self.log_interval = log_interval
        self.call_count = 0
        self.stats_history = []
    
    def process(self, layer, backreaction, **kwargs):
        self.call_count += 1
        
        # Compute statistics on backreaction
        if backreaction.dim() > 1:  # Vector backreaction
            norm = torch.norm(backreaction, dim=1)
            stats = {
                'mean_norm': float(torch.mean(norm).item()),
                'max_norm': float(torch.max(norm).item()),
                'min_norm': float(torch.min(norm).item()),
                'std_norm': float(torch.std(norm).item())
            }
        else:  # Scalar backreaction
            stats = {
                'mean': float(torch.mean(backreaction).item()),
                'max': float(torch.max(backreaction).item()),
                'min': float(torch.min(backreaction).item()),
                'std': float(torch.std(backreaction).item())
            }
        
        self.stats_history.append(stats)
        
        # Log periodically
        if self.call_count % self.log_interval == 0:
            layer_name = layer.__class__.__name__
            print(f"[{self.name}] Backreaction stats for {layer_name} (call {self.call_count}):")
            for key, value in stats.items():
                print(f"  {key}: {value:.6f}")
    
    def get_history(self):
        return self.stats_history
    
    def reset_history(self):
        self.stats_history = []

class AccumulationHook(BackreactionHook):
    def __init__(self, decay_factor=0.9,name="accumulation_hook"):
        super().__init__(name)
       self.accumulated = None
      self.decay_factor = decay_factor
       
    def process(self, layer, backreaction, **kwargs):
        # Initialize accumulated if not already done
        if self.accumulated is None:
            self.accumulated = torch.zeros_like(backreaction)
        
        # Handle shape mismatches (e.g. different batch sizes)
        if self.accumulated.shape != backreaction.shape:
            # If shapes differ, compute batch average and accumulate
            avg_backtrack = torch.mean(backreaction, dim=0, keepdim=True)
            self.accumulated = self.decay_factor * self.accumulated + (1 - self.decay_factor) * avg_backtrack
        else:
            # Update with exponential decay
            self.accumulated = self.decay_factor * self.accumulated + (1 - self.decay_factor) * backreaction
    
    def get_accumulated(self):
        return self.accumulated
    
    def reset_accumulation(self):
        self.accumulated = None
