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
