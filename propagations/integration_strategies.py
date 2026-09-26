import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BackreactionAwareLoss(nn.Module):
    def __init__(
        self,
        base_loss_fn,
        backreaction_weight=0.01,
        backreaction_mode='l2'
    ):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.backreaction_weight = backreaction_weight
        self.backreaction_mode = backreaction_mode
    
    def forward(
        self,
        predictions,
        targets,
        backreaction=None
    ):
        base_loss = self.base_loss_fn(predictions, targets)
        loss_components = {'base_loss': base_loss}
        
        if backreaction is not None:
            backreaction_loss = self._compute_backreaction_penalty(backreaction)
            loss_components['backreaction_loss'] = backreaction_loss
            total_loss = base_loss + self.backreaction_weight * backreaction_loss
        else:
            total_loss = base_loss
        
        return total_loss, loss_components
    
    def _compute_backreaction_penalty(self, backreaction):
        if self.backreaction_mode == 'l2':
            if backreaction.dim() > 1:
                penalty = torch.norm(backreaction, dim=-1).mean()
            else:
                penalty = torch.mean(backreaction ** 2)
        
        elif self.backreaction_mode == 'l1':
            penalty = torch.mean(torch.abs(backreaction))
        
        elif self.backreaction_mode == 'smooth_l1':
            penalty = torch.nn.functional.smooth_l1_loss(
                backreaction,
                torch.zeros_like(backreaction)
            )
        
        elif self.backreaction_mode == 'max':
            if backreaction.dim() > 1:
                penalty = torch.norm(backreaction, dim=-1).max()
            else:
                penalty = torch.max(backreaction ** 2)
        
        else:
            raise ValueError(f"Unknown backreaction_mode: {self.backreaction_mode}")
        
        return penalty


class BackreactionMonitor:
    def __init__(self):
        self.metrics = {
            'mean': [],
            'max': [],
            'min': [],
            'std': [],
            'total': []
        }
    
    def update(self, backreaction):
        backreaction = backreaction.detach()
        
        if backreaction.dim() > 1:
            norms = torch.norm(backreaction, dim=-1)
        else:
            norms = backreaction
        
        self.metrics['mean'].append(float(torch.mean(norms).item()))
        self.metrics['max'].append(float(torch.max(norms).item()))
        self.metrics['min'].append(float(torch.min(norms).item()))
        self.metrics['std'].append(float(torch.std(norms).item()))
        self.metrics['total'].append(float(torch.sum(norms).item()))
    
    def get_metrics(self):
        return self.metrics.copy()
    
    def get_latest(self):
        return {
            key: values[-1] if values else 0.0
            for key, values in self.metrics.items()
        }
    
    def reset(self):
        for key in self.metrics:
            self.metrics[key] = []
