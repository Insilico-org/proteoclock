import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Dict

@dataclass
class LossConfig:
    """Configuration for IPF loss components"""
    age_weight: float = 1.0  # Weight for age prediction loss
    pathway_weight: float = 1.0  # Weight for pathway prediction loss
    attention_reg_weight: float = 0.1  # Weight for attention regularization
    epsilon: float = 1e-6  # Small value for numerical stability

class ClockLoss(nn.Module):
    """Combined loss function for IPF model training with balanced components including attention regularization"""

    def __init__(self, config: LossConfig = None):
        super().__init__()
        self.config = config or LossConfig()

    @staticmethod
    def get_batch_size(tensor: torch.Tensor) -> int:
        """Helper to get batch size from any input tensor"""
        return tensor.size(0) if len(tensor.shape) > 1 else len(tensor)

    def compute_batch_losses(self, batch_size: int, loss_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Convert loss tensors to weighted per-batch values"""
        metrics = {
            k: (v.item() * batch_size if torch.is_tensor(v) else v * batch_size)
            for k, v in loss_dict.items() if k != 'debug'
        }
        metrics['debug'] = loss_dict['debug']
        return metrics

    def compute_age_loss(self,
                         pred_ages: torch.Tensor,
                         true_ages: torch.Tensor) -> torch.Tensor:
        """Compute age prediction loss using relative error"""
        relative_error = (pred_ages.squeeze() - true_ages) / (true_ages + self.config.epsilon)
        return torch.mean(relative_error ** 2)

    def compute_attention_loss(self,
                               attention_weights: torch.Tensor,
                               initial_attention: torch.Tensor) -> torch.Tensor:
        """Compute attention regularization loss using KL divergence and entropy"""
        if initial_attention is None:
            num_pathways = attention_weights.size(1)
            initial_attention = torch.ones(num_pathways, device=attention_weights.device) / num_pathways

        # Ensure initial_attention is on same device
        initial_attention = initial_attention.to(attention_weights.device)

        # KL divergence between current and initial attention
        kl_loss = F.kl_div(
            attention_weights.log(),
            initial_attention.expand_as(attention_weights),
            reduction='batchmean'
        )

        # Add entropy regularization to prevent attention collapse
        entropy = -torch.sum(attention_weights * torch.log(attention_weights + self.config.epsilon), dim=1).mean()
        entropy_reg = torch.exp(-entropy)  # Penalize low entropy more heavily

        # Combine KL divergence and entropy regularization
        attention_loss = kl_loss + 0.1 * entropy_reg  # Scale entropy term

        self.debug_info.update({
                                'attention_kl_loss': kl_loss.detach(),
                                'attention_entropy': entropy.detach(),
                                'attention_entropy_reg': entropy_reg.detach(),
                                'final_attn_loss': attention_loss.detach()
                            })

        return attention_loss

    def forward(self,
                pred_ages: torch.Tensor,
                true_ages: torch.Tensor,
                attention_weights: torch.Tensor,
                initial_attention: torch.Tensor = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute total balanced loss and individual components"""
        # Compute individual losses
        self.debug_info = dict()

        age_loss = self.compute_age_loss(pred_ages, true_ages)
        attention_loss = self.compute_attention_loss(attention_weights, initial_attention)

        age_weight, attention_weight = 1, 1
        age_weight *= self.config.age_weight
        attention_weight *= self.config.attention_reg_weight

        # # Combine losses with dynamic weights
        total_loss = (
            age_weight * age_loss +
            attention_weight * attention_loss
        )

        # Update debug info with weights
        self.debug_info.update({
            'age_weight': age_weight,#.detach(),
            'attention_weight': attention_weight,#.detach(),
        })

        loss_dict = {
            'total': total_loss,
            'age': age_loss,
            'attention_reg': attention_loss,
            'debug': {
                'attention_reg': {
                    'attention_kl_loss': self.debug_info['attention_kl_loss'],
                    'entropy': self.debug_info['attention_entropy'],
                    'attention_entropy_reg': self.debug_info['attention_entropy_reg'],
                    'weight': self.debug_info['attention_weight'],
                    'final_loss':self.debug_info['final_attn_loss']
                }
            }
        }

        return total_loss, loss_dict

class TripleLoss(nn.Module):
    """Combined loss function for IPF model training with balanced components including attention regularization"""

    def __init__(self, config: LossConfig = None):
        super().__init__()
        self.config = config or LossConfig()

    @staticmethod
    def get_batch_size(tensor: torch.Tensor) -> int:
        """Helper to get batch size from any input tensor"""
        return tensor.size(0) if len(tensor.shape) > 1 else len(tensor)

    def compute_batch_losses(self, batch_size: int, loss_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Convert loss tensors to weighted per-batch values"""
        metrics = {
            k: (v.item() * batch_size if torch.is_tensor(v) else v * batch_size)
            for k, v in loss_dict.items() if k != 'debug'
        }
        metrics['debug'] = loss_dict['debug']
        return metrics

    def compute_age_loss(self,
                         pred_ages: torch.Tensor,
                         true_ages: torch.Tensor) -> torch.Tensor:
        """Compute age prediction loss using relative error"""
        relative_error = (pred_ages.squeeze() - true_ages) / (true_ages + self.config.epsilon)
        return torch.mean(relative_error ** 2)

    def compute_pathway_loss(self,
                             pred_scores: torch.Tensor,
                             true_scores: torch.Tensor) -> torch.Tensor:
        """Compute pathway prediction loss with log scaling"""
        # Handle potential NaN or infinite values
        pred_scores = torch.nan_to_num(pred_scores, nan=0.0, posinf=1e6, neginf=-1e6)
        true_scores = torch.nan_to_num(true_scores, nan=0.0, posinf=1e6, neginf=-1e6)

        # Use relative error but clip extreme values
        relative_error = (pred_scores - true_scores) / (true_scores + self.config.epsilon)
        relative_error = torch.clamp(relative_error, min=-100, max=100)

        # Store intermediate values for debugging
        self.debug_info.update({'pred_scores': pred_scores.detach(),
                                'true_scores': true_scores.detach(),
                                'relative_errors': relative_error.detach(),
                                'raw_loss_per_pathway': torch.mean(relative_error ** 2, dim=0).detach()})

        # Compute raw loss and apply log scaling
        raw_loss = torch.mean(relative_error ** 2)
        scaled_loss = torch.log1p(raw_loss)

        pred_var = torch.var(pred_scores, dim=0)  # Variance per pathway
        true_var = torch.var(true_scores, dim=0)
        variance_loss = torch.mean((pred_var - true_var) ** 2)

        final_loss = scaled_loss + variance_loss

        self.debug_info['raw_total_loss'] = raw_loss.detach()
        self.debug_info['scaled_total_loss'] = scaled_loss.detach()
        self.debug_info['variance_loss'] = variance_loss.detach()
        self.debug_info['final_pathway_loss'] = final_loss.detach()

        return final_loss

    def compute_attention_loss(self,
                               attention_weights: torch.Tensor,
                               initial_attention: torch.Tensor) -> torch.Tensor:
        """Compute attention regularization loss using KL divergence and entropy"""
        if initial_attention is None:
            num_pathways = attention_weights.size(1)
            initial_attention = torch.ones(num_pathways, device=attention_weights.device) / num_pathways

        # Ensure initial_attention is on same device
        initial_attention = initial_attention.to(attention_weights.device)

        # KL divergence between current and initial attention
        kl_loss = F.kl_div(
            attention_weights.log(),
            initial_attention.expand_as(attention_weights),
            reduction='batchmean'
        )

        # Add entropy regularization to prevent attention collapse
        entropy = -torch.sum(attention_weights * torch.log(attention_weights + self.config.epsilon), dim=1).mean()
        entropy_reg = torch.exp(-entropy)  # Penalize low entropy more heavily

        # Combine KL divergence and entropy regularization
        attention_loss = kl_loss + 0.1 * entropy_reg  # Scale entropy term

        self.debug_info.update({
                                'attention_kl_loss': kl_loss.detach(),
                                'attention_entropy': entropy.detach(),
                                'attention_entropy_reg': entropy_reg.detach(),
                                'final_attn_loss': attention_loss.detach()
                            })

        return attention_loss

    def forward(self,
                pred_ages: torch.Tensor,
                true_ages: torch.Tensor,
                pred_pathways: torch.Tensor,
                true_pathways: torch.Tensor,
                attention_weights: torch.Tensor,
                initial_attention: torch.Tensor = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute total balanced loss and individual components"""
        # Compute individual losses
        self.debug_info = dict()

        age_loss = self.compute_age_loss(pred_ages, true_ages)
        pathway_loss = self.compute_pathway_loss(pred_pathways, true_pathways)
        attention_loss = self.compute_attention_loss(attention_weights, initial_attention)

        # Compute loss weights dynamically based on magnitudes
        # age_magnitude = torch.log1p(torch.abs(age_loss))
        # pathway_magnitude = torch.log1p(torch.abs(pathway_loss))
        # attention_magnitude = torch.log1p(torch.abs(attention_loss))
        #
        # # Normalize weights to sum to 3 (since we have 3 components)
        # total_magnitude = age_magnitude + pathway_magnitude + attention_magnitude
        # age_weight = 3 * age_magnitude / total_magnitude
        # pathway_weight = 3 * pathway_magnitude / total_magnitude
        # attention_weight = 3 * attention_magnitude / total_magnitude
        #
        # # Apply base weights from config
        age_weight, pathway_weight, attention_weight = 1, 1, 1

        age_weight *= self.config.age_weight
        pathway_weight *= self.config.pathway_weight
        attention_weight *= self.config.attention_reg_weight
        #
        # # Combine losses with dynamic weights
        total_loss = (
            age_weight * age_loss +
            pathway_weight * pathway_loss +
            attention_weight * attention_loss
        )

        # Update debug info with weights
        self.debug_info.update({
            'age_weight': age_weight,#.detach(),
            'pathway_weight': pathway_weight,#.detach(),
            'attention_weight': attention_weight,#.detach(),
        })

        loss_dict = {
            'total': total_loss,
            'age': age_loss,
            'pathway': pathway_loss,
            'attention_reg': attention_loss,
            'debug': {
                'pathway_scores': {
                    'predicted': self.debug_info['pred_scores'],
                    'actual': self.debug_info['true_scores'],
                    'relative_errors': self.debug_info['relative_errors'],
                    'raw_loss_per_pathway': self.debug_info['raw_loss_per_pathway'],
                    'raw_total_loss': self.debug_info['raw_total_loss'],
                    'scaled_total_loss': self.debug_info['scaled_total_loss'],
                    'variance_loss': self.debug_info['variance_loss'],
                    'final_loss': self.debug_info['final_pathway_loss'],
                },
                'attention_reg': {
                    'attention_kl_loss': self.debug_info['attention_kl_loss'],
                    'entropy': self.debug_info['attention_entropy'],
                    'attention_entropy_reg': self.debug_info['attention_entropy_reg'],
                    'weight': self.debug_info['attention_weight'],
                    'final_loss':self.debug_info['final_attn_loss']
                }
            }
        }

        return total_loss, loss_dict