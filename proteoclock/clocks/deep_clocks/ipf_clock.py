import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import mean_absolute_error, r2_score

from .losses import LossConfig
from proteoclock.other import IPFPathways, ScaleShift, UKBDataset

class IPFClock(nn.Module):

    def __init__(self,
                 olink_dim: int,
                 olink_to_gene: dict,
                 pathway_definitions,
                 feature_dim: int = 128,
                 dropout_rate: float = 0.2,
                 attention_reg_strength: float = 0.25):
        super().__init__()

        # Basic attributes
        self.feature_dim = feature_dim
        self.olink_to_gene = olink_to_gene
        self.pathway_definitions = pathway_definitions
        self.pathway_knowledge = None
        self.pathway_attention = None

        # Map OLINK IDs to column indices
        self.olink_columns = {id: idx for idx, id in enumerate(self.olink_to_gene.keys())}
        self.setup_protein_indices()

        self.loss_fn = LossConfig()

        # Feature extraction backbone
        self.feature_extractor = nn.Sequential(
            nn.Linear(olink_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU()
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.BatchNorm1d(64),
            nn.Tanh(),
            nn.Linear(64, 4),
            nn.Softmax(dim=1)
        )

        # Age prediction branch - enhanced to handle age scale
        self.age_transform = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.age_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus(),  # Ensures positive output
            ScaleShift(scale=85, shift=40)  # Scale to approximate age range [40, 125]
        )

        # Attention regularization
        self.initial_attention_weights = None
        self.attention_reg_strength = attention_reg_strength

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass"""
        # Extract shared features
        features = self.feature_extractor(x)

        # Generate attention weights
        attention_weights = self.attention(features)

        # Transform features separately for each task
        age_features = self.age_transform(features)

        # Generate predictions
        age_pred = self.age_head(age_features)

        # Apply attention weights to age scores
        pathway_scores = attention_weights * age_pred

        return age_pred, attention_weights

    def setup_protein_indices(self):
        """Map OLINK IDs to their column indices"""
        self.pathway_proteins = {
            'tgf_beta': [],
            'ecm': [],
            'inflammation': [],
            'oxidative': []
        }

        # Create reverse mapping from gene to OLINK ID
        gene_to_olink = {v: k for k, v in self.olink_to_gene.items()}

        pathways = self.pathway_definitions.get_all_pathways()

        # Map pathway proteins to column indices
        for gene in pathways['TGF_BETA']:
            if gene in gene_to_olink:
                olink_id = gene_to_olink[gene]
                if olink_id in self.olink_columns:
                    self.pathway_proteins['tgf_beta'].append(self.olink_columns[olink_id])

        for gene in pathways['ECM_REMODELING']:
            if gene in gene_to_olink:
                olink_id = gene_to_olink[gene]
                if olink_id in self.olink_columns:
                    self.pathway_proteins['ecm'].append(self.olink_columns[olink_id])

        for gene in pathways['INFLAMMATION']:
            if gene in gene_to_olink:
                olink_id = gene_to_olink[gene]
                if olink_id in self.olink_columns:
                    self.pathway_proteins['inflammation'].append(self.olink_columns[olink_id])

        for gene in pathways['OXIDATIVE_STRESS']:
            if gene in gene_to_olink:
                olink_id = gene_to_olink[gene]
                if olink_id in self.olink_columns:
                    self.pathway_proteins['oxidative'].append(self.olink_columns[olink_id])

    def encode_pathway_knowledge(self,
                                 rna_data: pd.DataFrame,
                                 pathway_annotations: Dict[str, List[str]],
                                 weights_config) -> None:
        """Encode RNA evidence into pathway knowledge"""
        pathway_embeddings = []
        pathway_importances = []

        for pathway, genes in pathway_annotations.items():
            # Get weighted evidence for pathway genes
            pathway_de = []
            pathway_significance = 0

            for gene in genes:
                fc, pval = weights_config.get_weighted_evidence(
                    rna_data,
                    gene_column='gene',
                    gene=gene,
                    fc_column='log2FC',
                    pvalue_column='pvalue'
                )
                if fc is not None:
                    pathway_de.append((fc, pval))
                    pathway_significance += -np.log10(pval)

            if pathway_de:
                # Create pathway embedding
                embedding = np.zeros(self.feature_dim)
                for fc, pval in pathway_de:
                    weight = -np.log10(pval)
                    embedding += weight * fc

                pathway_embeddings.append(embedding / len(pathway_de))
                pathway_importances.append(pathway_significance / len(pathway_de))

        if pathway_embeddings:
            self.pathway_knowledge = torch.tensor(
                pathway_embeddings,
                dtype=torch.float32,
                device=next(self.parameters()).device  # Put on same device as model
            )

            # Store initial attention weights
            importances = np.array(pathway_importances)
            normalized_importances = importances / importances.sum()

            # Store on same device as model
            self.initial_attention_weights = torch.tensor(
                normalized_importances,
                dtype=torch.float32,
                requires_grad=False,  # Make sure these don't get updated
                device=next(self.parameters()).device  # Put on same device as model
            )
    def get_attention_regularization_loss(self, current_attention: torch.Tensor) -> torch.Tensor:
        """Calculate regularization loss to maintain initial pathway importance"""
        if self.initial_attention_weights is None:
            return torch.tensor(0.0, device=current_attention.device)

        # Move initial weights to same device as current attention
        initial_weights = self.initial_attention_weights.to(current_attention.device)

        # Calculate KL divergence between current and initial attention distributions
        kl_div = F.kl_div(
            current_attention.log(),  # Current attention (log probabilities)
            initial_weights.expand_as(current_attention),  # Initial weights expanded to batch size
            reduction='batchmean'
        )

        return self.attention_reg_strength * kl_div

def analyze_samples(model: nn.Module,
                    data_loader: torch.utils.data.DataLoader,
                    device: str) -> Dict[str, float]:
    """Evaluate model on data loader using integrated loss calculation"""
    model.eval()
    total_losses = {
        'total': 0,
        'age': 0,
        'attention_reg': 0
    }

    age_preds = []
    true_ages = []

    with torch.no_grad():
        for batch in data_loader:
            olink = batch['olink'].to(device)
            ages = batch['age'].to(device)

            # Forward pass
            age_pred, attention_weights = model(olink)

            # Store predictions
            age_preds.append(age_pred.squeeze().cpu().numpy())
            true_ages.append(ages.cpu().numpy())

            # Calculate losses using model's loss function
            _, loss_dict = model.loss_fn(
                pred_ages=age_pred,
                true_ages=ages,
                attention_weights=attention_weights,
                initial_attention=model.initial_attention_weights
            )

            # Get batch size and compute batch losses
            batch_size = model.loss_fn.get_batch_size(ages)
            batch_losses = model.loss_fn.compute_batch_losses(batch_size, loss_dict)

            # Accumulate losses
            for k in total_losses:
                total_losses[k] += batch_losses[k]

    # Combine predictions for metrics calculation
    age_preds = np.concatenate(age_preds)
    true_ages = np.concatenate(true_ages)

    # Calculate metrics
    r2 = r2_score(true_ages, age_preds)
    mae = mean_absolute_error(true_ages, age_preds)

    # Average losses
    n_samples = len(data_loader.dataset)
    avg_losses = {k: v / n_samples for k, v in total_losses.items()}

    # Add metrics to results
    avg_losses.update({
        'r2': r2,
        'mae': mae
    })

    return avg_losses


def main():
    pass

if __name__ == "__main__":
    main()