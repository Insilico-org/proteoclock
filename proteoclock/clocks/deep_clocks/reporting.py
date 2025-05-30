import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
import torch
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any


class ModelEvaluator:
    """Unified evaluator that handles both simple aging clocks and pathway models"""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.has_pathways = self._check_pathway_capability()

    def _check_pathway_capability(self) -> bool:
        """Check if model has pathway prediction capability"""
        try:
            return hasattr(self.model, 'compute_pathway_scores')
        except:
            return False

    def evaluate_predictions(self,
                             data_loader: torch.utils.data.DataLoader,
                             device: str) -> Dict[str, Any]:
        """Evaluate model predictions - handles both age-only and age+pathway models"""
        self.model.eval()
        true_ages = []
        pred_ages = []

        if self.has_pathways:
            true_scores = []
            pred_scores = []

        with torch.no_grad():
            for batch in data_loader:
                olink = batch['olink'].to(device)
                ages = batch['age']

                if self.has_pathways:
                    age_pred, pathway_pred, _ = self.model(olink)
                    true_pathway = self.model.compute_pathway_scores(olink)

                    true_scores.append(true_pathway.cpu().numpy())
                    pred_scores.append(pathway_pred.cpu().numpy())
                else:
                    age_pred, _ = self.model(olink)

                true_ages.extend(ages.numpy())
                pred_ages.extend(age_pred.cpu().squeeze().numpy())

        # Convert to arrays
        true_ages = np.array(true_ages)
        pred_ages = np.array(pred_ages)

        # Calculate age metrics
        results = {
            'age_metrics': {
                'r2': r2_score(true_ages, pred_ages),
                'mae': mean_absolute_error(true_ages, pred_ages),
                'baseline_mae': mean_absolute_error(
                    true_ages,
                    np.full_like(true_ages, np.mean(true_ages))
                ),
                'true_ages': true_ages,
                'pred_ages': pred_ages
            }
        }

        # Add pathway metrics if available
        if self.has_pathways:
            true_scores = np.concatenate(true_scores)
            pred_scores = np.concatenate(pred_scores)

            # Calculate correlation for each pathway
            correlations = []
            for i in range(true_scores.shape[1]):
                corr = np.corrcoef(true_scores[:, i], pred_scores[:, i])[0, 1]
                correlations.append(corr)

            results['pathway_metrics'] = {
                'correlations': np.array(correlations),
                'true_scores': true_scores,
                'pred_scores': pred_scores
            }

        return results

    def plot_training_curves(self,
                             train_losses: List[Dict[str, float]],
                             val_losses: List[Dict[str, float]]) -> plt.Figure:
        """Plot training curves - adapts based on available metrics"""
        if self.has_pathways:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))

        epochs = range(1, len(train_losses) + 1)

        # Age loss
        ax1.plot(epochs, [x['age'] for x in train_losses], 'b-', label='Train')
        ax1.plot(epochs, [x['age'] for x in val_losses], 'r-', label='Validation')
        ax1.set_title('Age Prediction Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MSE Loss')
        ax1.legend()

        # Pathway loss if available
        if self.has_pathways:
            ax2.plot(epochs, [x['pathway'] for x in train_losses], 'b-', label='Train')
            ax2.plot(epochs, [x['pathway'] for x in val_losses], 'r-', label='Validation')
            ax2.set_title('Pathway Prediction Loss')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('MSE Loss')
            ax2.legend()

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_age_predictions(true_ages: np.ndarray,
                             pred_ages: np.ndarray) -> plt.Figure:
        """Plot age prediction scatter plot"""
        plt.figure(figsize=(8, 8))

        plt.scatter(true_ages, pred_ages, alpha=0.5)
        plt.plot([true_ages.min(), true_ages.max()],
                 [true_ages.min(), true_ages.max()],
                 'r--', label='Perfect prediction')

        plt.xlabel('True Age')
        plt.ylabel('Predicted Age')
        plt.title('Age Prediction Performance')
        plt.legend()

        return plt.gcf()

    @staticmethod
    def plot_pathway_correlations(correlations: np.ndarray) -> plt.Figure:
        """Plot pathway prediction correlations"""
        pathway_names = ['TGF-β', 'ECM', 'Inflammation', 'Oxidative Stress']

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.bar(pathway_names, correlations)
        ax.set_ylabel('Correlation')
        ax.set_title('Pathway Score Prediction Performance')
        ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        return fig

    def generate_report(self,
                        train_loader: torch.utils.data.DataLoader,
                        val_loader: torch.utils.data.DataLoader,
                        train_history: List[Dict[str, float]],
                        val_history: List[Dict[str, float]],
                        device: str) -> Dict:
        """Generate comprehensive training report"""
        # Evaluate predictions
        train_metrics = self.evaluate_predictions(train_loader, device)
        val_metrics = self.evaluate_predictions(val_loader, device)

        # Generate plots
        loss_curves = self.plot_training_curves(train_history, val_history)
        train_scatter = self.plot_age_predictions(
            train_metrics['age_metrics']['true_ages'],
            train_metrics['age_metrics']['pred_ages']
        )
        val_scatter = self.plot_age_predictions(
            val_metrics['age_metrics']['true_ages'],
            val_metrics['age_metrics']['pred_ages']
        )

        report = {
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'plots': {
                'loss_curves': loss_curves,
                'train_scatter': train_scatter,
                'val_scatter': val_scatter,
            }
        }

        # Add pathway plots if available
        if self.has_pathways:
            train_pathway = self.plot_pathway_correlations(
                train_metrics['pathway_metrics']['correlations']
            )
            val_pathway = self.plot_pathway_correlations(
                val_metrics['pathway_metrics']['correlations']
            )
            report['plots'].update({
                'train_pathway': train_pathway,
                'val_pathway': val_pathway
            })

        return report

    def diagnostic_checks(self,
                          val_loader: torch.utils.data.DataLoader,
                          device: str):
        """Run diagnostic checks on model outputs"""
        self.model.eval()

        with torch.no_grad():
            batch = next(iter(val_loader))
            olink = batch['olink'].to(device)

            if self.has_pathways:
                age_pred, pathway_pred, _ = self.model(olink)
                true_pathway = self.model.compute_pathway_scores(olink)

                print("\nDiagnostic Information:")
                print(f"Age predictions range: [{age_pred.min().item():.2f}, {age_pred.max().item():.2f}]")
                print(f"Pathway predictions range: [{pathway_pred.min().item():.2f}, {pathway_pred.max().item():.2f}]")
                print(f"True pathway scores range: [{true_pathway.min().item():.2f}, {true_pathway.max().item():.2f}]")

                print("\nPathway predictions variance:")
                print(torch.var(pathway_pred, dim=0))
                print("\nTrue pathway scores variance:")
                print(torch.var(true_pathway, dim=0))

                print("\nPathway protein indices:")
                for name, indices in self.model.pathway_proteins.items():
                    print(f"{name}: {indices}")
            else:
                age_pred, _ = self.model(olink)
                print("\nDiagnostic Information:")
                print(f"Age predictions range: [{age_pred.min().item():.2f}, {age_pred.max().item():.2f}]")

    @staticmethod
    def check_data_scaling(train_loader: torch.utils.data.DataLoader,
                           val_loader: torch.utils.data.DataLoader):
        """Check data scaling between train and validation sets"""

        def get_stats(loader):
            all_ages = []
            all_data = []
            for batch in loader:
                all_ages.extend(batch['age'].numpy())
                all_data.extend(batch['olink'].numpy())
            return np.array(all_ages), np.array(all_data)

        train_ages, train_data = get_stats(train_loader)
        val_ages, val_data = get_stats(val_loader)

        print("\nData Scaling Check:")
        print(f"Train ages: mean={train_ages.mean():.2f}, std={train_ages.std():.2f}")
        print(f"Val ages: mean={val_ages.mean():.2f}, std={val_ages.std():.2f}")
        print(f"Train data: mean={train_data.mean():.2f}, std={train_data.std():.2f}")
        print(f"Val data: mean={val_data.mean():.2f}, std={val_data.std():.2f}")

def debug_batch_losses(pred_ages: torch.Tensor,
                       true_ages: torch.Tensor,
                       loss_dict: Dict[str, float]):
    """Debug helper to print detailed loss calculation for a batch"""
    pred = pred_ages.detach().cpu().numpy().squeeze()
    true = true_ages.detach().cpu().numpy()

    abs_errors = np.abs(pred - true)
    rel_errors = np.abs((pred - true) / true)
    squared_rel_errors = rel_errors ** 2

    print("\nBatch Loss Analysis:")
    print(f"Sample size: {len(pred)}")
    print(f"True ages range: [{true.min():.1f}, {true.max():.1f}], mean: {true.mean():.1f}")
    print(f"Predicted ages range: [{pred.min():.1f}, {pred.max():.1f}], mean: {pred.mean():.1f}")

    print(f"\nError Metrics:")
    print(f"Mean Absolute Error: {np.mean(abs_errors):.2f} years")
    print(f"Mean Relative Error: {np.mean(rel_errors):.4f}")
    print(f"Loss Function Output: {loss_dict['age']:.6f}")

    print("\nDetailed examples (first 3 samples):")
    for i in range(min(3, len(pred))):
        print(f"\nSample {i + 1}:")
        print(f"True age: {true[i]:.1f}")
        print(f"Predicted age: {pred[i]:.1f}")
        print(f"Absolute error: {abs_errors[i]:.1f} years")
        print(f"Relative error: {rel_errors[i]:.4f}")
        print(f"Squared relative error: {squared_rel_errors[i]:.4f}")