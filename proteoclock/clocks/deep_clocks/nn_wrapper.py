import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error
import pickle
from pathlib import Path
from typing import Dict, Tuple, Union, Optional, List
from collections import defaultdict

from .reporting import ModelEvaluator
from .ipf_clock import IPFClock
from proteoclock.other import IPFPathways, UKBDataset
from proteoclock.data_preprocess.scalers import OLINKScaler

class AgingClockPredictor:
    """Wrapper class for making predictions with both types of aging clock models"""

    def __init__(self,
                 model_path: str,
                 feats: Union[Dict[str, str], str],
                 model_type: str = 'auto',
                 device: Optional[str] = None):
        """Initialize predictor

        Args:
            model_path: Path to saved model
            model_type: Type of model ('auto', 'ipf_clock', or 'ipf_transfer')
            device: Device to run predictions on
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(self.device)

        # Load model
        self._load_model(model_path, model_type, feats)

        # Initialize evaluator
        self.evaluator = ModelEvaluator(self.model)

    def _load_model(self,
                    model_path: str,
                    model_type: str,
                    feats: Union[Dict[str, str], str]) -> None:
        """Load the trained model

        Args:
            model_path: Path to saved model
            model_type: Type of model to load
            feats: Either a dictionary mapping features to genes or a path to a directory
                  containing feature_order.txt
        """
        try:
            # First try with weights_only=False (less secure but handles numpy objects)
            model_dict = torch.load(model_path, map_location=self.device, weights_only=False)
        except Exception as e:
            # As a fallback, try with allowlisting the numpy scalar type
            try:
                from torch.serialization import add_safe_globals
                add_safe_globals(['numpy.core.multiarray.scalar'])
                model_dict = torch.load(model_path, map_location=self.device)
            except Exception as nested_e:
                raise RuntimeError(f"Failed to load model: {e}\nTried fallback method but failed with: {nested_e}")

        # Handle feats as string (path to feature file)
        if isinstance(feats, str):
            feature_file = Path(feats)
            if not feature_file.exists():
                raise FileNotFoundError(f"Feature file not found: {feature_file}")
            
            # Load features from file - assuming one feature per line
            with open(feature_file, 'r') as f:
                feats = {x.split('\t')[0]:x.split('\t')[1] 
                            for x in f.read().splitlines()}

        # Get OLINK features from model
        self.olink_features = list(feats.keys())

        # Determine model type if auto
        if model_type == 'auto':
            # Check model architecture from state dict
            has_pathway_head = any('pathway_head' in key for key in model_dict['model_state_dict'].keys())
            model_type = 'ipf_transfer' if has_pathway_head else 'ipf_clock'

        # Initialize appropriate model
        if model_type == 'ipf_transfer':
            self.model = IPFTransferNet(
                olink_dim=len(self.olink_features),
                olink_to_gene=feats,
                pathway_definitions=IPFPathways()
            ).float()
        elif model_type == 'ipf_clock':
            self.model = IPFClock(
                olink_dim=len(self.olink_features),
                olink_to_gene=feats,
                pathway_definitions=IPFPathways()
            ).float()
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.model.to(self.device)
        self.model.load_state_dict(model_dict['model_state_dict'])
        self.model.eval()

        # Store model type
        self.model_type = model_type

    def _predict_batch(self,
                       batch: torch.Tensor) -> Tuple:
        """Make predictions for a single batch

        Args:
            batch: Input tensor of shape (batch_size, n_features)

        Returns:
            Tuple of predictions (age, pathway_scores, attention_weights)
        """
        if self.model_type == 'ipf_transfer':
            age_pred, pathway_pred, attention = self.model(batch)
            return age_pred, pathway_pred, attention
        else:
            age_pred, attention = self.model(batch)
            return age_pred, None, attention

    def predict_age(self, 
                data: Union[pd.DataFrame, np.ndarray],
                scaling: Optional[str] = None) -> np.ndarray:
        """
        Predict age from protein data, with interface matching simple clocks
        
        Args:
            data: OLINK protein measurements (DataFrame or NumPy array)
            scaling: Scaling parameter (ignored for deep learning models, included for API compatibility)
            
        Returns:
            numpy.ndarray: Array of predicted ages
        """
        # Handle data conversion internally
        if isinstance(data, pd.DataFrame):
            # Check if data is in long format (has gene_symbol and NPX columns)
            if 'gene_symbol' in data.columns and 'NPX' in data.columns:
                # Convert from long to wide format
                patient_ids = data['patient_id'].unique()
                wide_data = data.pivot_table(index='patient_id', columns='gene_symbol', values='NPX')
                wide_data = wide_data.reset_index()
                data_array = wide_data.iloc[:, 1:].values  # Exclude patient_id column
            else:
                # Assume it's already in wide format, just extract numeric columns
                numeric_cols = data.select_dtypes(include=['number']).columns
                data_array = data[numeric_cols].values
        else:
            # Already a NumPy array, use as is
            data_array = data
        
        # Ensure correct data type
        data_array = np.asarray(data_array, dtype=np.float32)
        
        # Call the existing predict method with the processed data
        results = self.predict(data_array)
        
        # Return just the age predictions to match simple clock interface
        return results['age_predictions']
    
    def predict(self,
                data: Union[pd.DataFrame, np.ndarray],
                true_ages: Optional[np.ndarray] = None,
                batch_size: int = 32) -> Dict[str, np.ndarray]:
        """Make predictions on new data

        Args:
            data: OLINK protein measurements
            true_ages: Optional array of true ages for evaluation
            batch_size: Batch size for prediction

        Returns:
            Dictionary containing predictions and optionally evaluation metrics
        """
        # Create dataset
        if true_ages is not None:
            dataset = UKBDataset(data, true_ages.astype(np.float32))
        else:
            dataset = UKBDataset(data, np.zeros(len(data)))

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False
        )

        # Get predictions
        age_preds = []
        attention_weights = []
        pathway_preds = [] if self.model_type == 'ipf_transfer' else None

        with torch.no_grad():
            for batch in loader:
                olink = batch['olink'].to(self.device)
                age_pred, pathway_pred, attention = self._predict_batch(olink)

                age_preds.append(age_pred.squeeze().cpu().numpy())
                attention_weights.append(attention.cpu().numpy())
                if pathway_pred is not None:
                    pathway_preds.append(pathway_pred.cpu().numpy())

        # Combine predictions
        results = {
            'age_predictions': np.concatenate(age_preds),
            'attention_weights': np.concatenate(attention_weights)
        }

        # Add pathway predictions if available
        if pathway_preds is not None:
            results['pathway_predictions'] = np.concatenate(pathway_preds)

            # Add true pathway scores if model supports it
            if hasattr(self.model, 'compute_pathway_scores'):
                true_pathway_scores = []
                with torch.no_grad():
                    for batch in loader:
                        olink = batch['olink'].to(self.device)
                        scores = self.model.compute_pathway_scores(olink)
                        true_pathway_scores.append(scores.cpu().numpy())
                results['true_pathway_scores'] = np.concatenate(true_pathway_scores)

                # Calculate correlations
                correlations = []
                for i in range(results['pathway_predictions'].shape[1]):
                    corr = np.corrcoef(
                        results['true_pathway_scores'][:, i],
                        results['pathway_predictions'][:, i]
                    )[0, 1]
                    correlations.append(corr)
                results['pathway_correlations'] = np.array(correlations)

        # Add evaluation metrics if true ages provided
        if true_ages is not None:
            results['true_ages'] = true_ages
            results['mae'] = mean_absolute_error(true_ages, results['age_predictions'])
            results['r2'] = r2_score(true_ages, results['age_predictions'])

        return results

    def plot_predictions(self,
                         predictions: Dict[str, np.ndarray],
                         output_dir: Optional[Path] = None):
        """Plot prediction results

        Args:
            predictions: Dictionary of predictions from predict()
            output_dir: Optional directory to save plots
        """
        if 'true_ages' not in predictions:
            raise ValueError("True ages required for plotting predictions")

        # Plot age predictions
        age_fig = self.evaluator.plot_age_predictions(
            predictions['true_ages'],
            predictions['age_predictions']
        )

        if output_dir:
            age_fig.savefig(output_dir / 'age_predictions.png',
                            bbox_inches='tight',
                            dpi=300)

        # Plot pathway predictions if available
        if 'pathway_predictions' in predictions:
            pathway_fig = self.evaluator.plot_pathway_correlations(
                predictions['pathway_correlations']
            )

            if output_dir:
                pathway_fig.savefig(output_dir / 'pathway_predictions.png',
                                    bbox_inches='tight',
                                    dpi=300)
        plt.show()

    def save_predictions(self,
                         predictions: Dict[str, np.ndarray],
                         output_dir: Path,
                         sample_ids: Optional[List[str]] = None):
        """Save predictions to files

        Args:
            predictions: Dictionary of predictions from predict()
            output_dir: Directory to save outputs
            sample_ids: Optional list of sample IDs
        """
        output_dir.mkdir(exist_ok=True)

        # Create DataFrame for age predictions
        if sample_ids is None:
            sample_ids = [f'Sample_{i}' for i in range(len(predictions['age_predictions']))]

        outputs = pd.DataFrame({
            'sample_id': sample_ids,
            'predicted_age': predictions['age_predictions']
        })

        if 'true_ages' in predictions:
            outputs['true_age'] = predictions['true_ages']

        outputs.to_csv(output_dir / 'age_predictions.csv', index=False)

        # Save pathway predictions if available
        if 'pathway_predictions' in predictions:
            pathway_names = ['TGF-β', 'ECM', 'Inflammation', 'Oxidative Stress']
            pathway_df = pd.DataFrame(
                predictions['pathway_predictions'],
                columns=[f'{p}_score' for p in pathway_names],
                index=sample_ids
            )

            if 'true_pathway_scores' in predictions:
                true_scores = pd.DataFrame(
                    predictions['true_pathway_scores'],
                    columns=[f'{p}_true_score' for p in pathway_names],
                    index=sample_ids
                )
                pathway_df = pd.concat([pathway_df, true_scores], axis=1)

            pathway_df.to_csv(output_dir / 'pathway_predictions.csv')

            # Save correlation summary
            corr_summary = pd.DataFrame({
                'pathway': pathway_names,
                'correlation': predictions['pathway_correlations']
            })
            corr_summary.to_csv(output_dir / 'pathway_correlations.csv', index=False)


def main():
    """Example usage"""
    # Set paths
    data_path = "./protein_data.csv"
    model_path = "best_clock.pt"
    output_dir = Path("./clock_outputs")
    output_dir.mkdir(exist_ok=True)

    # Load and initialize predictor
    predictor = AgingClockPredictor(model_path)

    # Initialize scaler with required features
    scaler = OLINKScaler(required_features=predictor.olink_features)

    # Load and format data
    raw_data = pd.read_csv(data_path)
    formatted_data = scaler.format_long_data(raw_data)

    # Fit scaler on training data (if not already fitted)
    scaler.fit(formatted_data)
    scaler.save(output_dir / 'olink_scaler.pkl')

    # Transform new data
    scaled_data = scaler.transform(formatted_data)

    # Make predictions
    results = predictor.predict(scaled_data)

    # Save predictions
    output = pd.DataFrame({
        'patient_id': formatted_data.index,
        'predicted_age': results['age_predictions']
    })
    output.to_csv(output_dir / 'predictions.csv', index=False)

    print(f"Results saved to {output_dir}")
    print(f"Predicted ages range: [{results['age_predictions'].min():.1f}, "
          f"{results['age_predictions'].max():.1f}]")