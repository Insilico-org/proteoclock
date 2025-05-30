"""
Simple clocks module for proteomic aging clock implementations.

This module provides implementations of various proteomic aging clocks
from published research, allowing users to predict biological age
from protein expression data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any, Literal
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error
from scikit_posthocs import posthoc_dunn
from scipy.stats import pearsonr, mannwhitneyu, wilcoxon, kruskal
import pickle
from abc import ABC, abstractmethod
import warnings

from proteoclock.data_preprocess.scalers import OLINKScaler


class BaseAgingClock(ABC):
    """Abstract base class for aging clock implementations.

    This class provides common functionality for loading coefficients,
    handling scalers, and preprocessing data. Specific clock implementations
    should inherit from this class and implement the predict_age method.
    """

    required_params: list[str] = []

    def __init__(self,
                 coef_file: Union[str, Path],
                 scaler_file: Optional[Union[str, Path]] = None,
                 sep: str = '\t',
                 header: Optional[int] = None,
                 index_col: Optional[Union[str, int]] = 0):
        """Initialize the aging clock with coefficients from a file.

        Args:
            coef_file: Path to a TSV file containing coefficients
            scaler_file: Optional path to a saved OLINKScaler for data preprocessing
            sep: Separator used in coefficient file
            header: Header row in coefficient file
            index_col: Column to use as index in coefficient file
        """
        # Load coefficients
        self.coef_df = pd.read_csv(coef_file, sep=sep, header=header, index_col=index_col)

        # Parse coefficients - to be implemented by subclasses
        self._valid_coefs()
        self._parse_coefs()

        # Initialize data preprocessing attributes
        self.scaler = None
        if scaler_file is not None:
            self._load_scaler(scaler_file)
        else:
            self._init_scaler()

    def _valid_coefs(self):
        missing_params = [p for p in self.required_params if p not in self.coef_df.index]
        if missing_params:
            raise ValueError(f"Missing required parameters in coefficient file: {missing_params}")

    def _setattr_coefs(self):
        for a in self.required_params:
            setattr(self, a.lower(), float(self.coef_df.loc[a].iloc[0]))
            
    @abstractmethod
    def _parse_coefs(self):
        """Parse coefficients from coef_df and set up model-specific attributes.

        This method should be implemented by each subclass to handle its specific
        coefficient parsing needs.
        """
        pass

    def _init_scaler(self):
        """Initialize a new OLINKScaler with required protein features."""
        self.scaler = OLINKScaler(required_features=list(self.protein_coefs.keys()))

    def _load_scaler(self, scaler_file: Union[str, Path]):
        """Load a saved OLINKScaler from file."""
        self.scaler = OLINKScaler.load(scaler_file)

    def _preprocess_data(self,
                         data: pd.DataFrame,
                         scaling: Optional[Literal['standard', 'minmax']] = None,
                         ignore_missing: bool = True) -> pd.DataFrame:
        """Preprocess protein data using the scaler.

        Args:
            data: DataFrame with protein measurements in long format.
                 Must have columns: ['patient_id', 'gene_symbol', 'NPX']
            scaling: Type of scaling to apply. Options: 'standard', 'minmax', or None
            ignore_missing: Whether to ignore missing features

        Returns:
            DataFrame with scaled protein measurements in wide format
        """
        if self.scaler is None:
            # If no scaler provided, just pivot the data
            return data.pivot(
                index='patient_id',
                columns='gene_symbol',
                values='NPX'
            )

        # Use scaler to convert to wide format and impute missing values
        wide_data = self.scaler.long_to_wide(data)

        missing_features = set(self.protein_coefs.keys()) - set(wide_data.columns)
        if missing_features:
            if ignore_missing:
                warnings.warn(f"Missing features in columns: {missing_features}")
            else:
                raise ValueError(f"Missing required protein features: {missing_features}")

        processed_data = self.scaler.prepare_wide(wide_data, impute_missing=True, transform=scaling)
        return pd.DataFrame(processed_data, index=wide_data.index, columns=wide_data.columns)

    @abstractmethod
    def predict_age(self,
                    data: pd.DataFrame,
                    scaling: Optional[Literal['standard', 'minmax']] = 'standard') -> pd.Series:
        """Predict biological age using the model.

        Args:
            data: DataFrame with protein measurements in long format.
                 Must have columns: ['patient_id', 'gene_symbol', 'NPX']
            scaling: Type of scaling to apply. Options: 'standard', 'minmax', or None

        Returns:
            Series with predicted ages indexed by patient_id
        """
        pass



class GompertzClock(BaseAgingClock):
    """Implementation of the Proteomic Aging Clock (PAC).

    This class implements a Gompertz-based aging clock that predicts biological age
    based on protein measurements and mortality risk over a 10-year period.
    The implementation follows the methodology described in the source paper.
    """

    required_params: list[str] =  ['shape_0', 'rate_0', 'age_coef_0', 'shape', 'rate', 'age_coef']

    def _parse_coefs(self):
        # Extract Gompertz parameters
        self._setattr_coefs()

        # Validate and compute constant term
        denom = self.rate_0 * (1 - np.exp(10 * self.shape_0))
        if abs(denom) < 1e-10:
            raise ValueError("Unstable PAC calculation: rate_0 or shape_0 parameters may need adjustment")
        self.const = self.shape_0 / denom

        # Create protein coefficients dictionary excluding parameters
        self.protein_coefs = dict(zip(self.coef_df.index, self.coef_df.iloc[:, 0]))
        self.protein_coefs = {k: v for k, v in self.protein_coefs.items()
                              if k not in self.required_params}

    def _init_scaler(self):
        """Initialize a new OLINKScaler with required protein features"""
        required_features = list(self.protein_coefs.keys())
        self.scaler = OLINKScaler(required_features=required_features)

    def _load_scaler(self, scaler_file: Union[str, Path]):
        """Load a saved OLINKScaler from file"""
        self.scaler = OLINKScaler.load(scaler_file)

    def _compute_gompertz_cdf(self, scores: np.ndarray, t: float = 10.0) -> np.ndarray:
        """Compute the Gompertz cumulative distribution function.

        Args:
            scores: Linear predictor scores
            t: Time period (default 10 years)

        Returns:
            Array of CDF values
        """
        b_x = self.rate * np.exp(scores)
        return 1 - np.exp((-b_x / self.shape) * (np.exp(self.shape * t) - 1))

    def predict_age(self,
                    data: pd.DataFrame,
                    age_data: pd.DataFrame,
                    scaling:Optional[Literal['standard', 'minmax']]=None) -> pd.Series:
        """Predict biological age using the PAC formula.

        Args:
            data: DataFrame with protein measurements in long format.
                 Must have columns: ['patient_id', 'gene_symbol', 'NPX']
            age_data: DataFrame with chronological ages.
                     Must have columns: ['patient_id', 'age']

        Returns:
            Series with predicted ages indexed by patient_id
        """
        # Convert to wide format and handle missing values
        data = self._preprocess_data(data, scaling=scaling)

        # Compute linear predictor scores
        scores = np.zeros(len(data))
        for protein, coef in self.protein_coefs.items():
            if protein in data.columns:
                scores += coef * data[protein].values

        # Add age contribution
        age_contribution = self.age_coef * age_data.set_index('patient_id')['age']
        scores += age_contribution[data.index].values

        # Compute Gompertz CDF
        cdf_values = self._compute_gompertz_cdf(scores)
        # Calculate PAC using the formula from the paper
        pac_values = (1 / self.age_coef_0) * np.log(self.const * np.log(1 - cdf_values))

        return pd.Series(pac_values, index=data.index, name='predicted_age')


class LinearClock(BaseAgingClock):
    """Implementation of a Linear Proteomic Aging Clock.

    This class implements a linear regression-based aging clock that predicts biological age
    based on protein measurements. The model is trained using elastic net regression.
    """

    required_params: list[str] = ['Intercept']

    def _parse_coefs(self):
        # Extract intercept
        self.intercept = float(self.coef_df.loc['Intercept'].iloc[0])
        
        # Create protein coefficients dictionary excluding intercept
        self.protein_coefs = dict(zip(self.coef_df.index, self.coef_df.iloc[:, 0]))
        self.protein_coefs = {k: v for k, v in self.protein_coefs.items()
                              if k != 'Intercept'}

    def predict_age(self,
                    data: pd.DataFrame,
                    scaling: Optional[Literal['standard', 'minmax']] = None) -> pd.Series:
        """Predict biological age using the linear model.

        Args:
            data: DataFrame with protein measurements in long format.
                 Must have columns: ['patient_id', 'gene_symbol', 'NPX']
            scaling: Type of scaling to apply

        Returns:
            Series with predicted ages indexed by patient_id
        """
        # Convert to wide format and handle missing values
        data = self._preprocess_data(data, scaling=scaling)

        # Compute linear predictor
        predictions = np.ones(len(data)) * self.intercept
        for protein, coef in self.protein_coefs.items():
            if protein in data.columns:
                predictions += coef * data[protein].values

        return pd.Series(predictions, index=data.index, name='predicted_age')


class CPHClock(BaseAgingClock):
    """Implementation of a Cox Proportional Hazards Clock.

    This class implements a mortality risk prediction model based on the Cox Proportional
    Hazards model. The model predicts mortality risk based on protein measurements.
    Risk scores can be converted to age-like predictions using a Gompertz model.
    """
    required_params:list[str] = []

    def _parse_coefs(self):

        gompertz_params = {
            'intercept': -9.94613787413831,
            'slope': 0.0897860500778604,
            'avg_age': 57.29426
        }

        self.time_horizon = 10.
        self.slope = gompertz_params['slope']
        self.intercept = gompertz_params['intercept']
        self.avg_rel_log_mort_hazard = (self.intercept + self.slope * gompertz_params['avg_age'])
        # -4.80

        self.protein_coefs = dict(zip(self.coef_df.index, self.coef_df.iloc[:, 0]))
        self.protein_coefs = {k: v for k, v in self.protein_coefs.items()}

    def predict_rel_log_hazard(self,
                     data: pd.DataFrame,
                     scaling:Optional[Literal['standard', 'minmax']]=None) -> pd.Series:
        """Predict mortality risk using the CPH model.

        The risk score is computed as the linear predictor (X @ beta).
        To get the actual survival probability, one would need the baseline
        hazard function, which is not typically provided with pre-trained models.

        Args:
            data: DataFrame with protein measurements in long format.
                 Must have columns: ['patient_id', 'gene_symbol', 'NPX']

        Returns:
            Series with predicted risk scores indexed by patient_id.
            Higher scores indicate higher mortality risk.
        """
        # Convert to wide format and handle missing values
        data = self._preprocess_data(data, scaling=scaling)

        # Calculate risk score (linear predictor)
        risk_score = pd.Series(0.0, index=data.index)
        for protein, coef in self.protein_coefs.items():
            risk_score += coef * data[protein]

        return risk_score

    def predict_age(self,
                    data: pd.DataFrame,
                    scaling:Optional[Literal['standard', 'minmax']]=None) -> pd.Series:
        """Predict biological age using the CPH model and Gompertz conversion.

        This method first computes the mortality risk score, then converts it
        to an age-like prediction using the Gompertz model parameters.

        Args:
            data: DataFrame with protein measurements in long format.
                 Must have columns: ['patient_id', 'gene_symbol', 'NPX']

        Returns:
            Series with predicted ages indexed by patient_id
        """
        # Get risk scores
        risk_scores = self.predict_rel_log_hazard(data, scaling)

        # Convert to ages using the Gompertz model
        ages = (risk_scores-self.avg_rel_log_mort_hazard) / self.slope - self.intercept

        return pd.Series(ages, index=risk_scores.index, name='predicted_age')


def generate_report(predictions: pd.Series, 
                    reference_data: pd.DataFrame, 
                    id_col: str = 'patient_id',
                    age_col: str = 'age') -> Dict:
    """Generate a basic performance report for age predictions.

    Args:
        predictions: Series with predicted ages
        reference_data: DataFrame with reference data including chronological ages
        id_col: Name of column containing patient IDs
        age_col: Name of column containing chronological ages

    Returns:
        Dictionary with performance metrics
    """
    # Ensure reference_data has patient_id as index
    if reference_data.index.name != id_col:
        reference_data = reference_data.set_index(id_col)
    
    # Get chronological ages for patients with predictions
    common_ids = set(predictions.index) & set(reference_data.index)
    if not common_ids:
        raise ValueError("No common patient IDs between predictions and reference data")
    
    pred_ages = predictions.loc[common_ids]
    real_ages = reference_data.loc[common_ids, age_col]
    
    # Compute performance metrics
    metrics = {
        'r2': r2_score(real_ages, pred_ages),
        'mae': mean_absolute_error(real_ages, pred_ages),
        'pearson_r': pearsonr(real_ages, pred_ages)[0],
        'n_samples': len(common_ids)
    }
    
    return metrics


class ModelPerformanceReporter:
    """Class for generating and visualizing performance reports for aging clocks."""
    
    def __init__(self, predictions: Dict[str, pd.Series], 
                 reference_data: pd.DataFrame,
                 id_col: str = 'patient_id',
                 age_col: str = 'age'):
        """Initialize reporter with predictions and reference data.
        
        Args:
            predictions: Dictionary mapping model names to prediction Series
            reference_data: DataFrame with reference data including chronological ages
            id_col: Name of column containing patient IDs
            age_col: Name of column containing chronological ages
        """
        self.predictions = predictions
        self.reference_data = reference_data
        self.id_col = id_col
        self.age_col = age_col
        
        # Ensure reference_data has patient_id as index
        if reference_data.index.name != id_col:
            self.reference_data = reference_data.set_index(id_col)
        
        # Generate reports for each model
        self.reports = {model: generate_report(preds, self.reference_data, 
                                              id_col=id_col, age_col=age_col) 
                       for model, preds in predictions.items()}
    
    def plot_correlation_matrix(self, figsize=(10, 8)):
        """Plot correlation matrix between different model predictions.
        
        Args:
            figsize: Figure size for the plot
            
        Returns:
            Matplotlib figure
        """
        # Create DataFrame with all predictions
        all_preds = pd.DataFrame({model: preds for model, preds in self.predictions.items()})
        
        # Add chronological age
        common_ids = set(all_preds.index) & set(self.reference_data.index)
        all_preds = all_preds.loc[common_ids]
        all_preds['Chronological Age'] = self.reference_data.loc[common_ids, self.age_col].values
        
        # Compute correlation matrix
        corr_matrix = all_preds.corr()
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
        ax.set_title('Correlation Matrix of Age Predictions')
        
        return fig
    
    def plot_prediction_vs_real(self, figsize=(12, 10)):
        """Plot predicted vs chronological age for each model.
        
        Args:
            figsize: Figure size for the plot
            
        Returns:
            Matplotlib figure
        """
        n_models = len(self.predictions)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_models == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for i, (model, preds) in enumerate(self.predictions.items()):
            ax = axes[i]
            
            # Get common IDs
            common_ids = set(preds.index) & set(self.reference_data.index)
            pred_ages = preds.loc[common_ids]
            real_ages = self.reference_data.loc[common_ids, self.age_col]
            
            # Plot
            ax.scatter(real_ages, pred_ages, alpha=0.6)
            
            # Add diagonal line
            min_age = min(real_ages.min(), pred_ages.min())
            max_age = max(real_ages.max(), pred_ages.max())
            ax.plot([min_age, max_age], [min_age, max_age], 'k--')
            
            # Add regression line
            from sklearn.linear_model import LinearRegression
            reg = LinearRegression().fit(real_ages.values.reshape(-1, 1), pred_ages)
            ax.plot([min_age, max_age], 
                    [reg.predict([[min_age]])[0], reg.predict([[max_age]])[0]], 
                    'r-')
            
            # Add metrics
            r2 = self.reports[model]['r2']
            mae = self.reports[model]['mae']
            ax.text(0.05, 0.95, f"R² = {r2:.3f}\nMAE = {mae:.3f}", 
                    transform=ax.transAxes, va='top')
            
            ax.set_xlabel('Chronological Age')
            ax.set_ylabel('Predicted Age')
            ax.set_title(model)
        
        # Hide unused subplots
        for i in range(n_models, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        return fig
    
    def print_summary(self):
        """Print summary of model performance metrics."""
        print("Model Performance Summary:")
        print("-" * 60)
        print(f"{'Model':<20} {'R²':<10} {'MAE':<10} {'Pearson r':<10} {'N':<10}")
        print("-" * 60)
        
        for model, report in self.reports.items():
            print(f"{model:<20} {report['r2']:<10.3f} {report['mae']:<10.3f} "
                  f"{report['pearson_r']:<10.3f} {report['n_samples']:<10}")
        
        print("-" * 60)
