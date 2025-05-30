"""
Scalers module for protein data preprocessing.

This module provides classes for preprocessing protein data,
including converting between long and wide formats and scaling.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Literal, Iterable
import pickle


class OLINKScaler:
    """Combined scaler and formatter for OLINK protein data"""

    def __init__(self, required_features: List[str]):
        """Initialize scaler with optional required features

        Args:
            required_features: List of protein features required by the model.
                             If None, will accept all features found in data.
        """
        self.data_min = None
        self.data_max = None
        self.data_median = None
        self.feature_means = None
        self.feature_stds = None
        self.required_features = required_features

    def long_to_wide(self,
                     data: pd.DataFrame,
                     id_col: str = 'patient_id',
                     gene_col: str = 'gene_symbol',
                     value_col: str = 'NPX') -> pd.DataFrame:
        """Format long-format protein data into wide format

        Args:
            data: DataFrame in long format (one measurement per row)
            id_col: Name of column containing sample/patient IDs
            gene_col: Name of column containing gene/protein names
            value_col: Name of column containing measurement values

        Returns:
            DataFrame in wide format with samples as rows and proteins as columns
        """
        # Validate input columns
        required_cols = [id_col, gene_col, value_col]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Pivot data to wide format
        wide_data = data.pivot(
            index=id_col,
            columns=gene_col,
            values=value_col
        )

        # Check for required features if specified
        missing_features = set(self.required_features) - set(wide_data.columns)
        if missing_features:
            print(f"Warning: Missing required features: {missing_features}")
            # Add missing columns
            for feature in missing_features:
                wide_data[feature] = np.nan

        # Ensure columns are in correct order
        wide_data = wide_data[list(self.required_features)]

        return wide_data
    
    def fit(self,
            data: Union[pd.DataFrame, np.ndarray]) -> 'OLINKScaler':
        """Fit scaler to training data and compute feature means

        Args:
            data: Training data to fit scaler on

        Returns:
            Self for chaining
        """

        if isinstance(data, pd.DataFrame):
            data = data.loc[:, self.required_features]
            data = data.values

        # Store raw means for imputation
        self.feature_means = np.nanmean(data, axis=0)
        self.feature_stds = np.nanstd(data, axis=0)

        # Compute scaling parameters
        self.data_min = np.nanmin(data, axis=0)
        self.data_max = np.nanmax(data, axis=0)
        self.data_range = (self.data_max - self.data_min)
        # Avoid division by zero
        nonzero_ranges = self.data_range[self.data_range != 0]
        if len(nonzero_ranges):
            pseudocount = min(nonzero_ranges)/100.
        else:
            pseudocount = min(self.data_min)/100.
        self.data_range[self.data_range == 0] = pseudocount

        # Compute median after min-max scaling
        scaled = (data - self.data_min) / self.data_range
        self.data_median = np.nanmedian(scaled, axis=0)
        return self

    def prepare_wide(self,
                  data: Union[pd.DataFrame, np.ndarray],
                  impute_missing: bool = True,
                  transform: Optional[Literal["standard","minmax"]] = None) -> np.ndarray:
        """Transform data using fitted parameters with optional mean imputation

        Args:
            data: Data to transform
            impute_missing: Whether to impute missing values using training means
            transform: Type of scaling to apply

        Returns:
            Transformed data as numpy array
        """
        if self.data_min is None:
            raise ValueError("Scaler must be fitted before transforming data")

        # Convert DataFrame to array if needed
        if isinstance(data, pd.DataFrame):
            data = data.loc[:, self.required_features]
            data = data.values

        # Make a copy to avoid modifying input
        data = data.copy()

        # Impute missing values if requested
        if impute_missing and self.feature_means is not None:
            data = self.impute_means(data)

        if not transform is None:
            match transform:
                case 'minmax':
                    data = self.transform_minmax(data)
                case 'standard':
                    data = self.transform_standard(data)
                case _:
                    pass
        return data.astype(np.float32)

    def transform_minmax(self, data: np.ndarray) -> np.ndarray:
        """Transform data using min-max scaling

        Args:
            data: Data to transform

        Returns:
            Transformed data
        """
        data = data.copy()

        sample_min = np.nanmin(data, axis=0)
        sample_max = np.nanmax(data, axis=0)

        var_mask = sample_max != sample_min
        data[:, var_mask] = data[:, var_mask] - sample_min[var_mask] / (sample_max[var_mask] - sample_min[var_mask])
        data[:, var_mask] = data[:, var_mask] * (self.data_max[var_mask] - self.data_min[var_mask]) + self.data_min[var_mask]

        return(data)

    def transform_standard(self, data: np.ndarray) -> np.ndarray:
        """Transform data using standardization (z-score)

        Args:
            data: Data to transform

        Returns:
            Transformed data
        """
        data = data.copy()

        sample_mean = np.nanmean(data, axis=0)
        sample_std = np.nanstd(data, axis=0)

        var_mask = sample_std != 0
        data[:, var_mask] = (data[:, var_mask] - sample_mean[var_mask])/sample_std[var_mask]
        data[:, var_mask] = (data[:, var_mask] * self.feature_stds[var_mask]) + self.feature_means[var_mask]

        return(data)
    
    def impute_means(self, data: np.ndarray) -> np.ndarray:
        """Impute missing values using feature means

        Args:
            data: Data with missing values

        Returns:
            Data with imputed values
        """
        data = data.copy()

        missing_mask = np.isnan(data)
        if missing_mask.any():
            n_missing = missing_mask.sum()
            print(f"Imputing {n_missing} missing values using training means")
            data[missing_mask] = np.take(self.feature_means,
                                         np.where(missing_mask)[1])
        return(data)

    def fit_transform(self,
                      data: Union[pd.DataFrame, np.ndarray],
                      impute_missing: bool = True) -> np.ndarray:
        """Convenience method to fit and transform in one step

        Args:
            data: Data to fit and transform
            impute_missing: Whether to impute missing values

        Returns:
            Transformed data as numpy array
        """
        return self.fit(data).prepare_wide(data, impute_missing)
    
    def filter_scaler(self, feats: Iterable[str]) -> 'OLINKScaler':
        """Create a new scaler with a subset of features

        Args:
            feats: Features to include in the new scaler

        Returns:
            New scaler with filtered features
        """
        miss_feats = [x for x in feats if not x in self.required_features]
        assert not miss_feats, f"Features missing in the original scaler: {miss_feats}"
        
        new_scaler = OLINKScaler(feats)
        new_scaler.required_features = [x for x in feats]
        
        # Copy fitted parameters if available
        if self.data_min is not None:
            feat_indices = [self.required_features.index(f) for f in feats]
            new_scaler.data_min = self.data_min[feat_indices]
            new_scaler.data_max = self.data_max[feat_indices]
            new_scaler.data_range = self.data_range[feat_indices]
            new_scaler.data_median = self.data_median[feat_indices]
            new_scaler.feature_means = self.feature_means[feat_indices]
            new_scaler.feature_stds = self.feature_stds[feat_indices]
        
        return new_scaler
    
    def save(self, filepath: str):
        """Save scaler parameters to file

        Args:
            filepath: Path to save scaler parameters
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'OLINKScaler':
        """Load saved scaler parameters

        Args:
            filepath: Path to saved scaler parameters

        Returns:
            Loaded scaler instance
        """
        with open(filepath, 'rb') as f:
            loaded_data = pickle.load(f)
            
            # If it's already an OLINKScaler instance, return it directly
            if isinstance(loaded_data, cls):
                return loaded_data
            
            # Otherwise, create a new instance and copy the attributes
            if isinstance(loaded_data, dict):
                # Create a new instance with required features
                scaler = cls(loaded_data.get('required_features', []))
                
                # Copy attributes from the loaded dictionary
                for key, value in loaded_data.items():
                    setattr(scaler, key, value)
                    
                return scaler
            
            # If it's neither an instance nor a dict, raise an error
            raise ValueError(f"Invalid loaded data type: {type(loaded_data)}")


class StdScaler(OLINKScaler):
    """Used with scalers that only scale the data based on standard deviation"""
    
    def transform_minmax(self, *args, **kwargs):
        """Not implemented for this scaler type"""
        raise NotImplementedError("MinMax scaling not supported for StdScaler")
