"""
ProteoClock - A Python package for proteomic aging clock analysis

This package provides implementations of various proteomic aging clocks
from published research, allowing users to predict biological age
from protein expression data.
"""

from proteoclock.clocks.deep_clocks.ipf_clock import IPFClock
from proteoclock.clocks.deep_clocks.reporting import ModelEvaluator
from proteoclock.clocks.deep_clocks.nn_wrapper import AgingClockPredictor
from proteoclock.clocks.deep_clocks.losses import LossConfig, ClockLoss, TripleLoss

from proteoclock.other import IPFPathways, ScaleShift, UKBDataset
from proteoclock.other import load_test_age_data, load_test_protein_data, load_kuo_test_age_data, load_kuo_test_protein_data
from proteoclock.clocks.simple_clocks.clocks import generate_report, ModelPerformanceReporter
from proteoclock.data_preprocess.scalers import OLINKScaler

from proteoclock.factory import ClockFactory

__version__ = "1.0.0"
__author__ = "Fedor Galkin"



def create_clock_factory():
    """Create a new ClockFactory instance."""
    return ClockFactory()


# direct import
__all__ = [
    'ClockFactory',
    'create_clock_factory',
    # Data utility functions
    'load_test_age_data',
    'load_test_protein_data',
    'load_kuo_test_age_data',
    'load_kuo_test_protein_data'
]