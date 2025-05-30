# ProteoClock: Proteomic Aging Clock Analysis

![ProteoClock Banner](https://img.shields.io/badge/ProteoClock-Proteomic%20Aging%20Analysis-blue)
![Python](https://img.shields.io/badge/Python-3.11%2B-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

ProteoClock is a Python package for analyzing and predicting biological age using proteomic data. This package implements various published proteomic aging clocks, allowing researchers to predict biological age from protein expression measurements.

## 🧬 Features

- **Multiple Clock Implementations**: Linear, Gompertz, and Cox Proportional Hazards (CPH) based aging clocks
- **Data Preprocessing**: Tools for handling protein expression data in various formats
- **Performance Analysis**: Utilities for evaluating and comparing clock performance
- **Visualization**: Functions for creating informative plots and visualizations

## 📋 Implemented Clocks

The package currently implements the following published proteomic aging clocks:

1. Goeminne's 2025 **OrganAge**: ["Plasma protein-based organ-specific aging and mortality models unveil diseases as accelerated aging of organismal systems"](https://doi.org/10.1016/j.cmet.2024.10.005)
2. Kuo's 2024 PAC: ["Proteomic aging clock (PAC) predicts age-related outcomes in middle-aged and older adults"](https://doi.org/10.1101/2023.12.19.23300228)
3. Galkin's 2025 ipfP3GPT: ["AI-Driven Toolset for IPF and Aging Research Associates Lung Fibrosis with Accelerated Aging"](https://doi.org/10.1101/2025.01.09.632065 )
4. Argintieri's 2024 **ProtAge**: ["Proteomic aging clock predicts mortality and risk of common age-related diseases in diverse populations"](https://doi.org/10.1038/s41591-024-03164-7)
5. JDJ Han's 2025 proteomic clock: *paper in review*

## 🚀 Installation

### Using pip

```bash
# Clone the repository
git clone https://github.com/yourusername/proteoclock.git
cd proteoclock

# Install the package
pip install -e .
```

### Using conda

```bash
# Clone the repository
git clone https://github.com/yourusername/proteoclock.git
cd proteoclock

# Create and activate conda environment
conda env create -f environment.yml
conda activate proteoclock

# Run this if you want to add the new env as a Jupyter Notebook kernel
pip install ipykernel
python -m ipykernel install --user --name proteoclock --display-name "proteoclock"

# Install the package
pip install -e .
```


## 🔍 Quick Start

```python
# Import the necessary components
from proteoclock import ClockFactory, load_test_age_data, load_test_protein_data

# Load sample data
age_data = load_test_age_data()
protein_data = load_test_protein_data()

# Initialize clock factory
factory = ClockFactory()

# View available clocks
factory.view_clocks()

# Get a specific clock
kuo_clock = factory.get_clock("kuo_2024", scaler="ukb_scaler")
goem_liver = factory.get_clock("goeminne_2025_full_chrono", subtype="liver", 
                              scaler="goeminne_2025_full")

# Predict biological age
predictions = kuo_clock.predict_age(protein_data, scaling='standard')
```

## 📊 Data Format

ProteoClock expects protein data in long format with the following columns:
- `patient_id`: Identifier for each sample/patient
- `gene_symbol`: Protein/gene name
- `NPX`: Normalized Protein eXpression value

Example:

| patient_id | gene_symbol | NPX   |
|------------|-------------|-------|
| SAMPLE_001 | IL6         | 5.23  |
| SAMPLE_001 | TNF         | 4.87  |
| SAMPLE_002 | IL6         | 6.12  |
| SAMPLE_002 | TNF         | 5.34  |

## 📚 Core Components

### Clock Implementations

All aging clock implementations inherit from the `BaseAgingClock` abstract base class.
All aging clock implementations inherit from the `BaseAgingClock` abstract base class, which provides common functionality for loading coefficients, handling scalers, and preprocessing data.

#### Clock Types

1. **LinearClock**: Simple regression-based clocks that directly predict age from protein measurements
2. **GompertzClock**: Based on mortality risk models like Kuo's 2024 PAC, using Gompertz distribution
3. **CPHClock**: Cox Proportional Hazards-based clocks that convert mortality risk to biological age
4. **DeepClock**: Neural network-based clocks for complex non-linear aging patterns

#### Usage Example

```python
# Get a clock through the factory (recommended approach)
kuo_clock = factory.get_clock("kuo_2024", scaler="ukb_scaler")
predicted_ages = kuo_clock.predict_age(protein_data, scaling='standard')

# Or initialize directly
from proteoclock.clocks.simple_clocks import GompertzClock
gompertz_clock = GompertzClock(
    coef_file="path/to/coefficients.txt",
    scaler_file="path/to/scaler.pckl"
)
predicted_ages = gompertz_clock.predict_age(protein_data, age_data, scaling='standard')
```

### Utility Components

#### Data Preprocessing

The `OLINKScaler` class handles protein data preprocessing, including:

- Conversion between long and wide formats
- Imputation of missing values
- Data scaling

## 📂 Project Structure

The package is organized into the following key directories:

- `proteoclock/clocks/`: Implementations of different clock types
- `proteoclock/data_preprocess/`: Data preprocessing utilities
- `proteoclock/materials/`: Model coefficients, scalers, and test data
- `notebooks/`: Example notebooks demonstrating package usage

For more details, see the README files in each directory.

## 📖 Documentation

For detailed usage examples, see the notebooks in the `notebooks/` directory, especially `Demo Proteoclock.ipynb`.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 References

- Kuo's 2024 PAC: ["Proteomic aging clock (PAC) predicts age-related outcomes in middle-aged and older adults"](https://onlinelibrary.wiley.com/doi/10.1111/acel.14195)
- Gladyshev's 2025 OrganAge: ["Plasma protein-based organ-specific aging and mortality models unveil diseases as accelerated aging of organismal systems"](https://doi.org/10.1016/j.cmet.2024.10.005)
- Argentieri's 2024 clock: ["Proteomic aging clock predicts mortality and risk of common age-related diseases in diverse populations"](https://www.nature.com/articles/s41591-024-03164-7)
- Review of older clocks: ["Systematic review and analysis of human proteomics aging studies unveils a novel proteomic aging clock and identifies key processes that change with age"](https://doi.org/10.1016/j.arr.2020.101070)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
