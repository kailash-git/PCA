# Chemical Feature Compression Engine

A production-ready Python tool that scales and compresses the 13 raw chemical features of the classic Wine dataset down to exactly 2 principal components using Principal Component Analysis (PCA).

## Features
- **Automatic Data Fetching**: Loads the classic Wine dataset directly from Scikit-learn.
- **Feature Scaling**: Standardizes the 13 chemical features using `StandardScaler` to ensure PCA is not biased by varying scales.
- **PCA Reduction**: Applies Principal Component Analysis (PCA) to compress the feature space into exactly 2 principal components.
- **Explained Variance Summary**: Programmatically calculates and displays the explained variance ratio and cumulative variance per component in a formatted Markdown table.
- **Visual Analytics**: Generates and saves a high-quality, publication-ready 2D scatter plot projection (`pca_projection.png`) with classes color-coded.

## Project Structure
```text
chemical_feature_compression/
├── README.md               # Documentation
├── requirements.txt         # Project dependencies
└── pca_compression.py      # Core python script
```

## Setup & Installation

### Prerequisites
- Python 3.9 or higher

### Installation
We recommend using a Python virtual environment to manage dependencies:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows (Command Prompt)
venv\Scripts\activate
# On Windows (PowerShell)
.\venv\Scripts\Activate.ps1
# On macOS/Linux
source venv/bin/activate

# Install the required packages
pip install -r requirements.txt
```

## Running the Engine

Execute the script with:
```bash
python pca_compression.py
```

### Options
You can customize the output image path using the `--output` command line argument:
```bash
python pca_compression.py --output custom_projection.png
```

## Mathematical Overview

Principal Component Analysis (PCA) performs a linear dimensionality reduction by projecting the features onto a lower-dimensional subspace where the variance of the data is maximized.

Given the standardized dataset $X$, PCA solves for the eigenvectors (principal components) $W$ of the covariance matrix $\Sigma = \frac{1}{n} X^T X$:

$$\Sigma W = \lambda W$$

Where $\lambda$ represents the eigenvalues (variance explained by each component). The original 13 features are projected onto the first 2 principal components $Z = X W_2$.
