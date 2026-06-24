"""
Chemical Feature Compression Engine

This script performs Principal Component Analysis (PCA) on the classic Wine dataset
from Scikit-learn. It scales the 13 raw chemical features using StandardScaler,
compresses the feature space down to 2 principal components, programmatically
calculates the explained variance, prints a Markdown summary table to the console,
and generates a high-quality 2D scatter plot of the projection.
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tabulate import tabulate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compress chemical features of the Wine dataset using PCA."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="pca_projection.png",
        help="Filename for the generated 2D scatter plot (default: pca_projection.png)"
    )
    return parser.parse_args()

def load_and_scale_data() -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """
    Load the Wine dataset and scale its 13 chemical features.
    
    Returns:
        tuple containing:
            - X_scaled: Standardized feature matrix
            - y: Target labels (wine classes)
            - feature_names: List of raw feature names
            - target_names: List of class names
    """
    logger.info("Loading Wine dataset from scikit-learn...")
    wine = load_wine()
    X = wine.data
    y = wine.target
    feature_names = wine.feature_names
    target_names = list(wine.target_names)
    
    logger.info(f"Loaded dataset with {X.shape[0]} samples and {X.shape[1]} features.")
    
    # Scale features
    logger.info("Standardizing chemical features using StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, feature_names, target_names

def perform_pca(X_scaled: np.ndarray, n_components: int = 2) -> tuple[PCA, np.ndarray]:
    """
    Perform PCA dimensionality reduction.
    
    Args:
        X_scaled: Standardized feature matrix
        n_components: Number of components to retain (default: 2)
        
    Returns:
        tuple containing:
            - pca: Fitted PCA object
            - X_pca: Reduced dimensional feature matrix
    """
    logger.info(f"Applying PCA (n_components={n_components})...")
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    return pca, X_pca

def print_variance_summary(pca: PCA) -> None:
    """
    Calculate and print a Markdown summary table of the explained variance.
    
    Args:
        pca: Fitted PCA object
    """
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    headers = ["Component", "Explained Variance Ratio", "Cumulative Explained Variance"]
    table_data = []
    
    for i in range(len(explained_variance)):
        table_data.append([
            f"PC{i+1}",
            f"{explained_variance[i]:.6f} ({explained_variance[i]*100:.2f}%)",
            f"{cumulative_variance[i]:.6f} ({cumulative_variance[i]*100:.2f}%)"
        ])
        
    markdown_table = tabulate(table_data, headers=headers, tablefmt="github")
    
    print("\n### PCA Explained Variance Summary\n")
    print(markdown_table)
    print("\n")

def plot_pca_projection(
    X_pca: np.ndarray,
    y: np.ndarray,
    target_names: list[str],
    pca: PCA,
    output_path: str
) -> None:
    """
    Generate and save a 2D scatter plot of the PCA projection.
    
    Args:
        X_pca: Reduced feature matrix (2 components)
        y: Target class labels
        target_names: Target class names
        pca: Fitted PCA object
        output_path: Path to save the generated image
    """
    logger.info("Generating 2D scatter plot...")
    
    # Set premium aesthetic styles
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    
    # Premium color palette (reds/browns/blues representing wine characters)
    colors = ["#8B0000", "#D2691E", "#4682B4"]  # Dark Red, Chocolate, Steel Blue
    markers = ["o", "^", "s"]
    
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FFFFFF")
    
    # Plot each class
    for class_idx, class_name in enumerate(target_names):
        mask = (y == class_idx)
        ax.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            c=colors[class_idx],
            label=f"Class {class_idx} ({class_name.capitalize()})",
            marker=markers[class_idx],
            s=70,
            alpha=0.85,
            edgecolors="none"
        )
        
    # Set labels with variance percentage
    ev_pc1 = pca.explained_variance_ratio_[0] * 100
    ev_pc2 = pca.explained_variance_ratio_[1] * 100
    ax.set_xlabel(f"Principal Component 1 (Explained Variance: {ev_pc1:.2f}%)", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_ylabel(f"Principal Component 2 (Explained Variance: {ev_pc2:.2f}%)", fontsize=11, fontweight="bold", labelpad=10)
    
    # Titles and labels
    ax.set_title("PCA Projection of Wine Dataset (13 Chemical Features → 2D)", fontsize=14, fontweight="bold", pad=20)
    
    # Grid and spines configuration
    ax.grid(True, linestyle="--", alpha=0.3, color="#CCCCCC")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")
    
    # Legend
    legend = ax.legend(
        loc="best",
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#E0E0E0",
        fontsize=10,
        title="Wine Classes",
        title_fontsize=11
    )
    legend.get_frame().set_boxstyle("round,pad=0.5")
    
    plt.tight_layout()
    
    # Save the figure
    try:
        plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
        logger.info(f"Scatter plot saved successfully to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save scatter plot: {e}")
        raise
    finally:
        plt.close()

def main() -> None:
    """Main execution function."""
    args = parse_arguments()
    try:
        X_scaled, y, _, target_names = load_and_scale_data()
        pca, X_pca = perform_pca(X_scaled, n_components=2)
        print_variance_summary(pca)
        plot_pca_projection(X_pca, y, target_names, pca, args.output)
        logger.info("PCA Compression executed successfully.")
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
