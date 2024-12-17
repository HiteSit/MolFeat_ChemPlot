"""
Tests for core functionality of the ChemPlot package.
"""
import pytest
import pandas as pd
import numpy as np
from chemplot import Plotter

def test_plotter_initialization(plotter_regression, sample_smiles, sample_regression_targets):
    """Test basic plotter initialization"""
    assert isinstance(plotter_regression, Plotter)
    assert len(plotter_regression._Plotter__mols) == len(sample_smiles)
    assert len(plotter_regression._Plotter__target) == len(sample_regression_targets)

def test_plotter_initialization_no_target(plotter_structural, sample_smiles):
    """Test plotter initialization without targets"""
    assert isinstance(plotter_structural, Plotter)
    assert len(plotter_structural._Plotter__mols) == len(sample_smiles)
    assert len(plotter_structural._Plotter__target) == 0

def test_invalid_smiles_handling(invalid_smiles):
    """Test handling of invalid SMILES"""
    with pytest.raises(Exception):
        Plotter.from_smiles(invalid_smiles)

def test_pca_reduction(plotter_regression):
    """Test PCA dimensionality reduction"""
    result = plotter_regression.pca()
    
    # Check if result is a dataframe
    assert isinstance(result, pd.DataFrame)
    
    # Check if we have 2 components + target column
    assert result.shape[1] == 3
    
    # Check column names
    assert 'PC-1' in result.columns[0]
    assert 'PC-2' in result.columns[1]
    assert 'target' in result.columns[2]

def test_tsne_reduction(plotter_regression):
    """Test t-SNE dimensionality reduction"""
    result = plotter_regression.tsne(random_state=42)
    
    # Check if result is a dataframe
    assert isinstance(result, pd.DataFrame)
    
    # Check if we have 2 components + target column
    assert result.shape[1] == 3
    
    # Check column names
    assert 't-SNE-1' in result.columns[0]
    assert 't-SNE-2' in result.columns[1]
    assert 'target' in result.columns[2]

def test_umap_reduction(plotter_regression):
    """Test UMAP dimensionality reduction"""
    result = plotter_regression.umap(random_state=42)
    
    # Check if result is a dataframe
    assert isinstance(result, pd.DataFrame)
    
    # Check if we have 2 components + target column
    assert result.shape[1] == 3
    
    # Check column names
    assert 'UMAP-1' in result.columns[0]
    assert 'UMAP-2' in result.columns[1]
    assert 'target' in result.columns[2]

def test_clustering(reduced_plotter):
    """Test clustering functionality"""
    result = reduced_plotter.cluster(n_clusters=2)
    
    # Check if result is a dataframe
    assert isinstance(result, pd.DataFrame)
    
    # Check if clusters column was added
    assert 'clusters' in result.columns
    
    # Check if we have the right number of clusters
    assert len(np.unique(result['clusters'])) == 2

def test_visualization(reduced_plotter):
    """Test visualization functionality"""
    # Test static plot
    ax = reduced_plotter.visualize_plot(size=10)
    assert ax is not None
    
    # Test interactive plot
    fig = reduced_plotter.interactive_plot(size=500)
    assert fig is not None

def test_classification_plotter(plotter_classification):
    """Test plotter with classification data"""
    result = plotter_classification.pca()
    assert isinstance(result, pd.DataFrame)
    assert result.shape[1] == 3
    assert all(isinstance(x, (int, np.integer)) for x in result['target'])

def test_structural_similarity(plotter_structural):
    """Test structural similarity mode"""
    result = plotter_structural.pca()
    assert isinstance(result, pd.DataFrame)
    assert result.shape[1] == 2  # No target column
    assert 'target' not in result.columns 