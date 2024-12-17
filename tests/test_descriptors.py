"""
Tests for molecular descriptor generation and feature selection functionality.
"""
import pytest
import pandas as pd
import numpy as np
from chemplot import descriptors
from rdkit import Chem

@pytest.fixture
def sample_mols(sample_smiles):
    """Convert sample SMILES to RDKit molecules"""
    return [Chem.MolFromSmiles(smi) for smi in sample_smiles]

def test_mordred_descriptors(sample_smiles, sample_regression_targets):
    """Test Mordred descriptor generation"""
    mols, df_desc, targets = descriptors.get_mordred_descriptors(
        sample_smiles, 
        sample_regression_targets
    )
    
    # Check outputs
    assert len(mols) > 0
    assert isinstance(df_desc, pd.DataFrame)
    assert len(targets) == len(sample_regression_targets)
    
    # Check descriptor properties
    assert df_desc.shape[0] == len(sample_smiles)
    assert df_desc.shape[1] > 0
    assert not df_desc.isnull().any().any()

def test_molfeat_descriptors(sample_smiles, sample_regression_targets):
    """Test MolFeat descriptor generation"""
    mols, df_desc, targets = descriptors.get_molfeat_descriptors(
        sample_smiles, 
        sample_regression_targets,
        fp_type="ecfp"
    )
    
    # Check outputs
    assert len(mols) > 0
    assert isinstance(df_desc, pd.DataFrame)
    assert len(targets) == len(sample_regression_targets)
    
    # Check descriptor properties
    assert df_desc.shape[0] == len(sample_smiles)
    assert df_desc.shape[1] > 0
    assert not df_desc.isnull().any().any()

def test_ecfp_generation(sample_smiles, sample_regression_targets):
    """Test ECFP fingerprint generation"""
    mols, df_fp, targets = descriptors.get_ecfp(
        sample_smiles,
        sample_regression_targets,
        radius=2,
        nBits=2048
    )
    
    # Check outputs
    assert len(mols) > 0
    assert isinstance(df_fp, pd.DataFrame)
    assert len(targets) == len(sample_regression_targets)
    
    # Check fingerprint properties
    assert df_fp.shape[0] == len(sample_smiles)
    assert df_fp.shape[1] <= 2048  # May be less due to bit pruning
    assert df_fp.isin([0, 1]).all().all()  # Only binary values

def test_feature_selection_regression(sample_smiles, sample_regression_targets):
    """Test feature selection for regression"""
    # Generate descriptors first
    _, df_desc, targets = descriptors.get_molfeat_descriptors(
        sample_smiles, 
        sample_regression_targets
    )
    
    # Test different feature selection methods
    for method in ['lasso', 'mutual_info', 'combined']:
        selected_features, selected_targets = descriptors.select_descriptors(
            df_desc,
            targets,
            method=method,
            target_type='R'
        )
        
        # Check outputs
        assert isinstance(selected_features, pd.DataFrame)
        assert len(selected_targets) == len(targets)
        assert selected_features.shape[0] == len(sample_smiles)
        assert selected_features.shape[1] <= df_desc.shape[1]
        assert not selected_features.isnull().any().any()

def test_feature_selection_classification(sample_smiles, sample_classification_targets):
    """Test feature selection for classification"""
    # Generate descriptors first
    _, df_desc, targets = descriptors.get_molfeat_descriptors(
        sample_smiles, 
        sample_classification_targets
    )
    
    # Test different feature selection methods
    for method in ['lasso', 'mutual_info', 'combined']:
        selected_features, selected_targets = descriptors.select_descriptors(
            df_desc,
            targets,
            method=method,
            target_type='C'
        )
        
        # Check outputs
        assert isinstance(selected_features, pd.DataFrame)
        assert len(selected_targets) == len(targets)
        assert selected_features.shape[0] == len(sample_smiles)
        assert selected_features.shape[1] <= df_desc.shape[1]
        assert not selected_features.isnull().any().any()

def test_invalid_feature_selection_method(sample_smiles, sample_regression_targets):
    """Test invalid feature selection method handling"""
    _, df_desc, targets = descriptors.get_molfeat_descriptors(
        sample_smiles, 
        sample_regression_targets
    )
    
    with pytest.raises(ValueError):
        descriptors.select_descriptors(
            df_desc,
            targets,
            method='invalid_method',
            target_type='R'
        )

def test_descriptor_error_handling():
    """Test error handling for invalid inputs"""
    invalid_smiles = ["invalid_smiles", "C1CC(invalid)CC1", "not_a_smiles"]
    empty_targets = []
    
    # Test Mordred descriptors
    mols, df_desc, targets = descriptors.get_mordred_descriptors(
        invalid_smiles,
        empty_targets
    )
    assert len(mols) == 0
    assert isinstance(df_desc, pd.DataFrame)
    assert df_desc.empty
    assert len(targets) == 0
    
    # Test MolFeat descriptors
    mols, df_desc, targets = descriptors.get_molfeat_descriptors(
        invalid_smiles,
        empty_targets
    )
    assert len(mols) == 0
    assert isinstance(df_desc, pd.DataFrame)
    assert df_desc.empty
    assert len(targets) == 0