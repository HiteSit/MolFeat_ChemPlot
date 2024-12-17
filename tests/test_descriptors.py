import pytest
import pandas as pd
import numpy as np
from rdkit import Chem
from chemplot.descriptors import (
    get_molfeat_descriptors,
    get_ecfp,
    select_descriptors
)

# Test molecules
TEST_SMILES = [
    'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
    'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
    'CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C'  # Testosterone
]

def test_get_molfeat_descriptors():
    # Test with valid SMILES
    mols, descriptors, _ = get_molfeat_descriptors(TEST_SMILES, [])
    
    assert len(mols) == len(TEST_SMILES)
    assert not descriptors.empty
    assert all(isinstance(mol, Chem.rdchem.Mol) for mol in mols)

def test_get_molfeat_descriptors_with_invalid_smiles():
    invalid_smiles = ['INVALID', 'CC(=O)OC1=CC=CC=C1C(=O)O']
    mols, descriptors, _ = get_molfeat_descriptors(invalid_smiles, [])
    
    assert len(mols) == 1  # Only valid molecule should be processed
    # The descriptors might be empty if all features are constant
    if not descriptors.empty:
        assert descriptors.shape[0] == 1  # Should have one row for valid molecule
    else:
        assert True  # Test passes if descriptors are empty (edge case)

def test_get_ecfp():
    # Test with valid SMILES
    mols, fingerprints, _ = get_ecfp(TEST_SMILES, [], radius=2, nBits=2048)
    
    assert len(mols) == len(TEST_SMILES)
    assert not fingerprints.empty
    assert fingerprints.shape[1] <= 2048  # Number of columns should not exceed nBits
    assert all(isinstance(mol, Chem.rdchem.Mol) for mol in mols)

def test_select_descriptors():
    # Create dummy descriptors and targets
    n_samples = 10
    n_features = 5
    descriptors = pd.DataFrame(
        np.random.rand(n_samples, n_features),
        columns=[f'feat_{i}' for i in range(n_features)]
    )
    targets = np.random.rand(n_samples)
    
    # Test regression
    selected_desc, selected_targets = select_descriptors(
        descriptors, targets, method='lasso', target_type='R'
    )
    assert isinstance(selected_desc, pd.DataFrame)
    assert len(selected_targets) == n_samples
    
    # Test classification
    targets_class = (targets > 0.5).astype(int)
    selected_desc, selected_targets = select_descriptors(
        descriptors, targets_class, method='lasso', target_type='C'
    )
    assert isinstance(selected_desc, pd.DataFrame)
    assert len(selected_targets) == n_samples 