import pytest
import pandas as pd
import numpy as np
from rdkit import Chem
import warnings

# Filter out deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

@pytest.fixture(scope="session")
def test_smiles():
    return [
        'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
        'CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C'  # Testosterone
    ]

@pytest.fixture(scope="session")
def test_mols(test_smiles):
    return [Chem.MolFromSmiles(smi) for smi in test_smiles]

@pytest.fixture(scope="session")
def test_descriptors():
    # Create dummy descriptors
    n_samples = 10
    n_features = 5
    return pd.DataFrame(
        np.random.rand(n_samples, n_features),
        columns=[f'feat_{i}' for i in range(n_features)]
    )

@pytest.fixture(scope="session")
def test_targets():
    # Create dummy targets
    return {
        'regression': np.array([1.5, 2.7, 3.2]),
        'classification': np.array([0, 1, 0])
    } 