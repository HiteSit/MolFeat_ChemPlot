"""
Test fixtures and configuration for pytest.
"""
import pytest
import pandas as pd
from pathlib import Path
from chemplot import Plotter

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "test_data"

# Sample SMILES and targets for testing
@pytest.fixture
def sample_smiles():
    return [
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",  # Testosterone
        "CC(C)(C)NCC(O)C1=CC(O)=CC(O)=C1",  # Salbutamol
        "CN1C=NC2=C1C(=O)NC(=O)N2C"  # Theophylline
    ]

@pytest.fixture
def sample_regression_targets():
    return [4.5, 2.3, 3.1, 1.8, 2.9]  # Example solubility values

@pytest.fixture
def sample_classification_targets():
    return [0, 1, 1, 0, 1]  # Binary classification example

@pytest.fixture
def plotter_regression(sample_smiles, sample_regression_targets):
    return Plotter.from_smiles(
        sample_smiles,
        target=sample_regression_targets,
        target_type="R",
        sim_type="tailored"
    )

@pytest.fixture
def plotter_classification(sample_smiles, sample_classification_targets):
    return Plotter.from_smiles(
        sample_smiles,
        target=sample_classification_targets,
        target_type="C",
        sim_type="tailored"
    )

@pytest.fixture
def plotter_structural(sample_smiles):
    return Plotter.from_smiles(
        sample_smiles,
        target=[],
        sim_type="structural"
    )

# Invalid SMILES for error testing
@pytest.fixture
def invalid_smiles():
    return [
        "invalid_smiles",
        "C1CC(C1)(C(=O)O)C(=O)O.N.N.[Pt]",  # Invalid metal complex
        "CC(Ccccccc1)NO",  # Invalid structure
        "non_smile",
        "[NH4][Pt]([NH4])(Cl)Cl"  # Invalid metal complex
    ]

@pytest.fixture
def error_plotter(invalid_smiles):
    with pytest.raises(Exception):
        Plotter.from_smiles(invalid_smiles)

# Fixture for testing dimensionality reduction
@pytest.fixture
def reduced_plotter(plotter_regression):
    plotter_regression.pca()
    return plotter_regression

# Fixture for testing clustering
@pytest.fixture
def clustered_plotter(reduced_plotter):
    reduced_plotter.cluster(n_clusters=2)
    return reduced_plotter
