# Authors: Murat Cihan Sorkun <mcsorkun@gmail.com>, Dajt Mullaj <dajt.mullai@gmail.com>, Jackson Warner Burns <jwburns@mit.edu>
# Descriptor operation methods
#
# License: BSD 3 clause
from __future__ import print_function

import math

import mordred
import numpy as np
import pandas as pd
from mordred import Calculator, descriptors
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.feature_selection import SelectFromModel, mutual_info_regression, mutual_info_classif
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.preprocessing import StandardScaler

import datamol as dm
from molfeat.calc import FPCalculator, FP_FUNCS
from molfeat.trans import MoleculeTransformer

def get_mordred_descriptors(smiles_list, target_list, **kwargs):
    """
    Calculates the Mordred descriptors for given smiles list

    :param smiles_list: List of smiles
    :type smiles_list: list
    :returns: The calculated descriptors list for the given smiles
    :rtype: Dataframe
    """
    return generate_mordred_descriptors(smiles_list, target_list, Chem.MolFromSmiles, "SMILES")

def get_mordred_descriptors_from_inchi(inchi_list, target_list, **kwargs):
    """
    Calculates the Mordred descriptors for given InChi list

    :param inchi_list: List of InChi
    :type inchi_list: list
    :returns: The calculated descriptors list for the given smiles
    :rtype: Dataframe
    """
    return generate_mordred_descriptors(inchi_list, target_list, Chem.MolFromInchi, "InChi")

def generate_mordred_descriptors(encoding_list, target_list, encoding_function, encoding_name):
    """
    Calculates the Mordred descriptors for list of molecules encodings

    :param smiles_list: List of molecules encodings
    :type smiles_list: list
    :returns: The calculated descriptors list for the given molecules encodings
    :rtype: Dataframe
    """
    calc = mordred.Calculator()

    calc.register(mordred.AtomCount)  # 16
    calc.register(mordred.RingCount)  # 139
    calc.register(mordred.BondCount)  # 9
    calc.register(mordred.HydrogenBond)  # 2
    calc.register(mordred.CarbonTypes)  # 10
    calc.register(mordred.SLogP)  # 2
    calc.register(mordred.Constitutional)  # 16
    calc.register(mordred.TopoPSA)  # 2
    calc.register(mordred.Weight)  # 2
    calc.register(mordred.Polarizability)  # 2
    calc.register(mordred.McGowanVolume)  # 1

    name_list = []
    for desc_name in calc.descriptors:
        name_list.append(str(desc_name))

    mols = []
    descriptors_list = []
    erroneous_encodings = []
    encodings_none_descriptors = []
    for encoding in encoding_list:
        mol = encoding_function(encoding)
        if mol is None:
            descriptors_list.append([None] * len(name_list))
            erroneous_encodings.append(encoding)
        else:
            mol = Chem.AddHs(mol)
            calculated_descriptors = calc(mol)
            for i in range(len(calculated_descriptors._values)):
                if math.isnan(calculated_descriptors._values[i]):
                    calculated_descriptors._values = [None] * len(name_list)
                    encodings_none_descriptors.append(encoding)
                    break
                if i == len(calculated_descriptors._values) - 1:
                    mols.append(mol)
            descriptors_list.append(calculated_descriptors._values)

    if len(erroneous_encodings) > 0:
        print(
            "The following erroneous {} have been found in the data:\n{}.\nThe erroneous {} will be removed from the data.".format(
                encoding_name, "\n".join(map(str, erroneous_encodings)), encoding_name
            )
        )

    if len(encodings_none_descriptors) > 0:
        print(
            "For the following {} not all descriptors can be computed:\n{}.\nThese {} will be removed from the data.".format(
                encoding_name, "\n".join(map(str, encodings_none_descriptors)), encoding_name
            )
        )

    df_descriptors = pd.DataFrame(descriptors_list, columns=name_list)
    df_descriptors = df_descriptors.select_dtypes(exclude=["object"])

    # Remove erroneous data
    if not isinstance(target_list, list):
        target_list = target_list.values
    df_descriptors = df_descriptors.assign(target=target_list)
    df_descriptors = df_descriptors.dropna(how="any")
    target_list = df_descriptors["target"].to_list()
    df_descriptors = df_descriptors.drop(columns=["target"])

    return mols, df_descriptors, target_list

def get_molfeat_descriptors(smiles_list, target_list, **kwargs):
    """
    Calculates molecular descriptors using molfeat fingerprints

    :param smiles_list: List of smiles
    :param target_list: List of target values
    :param kwargs: Additional arguments including:
                  - fp_type: Type of fingerprint to use (default: "ecfp")
                  - n_jobs: Number of parallel jobs (default: -1)
    :type smiles_list: list
    :type target_list: list
    :returns: The calculated descriptors list for the given smiles
    :rtype: tuple(list, DataFrame, list)
    """
    fp_type = kwargs.get("fp_type", "ecfp")
    n_jobs = kwargs.get("n_jobs", 1)
    
    # Convert SMILES to mols
    mols = []
    valid_smiles = []
    valid_targets = []
    
    # Handle empty or None target_list
    has_targets = target_list is not None and len(target_list) > 0
    
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mols.append(mol)
            valid_smiles.append(smi)
            if has_targets:
                valid_targets.append(target_list[i])
    
    if len(mols) == 0:
        return [], pd.DataFrame(), []
    
    try:
        # Calculate multiple types of descriptors to ensure we have enough features
        descriptors_list = []
        
        # 1. ECFP fingerprints
        calc1 = FPCalculator("ecfp", length=2048)
        mol_transf1 = MoleculeTransformer(calc1, n_jobs=n_jobs)
        features1 = mol_transf1(mols)
        df1 = pd.DataFrame(features1)
        descriptors_list.append(df1)
        
        # 2. MACCS keys
        calc2 = FPCalculator("maccs")
        mol_transf2 = MoleculeTransformer(calc2, n_jobs=n_jobs)
        features2 = mol_transf2(mols)
        df2 = pd.DataFrame(features2)
        descriptors_list.append(df2)
        
        # 3. Topological fingerprints
        calc3 = FPCalculator("topological", length=2048)
        mol_transf3 = MoleculeTransformer(calc3, n_jobs=n_jobs)
        features3 = mol_transf3(mols)
        df3 = pd.DataFrame(features3)
        descriptors_list.append(df3)
        
        # Combine all descriptors
        df_descriptors = pd.concat(descriptors_list, axis=1)
        
        # Remove constant features and ensure column names are unique
        df_descriptors = df_descriptors.loc[:, df_descriptors.nunique() > 1]
        df_descriptors.columns = [f'desc_{i}' for i in range(df_descriptors.shape[1])]
        
        return mols, df_descriptors, valid_targets if has_targets else []
    except Exception as e:
        print(f"Error calculating descriptors: {str(e)}")
        return [], pd.DataFrame(), []

def select_descriptors(df_descriptors, target_list, method="lasso", target_type="R", **kwargs):
    """
    Selects descriptors using various feature selection methods

    :param df_descriptors: descriptors of molecules
    :param target_list: list of target values
    :param method: feature selection method ('lasso', 'mutual_info', or 'combined')
    :param target_type: type of target ('R' for regression, 'C' for classification)
    :param kwargs: additional parameters for feature selection methods
    :type df_descriptors: DataFrame
    :type target_list: list
    :type method: str
    :type target_type: str
    :returns: The selected descriptors
    :rtype: tuple(DataFrame, list)
    """
    # If no target list is provided, return all descriptors
    if target_list is None or (isinstance(target_list, (list, np.ndarray)) and len(target_list) == 0):
        return df_descriptors, []
    
    # Handle empty dataframe
    if df_descriptors.empty:
        return df_descriptors, target_list
    
    try:
        df_descriptors_scaled = StandardScaler().fit_transform(df_descriptors)
        
        if method == "lasso":
            if target_type == "C":
                model = LogisticRegression(penalty="l1", solver="liblinear", random_state=1, **kwargs)
            else:
                model = Lasso(random_state=1, **kwargs)
            selector = SelectFromModel(model)
            
        elif method == "mutual_info":
            # Create a dummy estimator that will store MI scores
            class MIEstimator:
                def __init__(self):
                    self.coef_ = None
                
                def fit(self, X, y):
                    if target_type == "C":
                        self.coef_ = mutual_info_classif(X, y)
                    else:
                        self.coef_ = mutual_info_regression(X, y)
                    return self
                
                def get_params(self, deep=True):
                    return {}
                
                def set_params(self, **params):
                    return self
            
            # Use the dummy estimator with SelectFromModel
            selector = SelectFromModel(MIEstimator(), threshold="mean")
            
        elif method == "combined":
            # Calculate Lasso scores
            if target_type == "C":
                lasso_model = LogisticRegression(penalty="l1", solver="liblinear", random_state=1, **kwargs)
            else:
                lasso_model = Lasso(random_state=1, **kwargs)
            
            # Calculate MI scores
            if target_type == "C":
                mi_scores = mutual_info_classif(df_descriptors_scaled, target_list)
            else:
                mi_scores = mutual_info_regression(df_descriptors_scaled, target_list)
                
            # Fit Lasso
            selector_lasso = SelectFromModel(lasso_model)
            selector_lasso.fit(df_descriptors_scaled, target_list)
            mask_lasso = selector_lasso.get_support()
            
            # Use MI scores directly
            mi_threshold = np.mean(mi_scores)
            mask_mi = mi_scores > mi_threshold
            
            # Combine masks
            final_mask = mask_lasso | mask_mi
            
            # Return selected features
            return df_descriptors.iloc[:, final_mask], target_list
        else:
            print(f"Unknown method {method}, using all features")
            return df_descriptors, target_list
        
        # Fit and transform for lasso and mutual_info methods
        try:
            selector.fit(df_descriptors_scaled, target_list)
            selected_features_mask = selector.get_support()
            return df_descriptors.iloc[:, selected_features_mask], target_list
        except Exception as e:
            print(f"Feature selection failed: {str(e)}, using all features")
            return df_descriptors, target_list
            
    except Exception as e:
        print(f"Error in descriptor selection: {str(e)}, using all features")
        return df_descriptors, target_list

def get_ecfp(smiles_list, target_list, radius=2, nBits=2048):
    """
    Calculates the ECFP fingerprint for given SMILES list

    :param smiles_list: List of SMILES
    :param radius: The ECPF fingerprints radius.
    :param nBits: The number of bits of the fingerprint vector.
    :type radius: int
    :type smiles_list: list
    :type nBits: int
    :returns: The calculated ECPF fingerprints for the given SMILES
    :rtype: Dataframe
    """

    return generate_ecfp(smiles_list, Chem.MolFromSmiles, "SMILES", target_list, radius, nBits)


def get_ecfp_from_inchi(inchi_list, target_list, radius=2, nBits=2048):
    """
    Calculates the ECFP fingerprint for given InChi list

    :param inchi_list: List of InChi
    :param radius: The ECPF fingerprints radius.
    :param nBits: The number of bits of the fingerprint vector.
    :type inchi_list: list
    :type radius: int
    :type nBits: int
    :returns: The calculated ECPF fingerprints for the given InChi
    :rtype: Dataframe
    """

    return generate_ecfp(inchi_list, Chem.MolFromInchi, "InChi", target_list, radius, nBits)


def generate_ecfp(encoding_list, encoding_function, encoding_name, target_list, radius=2, nBits=2048):
    """
    Calculates the ECFP fingerprint for given list of molecules encodings

    :param encoding_list: List of molecules encodings
    :param encoding_function: Function used to extract the molecules from the encodings
    :param radius: The ECPF fingerprints radius.
    :param nBits: The number of bits of the fingerprint vector.
    :type encoding_list: list
    :type encoding_function: fun
    :type radius: int
    :type nBits: int
    :returns: The calculated ECPF fingerprints for the given molecules encodings
    :rtype: Dataframe
    """

    # Generate ECFP fingerprints
    mols = []
    ecfp_fingerprints = []
    erroneous_encodings = []
    for encoding in encoding_list:
        mol = encoding_function(encoding)
        if mol is None:
            ecfp_fingerprints.append([None] * nBits)
            erroneous_encodings.append(encoding)
        else:
            mol = Chem.AddHs(mol)
            mols.append(mol)
            list_bits_fingerprint = []
            fpgen = AllChem.GetRDKitFPGenerator(maxPath=radius, fpSize=nBits)
            list_bits_fingerprint[:0] = fpgen.GetFingerprint(mol)
            ecfp_fingerprints.append(list_bits_fingerprint)

    # Create dataframe of fingerprints
    df_ecfp_fingerprints = pd.DataFrame(data=ecfp_fingerprints, index=encoding_list)

    # Remove erroneous data
    if len(erroneous_encodings) > 0:
        print(
            "The following erroneous {} have been found in the data:\n{}.\nThe erroneous {} will be removed from the data.".format(
                encoding_name, "\n".join(map(str, erroneous_encodings)), encoding_name
            )
        )

    if len(target_list) > 0:
        if not isinstance(target_list, list):
            target_list = target_list.values
        df_ecfp_fingerprints = df_ecfp_fingerprints.assign(target=target_list)

    df_ecfp_fingerprints = df_ecfp_fingerprints.dropna(how="any")

    if len(target_list) > 0:
        target_list = df_ecfp_fingerprints["target"].to_list()
        df_ecfp_fingerprints = df_ecfp_fingerprints.drop(columns=["target"])

    # Remove bit columns with no variablity (all "0" or all "1")
    df_ecfp_fingerprints = df_ecfp_fingerprints.loc[:, (df_ecfp_fingerprints != 0).any(axis=0)]
    df_ecfp_fingerprints = df_ecfp_fingerprints.loc[:, (df_ecfp_fingerprints != 1).any(axis=0)]

    return mols, df_ecfp_fingerprints, target_list
