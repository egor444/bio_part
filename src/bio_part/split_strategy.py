import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from collections import Counter
from sklearn.decomposition import PCA
import math

class SplitStrategy:
    def split(self, df: pd.DataFrame, val_size: float = 0.15, test_size: float = 0.15, keep_old_test: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        raise NotImplementedError("Subclasses must implement this method.")
    

class RandomSplitStrategy(SplitStrategy):
    """
    A split strategy that randomly splits the data into train, validation, and test sets
    while keeping the class distribution (stratification) intact. The stratification group
    is derived from the target according to the task type, mirroring MaxDistanceSplitStrategy.
    """
    def __init__(self, data_type: str = "sequence_to_class", num_bins: int = 10):
        self.task_type = data_type
        self.num_bins = num_bins

    def split(self, df: pd.DataFrame, val_size: float = 0.15, test_size: float = 0.15, keep_old_test: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        from sklearn.model_selection import train_test_split
        

        self.task_type = self.task_type

        df = df.copy()

        build_group = getattr(self, f"_stratify_group_{self.task_type}")
        df = build_group(df)

        df['stratify_group'] = self._collapse_rare_groups(df['stratify_group'], val_size, test_size)

        if keep_old_test:
            test = df[df['set'] == 'test'].drop(columns=['stratify_group'], errors='ignore')
            df = df[df['set'] != 'test']
            train, val = train_test_split(
                df, test_size=val_size / (1 - test_size),
                stratify=df['stratify_group'], random_state=42
            )
        else:
            train, temp = train_test_split(
                df, test_size=(val_size + test_size),
                stratify=df['stratify_group'], random_state=42
            )
            val, test = train_test_split(
                temp, test_size=(test_size / (val_size + test_size)),
                stratify=temp['stratify_group'], random_state=42
            )
            

        drop_cols = ['stratify_group']
        train = train.drop(columns=drop_cols, errors='ignore')
        val = val.drop(columns=drop_cols, errors='ignore')
        if not keep_old_test:
            test = test.drop(columns=drop_cols, errors='ignore')

        return train, val, test

    def _collapse_rare_groups(self, series, val_size, test_size):
        smallest_share = min(val_size, test_size, 1 - val_size - test_size)
        min_count = max(2, math.ceil(1 / smallest_share))

        counts = series.value_counts()
        rare = counts[counts < min_count].index

        if len(rare) == 0:
            return series

        safe_counts = counts.drop(index=rare)
        if len(safe_counts) == 0:
            raise ValueError(
                "No stratify group meets the minimum count; cannot safely stratify this split."
            )

        target_group = safe_counts.idxmin() 
        return series.where(~series.isin(rare), other=target_group)

    def _stratify_group_sequence_to_class(self, df):
        df['stratify_group'] = df['target']
        return df

    def _stratify_group_sequence_to_value(self, df):
        df['stratify_group'] = pd.qcut(df['target'].astype(float), q=self.num_bins, labels=False, duplicates='drop')
        return df

    def _stratify_group_residue_to_class(self, df):
        df = df[df['sequence'].str.len() == df['target'].astype(str).str.len()]
        df['stratify_group'] = df['target'].apply(
            lambda x: Counter(str(x)).most_common(1)[0][0] if pd.notna(x) else 'X'
        )
        return df

    def _stratify_group_residue_to_value(self, df):
        def calculate_valid_mean(val_str):
            vals = [float(v) for v in str(val_str).split(';') if v.strip()]
            valid_vals = [v for v in vals if v != 999.0]
            return np.mean(valid_vals) if valid_vals else 0.0

        df['stratify_group'] = pd.qcut(
            df['target'].apply(calculate_valid_mean), q=self.num_bins, labels=False, duplicates='drop'
        )
        return df
    
class OriginalSplitStrategy(SplitStrategy):

    def __init__(self, data_type: str = "sequence_to_class"):
        self.data_type = data_type

    def split(self, df: pd.DataFrame, val_size: float = 0.15, test_size: float = 0.15, keep_old_test: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train = df[df['set'] == 'train']
        val = df[df['set'] == 'val']
        test = df[df['set'] == 'test']
        return train, val, test

class MaxDistanceSplitStrategy:
    def __init__(self, data_type: str = "sequence_to_class", num_bins: int = 10, exclude_features=None):
        self.data_type = data_type
        self.num_bins = num_bins
        self.exclude_features = exclude_features if exclude_features is not None else []

    def split(self, df: pd.DataFrame, val_size: float = 0.15, test_size: float = 0.15, keep_old_test: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        features = ["molecular_weight", "isoelectric_point", "hydrophobicity",
                    "aromaticity", "instability_index", "entropy", "charge", "length"]
        features = [f for f in features if f not in self.exclude_features]

        if not set(features).issubset(df.columns):
            from .physchem import get_physchem_df
            df = get_physchem_df(df, sequence_col="sequence")[0]

        if keep_old_test:
            test = df[df['set'] == 'test']
            df_to_split = df[df['set'] != 'test'].copy()
        else:
            df_to_split = df.copy()

        # Route to the appropriate private method based on prediction type
        split_method = getattr(self, f"_split_{self.data_type}")
        train, val, test_new = split_method(df_to_split, features, val_size, test_size, keep_old_test)

        if keep_old_test:
            return train, val, test
        return train, val, test_new

    def _split_sequence_to_class(self, df, features, val_size, test_size, keep_old_test):
        df['stratify_group'] = df['target']
        return self._apply_max_distance_split(df, features, val_size, test_size, keep_old_test)

    def _split_sequence_to_value(self, df, features, val_size, test_size, keep_old_test):
        # Bin the continuous target values
        df['stratify_group'] = pd.qcut(df['target'].astype(float), q=self.num_bins, labels=False, duplicates='drop')
        return self._apply_max_distance_split(df, features, val_size, test_size, keep_old_test)

    def _split_residue_to_class(self, df, features, val_size, test_size, keep_old_test):
        # drop all rows with sequences and targets of unequal lengths
        df = df[df['sequence'].str.len() == df['target'].astype(str).str.len()]
        # Stratify by the most common class in the residue string
        df['stratify_group'] = df['target'].apply(lambda x: Counter(str(x)).most_common(1)[0][0] if pd.notna(x) else 'X')
        return self._apply_max_distance_split(df, features, val_size, test_size, keep_old_test)

    def _split_residue_to_value(self, df, features, val_size, test_size, keep_old_test):
        def calculate_valid_mean(val_str):
            vals = [float(v) for v in str(val_str).split(';') if v.strip()]
            valid_vals = [v for v in vals if v != 999.0]
            return np.mean(valid_vals) if valid_vals else 0.0
            
        # Stratify by the mean of valid residue values
        df['stratify_group'] = pd.qcut(df['target'].apply(calculate_valid_mean), q=self.num_bins, labels=False, duplicates='drop')
        return self._apply_max_distance_split(df, features, val_size, test_size, keep_old_test)

    def _apply_max_distance_split(self, df, features, val_size, test_size, keep_old_test):
        train_list, val_list, test_list = [], [], []

        for _, group in df.groupby('stratify_group'):
            if len(group) > 1:
                X_scaled = StandardScaler().fit_transform(group[features])
                pca1 = PCA(n_components=1).fit_transform(X_scaled).flatten()
                group = group.assign(pca1=pca1).sort_values('pca1')
            
            n = len(group)
            
            if keep_old_test:
                v_prop = val_size / (1 - test_size)
                split_idx = int(n * (1 - v_prop))
                train_list.append(group.iloc[:split_idx])
                val_list.append(group.iloc[split_idx:])
            else:
                tr_end = int(n * (1 - val_size - test_size))
                val_end = int(n * (1 - test_size))
                train_list.append(group.iloc[:tr_end])
                val_list.append(group.iloc[tr_end:val_end])
                test_list.append(group.iloc[val_end:])
                
        drop_cols = ['pca1', 'stratify_group']
        train = pd.concat(train_list).drop(columns=drop_cols, errors='ignore')
        val = pd.concat(val_list).drop(columns=drop_cols, errors='ignore')
        
        test = None
        if not keep_old_test:
            test = pd.concat(test_list).drop(columns=drop_cols, errors='ignore')
            
        return train, val, test