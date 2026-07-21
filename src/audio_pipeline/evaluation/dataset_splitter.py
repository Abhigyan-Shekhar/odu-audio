"""
Dataset splitting for evaluation, ensuring patient stratification.
"""

from typing import Dict

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


class DatasetSplitter:
    """Split dataset at patient level."""

    def split(
        self,
        dataset: pd.DataFrame,
        strategy: str = "patient_stratified",
        train_size: float = 0.7,
        val_size: float = 0.15,
        random_state: int = 42,
    ) -> Dict[str, pd.DataFrame]:
        """
        Split ensuring no patient appears in multiple splits.
        We group by patient_id.
        """
        if "patient_id" not in dataset.columns:
            raise ValueError("dataset must contain a 'patient_id' column")

        if strategy != "patient_stratified":
            raise NotImplementedError(
                f"Strategy {strategy} is not implemented. Use 'patient_stratified'."
            )

        # First split: train vs temp (val + test)
        gss = GroupShuffleSplit(
            n_splits=1, train_size=train_size, random_state=random_state
        )
        train_idx, temp_idx = next(gss.split(dataset, groups=dataset["patient_id"]))

        train_set = dataset.iloc[train_idx].copy()
        temp_set = dataset.iloc[temp_idx].copy()

        # Second split: val vs test
        test_size_relative = 1.0 - (val_size / (1.0 - train_size))

        # If test_size_relative is very close to 0 or 1, handle carefully
        if test_size_relative <= 0.0:
            return {
                "train": train_set,
                "val": temp_set,
                "test": pd.DataFrame(columns=dataset.columns),
            }
        elif test_size_relative >= 1.0:
            return {
                "train": train_set,
                "val": pd.DataFrame(columns=dataset.columns),
                "test": temp_set,
            }

        gss_val = GroupShuffleSplit(
            n_splits=1, test_size=test_size_relative, random_state=random_state
        )
        val_idx, test_idx = next(gss_val.split(temp_set, groups=temp_set["patient_id"]))

        val_set = temp_set.iloc[val_idx].copy()
        test_set = temp_set.iloc[test_idx].copy()

        return {
            "train": train_set,
            "val": val_set,
            "test": test_set,
        }
