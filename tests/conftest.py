import os

import pytest
import pandas as pd
import numpy as np


@pytest.fixture()
def ovarian_cancer_dataset():
    dir_path = os.path.dirname(os.path.realpath(__file__))

    data = pd.read_csv(
        os.path.join(dir_path, "data", "ovarian_cancer.csv"), index_col="idx"
    )

    # Unfortunately it's unclear how the missing data was imputed in the article
    # We use median imputation here. As such AUROCs are the same but p-values differ.
    data.albumin = data.albumin.fillna(np.median(data.albumin.dropna()))
    data.total_protein = data.total_protein.fillna(
        np.median(data.total_protein.dropna())
    )

    return data
