from sklearn import model_selection

import pandas as pd
import os


if __name__ == "__main__":
    file_path = "../data/"
    df = pd.read_csv(os.path.join(file_path, "train.csv"))

    # create a new column that indicates the fold each row belongs to
    df["kfold"] = -1
    y = df.target.values

    # we want the ratio of positive to negative samples to be (somewhat) constant in each fold
    kf = model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, "kfold"] = f

    df.to_csv(os.path.join(file_path, "../data/train_with_folds.csv"), index=False)
