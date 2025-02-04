import os
import pandas as pd

def get_train_data(absolute_path, seed, removeNaNs = False, validation_proportion = 0, shuffle=True):
    """
    Returns two dataframes
    """
    train_df = pd.read_csv(absolute_path)

    if shuffle:
        train_df = train_df.sample(frac=1, random_state=seed)

    if removeNaNs:
        train_df = train_df.dropna()

    validation_df = train_df.sample(frac=validation_proportion, random_state=seed)
    train_df = train_df.drop(validation_df.index)

    return train_df, validation_df

def get_test_data(absolute_path) :
    """
    Returns a dataframe
    """
    test_df = pd.read_csv(absolute_path)

    return test_df