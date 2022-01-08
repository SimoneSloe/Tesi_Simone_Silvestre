import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit


data_df = pd.read_csv('Dataset/gare-empulia-cpv.csv')
labels = data_df.COD_CPV
data_df = data_df.drop(columns='COD_CPV')



# X = np.array(data_df)
# y = np.array(labels)


# count = 0
# labels = labels.apply(str)
# a = labels.value_counts()
# for i in a:
#     if i == 1:
#         count += 1
#
# print(count)



# sss = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
# for train_index, test_index in sss.split(X, y):
#     print("TRAIN:", train_index, "TEST:", test_index)
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]



# train, test = train_test_split(data_df, test_size=0.3, random_state=4, stratify=labels)



# test_ratio = 0.3
#
# def split_train_test(data: np.ndarray, distribution: list, test_ratio: float):
#     skf = StratifiedKFold(n_splits=int(test_ratio * 100), random_state=1374, shuffle=True)
#     return next(skf.split(data, distribution))
#
#
# split_train_test(data_df, labels, test_ratio)

