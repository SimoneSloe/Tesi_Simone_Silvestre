import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit


data_df = pd.read_csv('Dataset/gare-empulia-cpv.csv')
labels = data_df['COD_CPV']
data_df = data_df.drop(columns='COD_CPV')

# x = np.array(data_df)
# y = np.array(labels)


# all_indices = list(range(len(data_df)))
# train_indices, test_indices = train_test_split(all_indices, test_size=0.3)
#
# train = data_df.iloc[train_indices, :]
# test = data_df.iloc[test_indices, :]
#
# def get_class_counts(df):
#     grp = df.groupby(['COD_CPV']).nunique()
#     return{key: grp[key] for key in list(grp.keys())}
#
# def get_class_proportions(df):
#     class_counts = get_class_counts(df)
#     return {val[0]: round(val[1]/df.shape[0], 4) for val in class_counts.items()}
#
#
# train_class_proportions = get_class_proportions(train)
# test_class_proportions = get_class_proportions(test)
#
# print("Train data class proportions", train_class_proportions[0])
# print("Test data class proportions", test_class_proportions[0])


# count = 0
# labels = labels.apply(str)
# a = labels.value_counts()
# for i in a:
#     if i == 1:
#         count += 1
#
# print(count)


train, test = train_test_split(data_df, test_size=0.3, random_state=4, stratify=labels)




