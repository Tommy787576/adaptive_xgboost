import numpy as np
import pandas as pd
import json

def load_mapping():
    with open('mapping.json', 'r') as json_file:
        mapping = json.load(json_file)

    return mapping['col_map'], mapping['one_hot_col_map'], mapping['total_col']

def one_hot_encoding(data, total_col):

    one_hot = pd.get_dummies(data, prefix_sep='-') 

    # append 0 
    for col in total_col:
        if col not in one_hot.columns:
            one_hot[col] = 0

    return one_hot[total_col] # sort column


def preprocess(file_path, save=False):

    col_mapping, one_hot_col_mapping, total_col = load_mapping()
    data = pd.read_csv(file_path)

    catgorical = data.iloc[:,1:6]
    numerical = data.iloc[:,6:]
    ans = data.iloc[:,0]

    encoded_catgorical = catgorical.apply(lambda x: pd.Series([one_hot_col_mapping[c][3:] for c in x.astype(str)]))
    encoded_catgorical = encoded_catgorical.rename(columns=col_mapping).sort_index(axis=1)

    one_hot_catgorical = one_hot_encoding(encoded_catgorical, total_col)

    encoded_data = pd.concat([encoded_catgorical, numerical, ans], axis=1)
    one_hot_encoded_data = pd.concat([one_hot_catgorical, numerical, ans], axis=1)

    if save == True:
       encoded_data.to_csv('encoded_data.csv', index=False)
       one_hot_encoded_data.to_csv('one_hot_encoded_data.csv', index=False)

    return encoded_data.values, one_hot_encoded_data.values

if __name__ == '__main__':
    preprocess('UPH_222K.csv', save=True)

