import sys 
sys.path.append("..")

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from Tools.models.ranking import WideDeep, DeepFM, DCN
from Tools.trainers import CTRTrainer
from Tools.basic.features import DenseFeature, SparseFeature
from Tools.utils.data import DataGenerator

def convert_numeric_feature(val):
    v = int(val)
    if v > 2:
        return int(np.log(v)**2)
    else:
        return v - 2


def get_criteo_data_dict(data_path):
    if data_path.endswith(".gz"):  #if the raw_data is gz file:
        data = pd.read_csv(data_path, compression="gzip")
    else:
        data = pd.read_csv(data_path,nrows=100000)
#         data = pd.read_csv(data_path)
    print("data load finished")
    dense_features = [f for f in data.columns.tolist() if f[0] == "I"]
    sparse_features = [f for f in data.columns.tolist() if f[0] == "C"]

    data[sparse_features] = data[sparse_features].fillna('-996',)
    data[dense_features] = data[dense_features].fillna(0,)

    for feat in tqdm(dense_features):  #discretize dense feature and as new sparse feature
        sparse_features.append(feat + "_cat")
        data[feat + "_cat"] = data[feat].apply(lambda x: convert_numeric_feature(x))

    sca = MinMaxScaler()  #scaler dense feature
    data[dense_features] = sca.fit_transform(data[dense_features])

    for feat in tqdm(sparse_features):  #encode sparse feature
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    dense_feas = [DenseFeature(feature_name) for feature_name in dense_features]
    sparse_feas = [SparseFeature(feature_name, vocab_size=data[feature_name].nunique(), embed_dim=16) for feature_name in sparse_features]
    y = data["label"]
    del data["label"]
    x = data
    return dense_feas, sparse_feas, x, y