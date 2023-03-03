import sys
sys.path.append("..")

import os
import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from Tools.trainers import MatchTrainer
from Tools.basic.features import DenseFeature, SparseFeature, SequenceFeature
from Tools.utils.match import generate_seq_feature_match, gen_model_input
from Tools.utils.data import df_to_dict, MatchDataGenerator
from movielens_utils import match_evaluation
# from data import get_movielens_data

def get_movielens_data(data_path, load_cache=False):
    data = pd.read_csv(data_path)
    data["cate_id"] = data["genres"].apply(lambda x: x.split("|")[0])
    sparse_features = ['user_id', 'movie_id', 'gender', 'age', 'occupation', 'zip', "cate_id"]
    user_col, item_col = "user_id", "movie_id"

    feature_max_idx = {}
    for feature in sparse_features:
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature]) + 1
        feature_max_idx[feature] = data[feature].max() + 1
        if feature == user_col:
            user_map = {encode_id + 1: raw_id for encode_id, raw_id in enumerate(lbe.classes_)}  #encode user id: raw user id
        if feature == item_col:
            item_map = {encode_id + 1: raw_id for encode_id, raw_id in enumerate(lbe.classes_)}  #encode item id: raw item id
    np.save("../data/saved/raw_id_maps.npy", np.array((user_map, item_map), dtype=object))

    user_profile = data[["user_id", "gender", "age", "occupation", "zip"]].drop_duplicates('user_id')
    item_profile = data[["movie_id", "cate_id"]].drop_duplicates('movie_id')

    if load_cache:  #if you have run this script before and saved the preprocessed data
        x_train, y_train, x_test, y_test = np.load("../data/saved/data_preprocess.npy", allow_pickle=True)
    else:
        df_train, df_test = generate_seq_feature_match(data,
                                                       user_col,
                                                       item_col,
                                                       time_col="timestamp",
                                                       item_attribute_cols=[],
                                                       sample_method=1,
                                                       mode=0,
                                                       neg_ratio=3,
                                                       min_item=0)
        x_train = gen_model_input(df_train, user_profile, user_col, item_profile, item_col, seq_max_len=50)
        y_train = x_train["label"]
        x_test = gen_model_input(df_test, user_profile, user_col, item_profile, item_col, seq_max_len=50)
        y_test = x_test["label"]
        np.save("../data/saved/data_preprocess.npy", np.array((x_train, y_train, x_test, y_test), dtype=object))

    user_cols = ['user_id', 'gender', 'age', 'occupation', 'zip']
    item_cols = ['movie_id', "cate_id"]

    user_features = [
        SparseFeature(feature_name, vocab_size=feature_max_idx[feature_name], embed_dim=16) for feature_name in user_cols
    ]
    user_features += [
        SequenceFeature("hist_movie_id",
                        vocab_size=feature_max_idx["movie_id"],
                        embed_dim=16,
                        pooling="mean",
                        shared_with="movie_id")
    ]

    item_features = [
        SparseFeature(feature_name, vocab_size=feature_max_idx[feature_name], embed_dim=16) for feature_name in item_cols
    ]

    all_item = df_to_dict(item_profile)
    test_user = x_test
    return user_features, item_features, x_train, y_train, all_item, test_user