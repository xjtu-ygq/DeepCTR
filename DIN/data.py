import sys 
sys.path.append("..")

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from Tools.basic.features import DenseFeature, SparseFeature, SequenceFeature
from Tools.utils.data import DataGenerator, generate_seq_feature, df_to_dict, pad_sequences

def get_amazon_data_dict(dataset_path):
    data = pd.read_csv(dataset_path)
    print('========== Start Amazon ==========')
    # 注意这个函数里采用了LabelEncoder，数据发生了变化，实际使用注意修改
    train, val, test = generate_seq_feature(data=data, user_col="user_id", item_col="item_id", time_col='time', item_attribute_cols=["cate_id"])
#     print('INFO: Now, the dataframe named: ', train.columns)
    n_users, n_items, n_cates = data["user_id"].max(), data["item_id"].max(), data["cate_id"].max()
#     print(train)

    features = [SparseFeature("target_item_id", vocab_size=n_items+1, embed_dim=8), SparseFeature("target_cate_id", vocab_size=n_cates+1, embed_dim=8), SparseFeature("user_id", vocab_size=n_users+1, embed_dim=8)]
    target_features = features
    history_features = [
        SequenceFeature("hist_item_id", vocab_size=n_items+1, embed_dim=8, pooling="concat", shared_with="target_item_id"),
        SequenceFeature("hist_cate_id", vocab_size=n_cates+1, embed_dim=8, pooling="concat", shared_with="target_cate_id")
    ]

    print('========== Generate input dict ==========')
    train = df_to_dict(train)
    val = df_to_dict(val)
    test = df_to_dict(test)
    train_y, val_y, test_y = train["label"], val["label"], test["label"]

    del train["label"]
    del val["label"]
    del test["label"]
    train_x, val_x, test_x = train, val, test
    return features, target_features, history_features, (train_x, train_y), (val_x, val_y), (test_x, test_y)