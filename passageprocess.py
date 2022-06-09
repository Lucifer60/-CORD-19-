import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import json
import os
import matplotlib.pyplot as plt
import tables

'''在这一部分我们将元数据文件中的文章向量读取并进行归一化和PCA分析，随后存放在hdf5文件中便于下次读取，由于数据量较大，该处理过程需要近8小时'''

npassage = 1010000
dir_path = "F:\CORD-dataset"

'''读取metadata.csv文件'''
import time
time1 = time.time()
metadata_path = os.path.join(dir_path, "metadata.csv")
meta_df = pd.read_csv(metadata_path, nrows=npassage)
nanlist =meta_df["abstract"][meta_df["abstract"].isnull().values==True].index.tolist()
meta_df.drop(nanlist,inplace=True)
meta_df.reset_index(drop=True,inplace=True)
print(meta_df.index)
time2 = time.time() - time1
print("loading metada time:",time2)

embeddingdata_path = os.path.join(dir_path, "cord_19_embeddings_2022-04-28.csv")

names = ['cord_uid']
for i in range(768):
    names.append(i)

passagevector_df = pd.read_csv(embeddingdata_path, header=None, nrows=npassage, names=names)
passagevector_df.drop(nanlist,inplace=True)
passagevector_df.reset_index(drop=True, inplace=True)

Passage = []
for i in passagevector_df.index:
    Passage.append(passagevector_df.loc[i].values[1:].astype(np.float16))

from sklearn.preprocessing import StandardScaler  # 标准化工具

scaler = StandardScaler()
normalPassage = scaler.fit_transform(Passage)

from sklearn.decomposition import PCA

pca = PCA(n_components=0.95, random_state=42)
Passage_reduced = pca.fit_transform(normalPassage)
hdf5_path =dir_path+"\Passagereduced_data.hdf5"
hdf5_file = tables.open_file(hdf5_path, mode='w')
filters = tables.Filters(complevel=5, complib='blosc')
earray = hdf5_file.create_earray(
    hdf5_file.root,
    'data',
    tables.Atom.from_dtype(Passage_reduced.dtype),
    shape=(0,Passage_reduced.shape[1]),  # 第一维的 0 表示数据可沿行扩展
    filters=filters,
    expectedrows=800000  # 完整数据大约规模，可以帮助程序提高时空利用效率
)
print("hdf5done!")
# 将 data1 添加进 earray
earray.append(Passage_reduced)
# 写完之后记得关闭文件
hdf5_file.close()