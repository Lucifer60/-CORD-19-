import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import json
import os
import matplotlib.pyplot as plt
import tables



plt.style.use('ggplot')
dir_path = "F:\CORD-dataset" ##修改为源文件存放路径

pd.set_option("display.max.columns", None)
npassage = 1010000
# npassage = 101
import time
time1 = time.time()
metadata_path = os.path.join(dir_path, "metadata.csv")
meta_df = pd.read_csv(metadata_path, nrows=npassage)
nanlist =meta_df["abstract"][meta_df["abstract"].isnull().values==True].index.tolist()
meta_df.drop(nanlist,inplace=True)
meta_df.reset_index(drop=True,inplace=True)
time2 = time.time() - time1
print("loading metada time:",time2)

queries = ['What is known about transmission, incubation, and environmental stability?',
           'What do we know about COVID-19 risk factors?',
           'What do we know about virus genetics, origin, and evolution?',
           ' What do we know about vaccines and therapeutics?',
           'What do we know about non-pharmaceutical interventions?',
           ' What has been published about medical care?',
           'What do we know about diagnostics and surveillance?',
           ' What has been published about information sharing and inter-sectoral collaboration?',
           'What has been published about ethical and social science considerations?']
Queries = []

from biobert_embedding.embedding import BiobertEmbedding

# Class Initialization (You can set default 'model_path=None' as your finetuned BERT model path while Initialization)
biobert = BiobertEmbedding()
for query in queries:
    sentence_embedding = biobert.sentence_vector(query)
    Queries.append(sentence_embedding.numpy())
from sklearn.preprocessing import StandardScaler  # 标准化工具
scaler = StandardScaler()
normalQueries = scaler.fit_transform(Queries)

from sklearn.decomposition import PCA

pca = PCA(n_components=0.95, random_state=42)
Queries_reduced = pca.fit_transform(normalQueries)

hdf5_path =dir_path+"\Passagereduced_data.hdf5"
hdf5_file = tables.open_file(hdf5_path, mode='r')
# 数据名称为 'data'，我们可以通过 .root.data 访问到它
Passage_reduced = hdf5_file.root.data
print("Passage_reduced_shape:",Passage_reduced.shape) # (1000, 4096)

print("Standard Preprocess done!")




'''肘部法则求取k'''
from sklearn import metrics
from scipy.spatial.distance import cdist
# distortions = []
# K = range(30, 100)
# for k in K:
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     kmeans.fit(Passage_reduced)
#     distortions.append(sum(np.min(cdist(Passage_reduced, kmeans.cluster_centers_, 'euclidean'), axis=1)) / np.array(Passage_reduced).shape[0])
# X_line = [K[0], K[-1]]
# Y_line = [distortions[0], distortions[-1]]
# # Plot the elbow
# plt.plot(K, distortions, 'b-')
# plt.plot(X_line, Y_line, 'r')
# plt.xlabel('k')
# plt.ylabel('Distortion')
# plt.title('The Elbow Method showing the optimal k')
# plt.show()


'''聚类'''
from sklearn.cluster import KMeans
k = 20
kmeans = KMeans(n_clusters=k, random_state=42)
passage_labels = kmeans.fit_predict(Passage_reduced)
print("Passage_clustering_labels:",passage_labels)

meta_df['cluster'] = passage_labels


'''聚类可视化展示耗时较长'''
# from sklearn.manifold import TSNE
# tsne = TSNE(verbose=1, random_state=42, init="pca", method="barnes_hut")
# X_embedded_passage = tsne.fit_transform(Passage_reduced[:100000])
# from matplotlib import pyplot as plt
# import seaborn as sns
#
# sns.set(rc={'figure.figsize': (15, 15)})
# palette1 = sns.hls_palette(20, l=.4, s=.9)
# palette2 = sns.color_palette()
#
# sns.scatterplot(X_embedded_passage[:, 0], X_embedded_passage[:, 1], hue=passage_labels[:300000], legend='full',
#                 palette=palette1)
# # sns.scatterplot(X_embedded_query[:,0], X_embedded_query[:,1], hue=y_pred[npassage:], legend='full',markers='*')
# plt.title('t-SNE with Kmeans Labels')
# plt.show()
# plt.savefig("improved_cluster_tsne.png")


'''KNN匹配'''
print("KNNstart")
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors=int(npassage * 0.2), metric_params={"VI": 'mahalanobis'})
KNN.fit(Passage_reduced, passage_labels)
Queries_reduced = np.array(
    [np.pad(x, (0, Passage_reduced.shape[1] - Queries_reduced.shape[1])) for x in Queries_reduced])
querylabel = KNN.predict(Queries_reduced)
print(querylabel)


'''QA问答调取researchQA类'''
from research import ResearchQA
model_path = "ktrapeznikov/biobert_v1.1_pubmed_squad_v2"
Result=[]
for index in range(len(querylabel)):
    querydf = meta_df[meta_df['cluster'] == querylabel[index]]
    queryindex = querydf.index
    print("Start to Get Passagevector!")
    passagevector = Passage_reduced[queryindex]
    query = Queries_reduced[index]
    print("QAstart!")
    passagesort = np.argsort(np.dot(passagevector, query), axis=0)[:int(len(passagevector) * 0.3)]
    print(passagesort)
    querydf = querydf.iloc[passagesort]
    print(querydf.count())
    qa = ResearchQA(querydf, model_path)
    answers = qa.get_answers(queries[index], max_articles=100,answernums=10)
    answers['score'] = answers["start_score"] + answers["end_score"]
    answers["Question"] = queries[index]
    answers.sort_values(by="score", inplace=True, ascending=False)
    answers.drop_duplicates(subset="answer",inplace=True)
    print(answers.head())
    answers.to_csv("result/"+str(index)+"result.csv")
    Result.append(answers)
resultdf = pd.concat(Result,ignore_index=True)
print(resultdf.head())
resultdf.drop(labels=['start_score','end_score','context','doi','authors','journal','publish_time','title'],axis=1,inplace=True)
resultdf.to_csv("result_output.csv",index=None)#保存最终结果