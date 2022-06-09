import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import json
import os
import matplotlib.pyplot as plt
import tables
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler  # 标准化工具
from biobert_embedding.embedding import BiobertEmbedding
from research import ResearchQA

def demogetdata(question):
    question = str(question)
    print(question)
    plt.style.use('ggplot')

    dir_path = "F:\CORD-dataset" ##修改为源数据和预处理文章向量的存放路径

    npassage = 1010000
    metadata_path = os.path.join(dir_path, "metadata.csv")
    meta_df = pd.read_csv(metadata_path, nrows=npassage)
    nanlist = meta_df["abstract"][meta_df["abstract"].isnull().values == True].index.tolist()
    meta_df.drop(nanlist, inplace=True)
    meta_df.reset_index(drop=True, inplace=True)
    Queries=[]
    queries = [question]
    # Class Initialization (You can set default 'model_path=None' as your finetuned BERT model path while Initialization)
    biobert = BiobertEmbedding()
    for query in queries:
        sentence_embedding = biobert.sentence_vector(query)
        Queries.append(sentence_embedding.numpy())
    scaler = StandardScaler()
    normalQueries = scaler.fit_transform(Queries)
    pca = PCA(n_components=0.95, random_state=42)
    Queries_reduced = pca.fit_transform(normalQueries)
    print("StandardPCA Preprocess done!")
    hdf5_path = dir_path + "\Passagereduced_data.hdf5"
    hdf5_file = tables.open_file(hdf5_path, mode='r')
    Passage_reduced = hdf5_file.root.data
    print("Passage_reduced_shape:", Passage_reduced.shape)  # (1000, 4096)
    passage_labels = np.load("passagelabel.npy")
    print("Passage_clustering_labels:", passage_labels)
    meta_df['cluster'] = passage_labels
    print("KNNstart")
    from sklearn.neighbors import KNeighborsClassifier

    KNN = KNeighborsClassifier(n_neighbors=int(npassage * 0.2), metric_params={"VI": 'mahalanobis'})
    KNN.fit(Passage_reduced, passage_labels)
    Queries_reduced = np.array(
        [np.pad(x, (0, Passage_reduced.shape[1] - Queries_reduced.shape[1])) for x in Queries_reduced])
    querylabel = KNN.predict(Queries_reduced)
    print(querylabel)
    model_path = "ktrapeznikov/biobert_v1.1_pubmed_squad_v2"
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
        answers = qa.get_answers(queries[index], max_articles=100, answernums=10)
        answers['score'] = answers["start_score"] + answers["end_score"]
        answers["Question"] = queries[index]
        answers.sort_values(by="score", inplace=True, ascending=False)
        answers.drop_duplicates(subset="answer", inplace=True)
        print(answers.head())
        needcolums = ['answer', 'score', 'title', 'authors', 'publish_time', 'Question', 'doi']
        newresult = pd.DataFrame(answers, columns=needcolums)
        answers.to_csv("demoresult/question_answer_result.csv")
        return newresult
