

文本挖掘第二次实验

## 1. 实验环境：

1. torch≥1.2.0

2. bio-bertembedding：0.1.2	

    ```apl
    pip install biobert-embedding
    ```

3. transformers：4.18.0

4. scikit-learn

## 2. 数据

我们使用的是CORD-19数据集在4-28更新的文件

https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/historical_releases/cord-19_2022-04-28.tar.gz

## 3. 运行细节

1. 将原始数据cord_19_embeddings.tar.gz进行解压得到。

    dirpath-

      - metadata.csv（源数据文件）

    - cord_19_embeddings_2022-04-28.csv（文章向量文件）

2. 首先进入Passage_Preprocess.py

    将```dir_path```修改为存放最终生成文章向量结果hdf5结果的路径文件并且将metadata.csv放在该路径下。

    然后设定```npassage```数量。```npassage```为读取文章篇数，```dir_path```为保存文件的路径。（注意在这里后续进行问答时，```npassage```需要与之保持一致。）

    随后运行Passage_Preprocess.py：需要大约数小时，主要运行时间在归一化和对数据的PCA过程中。

    我们选取```npassage=1010000```时，最终结果接近2G。

    此时dirpath下面已经存放

    dirpath-

      - metadata.csv（源数据文件）

      - cord_19_embeddings_2022-04-28.csv（文章向量文件）

      - Passagereduced_data.hdf5(处理后的保存下的文章向量文件)

        （该文件是离线处理好的保存经过预处理后的Passage_reduced向量文件，需要运行数小时才能做完数据预处理。）

​		其中我们已经将bio-bertembedding模型预训练文件放在 ``biobert_v1.1_pubmed_pytorch_model``文件夹中。

2. 进入```main.py```中 将```dirpath```修改为metadata.csv的所在的目录路径

    注意这个地方npassage值应与初始Passage_preprocess处npassage值保持一致。（默认为101000）

    随后运行```main.py```即可

## 3.结果

最终我们本地生成的结果文件存放在``represent_result_output.csv``中

我们还将每个问题分别的结果存放在了represent_result文件夹下分别以x.csv 等命名。

## 4.Demo

为了便于调用，我们在本次实验中完成了Demo部分，我们将视频存放在DEMO视频展示.mp4文件中。

关于demo的使用可直接运行demo.py.需要修改demogetanswer.py部分的```dir_path```

输入问句后等待几分钟即可获得答案，答案将保存在demoresult文件夹内。（前提是已经做完了数据预处理部分）
