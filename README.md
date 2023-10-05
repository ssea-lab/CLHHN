# CLHHN

## Code
This is the official implementation of [CLHHN](https://dl.acm.org/doi/10.1145/3626569) (*CLHHN: Category-aware Lossless Heterogeneous Hypergraph
Neural Network for Session-based Recommendation*), which is just accepted by ACM Transactions on the Web.

## Environment
* Python = 3.8.13
* Pytorch = 1.12.0
* numpy = 1.22.4
* tqdm = 4.64.0

## Usage
### Datasets
Our data has been preprocessed and is available at [Dropbox](https://www.dropbox.com/sh/kiplsvwm1b64o5a/AABV17RsgwlLnqe4GNrDU7cBa?dl=0).
You need to download the dataset files and put it under the '/datasets' folder.
### Train & Test
Train and evaluate the model with the following commands.
You can also add command parameters to specify dataset/GPU id, or get fast execution by dataset sampling.

```shell
cd src
# --dataset: (str) dataset name
# --msl: (int) max sequence length
# --gpu: (int) gpu id
# --sample: (int) sample number for fast execution
python run_clhhn.py
```

## Citation
To be Added.
```text
@article{10.1145/3626569,
author = {Ma, Yutao and Wang, Zesheng and Huang, Liwei and Wang, Jian},
title = {CLHHN: Category-Aware Lossless Heterogeneous Hypergraph Neural Network for Session-Based Recommendation},
year = {2023},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
issn = {1559-1131},
url = {https://doi.org/10.1145/3626569},
doi = {10.1145/3626569},
abstract = {In recent years, session-based recommendation (SBR), which seeks to predict the target user’s next click based on anonymous interaction sequences, has drawn increasing interest for its practicality. The key to completing the SBR task is modeling user intent accurately. Due to the popularity of graph neural networks (GNNs), most state-of-the-art (SOTA) SBR approaches attempt to model user intent from the transitions among items in a session with GNNs. Despite their accomplishments, there are still two limitations. Firstly, most existing SBR approaches utilize limited information from short user-item interaction sequences and suffer from the data sparsity problem of session data. Secondly, most GNN-based SBR approaches describe pairwise relations between items while neglecting complex and high-order data relations. Although some recent studies based on hypergraph neural networks (HGNNs) have been proposed to model complex and high-order relations, they usually output unsatisfactory results due to insufficient relation modeling and information loss. To this end, we propose a category-aware lossless heterogeneous hypergraph neural network (CLHHN) in this article to recommend possible items to the target users by leveraging the category of items. More specifically, we convert each category-aware session sequence with repeated user clicks into a lossless heterogeneous hypergraph consisting of item and category nodes as well as three types of hyperedges, each of which can capture specific relations to reflect various user intents. Then, we design an attention-based lossless hypergraph convolutional network to generate session-wise and multi-granularity intent-aware item representations. Experiments on three real-world datasets indicate that CLHHN can outperform the SOTA models in making a better trade-off between prediction performance and training efficiency. An ablation study also demonstrates the necessity of CLHHN’s key components.},
note = {Just Accepted},
journal = {ACM Trans. Web},
month = {oct},
keywords = {additional information, information loss, heterogeneous hypergraphs, hypergraph neural networks, session-based recommendation}
}
```



