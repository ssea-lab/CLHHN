# CLHHN

## Code
This is the official implementation of CLHHN(*CLHHN: Category-aware Lossless Heterogeneous Hypergraph
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



