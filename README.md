# kt-algos
Implementation of knowledge tracing algorithms:
- [DAS3H](https://arxiv.org/pdf/1905.06873.pdf)
- [DKT](https://stanford.edu/~cpiech/bio/papers/deepKnowledgeTracing.pdf)

Install requirements in a conda environment, [PyTorch](https://pytorch.org) and the rest:

```
pip install -r requirements.txt
```

To prepare a given dataset:

```
python prepare_data.py --dataset assistments17 --remove_nan_skills
```

To encode sparse matrix with given features on given dataset:

```
python encode_das3h.py --dataset assistments17 --users --items --skills --wins --attempts --tw_kc
```

To train a logistic regression model with given sparse feature matrix encoded through encode_das3h.py:

```
python train_das3h.py data/assistments17/X-uiswat1.npz --dataset assistments17
```

To train DKT:

```
python train_dkt.py --dataset assistments17
```

