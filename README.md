# kt-algos
Implementation of knowledge tracing algorithms:
- [DAS3H](https://arxiv.org/pdf/1905.06873.pdf)
- [DKT](https://stanford.edu/~cpiech/bio/papers/deepKnowledgeTracing.pdf)

Install [PyTorch](https://pytorch.org) in a conda environment and the remaining requirements:

```
pip install -r requirements.txt
```

The code supports four datasets:
- [Assistments 2012-2013](https://sites.google.com/site/assistmentsdata/home/2012-13-school-data-with-affect) (assistments12)
- [Assistments Challenge 2017](https://sites.google.com/view/assistmentsdatamining) (assistments17)
- [Bridge to Algebra 2006-2007](https://pslcdatashop.web.cmu.edu/KDDCup/downloads.jsp) (bridge_algebra06)
- [Algebra I 2005-2006](https://pslcdatashop.web.cmu.edu/KDDCup/downloads.jsp) (algebra05)

Put a dataset under `data/<dataset codename>/data.csv` for Assistments dataset, and `data/<dataset codename>/data.txt` for the others. To prepare a dataset:

```
python prepare_data.py --dataset <dataset codename> --remove_nan_skills
```

To encode a sparse matrix with certain features on a dataset:

```
python encode_das3h.py --dataset <dataset codename> --users --items --skills --wins --attempts --tw_kc
```

To train a logistic regression model with a sparse feature matrix encoded through encode_das3h.py:

```
python train_das3h.py data/<dataset codename>/X-uiswat1.npz --dataset <dataset codename>
```

To train a DKT model:

```
python train_dkt.py --dataset <dataset codename>
```

