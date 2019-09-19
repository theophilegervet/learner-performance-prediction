# kt-algos
Implementation of knowledge tracing algorithms:
- [DAS3H](https://arxiv.org/pdf/1905.06873.pdf)
- [DKT](https://stanford.edu/~cpiech/bio/papers/deepKnowledgeTracing.pdf)

Install [PyTorch](https://pytorch.org) in a conda environment and the remaining requirements:

```
pip install -r requirements.txt
```

The code supports the following datasets:
- [Assistments 2009-2010](https://sites.google.com/site/assistmentsdata/home/assistment-2009-2010-data/skill-builder-data-2009-2010) (assistments09)
- [Assistments 2012-2013](https://sites.google.com/site/assistmentsdata/home/2012-13-school-data-with-affect) (assistments12)
- [Assistments 2015](https://sites.google.com/site/assistmentsdata/home/2015-assistments-skill-builder-data) (assistments15)
- [Assistments Challenge 2017](https://sites.google.com/view/assistmentsdatamining) (assistments17)
- [Bridge to Algebra 2006-2007](https://pslcdatashop.web.cmu.edu/KDDCup/downloads.jsp) (bridge_algebra06)
- [Algebra I 2005-2006](https://pslcdatashop.web.cmu.edu/KDDCup/downloads.jsp) (algebra05)

| First Header  | Second Header |
| ------------- | ------------- |
| Content Cell  | Content Cell  |
| Content Cell  | Content Cell  |

Place Assistments datasets under `data/<dataset codename>/data.csv` and the others under `data/<dataset codename>/data.txt`. To prepare a dataset:

```
python prepare_data.py --dataset <dataset codename> --remove_nan_skills
```

To encode a sparse matrix with certain features on a dataset:

```
python encode_lr.py --dataset <dataset codename> --users --items --skills --wins --attempts --tw_kc
```

To train a logistic regression model with a sparse feature matrix encoded through encode_das3h.py:

```
python train_lr.py data/<dataset codename>/X-uiswat1.npz --dataset <dataset codename>
```

To train a DKT model:

```
python train_dkt.py --dataset <dataset codename>
```

