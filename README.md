# kt-algos

Simple and performant implementations of knowledge tracing algorithms:
- [DAS3H](https://arxiv.org/pdf/1905.06873.pdf)
- [DKT](https://stanford.edu/~cpiech/bio/papers/deepKnowledgeTracing.pdf)
- [SAKT](https://arxiv.org/pdf/1907.06837.pdf)

## Setup

In a new conda environment with python 3, install [PyTorch](https://pytorch.org) and the remaining requirements:

```
pip install -r requirements.txt
```

The code supports the following datasets:
- [ASSISTments 2009-2010](https://sites.google.com/site/assistmentsdata/home/assistment-2009-2010-data/skill-builder-data-2009-2010) (assistments09)
- [ASSISTments 2012-2013](https://sites.google.com/site/assistmentsdata/home/2012-13-school-data-with-affect) (assistments12)
- [ASSISTments 2015](https://sites.google.com/site/assistmentsdata/home/2015-assistments-skill-builder-data) (assistments15)
- [ASSISTments Challenge 2017](https://sites.google.com/view/assistmentsdatamining) (assistments17)
- [Bridge to Algebra 2006-2007](https://pslcdatashop.web.cmu.edu/KDDCup/downloads.jsp) (bridge_algebra06)
- [Algebra I 2005-2006](https://pslcdatashop.web.cmu.edu/KDDCup/downloads.jsp) (algebra05)

| Dataset          | # Users  | # Items | # Skills | # Interactions | Mean # skills/item | Timestamps | Median length |
| ---------------- | -------- | ------- | -------- | -------------- | ------------------ | ---------- | ------------- |
| assistments09    | 3,241    | 26,634  | 124      | 397,137        | 1.20               | No         | 26            |
| assistments12    | 29,018   | 53,086  | 265      | 2,711,602      | 1.00               | Yes        | 49            |
| assistments15    | 14,567   | 100     | 100      | 658,887        | 1.00               | No         | 20            |
| assistments17    | 1,708    | 3,162   | 102      | 942,814        | 1.23               | Yes        | 441           |
| bridge_algebra06 | 1,146    | 129,263 | 493      | 1,817,476      | 1.01               | Yes        | 1,362         |
| algebra05        | 574      | 173,113 | 112      | 607,025        | 1.36               | Yes        | 574           |

To use a dataset, download the data from one of the links above and place the main file under `data/<dataset codename>/data.csv` if it is an ASSISTments dataset and under `data/<dataset codename>/data.txt` otherwise. To preprocess a dataset:

```
python prepare_data.py --dataset <dataset codename> --remove_nan_skills
```

## Training

#### Logistic regression

To encode a sparse feature matrix with specified features:

```
python encode_lr.py --dataset <dataset codename> --items --skills --wins --attempts --time_windows
```

To train a logistic regression model with a sparse feature matrix encoded through encode_lr.py:

```
python train_lr.py data/<dataset codename>/X-lr-iswa_tw.npz --dataset <dataset codename>
```

#### Feedforward neural network

To encode a sparse feature matrix with specified features:

```
python encode_ffw.py --dataset <dataset codename> --total --items --num_prev_interactions=1
```

To train a feedforward neural network model with a dense feature matrix encoded through encode_ffw.py:

```
python train_ffw.py data/<dataset codename>/X-ffw-ti-1.npz --dataset <dataset codename>
```

#### Deep knowledge tracing

To train a DKT model:

```
python train_dkt.py --dataset <dataset codename> --embed_inputs
```

#### Self-attentive knowledge tracing

To train a SAKT model:

```
python train_sakt.py --dataset <dataset codename> --embed_inputs 
```

## Results

| Algorithm | assistments09 | assistments12 | assistments15 | assistments17 | bridge_algebra06 | algebra05 |
| --------- | ------------- | ------------- | ------------- | ------------- | ---------------- | --------- | 
| IRT       | 0.69          | 0.71          | 0.64          | 0.68          | 0.75             | 0.77      |                  
| PFA       | 0.77          | 0.75          | 0.70          | 0.71          | 0.80             | 0.83      | 
| DAS3H     | -             | 0.75          | -             | 0.72          | 0.79             | 0.83      |
| FFW       | 0.78          |               | 0.71          | 0.71          |                  |           |
| DKT       |               |               |               |               |                  |           |
| SAKT      |               |               |               |               |                  |           |

Legend for results in table:
- IRT: logistic regression with `--item` flags
- PFA: logistic regression with `--item --skills --wins --attempts` flags
- DAS3H: logistic regression with `--item --skills --wins --attempts --time_windows` flags
- FFW: feedforward neural network with `--total --items --num_prev_interactions=1` flags
