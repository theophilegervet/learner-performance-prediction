# kt-algos

Simple and performant implementations of knowledge tracing algorithms:
- [Performance Factors Analysis (PFA)](http://pact.cs.cmu.edu/koedinger/pubs/AIED%202009%20final%20Pavlik%20Cen%20Keodinger%20corrected.pdf)
- [DAS3H](https://arxiv.org/pdf/1905.06873.pdf)
- [Deep Knowledge Tracing (DKT)](https://stanford.edu/~cpiech/bio/papers/deepKnowledgeTracing.pdf)
- [Self-Attentive Knowledge Tracing (SAKT)](https://arxiv.org/pdf/1907.06837.pdf)

## Setup

In a new conda environment with python 3, install [PyTorch](https://pytorch.org) and the remaining requirements:

```
pip install -r requirements_kt.txt
```

The code supports the following datasets:
- [ASSISTments 2009-2010](https://sites.google.com/site/assistmentsdata/home/assistment-2009-2010-data/skill-builder-data-2009-2010) (assistments09)
- [ASSISTments 2012-2013](https://sites.google.com/site/assistmentsdata/home/2012-13-school-data-with-affect) (assistments12)
- [ASSISTments 2015](https://sites.google.com/site/assistmentsdata/home/2015-assistments-skill-builder-data) (assistments15)
- [ASSISTments Challenge 2017](https://sites.google.com/view/assistmentsdatamining) (assistments17)
- [Bridge to Algebra 2006-2007](https://pslcdatashop.web.cmu.edu/KDDCup/downloads.jsp) (bridge_algebra06)
- [Algebra I 2005-2006](https://pslcdatashop.web.cmu.edu/KDDCup/downloads.jsp) (algebra05)
- [Spanish](https://github.com/robert-lindsey/WCRP) (spanish)
- [Statics](https://pslcdatashop.web.cmu.edu) (statics)

| Dataset          | # Users  | # Items | # Skills | # Interactions | Mean # skills/item | Timestamps | Median length |
| ---------------- | -------- | ------- | -------- | -------------- | ------------------ | ---------- | ------------- |
| assistments09    | 3,241    | 17,709  | 124      | 278,868        | 1.20               | No         | 35            |
| assistments12    | 29,018   | 53,086  | 265      | 2,711,602      | 1.00               | Yes        | 49            |
| assistments15    | 14,567   | 100     | 100      | 658,887        | 1.00               | No         | 20            |
| assistments17    | 1,708    | 3,162   | 102      | 942,814        | 1.23               | Yes        | 441           |
| bridge_algebra06 | 1,146    | 129,263 | 493      | 1,817,476      | 1.01               | Yes        | 1,362         |
| algebra05        | 574      | 173,113 | 112      | 607,025        | 1.36               | Yes        | 574           |
| spanish          | 182      | 409     | 221      | 578,726        | 1.00               | No         | 1,924         |
| statics          | 282      | 1,223   | 98       | 189,297        | 1.00               | No         | 635           |

To use a dataset, download the data from one of the links above and:
- place the main file under `data/<dataset codename>/data.csv` for an ASSISTments dataset
- place the main file under `data/<dataset codename>/data.txt` for a KDDCup dataset
- place the two data files under `data/<dataset codename>/{filename}` for the Spanish dataset
- the Statics dataset is already preprocessed under `data/statics/preprocessed_data.csv`

```
python prepare_data.py --dataset <dataset codename> --remove_nan_skills
```

## Training

#### Logistic Regression

To encode a sparse feature matrix with specified features:
- Item Response Theory (IRT): `--item` 
- PFA: `--item --skills --wins --attempts` 
- DAS3H: `--item --skills --wins --attempts --time_windows` 

```
python encode_lr.py --dataset <dataset codename> --items --skills --wins --attempts --time_windows
```

To train a logistic regression model with a sparse feature matrix encoded through encode_lr.py:

```
python train_lr.py --X_file data/<dataset codename>/X-lr-iswa_tw.npz --dataset <dataset codename>
```

#### Deep Knowledge Tracing

To train a DKT model taking skills as input and outputting skills:

```
python train_dkt.py --dataset <dataset codename> --skill_in --skill_out
```

#### Self-Attentive Knowledge Tracing

To train a SAKT model taking skills as input and querying with skills:

```
python train_sakt.py --dataset <dataset codename> --skill_in --skill_out
```

## Results

| Algorithm      | assist09      | assist12 | assist15      | assist17 | bridge06 | algebra05 | spanish  | statics  |
| -------------- | ------------- | -------- | ------------- | -------- | -------- | --------- | -------- | -------- |
| IRT            | 0.69          | 0.71     | 0.64          | 0.68     | 0.75     | 0.77      | 0.68     | 0.79     |       
| PFA            | **0.77**      | **0.75** | 0.70          | 0.71     | **0.80** | **0.83**  | **0.86** | 0.82     |
| DAS3H          | -             | **0.75** | -             | 0.72     | 0.79     | **0.83**  | -        | -        |
| DKT            | 0.75          | **0.75** | **0.73**      | **0.77** | 0.79     | 0.82      | 0.85     | **0.83** |
| SAKT           | 0.75          | 0.73     | 0.72          | 0.72     | 0.78     | 0.80      | -        | 0.81     |
| DAS3H reported | -             | 0.74     | -             | -        | 0.79     | **0.83**  | -        | -        |
| DKT reported   | 0.86 cheating | -        | **0.73**      | 0.73     | -        | -         | 0.83     | 0.83     |
| SAKT reported  | 0.85 cheating | -        | 0.85 cheating | 0.73     | -        | -         | -        | 0.85     |
| BKT+ reported  | 0.83 cheating | -        | -             | -        | -        | -         | 0.85     | 0.75     |
