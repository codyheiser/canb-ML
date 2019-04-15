# canb-ML
Supervised classification of neonatal data

---
Given high-dimensional dataset with live births as labels, build supervised machine learning model to predict classes in unseen dataset.
* __preprocess__ data into a form amenable to supervised ML algorithms ([`data_preprocessing.ipynb`](data_preprocessing.ipynb))
* evaluate information and __select features__ or __impute missing data__ to best inform classifier ([`imputation.ipynb`](imputation.ipynb))
* benchmark classifiers on training set using cross-validation to determine best algorithm for future predictions ([`cross_val.ipynb`](cross_val.ipynb))
---
Before starting, set up Python environment:
```
pip install -r requirements.text
```
