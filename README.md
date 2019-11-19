# credit_risk
Credit Risk Modeling in a Big-Data Setting 
(*keywords: Python, Pandas, Numpy, Keras, LightGBM, SKLEARN*)

Apply machine learning and logistic regression to credit decisioning in a big-data setting (~900K cases).

---

We are using a free dataset coming without documentation. It is used to support the argument that in practice Machine Learning models are difficult to implement in credit decisioning as they are difficult to interpret. We show as well, though, that Machine Learning produces significantly more predictive models.

In this exercise we neglect data not known at time of origination and develop models that can be used to assess personal loan applications. For our prediction to have any value we need to remove cases that are already delinquent at origination time. Not knowing specifics we observe the predictor "acc_now_delinq," which is 0 in all cases:

np.all(cred.acc_now_delinq == 0) is True

This probably means that according to our data already-delinquent accounts are filtered out. We do, however, delete, in addition, cases with any recoveries or late fees.

In real life, the best option when an analyst doesn't understand their dataset is to just ask the people who put the data together what the meaning of each predictor is! Here, by necessity, we depend on assumptions about the nature of the data at hand.

Refer to [Exploratory Data Analysis](https://www.kaggle.com/yanpapadakis/credit-default-risk-data-eda) for more information about this dataset.

This is not a well-structured study, for instance there is no reject inference. Credit decisioning models should use well documented data and follow standard methodologies. Our work, though, brings home the point that machine learning has pros and cons when applied to probability of default estimation.

---

## Transparent Models

1. [Logistic Regression](https://www.kaggle.com/yanpapadakis/credit-default-risk-standardmodel) (follows standard theory)

2. [Classification Tree](https://www.kaggle.com/yanpapadakis/credit-risk-model-dtree) (kept small for interpretability, arbitrary variables on top splits)

## Machine Learning

3. [Gradient Boosting](https://www.kaggle.com/yanpapadakis/credit-risk-model-gbm) (performs best here)

4. [Neural Network](https://www.kaggle.com/yanpapadakis/credit-risk-model-nnet)
