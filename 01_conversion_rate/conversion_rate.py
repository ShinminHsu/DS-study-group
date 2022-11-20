#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-

import importlib
import pandas as pd
import util_func
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
importlib.reload(util_func)

raw_data = pd.read_csv('conversion_project.csv')
raw_data = raw_data[~raw_data['age'].isin(raw_data['age'].nlargest(2).values)].reset_index(drop=True)

X = raw_data.drop(['converted'], axis=1)
y = raw_data['converted']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_train = util_func.feature_engineering(X_train)

logistic_model = LogisticRegression(penalty='none', max_iter=500)
logistic_model.fit(X_train, y_train)