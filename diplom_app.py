from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
import lightgbm as lgb
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from numpy import mean
from numpy import column_stack
# from cmath import exp
from numpy import log1p
from numpy import exp
# from math import round
import streamlit as st
import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from scipy.special import boxcox1p
from scipy.special import inv_boxcox

st.set_option('deprecation.showPyplotGlobalUse', False)

st.write("""
# Волошенко Артём Диплом
Разработка интернет приложения для прогнозирования цен на недвижимость с использованием технологий искусственного интеллекта
""")
st.write('---')


X_log = pd.read_csv('recreate_app_dataset.csv')
Y_log = pd.read_csv('y_app_dataset.csv')

X = pd.read_csv('recreate_real_data.csv')



st.sidebar.header('Введите параметры дома')


def user_input_features():
    TotalSF = st.sidebar.slider('Общая площадь дома, м2', float(X.TotalSF.min()*0.09290304), float(
        X.TotalSF.max()*0.09290304), float(X.TotalSF.mean())*0.09290304)
    OverallQual = st.sidebar.slider('Состояние дома', int(X.OverallQual.min()),
                                    int(X.OverallQual.max()), int(X.OverallQual.mean()))
    GarageArea = st.sidebar.slider('Площадь гаража, м2', float(X.GarageArea.min()*0.09290304), float(
        X.GarageArea.max()*0.09290304), float(X.GarageArea.mean())*0.09290304)
    GrLivArea = st.sidebar.slider('Общая жилая плошадь, м2', float(X.GrLivArea.min()*0.09290304),
                                  float(X.GrLivArea.max()*0.09290304), float(X.GrLivArea.mean())*0.09290304)
    GarageYrBlt = st.sidebar.slider('Год постройки гаража', int(X.GarageYrBlt.min()),
                                    int(X.GarageYrBlt.max()), int(X.GarageYrBlt.mean()))
    Fireplaces = st.sidebar.slider('Количество каминов', int(X.Fireplaces.min()), int(
        X.Fireplaces.max()), int(X.Fireplaces.mean()))
    YearBuilt = st.sidebar.slider('Год постройки дома', int(X.YearBuilt.min()),
                                  int(X.YearBuilt.max()), int(X.YearBuilt.mean()))

    lam = 0.15
    data_log = {'TotalSF': boxcox1p(TotalSF/0.09290304, lam),
                'OverallQual': boxcox1p(OverallQual, lam),
                'GarageArea': boxcox1p(GarageArea/0.09290304, lam),
                'GrLivArea': boxcox1p(GrLivArea/0.09290304, lam),
                'GarageYrBlt': boxcox1p(GarageYrBlt, lam),
                'Fireplaces': boxcox1p(Fireplaces, lam),
                'YearBuilt': boxcox1p(YearBuilt, lam),
                }
    data = {'TotalSF': (TotalSF),
            'OverallQual': (OverallQual),
            'GarageArea': (GarageArea),
            'GrLivArea': (GrLivArea),
            'GarageYrBlt': (GarageYrBlt),
            'Fireplaces': (Fireplaces),
            'YearBuilt': (YearBuilt),

            }
    features_log = pd.DataFrame(data_log, index=[0])
    features = pd.DataFrame(data, index=[0])
    return features_log, features


df_log, df = user_input_features()


df['TotalSF'] = int(df['TotalSF'])
df = df.rename(columns={'TotalSF': 'Общ_Площадь', 'OverallQual': 'Качество', 'Fireplaces': 'Камины',
               'GarageArea': 'Площадь_Гаража', 'GarageYrBlt': 'Год_гараж', 'GrLivArea': 'Жилая площадь', 'YearBuilt': 'Год_постройки'})

st.header('Введенные параметры')
st.write(df)
st.write('---')


n_folds = 5


lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))
ENet = make_pipeline(RobustScaler(), ElasticNet(
    alpha=0.0005, l1_ratio=.9, random_state=3))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin=55, bagging_fraction=0.8,
                              bagging_freq=5, feature_fraction=0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf=6, min_sum_hessian_in_leaf=11)


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]

    
        for model in self.models_:
            model.fit(X, y)

        return self

    def predict(self, X):
        predictions = column_stack([
            model.predict(X) for model in self.models_
        ])
        return mean(predictions, axis=1)


averaged_models = AveragingModels(models=(ENet, model_lgb, KRR, lasso))

avm = averaged_models.fit(X_log, Y_log)

prediction = avm.predict(df_log)

lam = 0.15
st.header('Такой дом предположительно будет стоить')
st.write(round(exp(prediction[0]), 2), '$')
st.write('---')



