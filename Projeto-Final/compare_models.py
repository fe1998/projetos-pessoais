import numpy as np
from sklearn.model_selection import train_test_split
from get_data_soccer import getTrainingData
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from filter_data import filter_features_avaliable


def acurrancy(y_res, y_ref):
    erro = 0
    acerto = 0
    for i in range(0, len(y_res)):
        if y_res[i] >= 0.55 and y_ref[i] > 0.5:
            acerto = acerto + 1
        elif y_res[i] <= 0.45 and y_ref[i] < 0.5:
            acerto = acerto + 1
        elif 0.45 < y_res[i] < 0.55 and y_ref[i] == 0.5:
            acerto = acerto + 1
        else:
            erro = erro + 1
    acurranc = acerto/(acerto+erro)
    return acurranc

filter_test = filter_features_avaliable(5)
years = range(2005, 2017)
xTrain, yTrain = getTrainingData(years, filter_test)
xTrain, X_test, yTrain, y_test = train_test_split(xTrain, yTrain)
x, y = np.array(xTrain), np.array(yTrain)

#############################
model = LinearRegression().fit(x, y)
y_pred = model.predict(X_test)
print(acurrancy(y_pred, y_test))
#############################
clf = SVR(gamma='scale', C=1.0, epsilon=0.2).fit(x,y)
y_pred= clf.predict(X_test)
print(acurrancy(y_pred,y_test))
#############################
clf2 = GaussianProcessRegressor().fit(x, y)
y_pred = clf2.predict(X_test)
print(acurrancy(y_pred, y_test))
############################
regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(x, y)
y_pred = regressor.predict(X_test)
print(acurrancy(y_pred, y_test))
############################
