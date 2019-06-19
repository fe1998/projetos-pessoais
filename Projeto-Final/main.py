import numpy as np
from get_data_soccer import getAnnualTeamData
from get_data_soccer import createAnnualDict
from get_data_soccer import getTrainingData
from get_data_soccer import get_teamlist
from get_data_soccer import mean_data_base
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from filter_data import filter_features_avaliable
import pandas as pd
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor

def result_all_competition_method(model, filter=[True, True,True,True,True,True,True,True,True,True,True,True,True,True]):
    X_real = list()
    name = list()
    game = list()
    for i in get_teamlist():
        for q in get_teamlist():
            if i != q:
                game.append(i)
                game.append(q)
                name.append(game.copy())
                #ERRO#
                a1 = getAnnualTeamData(i, 2016, filter)
                b1 = getAnnualTeamData(q, 2016, filter)
                diff = [a - b for a, b in zip(a1, b1)]
                #if len(diff) == 0:
                #    print(i,"   ",q)
                X_real.append(diff)
                game.clear()
    y_pred = model.predict(X_real)
    return name, y_pred

def result_all_squads_method(name_order,y_result):
    times = dict()
    ponts = list()
    for tim in get_teamlist():
        ponts.clear()
        for i, q in enumerate(name_order):
            if q[0] == tim or q[1] == tim:
                if y_result[i] >= 0.55:
                    ponts.append(3)
                elif 0.55 > y_result[i] > 0.45:
                    ponts.append(1)
                elif y_result[i] <= 0.45:
                    ponts.append(0)
        times[tim] = ponts.copy()
    return times

#Filter of features
#filter_test = filter_features_avaliable(5)
filter_test = [True, True,True,True,True,True,True,True,True,True,True,True,True,True]

years = [2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]
#years = [2016, 2017]
xTrain, yTrain = getTrainingData(years, filter_test)

x, y = np.array(xTrain), np.array(yTrain)

model = RandomForestRegressor(n_estimators=20, random_state=0).fit(x, y)

times = result_all_squads_method(result_all_competition_method(model,filter_test)[0], result_all_competition_method(model, filter_test)[1])

table = list()
total = list()

for tim in get_teamlist():
    total.clear()
    #print(tim)
    #print(times[tim])
    ponts = sum(times[tim])
    total.append(tim)
    total.append(ponts)
    table.append(total.copy())

table.sort(key=lambda row: row[1], reverse=True)


for hh in table:
    print(hh)

print()

#result = pd.DataFrame(table)
#result.to_excel("classificacaoRandomFlorest.xlsx")

#for h in mean_data_base():
#    print(h)
