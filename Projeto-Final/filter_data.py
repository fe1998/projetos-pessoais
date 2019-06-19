from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from get_data_soccer import getTrainingData

def filter_features_avaliable(oportunnites=3):
    years = range(2005, 2017)
    xTrain, yTrain = getTrainingData(years)

    # create the RFE model and select 3 attributes
    model1 = LinearRegression() #recursive feature elimination
    rfe = RFE(model1, oportunnites)
    rfe = rfe.fit(xTrain, yTrain)

    # summarize the selection of the attributes
    #print(rfe.support_)
    #print(rfe.ranking_)
    features_check = list()
    for i in rfe.support_:
        features_check.append(i)
    return features_check

print(filter_features_avaliable(5))