import pandas as pd
import numpy as np
import collections
import statistics

# Import the csv data
EPL_data = pd.read_csv('Training_Data.csv')
EPL_data['Year'] = EPL_data['Date'].str[-2:]

# Import teams sv
team_names = pd.read_csv('Team_Names.csv')
teamList = team_names['Team_Name'].tolist()

# get list of all teams
def get_teamlist():
    return teamList

# get annual vectors for each team
def getAnnualTeamData(teamName, year, filter=[True, True,True,True,True,True,True,True,True,True,True,True,True,True]):
    year = str(year)[-2:]
    annual_data = EPL_data[EPL_data['Year'] == year]

    # num goals scored in wins and losses
    gamesHome = annual_data[annual_data['HomeTeam'] == teamName]
    totalGoalsScored = gamesHome['FTHG'].sum()
    gamesAway = annual_data[annual_data['AwayTeam'] == teamName]
    totalGames = gamesHome.append(gamesAway)
    numGames = len(totalGames.index)
    # total goals scored
    totalGoalsScored += gamesAway['FTAG'].sum()
    # total goals allowed
    totalGoalsAllowed = gamesHome['FTAG'].sum()
    totalGoalsAllowed += gamesAway['FTHG'].sum()

    # discipline: total red cards, total yellow cards
    totalYellowCards = gamesHome['HY'].sum()
    totalYellowCards += gamesAway['AY'].sum()
    totalRedCards = gamesHome['HR'].sum()
    totalRedCards += gamesAway['AR'].sum()

    # total fouls
    totalFouls = gamesHome['HF'].sum()
    totalFouls += gamesAway['AF'].sum()

    # total Corners
    totalCorners = gamesHome['HC'].sum()
    totalCorners += gamesAway['AC'].sum()

    # shots per game (spg) = total shots / total games
    totalShots = gamesHome['HS'].sum()
    # avg shots per game
    totalShots += gamesAway['AS'].sum()
    if numGames != 0:
        spg = totalShots / numGames
    # avg shots allowed per game
    totalShotsAgainst = gamesHome['AS'].sum()
    totalShotsAgainst += gamesAway['HS'].sum()
    if numGames != 0:
        sag = totalShotsAgainst / numGames

    # Games Won Percentage = Games Won / (Games Won + Games Lost)
    gamesWon = annual_data[annual_data['Winner'] == teamName]
    gamesLost = annual_data[annual_data['Loser'] == teamName]
    numGamesWon = len(gamesWon.index)
    numGamesLost = len(gamesLost.index)
    if numGames != 0:
        gamesWonPercentage = numGamesWon / numGames

    # Defense stats
    # Goalie Saves = Shots on Goal - Goal Scored
    totalShotsOnGoal = gamesHome['HST'].sum()
    totalShotsOnGoal += gamesAway['AST'].sum()
    goalieSaves = totalShotsOnGoal - totalGoalsAllowed

    # Saves Percentage = Goalie Saves / Shots on Goal
    if totalShotsOnGoal != 0:
        savesPercentage = goalieSaves / totalShotsOnGoal

        # Saves Ratio = Shots On Goal / Goalie Saves
    if goalieSaves != 0:
        savesRatio = totalShotsOnGoal / goalieSaves

    # Offense stats
    # Scoring Percentage = (Scoring Attempts - Goals Scored ) / Scoring Attempts
    if totalShots != 0:
        scoringPercentage = (totalShots - totalGoalsScored) / totalShots

        # Scoring Ratio = Shots On Goal / Goals Scored
    if totalGoalsScored != 0:
        scoringRatio = totalShotsOnGoal / totalGoalsScored

    if numGames == 0:  # if team not in dataset
        gamesWon = 0
        gamesLost = 0
        totalGoalsScored = 0
        totalGoalsAllowed = 0
        totalYellowCards = 0
        totalRedCards = 0
        totalFouls = 0
        totalCorners = 0
        spg = 0
        sag = 0
        gamesWonPercentage = 0
        goalieSaves = 0
        savesPercentage = 0
        savesRatio = 0
        scoringPercentage = 0
        scoringRatio = 0


    data = [totalGoalsScored, totalGoalsAllowed, totalYellowCards, totalRedCards,
            totalFouls, totalCorners, spg, sag, gamesWonPercentage, goalieSaves, savesPercentage, savesRatio,
            scoringPercentage, scoringRatio]

    awser = list()
    for feat in range(0, len(filter)):
        if filter[feat] == True:
            awser.append(data[feat])
    return awser


# create annual dict of one team
def createAnnualDict(year, filter=[True, True,True,True,True,True,True,True,True,True,True,True,True,True]):
    annualDictionary = collections.defaultdict(list)
    for team in teamList:
        team_vector = getAnnualTeamData(team, year, filter)
        annualDictionary[team] = team_vector
    return annualDictionary


# data from from training
def getTrainingData(years, filter=[True, True,True,True,True,True,True,True,True,True,True,True,True,True]):
    totalNumGames = 0
    for year in years:
        year = str(year)[-2:]
        annual = EPL_data[EPL_data['Year'] == year]

        totalNumGames += len(annual.index)
    numFeatures = filter.count(True)
    #numFeatures = len(getAnnualTeamData('Arsenal',2015)) #random team, to find dimensionality
    xTrain = np.zeros(( totalNumGames, numFeatures))
    yTrain = np.zeros(( totalNumGames ))

    indexCounter = 0

    for year in years:
        year = str(year)[-2:]
        team_vectors = createAnnualDict(year, filter)
        #print(team_vectors)
        annual = EPL_data[EPL_data['Year'] == year]
        numGamesInYear = len(annual.index)
        xTrainAnnual = np.zeros(( numGamesInYear, numFeatures))
        yTrainAnnual = np.zeros(( numGamesInYear ))
        counter = 0
        #Table with the size of date
        for index, row in annual.iterrows():
            h_team = row['HomeTeam']
            h_vector = team_vectors[h_team]
            a_team = row['AwayTeam']
            a_vector = team_vectors[a_team]
            winner = row['Winner']
            loser = row['Loser']
            diff = [a - b for a, b in zip(h_vector, a_vector)]
            if h_team == winner and a_team == loser:
                if len(diff) != 0:
                    xTrainAnnual[counter] = diff
                yTrainAnnual[counter] = 1
            # the opposite of the difference of the vectors should be a true negative, where team 1 does not win
            elif winner =="None" and loser == "None":
                if len(diff) != 0:
                    xTrainAnnual[counter] = diff
                yTrainAnnual[counter] = 0.5
            elif h_team == loser and a_team == winner:
                if len(diff) != 0:
                    xTrainAnnual[counter] = diff
                yTrainAnnual[counter] = 0
            counter += 1
        xTrain[indexCounter:numGamesInYear+indexCounter] = xTrainAnnual
        yTrain[indexCounter:numGamesInYear+indexCounter] = yTrainAnnual
        indexCounter += numGamesInYear
    return xTrain, yTrain


#years = [2017]
#xTrain, yTrain = getTrainingData(years)

############################################################################

# Auxiliar if game and ponts
def get_real_data(years):
    totalNumGames = 0
    for year in years:
        year = str(year)[-2:]
        annual = EPL_data[EPL_data['Year'] == year]
        totalNumGames += len(annual.index)

    total = list()
    pontos = list()
    for year in years:
        year = str(year)[-2:]
        annual = EPL_data[EPL_data['Year'] == year]
        counter = 0
        #Table with the size of date
        for index, row in annual.iterrows():
            h_team = row['HomeTeam']
            a_team = row['AwayTeam']
            winner = row['Winner']
            loser = row['Loser']
            if h_team == winner and a_team == loser:
                total.append([winner, loser])
                pontos.append(3)
            # the opposite of the difference of the vectors should be a true negative, where team 1 does not win
            elif winner == "None" and loser == "None":
                total.append([h_team, a_team])
                pontos.append(1)
            elif h_team == loser and a_team == winner:
                total.append([loser, winner])
                pontos.append(0)
            counter += 1
    return total, pontos

# Organize games and ponts in a table
def mean_data_base(inicio=2015, fim=2016):
    #years = range(inicio, fim)
    years =[fim]
    jogos, pontos = get_real_data(years)
    num_jogos = len(jogos)

    all = dict()
    ponts = list()
    for time in get_teamlist():
        ponts.clear()
        for d in range(0, num_jogos):
            if jogos[d][0] == time or jogos[d][1] == time:
                ponts.append(pontos[d])
        all[time] = ponts.copy()
    test = list()
    for t in get_teamlist():
        #test.append([t, statistics.mean(all[t])])
        test.append([t, sum(all[t])])
    test.sort(key=lambda row: row[1], reverse=True)
    return test

#years = [2017]
#print(mean_data_base(2016))



