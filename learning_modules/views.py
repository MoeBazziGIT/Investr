from django.shortcuts import render
import json

import pandas as pd
import numpy as np
import math
import requests
import datetime
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression

api_key = "OjljOWZhNWRjYjhmMGVmMzI2YWQ5NmE3MTE5ZWU5NzYz"

def predict_values(ticker):
    data_elements = {
        "ebitda",
        "bookvaluepershare",
        "grossmargin",
        "roe",
        "basiceps"
    }
    date_now = datetime.datetime.now();
    end_date = date_now.strftime("%Y-%m-%d")
    start_date = (date_now - relativedelta(years=10)).strftime("%Y-%m-%d")
    data = {}

    # get prices from last 10 years
    url = "https://api.intrinio.com/prices?identifier="+ticker+"&end_date="+end_date+"&start_date="+start_date+"&frequency=quarterly"+"&api_key="+api_key
    price_data = requests.get(url).json()

    for item in price_data["data"]:
        data[item["date"][:-3]] = {
            "adj_close": item["adj_close"]
        }

    # add other data
    for element in data_elements:
        url = "https://api.intrinio.com/historical_data?identifier="+ticker+"&item="+element+"&api_key="+api_key
        element_data = requests.get(url).json()
        for item in element_data["data"]:
            item["date"] = item["date"][:-3]
            if (item["date"] in data):
                if (item["value"] == "nm"): item["value"] = np.nan
                data[item["date"]][element] = item["value"]

    # model data to turn into dataframe
    d = {
        "date": [],
        "adj_close": [],
        "ebitda": [],
        "bookvaluepershare": [],
        "grossmargin": [],
        "roe": [],
        "basiceps": []
    }

    dates = []

    for key in data:
        dates.append(key)

    dates = sorted(dates)

    for date in dates:
        d["date"].append(date)

        if ("adj_close" in data[date]): d["adj_close"].append(data[date]["adj_close"])
        else: d["adj_close"].append(np.nan)

        if ("ebitda" in data[date]): d["ebitda"].append(data[date]["ebitda"])
        else: d["ebitda"].append(np.nan)

        if ("bookvaluepershare" in data[date]): d["bookvaluepershare"].append(data[date]["bookvaluepershare"])
        else: d["bookvaluepershare"].append(np.nan)

        if ("grossmargin" in data[date]): d["grossmargin"].append(data[date]["grossmargin"])
        else: d["grossmargin"].append(np.nan)

        if ("roe" in data[date]): d["roe"].append(data[date]["roe"])
        else: d["roe"].append(np.nan)

        if ("basiceps" in data[date]): d["basiceps"].append(data[date]["basiceps"])
        else: d["basiceps"].append(np.nan)



    df = pd.DataFrame(data=d)
    order = ['date','adj_close', 'ebitda', 'bookvaluepershare', 'grossmargin', 'roe', 'basiceps']
    df = df[order]
    df = df.dropna()

    forecast_col = 'adj_close'
    forecast_out = int(math.ceil(0.05*len(df)))

    df['label'] = df[forecast_col].shift(-forecast_out)
    df = df.dropna()

    # start machine learning!
    X = np.array(df.drop(['label', 'date'], 1))
    y = np.array(df['label'])

    # # fig, ax = plt.subplots()
    # # ax.plot(df['bookvaluepershare'].values, df['adj_close'].values)
    # # ax.grid()
    # # plt.show()

    X = preprocessing.scale(X)
    y = df['label']
    # X = preprocessing.normalize(X,'l1',0)
    # # y = preprocessing.scale(y.reshape(-1,1))
    # # y = preprocessing.normalize(y.reshape(-1,1), 'l1', 0)

    clf = LinearRegression()

    accuracy = 0;
    while (accuracy < 0.8):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test,y_test)

    prices = {}
    for i, row in df.iterrows():
        prices[row['date']] = row['adj_close']

    labels = list(df['label'])
    prices[(date_now+relativedelta(months=3)).strftime("%Y-%m")] = labels[-2]
    prices[(date_now+relativedelta(months=6)).strftime("%Y-%m")] = labels[-1]


    return (prices, round(accuracy,2), forecast_out);

def index(request):
    return render(request, 'learning_modules/index.html')

def learning(request):
    return render(request, 'learning_modules/learning.html')

def compare(request):
    company1 = request.POST.get("Country1")
    # company2 = request.POST.get("Country2")

    prices, accuracy, q_out = predict_values(company1.upper())

    context = {'title':company1.upper(), "prices": json.dumps(prices), "accuracy": accuracy, "q_out": q_out}

    return render(request, 'learning_modules/compare.html', context)

def compare_info_form(request):
    return render(request, 'learning_modules/compare_info.html')

def introduction(request):
    return render(request, 'learning_modules/introduction.html')


def add_compare_info(request):
    tickers = []
    pass

def whatIsInvesting(request):
    return render(request, 'learning_modules/whatIsInvesting.html')

def riskAndDiversification(request):
    return render(request, 'learning_modules/riskAndDiversification.html')

def fixedIncome(request):
    return render(request, 'learning_modules/fixedIncome.html')

def stocks(request):
    return render(request, 'learning_modules/stocks.html')

def mutualFunds(request):
    return render(request, 'learning_modules/mutualFunds.html')
