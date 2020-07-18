#Need to keep analysis in relative terms, normalize and scale data
#Price / volume of diff coins will be different

#imports will go here
import pandas as pd 
import os
from sklearn import preprocessing
from collections import deque
import random
import numpy as np


def preprocess_df(df):
    df = df.drop('future', 1) #only needed future col to generate target
    #Normalizing price changes to percent value.
    for col in df.columns:
        if col != 'target':
            df[col] = df[col].pct_change()
            df.dropna(inplace=True)
            #Scaling 
            df[col] = preprocessing.scale(df[col].values)
    df.dropna(inplace=True) #just in case ;)

    sequential_data = []
    #Will pop out vals once maxlen is reached
    prev_days = deque(maxLen=SEQUENCE_LEN)
    
    for i in df.values: #list of lists
        prev_days.append([x for x in i[:-1]]) #Each col in list of lists, exluding target col
        if len(prev_days) == SEQUENCE_LEN:
            sequential_data.append([np.array(prev_days), i[-1]])

    random.shuffle(sequential_data)

#Classifying function
def classify(current, future):
    #Train network so that 1 is a good thing, e.g. should buy bc price goes up in this scenario
    if float(future) > float(current):
        return 1
    else: return 0



#Constants
SEQUENCE_LEN = 60   #minutes
FUTURE_PERIOD_PREDICTION = 3    #periods forward to predict
RATIO_TO_PREDICT = 'LTC-USD'    #desired ratio to predict

df = pd.read_csv('crypto_data/LTC-USD.csv', names=['time', 'low', 'high', 'open', 'close', 'vol'])


#Need to join all df's on shared axis of time
merged = pd.DataFrame()
#files to use
ratios = ['BTC-USD', 'LTC-USD', 'ETH-USD', 'BCH-USD']

for ratio in ratios:
    dataset = f"crypto_data/{ratio}.csv"

    df = pd.read_csv(dataset, names=['time', 'low', 'high', 'open', 'close', 'volume'])
    #print(df.head()) 
    df.rename(columns={'close':f"{ratio}_close", 'volume':f"{ratio}_volume"}, inplace=True)

    #Set time col as index and only keep closing price and volume
    df.set_index('time', inplace=True)
    df = df[[f"{ratio}_close", f"{ratio}_volume"]]
    

    #Merging cols together
    if len(merged) == 0: 
        merged = df
    else:
        merged = merged.join(df)

merged.fillna(method='ffill', inplace=True) #if gaps in data, use prev known mean vals to fill
merged.dropna(inplace=True)


#shift is to move a certain amount of periods into the future
merged['future'] = merged[f"{RATIO_TO_PREDICT}_close"].shift(-FUTURE_PERIOD_PREDICTION)
#map classify func to create new col called target
merged['target'] = list(map(classify, merged[f"{RATIO_TO_PREDICT}_close"], merged['future']))
#check curr price and 3 periods in future.
#print(merged[[f"{RATIO_TO_PREDICT}_close", 'future', 'target']].head(10))


#Need to build sequences and balance, normalize data.
#Need to scale data, out of sample, etc.

#all times, ordered
times = sorted(merged.index.values)
last_5pct = times[-int(0.05)*len(times)] #last 5%, for threshhold
#print(last_5pct)

#use validation data anywhere where timestamp is greater than val of last 5 pct
validation_merged = merged[(merged.index >= last_5pct)]
merged = merged[(merged.index < last_5pct)]

#Want a func to set train x and train y

preprocess_df(merged)

# train_x, train_y = preprocess_df(merged)
# validation_x, validation_y, = preprocess_df(validation_merged)


