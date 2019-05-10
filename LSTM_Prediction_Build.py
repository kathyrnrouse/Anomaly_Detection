'''
@author: Katie Rouse
May 10th, 2019

This code creates the function that will create the prediciton for a set of meters in a given range
The output of this function will be '.csv' files with 'meter_number/ Prediction'
'''


import os
# os.environ["THEANO_FLAGS"] = "mode=FAST_RUN, device=gpu, floatX=float32, lib.cnmem=0.8"

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from math import sqrt
import timeit
import datetime




## @lstm_prediction_build = function to build predictions using LSTM
## @ region = region being used for a given test, 
## @IEE_data = The name of the .csv file that contains the last IEE reading for given meters

## expected input example for @region:  'Memphis_0138' where the region is followed by the region code
## expected input example for @IEE_data: 'Memphis_0138\Memphis_0138_last_reading.csv'
## make sure to include the directory/ correct path name

def lstm_prediction_build(region, IEE_data):
    np.random.seed(7)
    
    outputDF = pd.DataFrame([])
    
    ## The comment below is if you do not want to use the function and want to test a file directly
    #df = pd.read_csv("Memphis_0138\Memphis_0138_last_reading.csv",names=['endtime','spid','kwh'], engine='python')

    file = str(region) + '\\' + str(IEE_data)
    df = pd.read_csv(file, names=['endtime','spid','kwh'], engine = 'python')
    
    df = pd.DataFrame(df)

    ## drop duplicates, this will be done for IEE because of reading groups
    df = df.drop_duplicates(subset=['endtime','spid'], keep='first').reset_index(drop=True)
    df['endtime'] = df['endtime'].astype(str)
    
    #pivot the dataframe with index time
    pivot_table = df.pivot(index='endtime', columns='spid', values='kwh')
    pivot_table = pivot_table.fillna(0)
    
    ## creates a list of distinctimes and stores the number of time intervals in hour 
    interval = df['endtime'].unique()
    interval.sort()
    interval = interval[0]
    interval = datetime.datetime.strptime(interval, "%Y-%m-%d %H:%M:%S.%f")
    table_time =interval
    interval = interval.minute
    interval = 60//interval
    
    # convet index to int value so can sum up to hour
    pIndex = pivot_table.reset_index()
    
    pSum = pIndex.groupby(pIndex.index//interval).sum()
    time = pd.date_range(start = pd.datetime(table_time.year, table_time.month, table_time.day), freq = 'h',periods=len(pSum))
    df = pSum.set_index(time+1)
    print(df.head())
    
    names = df.columns.values
    
    
    start = timeit.default_timer()
    for i in range(0, len(df.columns)-1 ):
        # X
        print(i)
    
        # Grabs meter power data and temperature data
        IEE_power_temp_name = df.iloc[:,i]
        #IEE_data = IEE_power_temp_name.values
        IEE_data = pd.DataFrame(IEE_power_temp_name)
        colName = names[i]
    
        print(colName)
    
        # Data preprocessing is required in order to meet the Keras dataset format requirement
        # 2015-2017
        # Total Rows = 26300, 70% is 18345.6
        remainderX = int(len(IEE_data)*(0.7)) % 168
        train_size = int((len(IEE_data)*(0.7)))-remainderX
         
        # 2016
        #test_size = len(IEE_data) - train_size
        #Total - train_size
        test_size = (len(IEE_data)-train_size) % 168
    
        print(test_size)
    
        # IEE used for both training and testing purposes (x)
        # need to start at 33 for train size so that trainX is divisible by 168 (used in flattening)
        trainX = IEE_data.iloc[:train_size, :]
        testX = IEE_data.iloc[train_size:len(IEE_data)-test_size, :]
        
        
        # Normalize X for deep learning
        IEE_norm_X = MinMaxScaler(feature_range=(0, 1))
        IEE_trainX_norm = IEE_norm_X.fit_transform(trainX)
        IEE_testX_norm = IEE_norm_X.transform(testX)
        
        print("about to read the csv file")
        
        # Y
        #IEE_power_temp_y = pd.read_csv("0102_Oracle_threeyears_pivotted.csv", usecols=[i], engine='python')
        IEE_power_temp_y = df.values
        IEE_data_y = pd.DataFrame(IEE_power_temp_y)
        
        # IEE used for both training and testing purposes (y)
        #trainY, testY = IEE_data_y[48:train_size, :], IEE_data_y[train_size:len(IEE_data_y)-48, :]
        trainY, testY = IEE_data_y.iloc[:train_size, :], IEE_data_y.iloc[train_size:len(IEE_data_y)-test_size, :]
        #print(len(trainY))
    
        # Normalize X for deep learning
        IEE_norm_X = MinMaxScaler(feature_range=(0, 1))
        IEE_trainX_norm = IEE_norm_X.fit_transform(trainX)
        IEE_testX_norm = IEE_norm_X.transform(testX)
    
        # Y
        IEE_power_temp_y = df.iloc[:,i]
        IEE_data_y = pd.DataFrame(IEE_power_temp_y.values)
    
        # IEE used for both training and testing purposes (y)
        # subtracting dimensions for flattening again like with trainX, testX
        trainY, testY = IEE_data_y.iloc[:train_size, :], IEE_data_y.iloc[train_size:len(IEE_data_y)-test_size, :]
    
        # Normalize y for deep learning
        IEE_norm_y = MinMaxScaler(feature_range=(0, 1))
        IEE_trainY_norm = IEE_norm_y.fit_transform(trainY)
        IEE_testY_norm = IEE_norm_y.transform(testY)
    
        # Reshape data format
        # Dividing the data by hours per week, 168 = 24*7
        IEE_trainX_norm_reshape = np.reshape(IEE_trainX_norm, (int(IEE_trainX_norm.shape[0]/168), 168, IEE_trainX_norm.shape[1]))
        IEE_testX_norm_reshape = np.reshape(IEE_testX_norm, (int(IEE_testX_norm.shape[0]/168), 168, IEE_testX_norm.shape[1]))
        IEE_trainY_norm_reshape = np.reshape(IEE_trainY_norm, (int(IEE_trainY_norm.shape[0]/168), 168, 1))
        IEE_testY_norm_reshape = np.reshape(IEE_testY_norm, (int(IEE_testY_norm.shape[0]/168), 168, 1))
    
    
    
        # Keras Deep learning model
        model = Sequential()
        model.add(LSTM(60, input_shape=(168, 1), return_sequences=True))
        #model.add(LSTM(60, input_shape=(1999, 1), return_sequences=True))
        model.add(LSTM(30, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        print(model.summary())
    
        # if using GPU use batch size of 1000, if not then can reduce batch size
        model.fit(IEE_trainX_norm_reshape, IEE_trainY_norm_reshape, batch_size=1000, epochs=80, verbose=1)
        train_predict = model.predict(IEE_trainX_norm_reshape)
        test_predict = model.predict(IEE_testX_norm_reshape)
    
    
        train_predict_sq = np.squeeze(train_predict).flatten()
        train_predict_rs = np.reshape(train_predict_sq, (train_predict_sq.shape[0], 1))
    
        test_predict_sq = np.squeeze(test_predict).flatten()
        test_predict_rs = np.reshape(test_predict_sq, (test_predict_sq.shape[0], 1))
    
        train_predict_inverse = IEE_norm_y.inverse_transform(train_predict_rs)
    
        # Our predicted y
        test_predict_inverse = IEE_norm_y.inverse_transform(test_predict_rs)
    
        print('RMSE for Testing Set: ', sqrt(mean_squared_error(testY, test_predict_inverse)))
        print('MAE for Testing Set: ', mean_absolute_error(testY, test_predict_inverse))
        print('R2 for Testing: ', r2_score(testY, test_predict_inverse))
    
    
    
    
         
        df_test = pd.DataFrame(test_predict_inverse)
        outputDF = outputDF.append(df_test)
        outputDF[colName]=df_test
        
        
        #save the model
        #save the weight of model
        #https://keras.io/getting-started/faq/
        import matplotlib.pyplot as plt
        plt.plot(model.history.history['loss'])
        plt.show()
        
        #https://conda.io/docs/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file
        
        stop= timeit.default_timer()
        
        print("Time: ", stop-start)
        #outputDF.rename(columns={0:meter_num}, inplace=True)
        outputDF.drop(columns =0)
        
        # This csv file will get used in Anomaly_Detection.py
    
       
        newpath = '%s' % (str(region)+'\\Prediction\\')
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        #path = '%s\\' % (str(colName))
        title = '%s' % (colName)
        outputDF.to_csv(newpath+title+'.csv', index=False)
        
        # outputDF = pd.DataFrame(test_predict_inverse)
        # outputDF[colName] = outputDF
    stop= timeit.default_timer()
    
    print("Time: ", stop-start)
    
    # This csv file will get used in Anomaly_Detection.py
    
