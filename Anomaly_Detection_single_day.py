'''
@author: Katie Rouse
May 10th, 2019

After running the LSTM_Prediction_Build.py and saved the output as: meter_number.csv
Will use this as the prediction input for running this code
Will use the "raw" IEE data from region_first_reading.csv as the test data
'''

import pandas as pd
import numpy as np
import numpy
import os
from statsmodels.tsa.arima_model import ARIMA
import datetime
#import timedelta


def anomaly_detection_single_day(region_code, meter, start, end):


    np.random.seed(7)

    #region_code = "Huntsville_0099"
    region = region_code
    region_reading = str(region) + '\\'+ str(region) + '_first_reading.csv'
    
    #df = pd.read_csv("nashville_0158_first_reading.csv",names=['endtime','spid','kwh'], engine='python')
    df = pd.read_csv(region_reading ,names=['endtime','spid','kwh'], engine='python')
    
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
    #print(df.head())
 
    
    ## date range to input that want the detected anomalies for
    startDate = start
    startDate2 = datetime.datetime.strptime(startDate, "%Y-%m-%d %H:%M:%S")
    
    endDate = end
    endDate2 = datetime.datetime.strptime(endDate, "%Y-%m-%d %H:%M:%S")
    num_days = abs((endDate2-startDate2).days)+1
    
    daterange = pd.date_range(start, end)
 
    
    
    powerDF = pd.DataFrame([])
    
    # Empty dataframe for keeping ARIMA residual values
    myDF = pd.DataFrame([])
    
    # Empty dataframe for keeping predicted kWh values
    predDF = pd.DataFrame([])

    i = True
    while i == True:
        # Read in power data that the user selected
        meter = meter.strip('.csv')
        colData = pd.DataFrame(df, columns=[meter])
        # print("colData:" ,colData)
    
        # Read predicted y
        # Output csv file from LSTM_Build.py is used here
        
        
        title = '%s' % (meter)
        
        y_hat = pd.read_csv(region+ '\\'+ 'Prediction\\' + title+
                            '.csv', usecols=[meter], engine='python')
    
    
        # Getting the list of meter names for power data and predicted data
        colName = list(colData.columns.values)
        #print("colName:" , colName)
        #predName = list(y_hat.columns.values)
    
        # Adding datetime information to the power data
        x = pd.Series(colData.iloc[:, 0])
    
        time = pd.date_range(start=pd.datetime(2015, 1, 1), freq='h', periods=len(x))
        x_time = pd.Series(x.values, index=time)
    
        # Index the power data with time by user specified time range (one-day interval)
        one_day = datetime.timedelta(1)

      
        for day in daterange:
            # next day
           
            startDay = day
            endDay = day + one_day
            userTime = x_time[startDay:endDay]
        
            # Adding datetime information to the predicted kWh
            x_pred = pd.Series(y_hat.iloc[:,0])
            time_pred = pd.date_range(start=pd.datetime(2017,1,1), freq='h', periods=len(x_pred))
            time_pred_data = pd.Series(x_pred.values, index=time_pred)
        
            # Run ARIMA on the power data; looking at 1 hr differencing
            model1 = ARIMA(userTime, order=(1, 1, 0))
        
            # Try and Catch for Singular Matrix Error
            
            try:
                model1_fit = model1.fit(disp=0)
            except numpy.linalg.linalg.LinAlgError as err:
                if ValueError:
                    print("Too many zeros in meter:", colName)
                    i = False
                    continue
                if 'Singular matrix' in str(err):
                    print('too many zeros in meter:', colName)
                else:
                    raise
        
            power = pd.DataFrame(x_time, columns=colName)
        
            # Save the ARIMA residual values into a new dataframe
            residual = pd.DataFrame(model1_fit.resid, columns=colName)
        
            # Save the predicted kwh value into a new dataframe
            pred_df = pd.DataFrame(time_pred_data, columns=colName)
        
            # Transpose both dataframe to have meter names as rows and time as columns
            pow_transposed = np.transpose(power)
            res_transposed = np.transpose(residual)
            pred_transposed = np.transpose(pred_df)
        
            
        
        
            # Append the transposed data into empty dataframe that was generated beforehand
            powerDF = powerDF.append(pow_transposed)
            myDF = myDF.append(res_transposed)
            predDF = predDF.append(pred_transposed)
        
            # Empty dataframe to keep the anomaly detection results
            outputData = pd.DataFrame(columns=['Time', 'MeterName', 'kWh', 
                                               'Predicted kWh', 'Difference',
                                               'Percent Difference'])
            
            # Iterate through meters
            for i in range(0, len(myDF)):
                meterName = myDF.index[i]
            
                kwh = powerDF.iloc[i, 1:]
            
                # Residue value for a specific meter
                col = myDF.iloc[i, 1:]
                col.fillna(0, inplace=True)
                # Voltage value for a specific meter
                # volt = voltageDF.iloc[i, 1:]
            
                # Y hat
                pred = predDF.iloc[i, 1:]
            
                # Calculate the mean and sigma by meter
                mean = np.mean(col)
                sigma = np.std(col)
            
                # Using the mean and sigma obtained from above, get the upper and lower limits
                upperBound = mean + (3.0 * sigma)
                lowerBound = mean - (3.0 * sigma)
            
                # Threshold for LSTM predicted values
                difference_upperBound = 200
                difference_lowerBound = -200
            
                # If the residue value is less than or greater than the limits, we save them in a result dataframe
                #print("forming outputData frame")
            # If the residue value is less than or greater than the limits, we save them in a result dataframe
                for j in range(0, len(col)):
                    if (col[j] > upperBound).all() or (col[j] < lowerBound).all():
                        #print((kwh[col.index[j]])-(pred[col.index[j]]))
                        if ((kwh[col.index[j]])-(pred[col.index[j]]) > difference_upperBound).all() or ((kwh[col.index[j]])-(pred[col.index[j]]) < difference_lowerBound).all():
                            entry = [{'Time': col.index[j], 
                                      'MeterName': meterName, 
                                      'kWh': kwh[col.index[j]],
                                      'Predicted kWh':pred[col.index[j]], 
                                      'Difference': (kwh[col.index[j]])-(pred[col.index[j]]),
                                      'Percent Difference': round(((kwh[col.index[j]]-pred[col.index[j]])/pred[col.index[j]]) * 100,3)}]
                            outputData = outputData.append(entry, ignore_index=True)
                     
                 
                dayData = outputData.copy()
                
                ## drops columns 'kWh', 'Predicted kWh' and 'Difference'
                dayData.drop(dayData[((dayData['Percent Difference'] > -50) & (dayData['Percent Difference'] < 30))].index, inplace = True)
            
                dayData.drop(['kWh', 'Predicted kWh', 'Difference'], axis = 1, inplace=True)
    
                
                if(len(dayData) > 0):
                        print(" ***************Anomalies Detected *******************")
                        print(dayData.sort_values(['Time']))
                        newpath = region+ '\\'+'Jerome_Anomalies'+ '\\' +r'%s_Anomalies' % (str(meter))
                      
                        if not os.path.exists(newpath):
                            os.makedirs(newpath)
                        path = '%s_Anomalies\\' % (str(meter))
                        title = day.strftime("%Y-%m-%d")
                        dayData.to_csv(region + '\\'+ 'Jerome_Anomalies' + '\\'+path+title+".csv")
                    #one_day += datetime.timedelta(1)
                    
                # clear dataframes    
                outputData = pd.DataFrame(columns=['Time', 'MeterName', 'kWh', 
                                                   'Predicted kWh', 'Difference',
                                                   'Percent Difference'])
                #dayData = []
                powerDF = pd.DataFrame([])
                myDF = pd.DataFrame([])
                predDF = pd.DataFrame([])
            i = False
            #print("\n")
        

#meter_test = ['0099070','0099150']
#directory = 'Huntsville_0099\Prediction'
#for file in os.listdir(directory):
#    anomaly_detection('Huntsville_0099', file, start = '2017-01-01 00:00:00', end = '2017-01-31 23:59:00')
        
            

#meter_test = ['0138819', '0138809','0138702', '0138701']
#meter_test =['0138819']
#directory = 'Memphis_0138\Prediction'
#for file in meter_test:
#    anomaly_detection_single_day('Memphis_0138', file, start = '2017-01-01 00:00:00', end = '2017-01-31 23:59:00')
#    print("next meter")