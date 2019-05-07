import pandas as pd
import numpy as np
import numpy
from statsmodels.tsa.arima_model import ARIMA

# test data
filename = 'IEE_28Meter_072718.csv'
# filename = 'tva_demo_testSet_120617.csv'

# convert the chosen file into pandas dataframe
data = pd.read_csv(filename)


startDate = '2016-12-01 00:00:00'
endDate = '2016-12-02 23:00:00'

powerDF = pd.DataFrame([])

# Empty dataframe for keeping ARIMA residual values
myDF = pd.DataFrame([])

# Empty dataframe for keeping predicted kWh values
predDF = pd.DataFrame([])

# Run the for loop for the number of meters that we have
for i in range(0, len(data.columns)):
    # Read in power data that the user selected
    colData = pd.read_csv(filename, usecols=[i], engine='python')

    # Read predicted y
    # Output csv file from TVA_LSTM_Prediction_010918.py is used here
    y_hat = pd.read_csv('LSTM_tva_result_072618.csv', usecols=[i], engine='python')
    # y_hat = pd.read_csv('JinAustin_DL_result_120617.csv', usecols=[i], engine='python')

    # Getting the list of meter names for power data and predicted data
    colName = list(colData.columns.values)
    predName = list(y_hat.columns.values)

    # Adding datetime information to the power data
    x = pd.Series(colData.iloc[:, 0])
    time = pd.date_range(start=pd.datetime(2011, 1, 1), freq='h', periods=len(x))
    x_time = pd.Series(x.values, index=time)

    # Index the power data with time by user specified time range (one-day interval)
    userTime = x_time[startDate:endDate]

    # Adding datetime information to the predicted kWh
    x_pred = pd.Series(y_hat.iloc[:,0])
    time_pred = pd.date_range(start=pd.datetime(2016,1,1), freq='h', periods=len(x_pred))
    time_pred_data = pd.Series(x_pred.values, index=time_pred)

    # Run ARIMA on the power data; looking at 1 hr differencing
    model1 = ARIMA(userTime, order=(1, 1, 0))

    # Try and Catch for Singular Matrix Error
    try:
        model1_fit = model1.fit(disp=0)
    except numpy.linalg.linalg.LinAlgError as err:
        if ValueError:
            print("Too many zeros in meter:", colName)
            continue
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
outputData = pd.DataFrame(columns=['Time', 'MeterName', 'kWh', 'Predicted kWh', 'Difference'])

# Iterate through meters
for i in range(0, len(myDF)):
    meterName = myDF.index[i]

    kwh = powerDF.iloc[i, 1:]

    # Residue value for a specific meter
    col = myDF.iloc[i, 1:]

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
    for j in range(0, len(col)):
        if (col[j] > upperBound).all() or (col[j] < lowerBound).all():
            print((kwh[col.index[j]])-(pred[col.index[j]]))
            if ((kwh[col.index[j]])-(pred[col.index[j]]) > difference_upperBound).all() or ((kwh[col.index[j]])-(pred[col.index[j]]) < difference_lowerBound).all():
                entry = [{'Time': col.index[j], 'MeterName': meterName, 'kWh': kwh[col.index[j]],
                        'Predicted kWh':pred[col.index[j]], 'Difference': (kwh[col.index[j]])-(pred[col.index[j]])}]
                outputData = outputData.append(entry, ignore_index=True)

print("\n")

# Print the result dataframe that contains meters with possible anomalies
print("             ******Anomalies Detected******")
print(outputData.sort_values(['Time']))

# Dataframe to csv
# outputData.sort_values(['Time']).to_csv('TVA_DL_ARIMA_RESULT_0524.csv', index=False)
