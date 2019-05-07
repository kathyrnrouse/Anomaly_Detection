import pandas as pd
import numpy as np

# test data
filename = 'IEE_28Meter_072718.csv'

# convert the chosen file into pandas dataframe
data = pd.read_csv(filename)

startDate = '2016-09-22 00:00:00'
endDate = '2016-09-24 23:00:00'

powerDF = pd.DataFrame([])

# Empty dataframe for keeping predicted kWh values
predDF = pd.DataFrame([])

# Run the for loop for the number of meters that we have
for i in range(0, len(data.columns)):
    # Read in power data that the user selected
    colData = pd.read_csv(filename, usecols=[i], engine='python')

    # Read predicted y
    # Output csv file from TVA_LSTM_Prediction_010918.py is used here
    y_hat = pd.read_csv('LSTM_tva_result_072618.csv', usecols=[i], engine='python')

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
    time_pred = pd.date_range(start=pd.datetime(2016, 1, 1), freq='h', periods=len(x_pred))
    time_pred_data = pd.Series(x_pred.values, index=time_pred)
    time_pred_data_1 = time_pred_data[startDate:endDate]

    power = pd.DataFrame(userTime, columns=colName)

    # Save the predicted kwh value into a new dataframe
    pred_df = pd.DataFrame(time_pred_data_1, columns=colName)

    # Transpose both dataframe to have meter names as rows and time as columns
    pow_transposed = np.transpose(power)
    pred_transposed = np.transpose(pred_df)

    # Append the transposed data into empty dataframe that was generated beforehand
    powerDF = powerDF.append(pow_transposed)
    predDF = predDF.append(pred_transposed)

# Empty dataframe to keep the anomaly detection results
outputData = pd.DataFrame(columns=['Time', 'MeterName', 'kWh', 'Predicted kWh', 'Difference'])

# Iterate through meters
for i in range(0, len(predDF)):
    meterName_1 = powerDF.index[i]
    print('This***', meterName_1)

    kwh = powerDF.iloc[i, 1:]

    # Y hat
    pred = predDF.iloc[i, 1:]

    # Calculate the mean and sigma by meter
    mean = np.mean(pred)
    sigma = np.std(pred)

    # Using the mean and sigma obtained from above, get the upper and lower limits
    # difference_upperBound = mean + (3.0 * sigma)
    # difference_lowerBound = mean - (3.0 * sigma)

    # Threshold for LSTM predicted values
    difference_upperBound = 200
    difference_lowerBound = -200

    for j in range(0, len(pred)):
        if ((kwh[j]) - (pred[j]) > difference_upperBound).all() or ((kwh[j]) - (pred[j]) < difference_lowerBound).all():
            entry = [{'Time': pred.index[j], 'MeterName': meterName_1, 'kWh': kwh[pred.index[j]],
                      'Predicted kWh': pred[j], 'Difference': (kwh[pred.index[j]]) - (pred[j])}]
            outputData = outputData.append(entry, ignore_index=True)


print("\n")

# Print the result dataframe that contains meters with possible anomalies
print("             ******Anomalies Detected******")
foo = (outputData.sort_values(['Time']))
df = foo[['Time','MeterName']]
# Dataframe to csv
df.to_csv('LSTM_RESULT_0923.csv', index=False)
