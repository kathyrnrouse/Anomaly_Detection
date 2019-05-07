import pandas as pd
import numpy as np
import numpy
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA


# filename = 'tva_demo_testSet_120617.csv'
# filename = 'IEE_Padded_2011_2016_Data_091517_test.csv'
filename = 'IEE_28Meter_072718.csv'

# convert the chosen file into pandas dataframe
data = pd.read_csv(filename)


startDate = '2016-05-22 00:00:00'
endDate = '2016-05-24 23:00:00'

#2016-11-00 00:00:00
powerDF = pd.DataFrame([])

# Empty dataframe for keeping ARIMA residual values
myDF = pd.DataFrame([])

# Empty dataframe for keeping voltage values2016-11-28 00:00:00
# voltageDF = pd.DataFrame([])

# Run the for loop for the number of meters that we have
for i in range(0, len(data.columns)):
    # Read in power data that the user selected
    colData = pd.read_csv(filename, usecols=[i], engine='python')

    # Getting the list of meter names for power data and voltage data
    colName = list(colData.columns.values)
    # Adding datetime information to the power data
    x = pd.Series(colData.iloc[:, 0])
    time = pd.date_range(start=pd.datetime(2011, 1, 1), freq='h', periods=len(x))
    x_time = pd.Series(x.values, index=time)

    # Index the power data with time by user specified time range (one-day interval)
    userTime = x_time[startDate:endDate]

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

    # Transpose both dataframe to have meter names as rows and time as columns
    pow_transposed = np.transpose(power)
    res_transposed = np.transpose(residual)

    # Append the transposed data into empty dataframe that was generated beforehand
    powerDF = powerDF.append(pow_transposed)
    myDF = myDF.append(res_transposed)

    mean_test = np.mean(model1_fit.resid)
    sig_test = np.std(model1_fit.resid)

    # print(mean_test, sig_test)
    # upperBound = mean_test + (3.0 * sig_test)
    # lowerBound = mean_test - (3.0 * sig_test)
    # print(upperBound, lowerBound)
    # plt.title(colName)
    # plt.plot(residual, color='blue',linestyle='-', label='Residue')
    # plt.axhline(y=upperBound, color='red', linestyle='--',label='Threshold')
    # plt.axhline(y=lowerBound, color='red', linestyle='--')
    # plt.grid(True)
    # plt.legend(prop={'size': 12})
    # plt.show()

# Empty dataframe to keep the anomaly detection results
# outputData = pd.DataFrame(columns=['MeterName', 'Time', 'kWh','Residue', 'Voltage'])
outputData = pd.DataFrame(columns=['Time', 'MeterName' , 'kWh','Residue'])

# Iterate through meters
for i in range(0, len(myDF)):
    meterName = myDF.index[i]

    kwh = powerDF.iloc[i, 1:]

    # Residue value for a specific meter
    col = myDF.iloc[i, 1:]

    # Calculate the mean and sigma by meter
    mean = np.mean(col)
    sigma = np.std(col)

    # Using the mean and sigma obtained from above, get the upper and lower limits
    upperBound = mean + (3.0 * sigma)
    lowerBound = mean - (3.0 * sigma)

    # If the residue value is less than or greater than the limits, we save them in a result dataframe
    for j in range(0, len(col)):
        if (col[j] > upperBound).all() or (col[j] < lowerBound).all():
                # entry = [{'MeterName':meterName, 'Time':col.index[j], 'kWh':kwh[col.index[j]] ,'Residue':col[j], 'Voltage':volt[col.index[j]]}]
                entry = [{'Time': col.index[j], 'MeterName': meterName, 'kWh': kwh[col.index[j]], 'Residue': col[j]}]
                outputData = outputData.append(entry, ignore_index=True)

print("\n")

# Print the result dataframe that contains meters with possible anomalies
print("             ******Anomalies Detected******")
print(outputData.sort_values(['Time']))

# Dataframe to csv
# outputData.sort_values(['Time']).to_csv('ARIMA_results_120817/ARIMA_0420.csv',index=False)


