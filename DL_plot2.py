import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv("IEE_28Meter_072718.csv", usecols=['198103'], engine='python')
# data = pd.read_csv("IEE_TVA_testData_120617_fixedData.csv", usecols=[1], engine='python')

x = pd.Series(data.iloc[:, 0])
time = pd.date_range(start=pd.datetime(2014, 1, 1), freq='h', periods=len(x))
x_time = pd.Series(x.values, index=time)

# data = data[17520:len(data)-48]
# data = data.reset_index()

data1 = pd.read_csv("LSTM_tva_result_072618.csv", usecols=['198103'], engine='python')
# data1 = pd.read_csv("JinAustin_DL_result_120617.csv", usecols=[1], engine='python')
x1 = pd.Series(data1.iloc[:, 0])
time1 = pd.date_range(start=pd.datetime(2016, 1, 1), freq='h', periods=len(x1))
x_time1 = pd.Series(x1.values, index=time1)

print(x_time.shape, x_time1.shape)

start = '2016-01-01 00:00:00'
end = '2016-01-30 23:00:00'

# diff_list = [(x_time['2016-04-19 00:00:00':'2016-04-21 23:00:00'])-
# (x_time1['2016-04-19 00:00:00':'2016-04-21 23:00:00'])]
plt.title(list(data.columns.values), fontsize=20)
plt.xlabel('Time', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel('kWh', fontsize=20)
plt.plot(x_time[start:end],
         color='r', label='Observed (IEE)', linestyle='-')
plt.plot(x_time1[start:end],
         color='b', label='Predicted (LSTM)', linestyle='--')
plt.plot((x_time[start:end])-(x_time1[start:end]),
         color='g', label='Difference', linestyle='-')
plt.grid(True)
plt.legend(prop={'size': 30})
plt.show()
