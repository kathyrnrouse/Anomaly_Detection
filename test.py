# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 19:35:09 2019

@author: krouse
"""

from LSTM_Prediction_Build import lstm_prediction_build
from Anomaly_Detection_single_day import anomaly_detection_single_day
import os


## Here we call the function lstm_prediction_build to build the predicitons for each meter in the region
lstm_prediction_build('Huntsville_0099','Huntsville_0099_last_reading.csv')


## The following is how one can run the anomaly_detection function over each meter based on the output of the prediction above
directory = 'Huntsville_0099\Prediction'
location = 'Huntsville_0099'
meter_test = ['0099070']
for file in meter_test:
    anomaly_detection_single_day(location, file, start = '2017-02-01 00:00:00',
                                 end= '2017-2-04 23:59:59')

