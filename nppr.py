from flask import Flask
from flask import request
import osmnx as ox
import networkx as nx
import requests
import pandas as pd
import random
import tensorflow as tf
import numpy as np
import warnings
import sys
import requests
from geopy.distance import geodesic
warnings.filterwarnings('ignore')
from flask import Flask
from flask import request
import airpyllution
from airpyllution.airpyllution import get_pollution_history
import pandas as pd
import datetime
import math
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense, LSTM
import numpy as np

def futurePrediction(lat,lon,nom,d):
  random.seed(0)
  np.random.seed(0)
  tf.random.set_seed(0)
  api_key = 'e38917613f43372c3e61d01cbbcb010d'

  data = get_pollution_history(1632821940,1695893940,lat,lon, api_key)
  df = data


  # Assuming you already have a DataFrame named 'df' with your data

  # Calculate the maximum value for each row and store it in a new column 'max_value'
  df['aqi'] = df.iloc[:, :-1].max(axis=1)

  # Print the updated DataFrame
  print(df)
  df2=df
  min_val = df2['aqi'].min()
  max_val = df2['aqi'].max()
  #print(min_val, max_val)
  max_range=1000
  min_range=1
  df2['aqi'] =((df2['aqi'] - min_val) / (max_val - min_val)) * (max_range - min_range) + min_range # (df2['aqi'] - min_val) / (max_val - min_val)

  print(df2)
  df=df2
  data = df
  df2 = df[['aqi','dt']].copy()
  df = df2
  df['dt'] = pd.to_datetime(df['dt'])
  df['Date'] = df['dt'].dt.date
  df_date=pd.DataFrame(df.groupby('Date')['aqi'].mean())
  dataset  = df_date.values
  training_data_len = math.ceil(len(dataset)*.8)
  training_data_len
  sc = MinMaxScaler(feature_range=(0,1))
  scaled_data = sc.fit_transform(dataset)
  train_data = scaled_data[0:training_data_len, :]
  x_train = []
  y_train = []
  num = 60
  for i in range(num, len(train_data)):
      x_train.append(train_data[i-num:i , 0])
      y_train.append(train_data[i , 0])
  x_train, y_train = np.array(x_train), np.array(y_train)
  x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
  model = Sequential()
  model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1],1)))
  model.add(LSTM(50, return_sequences=False))
  model.add(Dense(25))
  model.add(Dense(1))
  model.compile(optimizer = "adam", loss = "mean_squared_error")
  model.fit(x_train,y_train, batch_size=1, epochs=1)
  test_data = scaled_data[training_data_len-60: , :]
  x_test = []
  y_test = dataset[training_data_len:,:]
  for i in range(num, len(test_data)):
        x_test.append(test_data[i-num:i, 0])
  x_test = np.array(x_test)
  x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
  predictions = model.predict(x_test)
  predictions = sc.inverse_transform(predictions)
  rmse = np.sqrt(np.mean(predictions - y_test)**2)
  train = df_date[:training_data_len]
  valid = df_date[training_data_len:]
  valid["Predictions"] = predictions
  # Existing code up to the training and validation data setup

  # Define the number of future time steps to predict
  n_future = nom  # Change this to the number of future time steps you want to predict

  # Extend your data for predicting future values
  extended_data = np.copy(scaled_data)
  x_ext = []

  for i in range(n_future):
      x_ext.append(extended_data[-num:])
      prediction = model.predict(np.array(x_ext[-1]).reshape(1, num, 1))
      extended_data = np.append(extended_data, prediction)

  # Reshape the extended data
  extended_data = extended_data.reshape(-1, 1)

  # Prepare the test data for the extended data
  x_test_ext = []
  y_test_ext = []

  for i in range(num, len(extended_data) - n_future):
      x_test_ext.append(extended_data[i - num : i, 0])
      y_test_ext.append(extended_data[i + n_future, 0])

  x_test_ext = np.array(x_test_ext)
  x_test_ext = np.reshape(x_test_ext, (x_test_ext.shape[0], x_test_ext.shape[1], 1))

  # Predict for the extended data
  predictions_ext = model.predict(x_test_ext)
  predictions_ext = sc.inverse_transform(predictions_ext)

  # Separate the extended predictions for visualization
  predictions_future = predictions_ext[-n_future:]

  # Continue with the rest of your code for visualization
  # return (str(predictions_future[2]))
  # res=str(predictions_future[2][0])
  # print(res,type(res))
  
  print(str(predictions_future[0][0]))
  return (str(predictions_future[0][0]))
  # return "5"

def interpolate_aqi(lat, lon):
    nearby_aqi_values = []
    nearby_points = [(lat + 0.0001, lon), (lat, lon + 0.0001), (lat - 0.0001, lon), (lat, lon - 0.0001)]
    for point in nearby_points:
        aqi = fetchAqi(point[0], point[1])
        if aqi:
            nearby_aqi_values.append(aqi)

    if nearby_aqi_values:
        # Interpolate by averaging nearby AQI values
        interpolated_aqi = sum(nearby_aqi_values) / len(nearby_aqi_values)
        return interpolated_aqi
    else:
        return 0

def fetchAqi(lat, lon):
    try:
      url=f"https://api.waqi.info/feed/geo:{lat};{lon}/?token=b251e1662957c47d186f229f40cbc56fe0413e3a"
      response= requests.get(url)
      json_data= response.json()
      json_data
      aqi = response.json()['data']['aqi']
      return aqi
    except KeyError:
        return interpolate_aqi(lat, lon)
    except Exception as e:
        print(f"An error occurred: {e}")
        return 0

def fetchTraffic(lon1,lat1,lon2, lat2):

  url=f"https://api.tomtom.com/routing/1/calculateRoute/{lat1},{lon1}:{lat2},{lon2}/json?key=Zq4ACA9IZ1RZknhpscoJsqfcVv2cyEBV&traffic=true"
  #print(url)
  response= requests.get(url)
  json_data= response.json()
  print(json_data['routes'][0]["summary"]["trafficDelayInSeconds"])
  return (json_data['routes'][0]["summary"]["trafficDelayInSeconds"])

def magical(lat1,lon1,lat2,lon2,prio):
  print(lat1,lon1,lat2,lon2)
  url=f"https://api.tomtom.com/routing/1/calculateRoute/{lat1},{lon1}:{lat2},{lon2}/json?key=Zq4ACA9IZ1RZknhpscoJsqfcVv2cyEBV&traffic=true&maxAlternatives=3"
  print(url)
  response= requests.get(url)
  json_data= response.json()
  #print(json_data)
  data=json_data
  raaste=[]
  for route in data.get("routes", []):
      for leg in route.get("legs", []):
          one=[]
          for point in leg.get("points", []):
              latitude = point.get("latitude")
              longitude = point.get("longitude")
              one.append([latitude,longitude])
              #print(f"{latitude},{longitude}")
          raaste.append(one)
  #print(len(raaste))
  time_delays = [route["summary"]["travelTimeInSeconds"] for route in data["routes"]]
  traffic_delays = [route["summary"]["trafficDelayInSeconds"] for route in data["routes"]]
  distance = [route["summary"]["lengthInMeters"] for route in data["routes"]]
  #for i in range(0,len(distance)):
    #print(distance[i], time_delays[i])
  
  aqiList=[]
  timeList=[]
  distList=[]
  delayList=[]
  df = pd.DataFrame(columns=['Latitude', 'Longitude', 'AQI'])
  for i in range(0,len(raaste)):
    #print(raaste[i])
    print("\n\nRaasta", i)
    aqi=0
    cnt=0
    leng=0
    for lat,lon in raaste[i]:
      cnt+=1
      if(cnt%4==0):
        leng=leng+1
        result = df[(df['Latitude'] == lat) & (df['Longitude'] == lon)]
        #print(aqi)
        if not result.empty:
            aqi =aqi+result.iloc[0]['AQI']
        else:
            aqi = aqi + fetchAqi(lat, lon)
      #aqi=aqi+fetchAqi(lat,lon)
    aqiList.append(aqi/leng)
    timeList.append(time_delays[i])
    distList.append(distance[i])
    delayList.append(time_delays[i])
    print(i,"ka avg", aqi/(leng), "time",time_delays[i], "distance",distance[i])

  minAqi=sys.maxsize
  minAqiIdx=0

  minDis=sys.maxsize
  minDisIdx=0

  minDel=sys.maxsize
  minDelIdx=0

  minTime=sys.maxsize
  minTimeIdx=0

  for i in range (0,len(raaste)):
    if(aqiList[i]<minAqi):
      minAqi=aqiList[i]
      minAqiIdx=i
    if(distList[i]<minDis):
      minDis=distList[i]
      minDisIdx=i
    if(timeList[i]<minTime):
      minTime=timeList[i]
      minTimeIdx=i
    if(delayList[i]<minDel):
      minDel=delayList[i]
      minDelIdx=i
  ans=[]
  if(prio==2): # health optimal
    ans.append([distList[minAqiIdx]])
    ans.append([aqiList[minAqiIdx]])
    ans.append(raaste[minAqiIdx])

  if(prio==1):
    weight_avg_travel_time = 0.5
    weight_traffic_delay = 0.5
    csMin=sys.maxsize
    csMinIdx=0
    min=sys.maxsize
    max=0
    for i in range(0,len(raaste)):
      if(min>timeList[i]):
        min=timeList[i]
      if(max<timeList[i]):
        max=timeList[i]

    for i in range (0, len(raaste)):

      normalized_value = (timeList[i] - min) * (60 - 20) / (max - min) + 20

      combined_score = (weight_avg_travel_time * normalized_value) + (weight_traffic_delay * aqiList[i])
      print(combined_score)
      if(combined_score<csMin):
        csMin=combined_score
        csMinIdx=i
    #print(raaste[csMinIdx])
    ans.append([distList[csMinIdx]])
    ans.append([aqiList[csMinIdx]])
    ans.append(raaste[csMinIdx])


  if(prio==0):
    ans.append([distList[minTimeIdx]])
    ans.append([aqiList[minTimeIdx]])
    ans.append(raaste[minTimeIdx])
  return ans

app=Flask(__name__)
@app.route('/')
def hello():
    return "Hello prado"

@app.route('/post', methods=["POST"])
def aStarPath():
    lat1 = float(request.form['lat1'])
    lon1 = float(request.form['lon1'])
    lat2 = float(request.form['lat2'])
    lon2 = float(request.form['lon2'])
    prio = float(request.form['prio'])
    arr = magical(lat1, lon1, lat2, lon2, prio)
    return arr

@app.route('/post2', methods=["POST"])
def predict():
    
    # arr=futurePrediction(12.928279931565662, 77.66997930688865,30,d)
    lat = float(request.form['lat'])
    lon = float(request.form['lon'])
    d = int(request.form['d'])
    print("Hello",lat,lon,d)
    arr=futurePrediction(lat, lon,30,d)
    return (arr)

if __name__=="__main__":
    app.run(debug=False,host='0.0.0.0')

  
