import numpy as np
import pandas as pd
import glob
import math
import os.path
from scipy import interpolate
import datetime as dt
from datetime import datetime
from mpl_toolkits.basemap import Basemap
from matplotlib.mlab import griddata
from scipy import interpolate
import csv
import matplotlib.pyplot as plt
import pickle
from sklearn import linear_model
import matplotlib.dates as mdates

#Takes in a set of desired attributes, the species, and the year range
#returns an observation x [lat, lon, season, attribute_1, attribute_2,...attribute_n] matrix 
def load_observations(ATTRIBUTES, SPECIES, START_YEAR, END_YEAR):
	path = os.path.normpath('Birdgriddata') 
	allFiles = glob.glob(path + "/*.csv")
	observations = pd.DataFrame()
	ColumnNames=np.append(ATTRIBUTES, SPECIES)
	list_ = []
	for file_ in allFiles:
		df = pd.read_csv(file_,index_col=None,header=0,usecols=ColumnNames)
		list_.append(df)
	observations= pd.concat(list_)
	observations=observations[(observations['YEAR']>=START_YEAR)&(observations['YEAR']<=END_YEAR)]
	observations=observations.replace('X',1) 
	observations=observations.replace('?',0)
	return observations

#Takes in the matrix of observations and bins it into observations based on the provided grid size.
#Returns an array of dicts, each dict represents one location and contains lat, lon and data for each timestep
#Returns a dataframe of attributes that are divided into grids, where as each grid square represents the total count of the species found in that location

def init_birdgrid(observations,GRID_SIZE,SPECIES,TIME_STEP):
	lats=observations['LATITUDE']
	lons=observations['LONGITUDE']
	observations=observations.convert_objects(convert_numeric=True)
	lat_min = int(math.floor(min(lats)))
	lat_max = int(math.floor(max(lats)))
	lon_min =int(math.floor(min(lons)))
	lon_max =int(math.floor(max(lons)))
	GridSquare=[]
	df=pd.DataFrame([])
	if TIME_STEP =='monthly':
		for i in range(lat_min,lat_max,GRID_SIZE):
			for j in range(lon_min,lon_max,GRID_SIZE):
				GridSquare=observations[(observations['LATITUDE']>=i)&(observations['LATITUDE']<i+GRID_SIZE)&(observations['LONGITUDE']>=j)&(observations['LONGITUDE']<j+GRID_SIZE)]
				GridSquare['LATITUDE']=i
				GridSquare['LONGITUDE']=j
				GridwiseCount=GridSquare.groupby(['LATITUDE','LONGITUDE','YEAR','MONTH'],as_index=False)[SPECIES].sum()
				df=df.append(GridwiseCount)
		
	elif TIME_STEP =='yearly':
		for i in range(lat_min,lat_max,GRID_SIZE):
			for j in range(lon_min,lon_max,GRID_SIZE):
				GridSquare=observations[(observations['LATITUDE']>=i)&(observations['LATITUDE']<i+GRID_SIZE)&(observations['LONGITUDE']>=j)&(observations['LONGITUDE']<j+GRID_SIZE)]
				GridSquare['LATITUDE']=i
				GridSquare['LONGITUDE']=j
				GridwiseCount=GridSquare.groupby(['LATITUDE','LONGITUDE','YEAR'],as_index=False)[SPECIES].sum()
				df=df.append(GridwiseCount)
	return df
	

	

#Plot the actual species frequency (from the data) on a map

def plot_observation_frequency(locations,SEASONS,GRID_SIZE,START_YEAR,END_YEAR):
	for year in range(START_YEAR,END_YEAR+1):
		for season in SEASONS:
			wanted=SEASONS[season]
			Yearly_Data=(locations.loc[locations['YEAR']==year])
			Seasonal_Data=(Yearly_Data.loc[Yearly_Data['MONTH'].isin(wanted)])
			lats = np.asarray(Seasonal_Data['LATITUDE'])
			lons = np.asarray(Seasonal_Data['LONGITUDE'])
			Species_count = np.asarray(Seasonal_Data.iloc[:,-1])
			lat_min = min(lats)
			lat_max = max(lats)
			lon_min = min(lons)
			lon_max = max(lons)
			spatial_resolution = 1
			fig = plt.figure()
			x = np.array(lons)
			y = np.array(lats)
			z = np.array(Species_count)
			xinum = (lon_max - lon_min) / spatial_resolution
			yinum = (lat_max - lat_min) / spatial_resolution
			xi = np.linspace(lon_min, lon_max + spatial_resolution, xinum)        
			yi = np.linspace(lat_min, lat_max + spatial_resolution, yinum)        
			xi, yi = np.meshgrid(xi, yi)
			zi = griddata(x, y, z, xi, yi, interp='linear')
			m = Basemap(projection = 'merc',llcrnrlat=lat_min, urcrnrlat=lat_max,llcrnrlon=lon_min, urcrnrlon=lon_max,rsphere=6371200., resolution='l', area_thresh=10000)
			m.drawcoastlines()
			m.drawstates()
			m.drawcountries()
			m.drawparallels(np.arange(lat_min,lat_max,GRID_SIZE),labels=[False,True,True,False])
			m.drawmeridians(np.arange(lon_min,lon_max,GRID_SIZE),labels=[True,False,False,True])
			lat, lon = m.makegrid(zi.shape[1], zi.shape[0])
			x,y = m(lat, lon)
			z=zi.reshape(xi.shape)
			levels=np.linspace(0,z.max(),25)
			cm=plt.contourf(x, y, zi,levels=levels,cmap=plt.cm.Greys)
			plt.colorbar()
			plt.title(str(year)+"-"+str(season))
			plt.savefig(str(year)+"-"+str(season)+".png")
			plt.close()
	return
'''	
#Plots the frequency (Y axis) against the timesteps (X axis) for the given location.
#Uses the location's included coordinates to provide a map insert showing a dot for the location on the US map (this should use matplotlib's "axes" interface as with here http://matplotlib.org/examples/pylab_examples/axes_demo.html)
#The optional "predictor" object overlays the expectations of a particular predictor (which is associated with a particular timestamp)
def plot_birds_over_time(location,SPECIES):
	fig=plt.figure()
	Gridpoint_Data=location
	Gridpoint_Data['Date']=Gridpoint_Data.MONTH.astype(str).str.cat(Gridpoint_Data.YEAR.astype(str),sep='/')
	lat=np.unique(Gridpoint_Data['LATITUDE'])
	lon=np.unique(Gridpoint_Data['LONGITUDE'])
	Timestamp=Gridpoint_Data['Date']
	Species_Frequency=Gridpoint_Data[SPECIES]
	Timestamp = [dt.datetime.strptime(d,'%m/%Y').date() for d in Timestamp]
	plt.title(str(lat)+"-"+str(lon))
	plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
	plt.plot_date(x=Timestamp,y=Species_Frequency)
	plt.gcf().autofmt_xdate()
	plt.savefig(str(lat)+"-"+str(lon)+".png")
	return
'''	
#Makes a prediction of the observation for each timestep as a sklearn Pipeline object
#To start with, try predicting for each month using the data only for that season.  That should allow you to use linear regression.
def model_location_novelty_over_time(location,SPECIES,SEASONS,START_YEAR,END_YEAR):
	Regression_Model=[]
	Maximum_Error=[]
	Mean_Error=[]
	Regression_Coefficient=[]
	Predictions=[]
	ActualSpeciesCount=[]
	TestDataforplotting=[]
	seasonlist=[]
	predictingyearlist=[]
	latitude=[]
	longitude=[]
	TrainDataforplotting=[]
	TrainDataSpeciesCount=[]
	SeasonwiseTrainData=[]
	SeasonwiseTrainDataFrequency=[]
	LocationData = location
	d={}
	Training_years=[]
	for year in range(START_YEAR,END_YEAR):
		Training_years.append(year)
		predicting_year=[year+1]
		for season in SEASONS:
			wanted=SEASONS[season]
			NonSeasonal_Data=(LocationData.loc[~LocationData['MONTH'].isin(wanted)])
			Train_DataPlotting=(NonSeasonal_Data.loc[NonSeasonal_Data['YEAR'].isin(Training_years)])
			Train_DataPlotting['PERIOD'] = Train_DataPlotting.YEAR.astype(str).str.cat(Train_DataPlotting.MONTH.astype(str),sep='/')
			Train_Data_Plotting=Train_DataPlotting['PERIOD']
			Train_Data_Plotting_Frequency=Train_DataPlotting[SPECIES]
			Seasonal_Data=(LocationData.loc[LocationData['MONTH'].isin(wanted)])
			Train_Data=(Seasonal_Data.loc[Seasonal_Data['YEAR'].isin(Training_years)])
			Test_Data=(Seasonal_Data.loc[Seasonal_Data['YEAR'].isin(predicting_year)])
			Train_Data['PERIOD'] = Train_Data.YEAR.astype(str).str.cat(Train_Data.MONTH.astype(str))
			Test_Data['PERIOD'] = Test_Data.YEAR.astype(str).str.cat(Test_Data.MONTH.astype(str))
			TrainData=Train_Data['PERIOD']
			TrainData=TrainData.reshape(-1, 1)
			TrainData=TrainData.astype(np.float)
			Seasonwise_TrainData=Train_Data.YEAR.astype(str).str.cat(Train_Data.MONTH.astype(str),sep='/')
			Seasonwise_Traindata_Frequency=Train_Data[SPECIES]
			TrainData_Target = Seasonwise_Traindata_Frequency.as_matrix()
			TrainData_Target=TrainData_Target.astype(np.float)
			TestData=Test_Data['PERIOD']
			TestData=TestData.reshape(-1, 1)
			TestData=TestData.astype(np.float)
			Actual_Species_Count=Test_Data[SPECIES]
			latt=Test_Data['LATITUDE']
			long=Test_Data['LONGITUDE']
			regr = linear_model.LinearRegression()
			if len(Train_Data)!=0 and len(TestData)!=0:
				regr.fit(TrainData,TrainData_Target)
				Regression_Model.append(regr)
				Predicted_Species_Count=regr.predict(TestData)
				MaxError=np.max(abs(Predicted_Species_Count-Actual_Species_Count))
				Maximum_Error.append(MaxError)				
				MeanError=np.mean((regr.predict(TestData) - Actual_Species_Count) ** 2)
				Mean_Error.append(MeanError)
				Regression_Coefficient.append(regr.coef_)
				Predictions.append(Predicted_Species_Count)
				ActualSpeciesCount.append(Actual_Species_Count)
				TestDataforplotting.append(TestData)
				seasonlist.append(season)
				predictingyearlist.append(predicting_year)
				latitude.append(latt)
				longitude.append(long)
				TrainDataforplotting.append(Train_Data_Plotting)
				TrainDataSpeciesCount.append(Train_Data_Plotting_Frequency)
				SeasonwiseTrainData.append(Seasonwise_TrainData)
				SeasonwiseTrainDataFrequency.append(Seasonwise_Traindata_Frequency)
			else:
				continue
		
	d['model']=Regression_Model
	d['location']={}
	d['stats']={}
	d['location']['latitude']=latitude
	d['location']['longitude']=longitude
	d['stats']['score']=np.reshape(Regression_Coefficient, len(Regression_Coefficient))
	d['stats']['max_error']=np.reshape(Maximum_Error, len(Maximum_Error))
	d['stats']['mean_error']=np.reshape(Mean_Error, len(Mean_Error))
	d['predictions']=Predictions
	d['actualspeciescount']=ActualSpeciesCount
	d['TestDataforplotting']=TestDataforplotting
	d['seasonlist']=seasonlist
	d['predictingyearlist']=np.reshape(predictingyearlist,len(predictingyearlist))
	d['traindataspeciescount']=TrainDataSpeciesCount
	d['traindataforplotting']=TrainDataforplotting
	d['seasonwisetraindata']=SeasonwiseTrainData
	d['seasonwisetrainDatafrequency']=SeasonwiseTrainDataFrequency
	return d
	
def plot_birds_over_time(predictors):	
	for p in predictors:
		for q,r,z,m,t,u,v,w,o,n,k in zip(p["predictions"],p["actualspeciescount"],p["TestDataforplotting"],p["location"]["latitude"],p["location"]["longitude"],p['traindataspeciescount'],p['traindataforplotting'],p['seasonlist'],p['predictingyearlist'],p['seasonwisetraindata'],p['seasonwisetrainDatafrequency']):
			plt.figure()
			predictions=q
			actualspeciescount=r
			TestDataforplotting=z
			TestDataforplotting = [l[0] for l in TestDataforplotting]
			TestDataforplotting = [str(int(s)) for s in TestDataforplotting]
			TestDataforplotting = [s[:4]+'/'+s[4:] for s in TestDataforplotting]
			TestDataforplotting = [dt.datetime.strptime(d,'%Y/%m').date() for d in TestDataforplotting]
			latitude=m
			longitude=t
			TrainData_Frequency=u
			traindataforplotting=v
			traindataforplotting = [dt.datetime.strptime(d,'%Y/%m').date() for d in traindataforplotting]
			season=w
			predicting_year=o
			SeasonTrainData=n
			SeasonTrainData = [dt.datetime.strptime(d,'%Y/%m').date() for d in SeasonTrainData]
			SeasonTrainDataFrequency=k
			lat=np.unique(latitude)
			lon=np.unique(longitude)
			plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m'))
			plt.scatter(traindataforplotting,TrainData_Frequency,alpha=0.5)
			plt.plot_date(x=SeasonTrainData,y=SeasonTrainDataFrequency)
			plt.gcf().autofmt_xdate()
			plt.scatter(TestDataforplotting,actualspeciescount,color='black')
			plt.plot(TestDataforplotting,predictions,color='blue',linewidth=3)
			plt.title(str(lat)+"-"+str(lon)+"-"+str(predicting_year)+"-"+str(season))
			plt.savefig("image"+str(lat)+"-"+str(lon)+"-"+str(predicting_year)+"-"+str(season)+".png")
			plt.close()
	
'''	
def plot_predictors(predictors,max_size=10, out_fname = "predictor_plot.png"):
	predictor_coefs = []
	predictor_intercepts = []
	predictor_variance = []
	for preds in predictors:
		for p in preds:
			predictor_coefs.append(p["model"].coef_)
			predictor_intercepts.append(p["model"].predict([0])[0])
			predictor_errors.append(p["stats"]["score"])
	plt.figure(figsize=(10,10))
	plt.scatter(predictor_coefs,predictor_intercepts,s=[e * max_size for e in predictor_errors])
	plt.xlabel("Regression coefficient")
	plt.ylabel("Regression intercept")
	plt.savefig(out_fname)
	
'''