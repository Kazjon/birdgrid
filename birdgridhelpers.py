import numpy as np
import pandas as pd
import glob
import math
from scipy import interpolate
from datetime import datetime
from mpl_toolkits.basemap import Basemap
from matplotlib.mlab import griddata
from scipy import interpolate
import csv
import matplotlib.pyplot as plt
import pickle
from sklearn import linear_model

#Takes in a set of desired attributes, the species, and the year range
#returns an observation x [lat, lon, season, attribute_1, attribute_2,...attribute_n] matrix 
def load_observations(attributes,species,start_year,end_year):
	path =r'\birdgrid\birdgrid_data' 
	allFiles = glob.glob(path + "/*.csv")
	observations = pd.DataFrame()
	ColumnNames=np.append(attributes,species)
	list_ = []
	for file_ in allFiles:
		df = pd.read_csv(file_,index_col=None,header=0,usecols=ColumnNames)
		list_.append(df)
	observations= pd.concat(list_)
	observations=observations[(observations['YEAR']>=start_year)&(observations['YEAR']<=end_year)]
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

def plot_observation_frequency(locations):
	return
	
#Plots the frequency (Y axis) against the timesteps (X axis) for the given location.
#Uses the location's included coordinates to provide a map insert showing a dot for the location on the US map (this should use matplotlib's "axes" interface as with here http://matplotlib.org/examples/pylab_examples/axes_demo.html)
#The optional "predictor" object overlays the expectations of a particular predictor (which is associated with a particular timestamp)
def plot_birds_over_time(location, predictor=None):
	pass
	
#Makes a prediction of the observation for each timestep as a sklearn Pipeline object
#To start with, try predicting for each month using the data only for that season.  That should allow you to use linear regression.
def model_location_novelty_over_time(location,SPECIES):
	Locationpredictors=[]
	LocationData = location
	seasons = {"winter": [12,1,2],"spring": [3,4,5],"summer":[6,7,8],"fall":[9,10,11]}
	years=[2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012]
	Training_years=[]
	for year in years:
		Training_years.append(year)
		predicting_year=[year+1]
		for season in seasons:
			wanted=seasons[season]
			Seasonal_Data=(LocationData.loc[LocationData['MONTH'].isin(wanted)])
			Train_Data=(Seasonal_Data.loc[Seasonal_Data['YEAR'].isin(Training_years)])
			Test_Data=(Seasonal_Data.loc[Seasonal_Data['YEAR'].isin(predicting_year)])
			Train_Data['PERIOD'] = Train_Data.YEAR.astype(str).str.cat(Train_Data.MONTH.astype(str),sep='0')
			Test_Data['PERIOD'] = Test_Data.YEAR.astype(str).str.cat(Test_Data.MONTH.astype(str),sep='0')
			TrainData=Train_Data['PERIOD']
			TrainData=TrainData.reshape(-1, 1)
			TrainData=TrainData.astype(np.float)
			TrainData_Target=Train_Data[SPECIES]
			TrainData_Target = TrainData_Target.as_matrix()
			TrainData_Target=TrainData_Target.astype(np.float)
			TestData=Test_Data['PERIOD']
			TestData=TestData.reshape(-1, 1)
			TestData=TestData.astype(np.float)
			ActualResult=Test_Data[SPECIES]
			regr = linear_model.LinearRegression()
			if len(Train_Data)!=0 and len(TestData)!=0:
				regr.fit(TrainData,TrainData_Target)
				Locationpredictors.append(regr)
			else:
				continue
	return Locationpredictors
	
def plot_predictors(predictors,max_size=10, out_fname = "predictor_plot.png"):
	predictor_coefs = []
	predictor_intercepts = []
	predictor_variance = []
	for preds in predictors:
		for p in preds:
			predictor_coefs.append(p["model"].coef_)
			predictor_intercepts.append(p["model"]predict([0])[0])
			predictor_errors.append(p["stats"]["score"])
	plt.figure(figsize=(10,10))
	plt.scatter(predictor_coefs,predictor_intercepts,s=[e * max_size for e in predictor_errors])
	plt.xlabel("Regression coefficient")
	plt.ylabel("Regression intercept")
	plt.savefig(out_fname)
	
