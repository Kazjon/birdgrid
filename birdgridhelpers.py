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
from mpl_toolkits.axes_grid.inset_locator import inset_axes
import matplotlib.dates as mdates

#Takes in a set of desired attributes, the species, and the year range
#returns an observation x [lat, lon, season, attribute_1, attribute_2,...attribute_n] matrix 
def load_observations(ATTRIBUTES, SPECIES, START_YEAR, END_YEAR):
	path = os.path.normpath('Birdgriddata') 
	allFiles = glob.glob(path + "/*.csv")
	observations = pd.DataFrame()
	ColumnNames=np.append(ATTRIBUTES,SPECIES)
	list_ = []
	for file_ in allFiles:
		df = pd.read_csv(file_,index_col=None,header=0,usecols=ColumnNames)
		df=df[(df['YEAR']>=START_YEAR)&(df['YEAR']<=END_YEAR)]
		df=df.replace('X',1)
		df=df[(df[SPECIES[0]]!='0')]
		df=df[(df[SPECIES[0]]!=0)]
		list_.append(df)
	observations= pd.concat(list_)
	return observations

#Takes in the matrix of observations and bins it into observations based on the provided grid size.
#Returns an array of dicts, each dict represents one location and contains lat, lon and data for each timestep
#Returns a dataframe of attributes that are divided into grids, where as each grid square represents the total count of the species found in that location

def init_birdgrid(observations,GRID_SIZE,SPECIES,TIME_STEP,START_YEAR,END_YEAR,PICKLE_NAME):
	lats=observations['LATITUDE']
	lons=observations['LONGITUDE']
	observations=observations.convert_objects(convert_numeric=True)
	lat_min = int(math.floor(min(lats)))
	lat_max = int(math.floor(max(lats)))
	lon_min =int(math.floor(min(lons)))
	lon_max =int(math.floor(max(lons)))
	GridSquare=[]
	df=pd.DataFrame([])
	nw=pd.DataFrame([])
	
	if TIME_STEP =='monthly':
		for i in range(lat_min,lat_max,GRID_SIZE):
			for j in range(lon_min,lon_max,GRID_SIZE):
				GridSquare=observations[(observations['LATITUDE']>=i)&(observations['LATITUDE']<i+GRID_SIZE)&(observations['LONGITUDE']>=j)&(observations['LONGITUDE']<j+GRID_SIZE)]
				GridSquare['LATITUDE']=i
				GridSquare['LONGITUDE']=j
				GridwiseCount=GridSquare.groupby(['LATITUDE','LONGITUDE','YEAR','MONTH'],as_index=False)[SPECIES[0]].sum()								
				df=df.append(GridwiseCount)
		monthnumber=0
		for year in range(START_YEAR,END_YEAR+1):
			for month in range(1,13):
				obs=df[(df['YEAR']==year)&(df['MONTH']==month)]
				obs['timeframe']=monthnumber
				nw=nw.append(obs)
				monthnumber += 1
	nw=nw.reset_index()
	nw['Date_Format']=pd.Series("/".join(a) for a in zip(nw.YEAR.astype("int").astype(str),nw.MONTH.astype("int").astype(str)))
	nw.to_pickle(PICKLE_NAME+".p") 
	return nw

#Plot the actual species frequency (from the data) on a map
def plot_observation_frequency(locations,SEASONS,GRID_SIZE,START_YEAR,END_YEAR,SPECIES):
	for year in range(START_YEAR,END_YEAR+1):
		for season in SEASONS:
			wanted=SEASONS[season]
			latitude = np.asarray(locations['LATITUDE'])
			longitude = np.asarray(locations['LONGITUDE'])
			Yearly_Data=(locations.loc[locations['YEAR']==year])
			Seasonal_Data=(Yearly_Data.loc[Yearly_Data['MONTH'].isin(wanted)])
			lats = np.asarray(Seasonal_Data['LATITUDE'])
			lons = np.asarray(Seasonal_Data['LONGITUDE'])
			Species_count=np.asarray(Seasonal_Data[SPECIES[0]])
			Species_count=np.reshape(Species_count,len(Species_count))
			lat_min = min(latitude)
			lat_max = max(latitude)
			lon_min = min(longitude)
			lon_max = max(longitude)
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
			plt.title(str(SPECIES[0])+"-"+str(year)+"-"+str(season))
			#plt.show()
			plt.savefig(str(SPECIES[0])+"-"+str(year)+"-"+str(season)+".png")
			plt.close()
	return

#Plots the frequency (Y axis) against the timesteps (X axis) for the given location.
#Uses the location's included coordinates to provide a map insert showing a dot for the location on the US map (this should use matplotlib's "axes" interface as with here http://matplotlib.org/examples/pylab_examples/axes_demo.html)
#The optional "predictor" object overlays the expectations of a particular predictor (which is associated with a particular timestamp)


def model_location_novelty_over_time(location,SPECIES,SEASONS,START_YEAR,END_YEAR):
	ModelObject=[]
	Maximum_Error=[]
	Mean_Error=[]
	Model_Name=[]
	Regression_Coefficient=[]
	Regression_Intercepts=[]
	Regression_Score=[]
	Predictions=[]
	ActualSpeciesCount=[]
	TestDataforplotting=[]
	seasonlist=[]
	predictingyearlist=[]
	latitude=[]
	longitude=[]
	Nonseasonaldata=[]
	Nonseasonaldata_Frequency=[]
	SeasonwiseTrainData=[]
	SeasonwiseTrainDataFrequency=[]
	train=[]
	Nonseasonaldata_timeframe=[]
	tr= pd.DataFrame()
	LocationData = location
	d={}
	Training_years=[]
	for year in range(START_YEAR,END_YEAR):
		Training_years.append(year)
		predicting_year=[year+1]
		for season in SEASONS:
			wanted=SEASONS[season]
			NonSeasonal_Data=(LocationData.loc[~LocationData['MONTH'].isin(wanted)])
			NonSeasonalData=(NonSeasonal_Data.loc[NonSeasonal_Data['YEAR'].isin(Training_years)])
			#NonSeasonalData=NonSeasonalData.append(NonSeasonal_Data[NonSeasonal_Data['YEAR']==predicting_year],ignore_index=True)
			NonSeasonalDataTimeframe=NonSeasonalData['timeframe']
			NonSeasonalDataTimeframe=NonSeasonalDataTimeframe.reshape(-1,1)
			NonSeasonalDataforPlotting = NonSeasonalData['Date_Format']
			NonSeasonalDataFrequency=NonSeasonalData[SPECIES[0]]
			Seasonal_Data=LocationData[LocationData['MONTH'].isin(wanted)]
			Train_Data=Seasonal_Data[Seasonal_Data['YEAR'].isin(Training_years)]
			max_train_year=max(Training_years)
			Test_Data=(Seasonal_Data.loc[Seasonal_Data['YEAR'].isin(predicting_year)])
			if season =='winter':
				Test_Data=Test_Data[(Test_Data['YEAR']==predicting_year)&(Test_Data['MONTH']!=12)]
				Test_Data=Test_Data.append(Train_Data[(Train_Data['YEAR']==max_train_year)&(Train_Data['MONTH']==12)], ignore_index=True)
				tr=Train_Data[(Train_Data['YEAR']!=max_train_year)]
				Train_Data=tr.append(Train_Data[(Train_Data['YEAR']==max_train_year)&(Train_Data['MONTH']!=12)],ignore_index=True)
				
			TrainData=Train_Data['timeframe']
			TrainData=TrainData.reshape(-1,1)
			Seasonwise_TrainData=Train_Data['Date_Format']
			Seasonwise_Traindata_Frequency=Train_Data[SPECIES[0]]
			TrainData_Target = Seasonwise_Traindata_Frequency.as_matrix()
			TrainData_Target=TrainData_Target.astype(np.float)
			TestData=Test_Data['timeframe']
			TestData=TestData.reshape(-1,1)
			TestData_Plotting=Test_Data['Date_Format']
			Actual_Species_Count=Test_Data[SPECIES[0]]
			lat=Test_Data['LATITUDE']
			lon=Test_Data['LONGITUDE']
			
			if len(Train_Data)!=0 and len(TestData)!=0:
				residual_threshold=np.median([abs(x - np.median(TrainData_Target)) for x in TrainData_Target])
				try:
					model_name='Ransac_RobustRegression'
					min_samples=round(0.8*len(TrainData))
					regr = linear_model.RANSACRegressor(linear_model.LinearRegression(),min_samples,residual_threshold)
					regr.fit(TrainData,TrainData_Target)
					Predicted_Species_Count=regr.predict(TestData)
				except ValueError:
					model_name='Linear_Regression'
					regr = linear_model.LinearRegression()
					regr.fit(TrainData,TrainData_Target)
					#Regression_Model.append(regr)
					Predicted_Species_Count=regr.predict(TestData)
					#MaxError=np.max(abs(Predicted_Species_Count-Actual_Species_Count))
					#Maximum_Error.append(MaxError)				
					#MeanError=np.mean((regr.predict(TestData) - Actual_Species_Count) ** 2)
					#Mean_Error.append(MeanError)
					#Score=regr.score(TestData,Actual_Species_Count)
					#Regression_Score.append(Score)
				
				ActualSpeciesCount.append(Actual_Species_Count)
				TestDataforplotting.append(TestData_Plotting)
				seasonlist.append(season)
				predictingyearlist.append(predicting_year)
				latitude.append(lat)
				longitude.append(lon)
				Nonseasonaldata.append(NonSeasonalDataforPlotting)
				Nonseasonaldata_Frequency.append(NonSeasonalDataFrequency)
				SeasonwiseTrainData.append(Seasonwise_TrainData)
				SeasonwiseTrainDataFrequency.append(Seasonwise_Traindata_Frequency)
				train.append(TrainData)
				Nonseasonaldata_timeframe.append(NonSeasonalDataTimeframe)
				ModelObject.append(regr)
				Predictions.append(Predicted_Species_Count)	
				Model_Name.append(model_name)
			else:
				continue
			
	d['location']={}
	#d['stats']={}
	d['location']['latitude']=latitude
	d['location']['longitude']=longitude
	#d['stats']['score']=Regression_Score
	#d['stats']['max_error']=np.reshape(Maximum_Error, len(Maximum_Error))
	#d['stats']['mean_error']=np.reshape(Mean_Error, len(Mean_Error))
	d['predictions']=Predictions
	d['actualspeciescount']=ActualSpeciesCount
	d['TestDataforplotting']=TestDataforplotting
	d['seasonlist']=seasonlist
	d['predictingyearlist']=np.reshape(predictingyearlist,len(predictingyearlist))
	d['NonSeasonalDataFrequency']=Nonseasonaldata_Frequency
	d['NonSeasonalData']=Nonseasonaldata
	d['seasonwisetraindata']=SeasonwiseTrainData
	d['seasonwisetrainDatafrequency']=SeasonwiseTrainDataFrequency
	d['traindata']=train
	d['Model_object']=ModelObject
	d['Nonseasonaldata-timeframe']=Nonseasonaldata_timeframe
	d['Model_Name']=Model_Name
	return d


def plot_birds_over_time(predictors,SPECIES,locations,DIRECTORY_NAME):	
	locationslatitude = np.asarray(locations['LATITUDE'])
	locationslongitude = np.asarray(locations['LONGITUDE'])
	lat_min = min(locationslatitude)
	lat_max = max(locationslatitude)
	lon_min = min(locationslongitude)
	lon_max = max(locationslongitude)
	for p in predictors:
		for Model_Name,Model_Object,Predictions,Actualspecies_count,TestDataforplotting,latitude,longitude,NonSeasonalDataFrequency,NonSeasonalData,season,predicting_year,SeasonTrainData,SeasonTrainDataFrequency,TrainData,Nonseasonaldatamonths in zip(p["Model_Name"],p["Model_object"],p["predictions"],p["actualspeciescount"],p["TestDataforplotting"],p["location"]["latitude"],p["location"]["longitude"],p['NonSeasonalDataFrequency'],p['NonSeasonalData'],p['seasonlist'],p['predictingyearlist'],p['seasonwisetraindata'],p['seasonwisetrainDatafrequency'],p['traindata'],p['Nonseasonaldata-timeframe']):
			plt.figure()
			TestDataforplotting = [dt.datetime.strptime(d,'%Y/%m').date() for d in TestDataforplotting]
			NonSeasonalData = [dt.datetime.strptime(d,'%Y/%m').date() for d in NonSeasonalData]   #Converting the integer monthvalues into year/month format for plotting
			lat=np.unique(latitude)
			lon=np.unique(longitude)
			plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m'))
			plt.gcf().autofmt_xdate()
			plt.scatter(NonSeasonalData,NonSeasonalDataFrequency,alpha=0.3,label='Non Seasonal Datapoints')    #Scatter plot of Non-Seasonal-Data over Training years
			if Model_Name=='Linear_Regression':
				SeasonTrainData = [dt.datetime.strptime(d,'%Y/%m').date() for d in SeasonTrainData]
				seasontraindatapredictions=Model_Object.predict(TrainData)     # Predicting the frequency for the Seasonal Train Data using Linear regression model
				plt.scatter(SeasonTrainData,SeasonTrainDataFrequency,label='Train Datapoints')
				plt.scatter(TestDataforplotting,Actualspecies_count,color='black',label='Test Data')              #Scatter plot of Test Data with Actual Frequencies
				plt.plot(TestDataforplotting,Predictions,'r-',linewidth=1,label='Linear Regressor Line')   #Plotting linear Regressor for Test Data
				plt.plot(SeasonTrainData,seasontraindatapredictions,'r-',linewidth=1)  #plotting predictor line for sesonal train data
				if len(NonSeasonalData)!=0:
					Nonseasonaldatapredictions=Model_Object.predict(Nonseasonaldatamonths)
					plt.plot(NonSeasonalData,Nonseasonaldatapredictions,'r-',linewidth=1)
			
			elif Model_Name=='Ransac_RobustRegression':
				seasontraindata_ransacpredictions=Model_Object.predict(TrainData)  # Predicting the frequency for the Seasonal Train Data using Robust regression model
				
				inlier_mask = Model_Object.inlier_mask_                           
				outlier_mask = np.logical_not(inlier_mask)                       
				seasonalinliers=SeasonTrainData[inlier_mask]						#Finding inliers				
				sesonaloutliermask=SeasonTrainData[outlier_mask]					#Finding outliers
				seasonalinliers = [dt.datetime.strptime(d,'%Y/%m').date() for d in seasonalinliers]
				sesonaloutliermask = [dt.datetime.strptime(d,'%Y/%m').date() for d in sesonaloutliermask]
				SeasonTrainData = [dt.datetime.strptime(d,'%Y/%m').date() for d in SeasonTrainData]
				plt.plot(seasonalinliers,SeasonTrainDataFrequency[inlier_mask],'.g', label='Inliers',ms=10) 			#Scatterplot of Seasonal inlier Train Data 
				plt.plot(sesonaloutliermask,SeasonTrainDataFrequency[outlier_mask],'.r', label='Outliers',ms=10)		#Scatterplot of Seasonal outliers in the Train Data
				plt.plot(TestDataforplotting,Predictions,'b-',label='Robust Regressor Line',linewidth=2)  #Plotting Robust Regressor line for Test Data
				plt.plot(SeasonTrainData,seasontraindata_ransacpredictions,'b-',linewidth=2)
				plt.scatter(TestDataforplotting,Actualspecies_count,color='black',label='Test Data')
				if len(NonSeasonalData)!=0:
					Nonseasonal_ransacpredictions=Model_Object.predict(Nonseasonaldatamonths)
					plt.plot(NonSeasonalData,Nonseasonal_ransacpredictions,'b-',linewidth=2)
				
			plt.title("Species Name -"+str(SPECIES[0])+"-"+str(predicting_year)+"-"+str(season))
			plt.legend(fontsize ='x-small',labelspacing=0.2)
			plt.xlabel("Timeseries")
			plt.ylabel("Predicted frequency values of the species")
			x1,x2,y1,y2=plt.axis()
			plt.axis((x1,x2,0,y2))
			insetfig= plt.axes([0.15,0.65,0.2,0.2])															#Setting coordinates and width,height of inset 
			themap= Basemap(projection = 'merc',llcrnrlat=lat_min,urcrnrlat=lat_max,llcrnrlon=lon_min, urcrnrlon=lon_max,rsphere=6371200., resolution='l', area_thresh=10000)
			#themap = Basemap(llcrnrlon=-126, llcrnrlat=22, urcrnrlon=-64,urcrnrlat=49, projection='lcc', lat_1=33, lat_2=45,lon_0=-95, resolution='h', area_thresh=10000)
			themap.bluemarble()
			themap.drawcoastlines()
			themap.drawcountries(linewidth=2)
			themap.drawstates()
			themap.fillcontinents(color = 'gainsboro')
			themap.drawmapboundary(fill_color='steelblue')
			longg,latt=lon,lat
			x, y = themap(longg,latt)
			lonpoint, latpoint = themap(x,y,inverse=True)       
			themap.plot(x,y,'ro',markersize=7)
			plt.title("Location"+'(%5.1fW,%3.1fN)'%(lonpoint,latpoint),fontsize=10)
			#plt.text(x+100000,y+100000,'Grid(%5.1fW,%3.1fN)'% (lonpoint,latpoint))
			plt.xticks([])
			plt.yticks([])
			#plt.show()
			figure_name=str(SPECIES[0])+"-"+str(lat)+"-"+str(lon)+"-"+str(predicting_year)+"-"+str(season)+".png"
			if not os.path.isdir(DIRECTORY_NAME):
				os.mkdir(DIRECTORY_NAME)
			destination_dir=os.path.abspath(DIRECTORY_NAME)
			plt.savefig(os.path.join(destination_dir,figure_name))
			plt.close()
	return
'''
def plot_predictors(predictors,max_size,out_fname):
	predictor_coefs = []
	predictor_intercepts = []
	predictor_variance = []
	for p in predictors:
		for model,score in zip(p["model"],p["stats"]["score"]):
			predictor_coefs.append(model.coef_)
			predictor_intercepts.append(model.intercept_)
			predictor_variance.append(score)
	plt.figure(figsize=(10,10))
	variance=predictor_variance
	plt.scatter(predictor_coefs,predictor_intercepts)
	plt.xlabel("Regression coefficient")
	plt.ylabel("Regression intercept")
	#plt.show()
	plt.savefig(str(out_fname)+".png")
	plt.close()
	return
'''	