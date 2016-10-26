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
from matplotlib.ticker import MultipleLocator
from scipy import interpolate
import csv
import matplotlib.pyplot as plt
import pickle
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error,explained_variance_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from mpl_toolkits.axes_grid.inset_locator import inset_axes
import matplotlib.dates as mdates
from matplotlib.patches import Polygon


#Takes in a set of desired attributes, the species, and the year range
#returns an observation x [lat, lon, season, attribute_1, attribute_2,...attribute_n] matrix 
def load_observations(config):
	path = os.path.normpath('Birdgriddata') 
	allFiles = glob.glob(path + "/*.csv")
	observations = pd.DataFrame()
	ColumnNames=np.append(config["ATTRIBUTES"],config['SPECIES'])
	list_ = []
	for file_ in allFiles:
		df = pd.read_csv(file_,index_col=None,header=0,usecols=ColumnNames)
		#df=df[(df['YEAR']>=config['START_YEAR'])&(df['YEAR']<=config['END_YEAR'])]
		#df=df.replace('X',1)
		#df=df[(df[SPECIES[0]]!='0')]
		#df=df[(df[SPECIES[0]]!=0)]
		list_.append(df)
	observations= pd.concat(list_)
	observations=observations[(observations['YEAR']>=config['START_YEAR']-1)&(observations['YEAR']<=config['END_YEAR'])]
	observations=observations.replace('X',1) 
	observations=observations.replace('?',0)
	return observations

#Takes in the matrix of observations and bins it into observations based on the provided grid size.
#Returns an array of dicts, each dict represents one location and contains lat, lon and data for each timestep
#Returns a dataframe of attributes that are divided into grids, where as each grid square represents the total count of the species found in that location

def init_birdgrid(observations,config):
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
	
	if config["TIME_STEP"] =='monthly':
		for i in range(lat_min,lat_max,config['GRID_SIZE']):
			for j in range(lon_min,lon_max,config['GRID_SIZE']):
				GridSquare=observations[(observations['LATITUDE']>=i)&(observations['LATITUDE']<i+config['GRID_SIZE'])&(observations['LONGITUDE']>=j)&(observations['LONGITUDE']<j+config['GRID_SIZE'])]
				GridSquare['LATITUDE']=i
				GridSquare['LONGITUDE']=j
				if config['use_chance_not_count']:
					counts={}
					counts['LATITUDE']=[]
					counts['LONGITUDE'] =[]
					counts['YEAR']=[]
					counts['MONTH']=[]
					counts[config['SPECIES']]=[]
					for (lat,lon,y,m),g in GridSquare.groupby(['LATITUDE','LONGITUDE','YEAR','MONTH'],as_index=False):
						counts['LATITUDE'].append(lat)
						counts['LONGITUDE'].append(lon)
						counts['YEAR'].append(y)
						counts['MONTH'].append(m)
						counts[config['SPECIES']].append(float(np.sum(g[config['SPECIES']].values>0))/g.shape[0])
					GridwiseCount = pd.DataFrame.from_dict(counts)
				else:
					GridwiseCount=GridSquare.groupby(['LATITUDE','LONGITUDE','YEAR','MONTH'],as_index=False)[config['SPECIES']].sum()
				
				df=df.append(GridwiseCount)
		monthnumber=0
		for year in range(config['START_YEAR']-1,config['END_YEAR']+1):
			for month in range(1,13):
				obs=df[(df['YEAR']==year)&(df['MONTH']==month)]
				obs['timeframe']=monthnumber
				nw=nw.append(obs)
				monthnumber += 1
	elif config["TIME_STEP"] == "weekly":
		raise NotImplementedError
	nw=nw.reset_index()
	nw['Date_Format']=pd.Series("-".join(a) for a in zip(nw.YEAR.astype("int").astype(str),nw.MONTH.astype("int").astype(str)))
	nw.to_pickle(config["RUN_NAME"]+".p")
	return nw

#Plot the actual species frequency (from the data) on a map
def plot_observation_frequency(locations,SEASONS,config):
	for year in range(config['START_YEAR'],config['END_YEAR']+1):
		for season in SEASONS:
			wanted=SEASONS[season]
			latitude = np.asarray(locations['LATITUDE'])
			longitude = np.asarray(locations['LONGITUDE'])
			Yearly_Data=(locations.loc[locations['YEAR']==year])
			Seasonal_Data=(Yearly_Data.loc[Yearly_Data['MONTH'].isin(wanted)])
			lats = np.asarray(Seasonal_Data['LATITUDE'])
			lons = np.asarray(Seasonal_Data['LONGITUDE'])
			Species_count=np.asarray(Seasonal_Data[config['SPECIES']])
			Species_count=np.reshape(Species_count,len(Species_count))
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
			m.drawparallels(np.arange(lat_min,lat_max,config['GRID_SIZE']),labels=[False,True,True,False])
			m.drawmeridians(np.arange(lon_min,lon_max,config['GRID_SIZE']),labels=[True,False,False,True])
			lat, lon = m.makegrid(zi.shape[1], zi.shape[0])
			x,y = m(lat, lon)
			z=zi.reshape(xi.shape)
			levels=np.linspace(0,z.max(),25)
			cm=plt.contourf(x, y, zi,levels=levels,cmap=plt.cm.Greys)
			plt.colorbar()
			plt.title(config['SPECIES']+"-"+str(year)+"-"+str(season))
			#plt.show()
			plt.savefig(config['SPECIES']+"-"+str(year)+"-"+str(season)+".png")
			plt.close()
	return

#Plots the frequency (Y axis) against the timesteps (X axis) for the given location.
#Uses the location's included coordinates to provide a map insert showing a dot for the location on the US map (this should use matplotlib's "axes" interface as with here http://matplotlib.org/examples/pylab_examples/axes_demo.html)
#The optional "predictor" object overlays the expectations of a particular predictor (which is associated with a particular timestamp)


def model_location_novelty_over_time(location,SEASONS,config):
	ModelObject=[]
	Maximum_Error=[]
	Mean_Error=[]
	Model_Name=[]
	mean_abs_errors = []
	explained_var = []
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
	Train_Data_years=[]
	Train_Data_months=[]
	Nonseasonaldata=[]
	Nonseasonaldata_Frequency=[]
	SeasonwiseTrainData=[]
	SeasonwiseTrainDataFrequency=[]
	train=[]
	Nonseasonaldata_timeframe=[]
	tr= pd.DataFrame()
	LocationData = location
	d={}
	
	for predictingyear in range(config['PREDICTION_START_YEAR'],config['END_YEAR']+1):
		predicting_year=[predictingyear]
		Training_years=[]
		for year in range(config['START_YEAR'],predictingyear,1):
			Training_years.append(year)
		for season in SEASONS:
			wanted=SEASONS[season]
			NonSeasonal_Data=(LocationData.loc[~LocationData['MONTH'].isin(wanted)])
			NonSeasonalData=(NonSeasonal_Data.loc[NonSeasonal_Data['YEAR'].isin(Training_years)])
			#NonSeasonalData=NonSeasonalData.append(NonSeasonal_Data[NonSeasonal_Data['YEAR']==predicting_year],ignore_index=True)
			NonSeasonalDataTimeframe=NonSeasonalData['timeframe']
			NonSeasonalDataTimeframe=NonSeasonalDataTimeframe.reshape(-1,1)
			NonSeasonalDataforPlotting = NonSeasonalData['Date_Format']
			NonSeasonalDataFrequency=NonSeasonalData[config['SPECIES']]
			Seasonal_Data=LocationData[LocationData['MONTH'].isin(wanted)]
			Train_Data=Seasonal_Data[Seasonal_Data['YEAR'].isin(Training_years)]
			max_train_year=max(Training_years)
			min_train_year=min(Training_years)
				
			Test_Data=(Seasonal_Data.loc[Seasonal_Data['YEAR'].isin(predicting_year)])
			if season =='WINTER':
				Test_Data=Test_Data[(Test_Data['YEAR']==predicting_year)&(Test_Data['MONTH']!=12)]
				Test_Data=Test_Data.append(Train_Data[(Train_Data['YEAR']==max_train_year)&(Train_Data['MONTH']==12)], ignore_index=True)
				tr=Train_Data[(Train_Data['YEAR']!=max_train_year)]
				Train_Data=tr.append(Train_Data[(Train_Data['YEAR']==max_train_year)&(Train_Data['MONTH']!=12)],ignore_index=True)
				Train_Data=Train_Data.append(Train_Data[(Train_Data['YEAR']==max_train_year)&(Train_Data['MONTH']!=12)],ignore_index=True)
				if min_train_year >=2003:
					Starting_yearData=LocationData[LocationData['MONTH'].isin(SEASONS['WINTER'])]
					Train_Data=Train_Data.append(Starting_yearData[(Starting_yearData['YEAR']==min_train_year-1)&(Starting_yearData['MONTH']==12)],ignore_index=True)
			TrainData_years=Train_Data['YEAR']	
			TrainData_months=Train_Data['MONTH']
			TrainData=Train_Data['timeframe']
			TrainData=TrainData.reshape(-1,1)
			Seasonwise_TrainData=Train_Data['Date_Format']
			Seasonwise_Traindata_Frequency=Train_Data[config['SPECIES']]
			TrainData_Target = Seasonwise_Traindata_Frequency.as_matrix()
			TrainData_Target=TrainData_Target.astype(np.float)
			TestData=Test_Data['timeframe']
			TestData=TestData.reshape(-1,1)
			TestData_Plotting=Test_Data['Date_Format']
			Actual_Species_Count=Test_Data[config['SPECIES']]
			lat=Test_Data['LATITUDE']
			lon=Test_Data['LONGITUDE']
			
			if len(Train_Data)!=0 and len(TestData)!=0:
				if config['PREDICTOR']=='linear':
					regr = linear_model.LinearRegression()
				elif config['PREDICTOR']=='theilsen':
					regr = linear_model.TheilSenRegressor()
				regr.fit(TrainData,TrainData_Target)
				Predicted_Species_Count=regr.predict(TestData)
				MaxError=np.max(abs(Predicted_Species_Count-Actual_Species_Count))
				Maximum_Error.append(MaxError)
				MeanError=np.mean((Predicted_Species_Count - Actual_Species_Count) ** 2)
				Mean_Error.append(MeanError)
				r2_test=regr.score(TestData,Actual_Species_Count)
				r2_train=regr.score(TrainData,TrainData_Target)
				mae_test = mean_absolute_error(Actual_Species_Count,Predicted_Species_Count)
				mae_train = mean_absolute_error(TrainData_Target,regr.predict(TrainData))
				e_v = explained_variance_score(TrainData_Target,regr.predict(TrainData))
				explained_var.append(e_v)
				Regression_Score.append((r2_train,r2_test))
				mean_abs_errors.append((mae_train,mae_test))
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
				Train_Data_years.append(TrainData_years)
				Train_Data_months.append(TrainData_months)
			else:
				continue
			
	d['location']={}
	d['stats']={}
	d['location']['latitude']=latitude
	d['location']['longitude']=longitude
	d['stats']['score']=Regression_Score
	d['stats']['max_error']=np.reshape(Maximum_Error, len(Maximum_Error))
	d['stats']['mean_error']=np.reshape(Mean_Error, len(Mean_Error))
	d["stats"]["mean_abs_errors"] = mean_abs_errors
	d["stats"]["expvar"] = explained_var
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
	d['TrainData_years']=Train_Data_years
	d['TrainData_months']=Train_Data_months
	return d


def plot_birds_over_time(predictors,locations,config):
	locationslatitude = np.asarray(locations['LATITUDE'])
	locationslongitude = np.asarray(locations['LONGITUDE'])
	lat_min = min(locationslatitude)
	lat_max = max(locationslatitude)
	lon_min = min(locationslongitude)
	lon_max = max(locationslongitude)
	if config['use_chance_not_count']:
		SeasonTrainData_Label="Sighting chance (months)"
		TestData_Label="Sighting chance( )"
		NonSeasonalData_Label="Sighting chance (all months)" 
		RegressorLine_Label="Expected sighting chance"
		YAxis_Label="Chance to see"
	else:
		SeasonTrainData_Label="Sightings( months)"
		TestData_Label="Sightings( )"
		NonSeasonalData_Label="Sightings(all months)"
		RegressorLine_Label="Expected sightings"
		YAxis_Label="Number of sightings"
		
	for p in predictors:
		for Model_Object,Predictions,Actualspecies_count,TestDataforplotting,latitude,longitude,NonSeasonalDataFrequency,NonSeasonalData,season,predicting_year,SeasonTrainData,SeasonTrainDataFrequency,TrainData,Nonseasonaldatamonths,TrainData_years,TrainData_months,mean_abs_errors in zip(p["Model_object"],p["predictions"],p["actualspeciescount"],p["TestDataforplotting"],p["location"]["latitude"],p["location"]["longitude"],p['NonSeasonalDataFrequency'],p['NonSeasonalData'],p['seasonlist'],p['predictingyearlist'],p['seasonwisetraindata'],p['seasonwisetrainDatafrequency'],p['traindata'],p['Nonseasonaldata-timeframe'],p['TrainData_years'],p['TrainData_months'],p["stats"]["mean_abs_errors"]):
			plt.figure(figsize=(25,20))
			#fig.set_size_inches(18.5, 10.5, forward=True)
			TestDataforplotting = [dt.datetime.strptime(d,'%Y-%m').date() for d in TestDataforplotting]
			AllData = pd.concat([NonSeasonalData,SeasonTrainData])
			AllDataFrequency = pd.concat([NonSeasonalDataFrequency,SeasonTrainDataFrequency])
			AllDataSeasonal_frame=pd.DataFrame({"dates":[dt.datetime.strptime(d,'%Y-%m').date() for d in SeasonTrainData],"freqs":SeasonTrainDataFrequency,"years":TrainData_years,"season":season,"MONTH":TrainData_months}).sort_values("dates")
			AllDataNonSeasonal_frame=pd.DataFrame({"dates":[dt.datetime.strptime(d,'%Y-%m').date() for d in NonSeasonalData],"freqs":NonSeasonalDataFrequency}).sort_values("dates")
			AllData_frame = pd.DataFrame({"dates":[dt.datetime.strptime(d,'%Y-%m').date() for d in AllData],"freqs":AllDataFrequency}).sort_values("dates")
			alldata_array=np.asarray(AllData_frame)
			#season_array=np.asarray(AllDataSeasonal_frame)
			non_season_array=np.asarray(AllDataNonSeasonal_frame)
			lat=np.unique(latitude)
			lon=np.unique(longitude)
			plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
			plt.gca().xaxis.set_minor_formatter(mdates.DateFormatter('%M'))
			plt.gca().xaxis.set_minor_locator(mdates.MonthLocator())
			plt.gcf().autofmt_xdate()
			SeasonTrainData = [dt.datetime.strptime(d,'%Y-%m').date() for d in SeasonTrainData]
			NonSeasonalData=[dt.datetime.strptime(d,'%Y-%m').date() for d in NonSeasonalData]
			plt.ylim([0,100])
			#plt.scatter(NonSeasonalData,NonSeasonalDataFrequency*100,alpha=0.3,label='Non Seasonal Datapoints')    #Scatter plot of Non-Seasonal-Data over Training years
			seasontraindatapredictions=Model_Object.predict(TrainData)     # Predicting the frequency for the Seasonal Train Data using regression model
			plt.scatter(SeasonTrainData,SeasonTrainDataFrequency*100,label=SeasonTrainData_Label)
			plt.scatter(TestDataforplotting,Actualspecies_count*100,color='black',label=TestData_Label)              #Scatter plot of Test Data with Actual Frequencies
			grouped_seasonedTraindata=AllDataSeasonal_frame.groupby('years')
			plt.plot(AllData_frame["dates"].tolist(),AllData_frame["freqs"]*100,linewidth=0.6,alpha=0.7,label=NonSeasonalData_Label)
			season_name=AllDataSeasonal_frame.season.unique()
			if season_name=='WINTER':
				year_values=AllDataSeasonal_frame.years.unique()
				for year in year_values:
					seasontraindata_winter=AllDataSeasonal_frame[(AllDataSeasonal_frame['years']==year)&(AllDataSeasonal_frame['MONTH']!=12)]
					seasontraindata_winter=seasontraindata_winter.append(AllDataSeasonal_frame[(AllDataSeasonal_frame['years']==year-1)&(AllDataSeasonal_frame['MONTH']==12)],ignore_index=True)
					seasontraindata_winter=seasontraindata_winter.sort(['years','MONTH'], ascending=[True,True])
					yearslist=seasontraindata_winter.years.unique()
					season_array=np.asarray(seasontraindata_winter[['dates','freqs']])
					for start, stop in zip(season_array[:-1], season_array[1:]):
						x, y = zip(start, stop)
						plt.plot(x,y*100,color='blue')
					
					if len(seasontraindata_winter)>=2:
						if year-1 in yearslist:
							verticalarea_startdate=np.asarray(seasontraindata_winter[(seasontraindata_winter['years']==year-1)]['dates'])
							area_startdate=verticalarea_startdate[0]
							dft=seasontraindata_winter[(seasontraindata_winter['years']==year)]
							if len(dft)==1:
								verticalarea_enddate=np.asarray(seasontraindata_winter[(seasontraindata_winter['years']==year)]['dates'])
								area_enddate=verticalarea_enddate[0]
							else:
								verticalarea_enddate=np.asarray(dft[(dft['MONTH']==dft['MONTH'].max())]['dates'])
								area_enddate=verticalarea_enddate[0]	
						else:
							verticalarea_startdate=np.asarray(seasontraindata_winter[(seasontraindata_winter['years']==year)&(seasontraindata_winter['MONTH']==seasontraindata_winter['MONTH'].min())]['dates'])
							area_startdate=verticalarea_startdate[0]	
							verticalarea_enddate=np.asarray(seasontraindata_winter[(seasontraindata_winter['years']==year)&(seasontraindata_winter['MONTH']==seasontraindata_winter['MONTH'].max())]['dates'])
							area_enddate=verticalarea_enddate[0]
						plt.axvspan(area_startdate,area_enddate,color='b',alpha=0.1,lw=1)					
			else:		
				for name,group in grouped_seasonedTraindata:
					season_array=np.asarray(group[['dates','freqs']])
					for start, stop in zip(season_array[:-1], season_array[1:]):
						x, y = zip(start, stop)
						plt.plot(x,y*100,color='blue')
					if len(group) >=2:
						verticalarea_startdate=np.asarray(group[(group['MONTH']==group['MONTH'].min())]['dates'])
						area_startdate=verticalarea_startdate[0]
						verticalarea_enddate=np.asarray(group[(group['MONTH']==group['MONTH'].max())]['dates'])
						area_enddate=verticalarea_enddate[0]
						plt.axvspan(area_startdate,area_enddate,color='b',alpha=0.1,lw=1)
			plt.plot(TestDataforplotting,Predictions*100,'r-',linewidth=1.5,label=RegressorLine_Label)   #Plotting Regressor line for Test Data
			plt.plot(SeasonTrainData,seasontraindatapredictions*100,'r-',linewidth=1)  #plotting predictor line for sesonal train data
			#plt.plot(NonSeasonalData,NonSeasonalDataFrequency,linewidth=0.25,alpha=0.3,label=NonSeasonalData_Label)
			#plt.plot(SeasonTrainData,SeasonTrainDataFrequency,'b-',linewidth=0.6,alpha=0.7,label='SeasonalData_Line')
			plt.title(config["SPECIES"]+"\n"+str(config['PREDICTION_START_YEAR'])+"-"+str(config['END_YEAR'])+"\n"+str(season),loc='left')
			#plt.legend(fontsize ='x-small',labelspacing=0.2,bbox_to_anchor=(1, 1),bbox_transform=plt.gcf().transFigure)
			plt.tight_layout(pad=20)
			plt.xlabel("Time")
			plt.ylabel(YAxis_Label)
			plt.xticks(rotation='horizontal')
			plt.setp(plt.gca().get_xminorticklabels(),visible=False)
			#labels = [item.get_text() for item in plt.gca().get_xminorticklabels()]
			#empty_string_labels = ['']*len(labels)
			#plt.gca().set_xticklabels(empty_string_labels)
			#plt.minorticks_off()
			x1,x2,y1,y2=plt.axis()
			plt.axis((x1,x2,0,y2))
			insetfig= plt.axes([0.6,0.8,0.2,0.2])							#Setting coordinates and width,height of inset 
			themap=Basemap(projection='merc',llcrnrlat=lat_min-config['GRID_SIZE'],urcrnrlat=lat_max+config['GRID_SIZE'],llcrnrlon=lon_min-config['GRID_SIZE'],urcrnrlon=lon_max+config['GRID_SIZE'],rsphere=6371200.,resolution='l',area_thresh=10000)
			#themap = Basemap(llcrnrlon=-126, llcrnrlat=16, urcrnrlon=-64,urcrnrlat=49, projection='lcc', lat_1=33, lat_2=45,lon_0=-95, resolution='h', area_thresh=10000)
			reclats=[lat,lat+config['GRID_SIZE'],lat+config['GRID_SIZE'],lat]   #Rectangular latitude coordinates for displaying grid in plot
			reclons=[lon,lon,lon+config['GRID_SIZE'],lon+config['GRID_SIZE']]	#Rectangular longitude coordinates for displaying grid in plot
			#themap.bluemarble()
			themap.drawcoastlines()
			themap.drawcountries(linewidth=2)
			themap.drawstates()
			#themap.fillcontinents(color='gainsboro')
			#themap.drawmapboundary(fill_color='steelblue')
			x3,y3=themap(reclons,reclats)
			x3y3=zip(x3,y3)
			p= Polygon(x3y3, facecolor='red', alpha=0.4)       #Plotting rectangular polygon grid in Basemap
			plt.gca().add_patch(p)
			longg,latt=lon,lat
			x, y = themap(longg,latt)
			lonpoint, latpoint = themap(x,y,inverse=True)       
			plt.title("Location"+'(%5.1fW,%3.1fN)'%(lonpoint,latpoint),fontsize=10)
			plt.xticks([])
			plt.yticks([])
			figure_name=str(config['SPECIES'])+"-"+str(lat)+"-"+str(lon)+"-"+str(predicting_year)+"-"+str(season)+"-"+config['PREDICTOR']+".png"
			if not os.path.isdir(config["RUN_NAME"]):
				os.mkdir(config["RUN_NAME"])
			destination_dir=os.path.abspath(config["RUN_NAME"])
			plt.savefig(os.path.join(destination_dir,figure_name))
			#plt.show()
			plt.close()

def plot_predictors(predictors,config, max_size,out_fname, minlimit = -100):
	predictor_coefs = []
	predictor_intercepts = []
	predictor_variance = []
	predictor_surprise = []
	predictor_train_maes = []
	predictor_test_maes = []
	predictor_names = []
	predictor_expvar = []
	for p in predictors:
		for model,score,errors,season,predicting_year,lat,long,expvar in zip(p["Model_object"],p["stats"]["score"],p["stats"]["mean_abs_errors"],p["seasonlist"],p["predictingyearlist"],p["location"]["latitude"],p["location"]["longitude"],p["stats"]["expvar"]):
			if predicting_year >= config['PREDICTION_START_YEAR']:
				predictor_coefs.append(model.coef_[0])
				predictor_intercepts.append(model.intercept_)
				predictor_expvar.append(expvar)
				predictor_variance.append(score[0])
				predictor_surprise.append(score[1])
				predictor_train_maes.append(errors[0])
				predictor_test_maes.append(errors[1])
				predictor_names.append(str(lat.values[0])+"_"+str(long.values[0])+"_"+str(predicting_year)+"_"+str(season))
	pred_df = pd.DataFrame({"name":predictor_names,"coef":predictor_coefs,"intercept":predictor_intercepts,"train_maes":predictor_train_maes,"test_maes":predictor_test_maes,"expvar":predictor_expvar,"test/train_mae":[tr/te for tr,te in zip(predictor_test_maes,predictor_train_maes)]}) #"r2_train":predictor_variance,"r2_test":predictor_surprise,
	pd.set_option('expand_frame_repr', False)
	print pred_df
	sorted_pred_df = pred_df.sort_values("test/train_mae")
	print sorted_pred_df
	plt.figure(figsize=(10,10))
	plt.scatter(predictor_train_maes,predictor_test_maes)
	plt.xlabel("Train MAE")
	plt.ylabel("Test MAE")
	#plt.show()
	plt.savefig(str(out_fname)+"variance_by_surprise.png")
	plt.close()
