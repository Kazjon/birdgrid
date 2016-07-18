import numpy as np
import pandas as pd
import glob

#Takes in a set of desired attributes, the species, and the year range
#returns an observation x [lat, lon, season, attribute_1, attribute_2,...attribute_n] matrix 
def load_observations(attributes,species,start_year,end_year):
	path =r'C:\Users\swarna\Desktop\DENSITY\Grid\glob' 
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
'''
attributes=['LATITUDE','LONGITUDE','YEAR','MONTH']
species=['Turdus_migratorius']
start_year=2002
end_year=2004
observations=load_observations(attributes,species,start_year,end_year)
grid_size=1
'''
#Takes in the matrix of observations and bins it into observations based on the provided grid size.
#Returns an array of dicts, each dict represents one location and contains lat, lon and data for each timestep
#Returns a dataframe of attributes that are divided into grids, where as each grid square represents the total count of the species found in that location
def init_birdgrid(observations,grid_size,species):
	#headers=observations.columns.values
	lats=observations['LATITUDE']
	lons=observations['LONGITUDE']
	observations=observations.convert_objects(convert_numeric=True)
	lat_min = int(math.floor(min(lats)))
	lat_max = int(math.floor(max(lats)))
	lon_min =int(math.floor(min(lons)))
	lon_max =int(math.floor(max(lons)))
	GridSquare=[]
	df=pd.DataFrame([])
	
	#starting the grid value with the minimum latitude and longitude values and iterating through the size of the grid 

	for i in range(lat_min,lat_max,grid_size):
		for j in range(lon_min,lon_max,grid_size):
			GridSquare=observations[(observations['LATITUDE']>=i)&(observations['LATITUDE']<i+grid_size)&(observations['LONGITUDE']>=j)&(observations['LONGITUDE']<j+grid_size)]
			GridSquare['LATITUDE']=i #replacing the lat value with the starting value of that particular grid
			GridSquare['LONGITUDE']=j #replacing the lon value with that of the starting value of that particular grid
			#replaced lat,lon in order to group the species count
			GridwiseCount=GridSquare.groupby(['LATITUDE','LONGITUDE','YEAR','MONTH'],as_index=False)[species].sum() 
			df=df.append(GridwiseCount)
	return df

	

#Plot the actual species frequency (from the data) on a map
def plot_observation_frequency(locations):
	pass
	
#Plots the frequency (Y axis) against the timesteps (X axis) for the given location.
#Uses the location's included coordinates to provide a map insert showing a dot for the location on the US map (this should use matplotlib's "axes" interface as with here http://matplotlib.org/examples/pylab_examples/axes_demo.html)
#The optional "predictor" object overlays the expectations of a particular predictor (which is associated with a particular timestamp)
def plot_birds_over_time(location, predictor=None):
	pass
	
#Makes a prediction of the observation for each timestep as a sklearn Pipeline object
#To start with, try predicting for each month using the data only for that season.  That should allow you to use linear regression.
location=[41,-88] #selecting a gridsquare coordinates
def model_location_novelty_over_time(location):
	Lat=location[0]
	Lon=location[1]
	LocationData = locations[((locations['LATITUDE']==Lat) & (locations['LONGITUDE']==Lon))]
	seasons = {"winter": [12,1,2],"spring": [3,4,5],"summer":[6,7,8],"fall":[9,10,11]}
	years=[2002,2003,2004,2005,2006,2007,2008,2009,2010,2011]
	Training_years=[]
	#looping through each and every year. Starting from year 2002. Trains on 2002 and tests on 2003. later trains on 2002,2003 and tests on 2004..
	for year in years:  
		Training_years.append(year)
		predicting_year=[year+1]
		for i,season in zip(range(8),seasons):
			plt.subplot(4,2,i+1)
			wanted=seasons[season]
			Seasonal_Data=(LocationData.loc[LocationData['MONTH'].isin(wanted)])
			Train_Data=(Seasonal_Data.loc[Seasonal_Data['YEAR'].isin(Training_years)])
			Test_Data=(Seasonal_Data.loc[Seasonal_Data['YEAR'].isin(predicting_year)])
			Train_Data['PERIOD'] = Train_Data.YEAR.astype(str).str.cat(Train_Data.MONTH.astype(str),sep='0')
			Test_Data['PERIOD'] = Test_Data.YEAR.astype(str).str.cat(Test_Data.MONTH.astype(str),sep='0')
			TrainData=Train_Data['PERIOD']
			TrainData=TrainData.reshape(-1, 1)
			TrainData=TrainData.astype(np.float)
			TrainData_Target=Train_Data[species]
			TrainData_Target = TrainData_Target.as_matrix()
			TrainData_Target=TrainData_Target.astype(np.float)
			TestData=Test_Data['PERIOD']
			TestData=TestData.reshape(-1, 1)
			TestData=TestData.astype(np.float)
			ActualResult=Test_Data[species]
			regr = linear_model.LinearRegression()
			regr.fit(TrainData,TrainData_Target)
			PredictedSpeciesCount=regr.predict(TestData)
			print('Predicted Species Count:',PredictedSpeciesCount)
			print('Actual Count of Species:',ActualResult)
			ErrorInPrediction=abs(PredictedSpeciesCount-ActualResult)
			print('Individual Error:',ErrorInPrediction) 

			MaximumError=max(ErrorInPrediction)
			print('Highestindividual error:',MaximumError)

			print("Residual sum of squares: %.2f",np.mean((regr.predict(TestData) - ActualResult) ** 2))
			print('Coefficients: \n', regr.coef_)
			plt.scatter(TestData,ActualResult,color='black')
			
			plt.plot(TestData, regr.predict(TestData), color='blue',linewidth=3)
			plt.title(season)
			plt.xticks(())
			plt.yticks(())
			 
		plt.show()
	return

	
