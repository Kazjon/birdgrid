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
	observations=observations.replace('X',1) 
	observations=observations.replace('?',0)
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
Grid_Data=init_birdgrid(observations,grid_size,species)
	

#Plot the actual species frequency (from the data) on a map
def plot_observation_frequency(observations):
	pass

#Plot the locations of each predictor on a map, overlaid on plot_observation_frequency
def plot_predictor_locations(predictors,observations):
	pass
	
#Plots the frequency (Y axis) against the timesteps (X axis) for the given location.
#Uses the location's included coordinates to provide a map insert showing a dot for the location on the US map (this should use matplotlib's "axes" interface as with here http://matplotlib.org/examples/pylab_examples/axes_demo.html)
#The optional "predictor" object overlays the expectations of a particular predictor (which is associated with a particular timestamp)
def plot_birds_over_time(location, predictor=None):
	pass
	
#Makes a prediction of the observation for each timestep as a sklearn Pipeline object
#To start with, try predicting for each month using the data only for that season.  That should allow you to use linear regression.
def model_location_novelty_over_time(location):
	pass