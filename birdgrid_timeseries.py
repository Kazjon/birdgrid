import numpy as np
from birdgridhelpers import load_observations,init_birdgrid,plot_observation_frequency,model_location_novelty_over_time,plot_birds_over_time
import numpy as np
import glob
import math
import pickle
import os.path
import pandas as pd
from scipy import interpolate
import datetime as dt
from datetime import datetime
from mpl_toolkits.basemap import Basemap
from matplotlib.mlab import griddata
from scipy import interpolate
import csv 
import matplotlib.pyplot as plt
from sklearn import linear_model
import matplotlib.dates as mdates

SPECIES = ['Turdus_migratorius']
ATTRIBUTES = ['LATITUDE','LONGITUDE','YEAR','MONTH']
TIME_STEP = "monthly" 
predictors = []
SEASONS = {"winter": [12,1,2],"spring": [3,4,5],"summer":[6,7,8],"fall":[9,10,11]}
config={}
config['SPECIES-Name']=SPECIES[0]
config['START_YEAR']=2002
config['PREDICTION_START_YEAR']=2008
config['END_YEAR']=2012
config['GRID_SIZE']=5 #Side length of each grid square (in degrees lat/lon)
config['PREDICTOR']="theilsen"
config['use_chance_not_count']=False
DIRECTORY_NAME=str(config['SPECIES-Name'])+"-"+str(config['START_YEAR'])+"-"+str(config['END_YEAR'])+"-"+str(config['GRID_SIZE'])+"-"+str(config['PREDICTOR'])
PICKLE_NAME=str(config['SPECIES-Name'])+"-"+str(config['START_YEAR'])+"-"+str(config['END_YEAR'])+"-"+str(config['GRID_SIZE'])+"-"+str(config['PREDICTOR'])

if os.path.isfile(PICKLE_NAME+".p"):
	locations=pd.read_pickle(PICKLE_NAME+".p")

else:
	observations = load_observations(ATTRIBUTES,SPECIES,config) #Load these in from somewhere, one row per observation, columns 0 and 1 are lat and lon
	locations=init_birdgrid(observations,SPECIES,TIME_STEP,PICKLE_NAME,config)  #Calculate these from the above, Array of dicts, each dict contains lat, lon and data for each timestep

#Plot our species frequency observations
plot_observation_frequency(locations,SEASONS,SPECIES,config)

# matrix of models of shape locations x timesteps. 
for k,location in locations.groupby(['LATITUDE','LONGITUDE'],as_index=False):
	predictors.append(model_location_novelty_over_time(location,SPECIES,SEASONS,config))


plot_birds_over_time(predictors,SPECIES,locations,DIRECTORY_NAME,config)

'''
plot_predictors(predictors,max_size=100, out_fname ='predictor_plot')
'''