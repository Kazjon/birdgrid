import numpy as np
from birdgridhelpers import load_observations,init_birdgrid,plot_observation_frequency,model_location_novelty_over_time,plot_birds_over_time, plot_predictors
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

predictors = []
SEASONS = {"WINTER": [12,1,2],"SPRING": [3,4,5],"SUMMER":[6,7,8],"FALL":[9,10,11]}
config={}
config['SPECIES']='Cathartes_aura'
config["TIME_STEP"] = "monthly"
config["ATTRIBUTES"] = ['LATITUDE','LONGITUDE','YEAR','MONTH']
config['START_YEAR']=2003
config['PREDICTION_START_YEAR']=2010
config['END_YEAR']=2012
config['GRID_SIZE']=5
config['PREDICTOR']="theilsen"
config['use_chance_not_count']=True
config['REGRESSION_LINE']=True
if config['use_chance_not_count']:
	Model_mode="chance_mode"
else:
	Model_mode="count_mode"

config["RUN_NAME"]=str(config['SPECIES'])+"-"+str(config['START_YEAR'])+"-"+str(config['END_YEAR'])+"-"+str(config['GRID_SIZE'])+"-"+str(config['PREDICTOR'])+"-"+Model_mode

if os.path.isfile(config["RUN_NAME"]+"_predictors.p") and os.path.isfile(config["RUN_NAME"]+"_locations.p"):
	locations=pd.read_pickle(config["RUN_NAME"]+"_locations.p")
	with open(config["RUN_NAME"]+'_predictors.p',"rb") as pf:
		predictors = pickle.load(pf)
else:
	if os.path.isfile(config["RUN_NAME"]+"_locations.p"):
		locations=pd.read_pickle(config["RUN_NAME"]+"_locations.p")
	else:
		observations = load_observations(config) #Load these in from somewhere, one row per observation, columns 0 and 1 are lat and lon
		locations=init_birdgrid(observations,config)  #Calculate these from the above, Array of dicts, each dict contains lat, lon and data for each timestep


	#Plot our species frequency observations
	plot_observation_frequency(locations,SEASONS,config)
	
	# matrix of models of shape locations x timesteps.
	
	for k,location in locations.groupby(['LATITUDE','LONGITUDE'],as_index=False):
		predictors.append(model_location_novelty_over_time(location,SEASONS,config))
	with open(config["RUN_NAME"]+'_predictors.p',"wb") as pf:
		pickle.dump(predictors,pf)

plot_birds_over_time(predictors,locations,config)

plot_predictors(predictors, config, max_size=100, out_fname =config['RUN_NAME'])