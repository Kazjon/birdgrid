import numpy as np
from birdgridhelpers import load_observations,init_birdgrid,plot_observation_frequency,model_location_novelty_over_time,plot_birds_over_time
import numpy as np
import glob
import math
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
START_YEAR = 2002
END_YEAR = 2012
TIME_STEP = "monthly" 
GRID_SIZE = 1 #Side length of each grid square (in degrees lat/lon)
predictors = []
SEASONS = {"winter": [12,1,2],"spring": [3,4,5],"summer":[6,7,8],"fall":[9,10,11]}

observations = load_observations(ATTRIBUTES, SPECIES, START_YEAR, END_YEAR) #Load these in from somewhere, one row per observation, columns 0 and 1 are lat and lon
locations=init_birdgrid(observations,GRID_SIZE,SPECIES,TIME_STEP,START_YEAR,END_YEAR)  #Calculate these from the above, Array of dicts, each dict contains lat, lon and data for each timestep

#Plot our species frequency observations
plot_observation_frequency(locations,SEASONS,GRID_SIZE,START_YEAR,END_YEAR)
'''
#For each location (grid square), plot the birdcount over time. Additionally display the location within the US on a small inset mapbox
for k,location in locations.groupby(['LATITUDE','LONGITUDE'],as_index=False):
	plot_birds_over_time(location,SPECIES)
'''
# matrix of models of shape locations x timesteps. 
for k,location in locations.groupby(['LATITUDE','LONGITUDE'],as_index=False):
	predictors.append(model_location_novelty_over_time(location,SPECIES,SEASONS,START_YEAR,END_YEAR))
	
plot_birds_over_time(predictors)

'''
#Each location will have an array of predictors associated with it -- one per timestep.
for preds,loc in zip(predictors,locations):
	for p in preds:
		plot_birds_over_time(loc, predictor=p)
		
plot_predictors(predictors)
'''