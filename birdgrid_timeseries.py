import numpy as np
from geopy.distance import vincenty
from birdgridhelpers import load_observations,init_birdgrid,plot_observation_frequency,plot_predictor_locations

SPECIES = ""
ATTRIBUTES = []
START_YEAR = 2002
END_YEAR = 2016
GRID_SIZE = 1 #Side length of each grid square (in degrees lat/lon)

observations = load_observations(ATTRIBUTES, SPECIES, START_YEAR, END_YEAR) #Load these in from somewhere, one row per observation, columns 0 and 1 are lat and lon
locations = init_birdgrid(observations, GRID_SIZE, TIME_STEP) #Calculate these from the above, Array of dicts, each dict contains lat, lon and data for each timestep

#Plot our species frequency observations
plot_observation_frequency(locations)

#For each location (grid square), plot the birdcount over time. Additionally display the location within the US on a small inset mapbox
for loc in locations:
	plot_birds_over_time(loc)
	
predictors = [] # matrix of models of shape locations x timesteps. 
for loc in locations:
	predictors.append(model_location_novelty_over_time(loc))

#Each location will have an array of predictors associated with it -- one per timestep.
for preds,loc in zip(predictors,locations):
	for p in preds:
		plot_birds_over_time(loc, predictor=p)
		
plot_predictors(predictors)