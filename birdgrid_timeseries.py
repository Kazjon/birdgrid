import numpy as np
from geopy.distance import vincenty
from birdgridhelpers import load_observations,init_birdgrid,plot_observation_frequency,plot_predictor_locations

SPECIES = ""
ATTRIBUTES = []
START_YEAR = 2002
END_YEAR = 2016
TIME_STEP = "monthly" #Can switch to "yearly" or "seasons"
GRID_SIZE = 1 #Side length of each grid square (in degrees lat/lon)

observations = load_observations(ATTRIBUTES, SPECIES, START_YEAR, END_YEAR) #Load these in from somewhere, one row per observation, columns 0 and 1 are lat and lon
locations = init_birdgrid(observations, GRID_SIZE, TIME_STEP) #Calculate these from the above, one row per location, columns 0 and 1 are lat and lon

#Plot our species frequency observations
plot_observation_frequency(observations)

#For each location (grid square), plot the birdcount over time. Additionally display the location within the US on a small inset mapbox
for loc in locations:
	plot_birds_over_time(loc)
	
predictors = [] # Array of models of same length as locations. The format of each entry will depend on whether we're doing one predictor for the whole year, or one per season.
for loc in locations:
	predictors.append(model_bird_novelty(loc))