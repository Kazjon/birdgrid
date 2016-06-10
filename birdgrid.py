import numpy as np
from geopy.distance import vincenty
from birdgridhelpers import load_observations,init_birdgrid,plot_observation_frequency,plot_predictor_locations

SPECIES = ""
ATTRIBUTES = []
START_YEAR = 2002
END_YEAR = 2016
LOCATIONS_PER_PREDICTOR = 1000 # Roughly how many gridpoints will be associated with each predictor
MIN_DIST_BETWEEN_PREDICTORS = 100 # Number of kilometers between a new predictor and existing ones below which we'll re-sample
GRID_SIZE = 0.1 #Side length of each grid square (in degrees lat/lon)

observations = load_observations(ATTRIBUTES, SPECIES, START_YEAR, END_YEAR) #Load these in from somewhere, one row per observation, columns 0 and 1 are lat and lon, and column 3 is the season
locations = init_birdgrid(observations, GRID_SIZE) #Calculate these from the above, Array of dicts, each dict contains lat, lon and data for each timestep
num_predictors = len(locations) / LOCATIONS_PER_PREDICTOR
predictor_locations = np.zeros((2,num_predictors))

#Plot our species frequency observations
plot_observation_frequency(observations)

#Sample uniformly from observations to determine predictor sites, throwing back any that are too close to existing sites
valid_predictors_found = 0
while valid_predictors_found < len(predictor_locations):
	loc = np.random.choice(observations)[:2]
	valid = True
	for p in predictor_locations:
		if vincenty(p,loc) < MIN_DIST_BETWEEN_PREDICTORS:
			valid = False
			break
	if valid:
		predictor_locations[valid_predictors_found,:] = loc
		
#Display our predictor locations 
plot_predictor_locations(predictors,observations)

#Determine the 