import numpy as np

#Takes in a set of desired attributes, the species, and the year range
#returns an observation x [lat, lon, season, attribute_1, attribute_2,...attribute_n] matrix 
def load_observations(attributes,species, start_year, end_year):
	return np.array([])

#Takes in the matrix of observations and bins it into observations based on the provided grid size.
#Returns an array of dicts, each dict represents one location and contains lat, lon and data for each timestep
def init_birdgrid(observations, grid_size):
	return []

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