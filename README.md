# birdgrid
Unexpectedness analytics in eBird data

# birdgrid_timeseries.py

This file will have function calls which are defined in the birdgridhelpers.py file
All the parameter values will be stored in the Config dictionary object.

Config['ATTRIBUTES'] will consider only the specified columns from the data file.
Config['START_YEAR'] will load the data from the specified year till Config['END_YEAR']
config['PREDICTION_START_YEAR'] will start the prediction from the specified year till the config['END_YEAR']
config['GRID_SIZE'] will divide the latitude and longitude into the specified grid size.                                                                             

config['PREDICTOR'] takes either "theilsen"  or "linear".
"Theilsen" regression algorithm is implemented when "theilsen" is considered otherwise "linearregression" is implemented.                                                               

config['use_chance_not_count'] is a boolean value which is either "True" or "False" . 
If "True" chance mode is considered otherwise species frequency is considered.                                                              

config['REGRESSION_LINE'] will take any one value from ['True','False','nodata']
If it is "True" then the plot will display the regression line
If it is "False" then the regression line will be hidden from the plot.
If it is "nodata" then the dark blue line,regression line,vertical bar and test data points will be hidden from the plot. 

