# ecs171project

Team Members: 
Jamie Wu
Jordan Lim
Rohan Arumugam
Elson Jian
Hyunkyong (HK) Boo 	
Juntao Wang

Data Exploration Step:
1. Evaluate Data
2. Number of Observations - 10.1k (insert precise number)
3. Details about data distributions - normal, poisson, etc.?
4. Scales - any transformations we want to put such as log transformation, this can be based on any trends we see on pairplots 
5. Missing Data - code stuff to check for NA values
6. Column Descriptions - how sensors are grouped and labeled into different columns, units and how they might impact outcome (ex. cm for one attribute vs m for another)
   
Plot Data:
1. Pairplots
2. Correlation plot
3. Plot of how different sensor data changes over time

How will the data be preprocessed?
1. Do we want to standardize or normalize? why
2. How can we preprocess data and time data from a text to numerical format? Ex. Make the first date the starting point, and the rest be numerical datapoints showing time elapsed since then?

Running Questions:
1. How do we want to handle the time-series data, and track changes in sensor data over time
2. How will we interpret sensor data coming from multiple of the same type of sensor?
3. How will we detect changes in room occupancy? Should we be primarily looking at inflection points where the number of people in the room change, and instead of looking at raw sensor data, change the datapoints to be the amount of change compared to the previous time stamp? How might we process the data to show this

Installs:
!wget !unzip like functions as well as !pip install functions for non standard libraries not available in colab are required to be in the top section of your jupyter lab notebook

Note: I will grade this part of your submission as if it were finalized! You are expected to finalize the data exploration by Sunday.
