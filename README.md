# ECS 171 Project - Data Exploration
Dataset: https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation

Team Members: \
Jamie Wu - jmewu@ucdavis.edu\
Jordan Lim - jflim@ucdavis.edu\
Rohan Arumugam - rarumugam@ucdavis.edu\
Elson Jian - egjian@ucdavis.edu\
Hyunkyong (HK) Boo - hboo@ucdavis.edu\
Juntao Wang - jutwang@ucdavis.edu

Attribute Description: 10129 Observations, 19 attributes with 1 class attribute (Room Occupancy)
1. Date: formatted as YYYY/MM/DD
2. Time: formatted as HH:MM:SS
3. S1_Temp: Sensor 1 temperature readings, measured in ℃
4. S2_Temp: Sensor 2 temperature readings, measured in ℃
5. S3_Temp: Sensor 3 temperature readings, measured in ℃
6. S4_Temp: Sensor 4 temperature readings, measured in ℃
7. S1_Light: Sensor 1 light intensity, measured in lux
8. S2_Light: Sensor 2 light intensity, measured in lux
9. S3_Light: Sensor 3 light intensity, measured in lux
10. S4_Light: Sensor 4 light intensity, measured in lux
11. S1_Sound: Sound level from sensor 1 in volts
12. S2_Sound: Sound level from sensor 2 in volts
13. S3_Sound: Sound level from sensor 3 in volts
14. S4_Sound: Sound level from sensor 4 in volts
15. S5_CO2: CO2 levels in the air of the room, measured in parts per million (PPM)
16. S5_CO2_Slope: Rate of change of CO2 levels taken on a sliding scale
17. S6_PIR: Motion detection readings from digital passive infrared (PIR) sensors at sensor 6, binary value that ranges from 0-1 (0 for no motion, 1 for motion detected)
18. S7_PIR: Motion detection readings from digital passive infrared (PIR) sensors at sensor 7, binary value that ranges from 0-1 (0 for no motion, 1 for motion detected)
19. Room_Occupancy_Count: Ground truth for number of occupants inside room

Dataset General Observations:
1. We want to track changes in sensor output over time, so we processed the “Date” and “Time” attributes into numerical values. We used Pandas to.datetime() to turn the two attributes into one attribute to plot overall changes in sensor output over time. We noticed many peaks in sensor output intensity during the first 3 and last 2 days of the experiment, with more minimal steady growth for the middle 10-14 days.
2. Our dataset is structured in groups of sensor types, with multiple of the same type of sensor placed around the room. In the data exploration phase our team noticed that different sensors (though the same type) exhibited differences in predictive impact for room occupancy. For example, in the correlation matrix S1_Temp has a 0.7 correlation with room occupancy and S1-3_Temp all have correlations of 0.65+, but S4_temp has a much lower correlation value of 0.53. To make sense of these sort of differences, we sourced a diagram of sensor placement that was associated with the dataset. This helps provide some context to differences in sensor outputs that we saw during data exploration. For example, one explanation for the differences in Temp sensor correlation could be because the placement of sensor S4 is in the upper right corner of the room, far from central node N and by a window.\
![Screenshot 2023-08-24 231529](https://github.com/jordlim/ecs171project/assets/115687850/bbcb0763-c36e-4c92-ba9d-3cb4c5e03cdf)

Transformations and Normalization:
1. During data exploration, we noticed that attributes like temperature (in Celsius) changed by the hundredths, while other attributes such as light (in Lux) changed much more drastically on the order of ones and tens. Because the range of the different axis are very different between each sensor type, we felt that the relationship was unclear. Some data points from the pair plots were also skewed or cluttered together with several points being significantly more distinct.
2. Because the sensors from the experiment measured different variables with different units, we decided to normalize the data in order to reduce the rate of change differences between variable measurements for different observations. 

Data Segmentation:
1. Our project aims to gain insight into how specific sensor values correlate with the room occupancy at a discrete point in time, and also how changes in sensor outputs over time can predict increases and decreases in room occupancy. To do this, we felt that it was necessary to process the original dataframe into two separate dataframes, retaining specific attributes for the “discrete” dataframe and processing its data into a secondary “delta” dataframe.
- Discrete Dataframe: From the unmodified dataset, we dropped the “S5_CO2_Slope” attribute because it described changes in “S5_CO2" which we felt aligns better with the delta dataframe.
- Delta Dataframe: Using the discrete dataframe, we use a loop to generate a new dataframe that instead records changes in sensor values in comparison to the previous timestamp. We retained the “S6_PIR” and “S7_PIR” attributes, which are binary values indicating the presence of movement in the room. These two attributes will help reinforce our calculation of how the room occupancy may have changed.
Below is an example diagram of how the discrete dataframe can be processed into a delta dataframe. \
![Screenshot 2023-08-24 231613](https://github.com/jordlim/ecs171project/assets/115687850/6f3f6ed3-5946-4072-8a4f-0457709918f2)



