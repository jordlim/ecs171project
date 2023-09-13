# ECS 171 Project - Room Occupancy Estimation
[Dataset Link](https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation)

Colab Link: https://colab.research.google.com/github/jordlim/ecs171project/blob/main/occupancy.ipynb

Team Members: \
Jamie Wu - jmewu@ucdavis.edu - GitHub: jamtoabeat\
Jordan Lim - jflim@ucdavis.edu - GitHub: jordlim \
Rohan Arumugam - rarumugam@ucdavis.edu - GitHub: rohan-arumugam\
Elson Jian - egjian@ucdavis.edu - GitHub: ElsonJian\
Hyunkyong (HK) Boo - hboo@ucdavis.edu - GitHub: hboo0507\
Juntao Wang - jutwang@ucdavis.edu - GitHub: JuWT

# Introduction

## Abstract
The dataset we chose is called Room Occupancy Estimation, and it is something that our group was interested in because of its possible applications to modern society. The purpose of the experiment was to study how different factors, such as time/day, light and sound sensors, were related to one another and the implications each factor has on room occupancy. Our group’s goal was to use different machine learning methods on the dataset to predict the occupancy of a room using those factors. Being able to predict room occupancy in a way that does not disturb the residents inside the room itself can be beneficial in many ways. Knowing when some areas are expected to be empty or full, and being able to determine the best date and time to do a certain thing are helpful in many situations, both small and big: for instance, construction workers will be able to monitor the number of residents living in areas near construction sites and adjust working times in order minimize disturbance in the area, and property owners could be able to monitor the number of people on the property at a specific time and use that information to optimize energy usage, preventing the wasting of unnecessary resources in unoccupied rooms. Being able to predict room occupancy quickly and accurately can even have life-saving applications. Every year, law enforcement and other government agencies face emergency situations like hostage negotiations where it is unsafe, resource intensive, or simply infeasible to enter a room. In emergency situations such as fires or natural disasters, being able to quickly discern the number of people in an area is crucial and can lead to better preparation and help first responders evacuate everyone safely. One way to combat the dangers of these situations is to gain intelligence about occupancy passively through various sensors. The ability to detect varying occupancy in a room without being inside has commercial applications as well, such as for monitoring isolated but hazardous rooms like storage freezers and implementing more eco-friendly demand-based ventilation systems. Using data collected from multiple non-intrusive sensors that collected ambient room measurements such as temperature, sound, and CO2 levels, we created a model that can predict the number of people inside a room. After preprocessing the data, we conducted exploratory data analysis using correlation matrices, box plots, and pair plots to understand the normal tendencies of the room and identify important features for predicting the number of people inside the room. Next, we compared the performance of multiple machine learning techniques, such as logistic regression, decision trees, random forest classification, and support vector machines, in order to determine the model that is best suited for predicting the number of people inside a room and make predictions on how fluctuations in sensor output overtime correlated with occupant detection. To validate and test our results, we split our data into train and test segments using cross-validation techniques.

## Dataset Explanation
This dataset is the result of an experiment performed by the International Institute of Information Technology in Hyderabad to determine room occupancy in a non-intrusive way. Over a period of 4 days, the research team tracked sensor data every 30 seconds for 7 different “sensor nodes,” collecting 10,129 complete records and 16 total attributes describing numerical time series data. Attributes range from light in lux to CO2 slope and aim to estimate the occupancy in a room at a specific time which could range from 0 to 3 individuals. The sensors were labeled from S1 to S7, divided based on their function; S1-S4 measured temperature, light and sound sensors; S5 tracked CO2 levels, and S6 and S7 were both passive infrared (PIR) motion sensors. The sensors were arranged in a star configuration. The PIR motion sensors were deployed on the ceiling to maximize their field of view for optimal motion detection. An edge node periodically compiled data from all sensor nodes.


# Figures

1. Architectural Diagram of Neural Network Model: A schematic visualization delineating the architecture, inclusive of layers, nodes, and activation functions employed, will provide insights into the neural network's structure.
2. Confusion Matrices Across Models: Visual confusion matrices for each classification algorithm will facilitate a more immediate understanding of class misclassifications.
3. Feature Importance Visualization: For ensemble tree-based models like Random Forest and Gradient Boosting, a bar graph quantifying feature importance could offer additional interpretive dimensions.
4. Principal Component Analysis (PCA) Results: A bar chart elucidating the variance explained by each principal component can validate the dimensionality reduction approach.
5. Performance Metrics Comparison: Utilize bar graphs or line charts to compare performance metrics such as accuracy and MSE across various models for both the original dataset and the PCA-reduced dataset.
6. Hyperparameter Sensitivity Analysis: If hyperparameter tuning was executed, a graphical representation depicting performance variations against different hyperparameters can be presented.
7. Learning Curves for Model Training: Learning curves plotted against epochs or iterations can aid in diagnosing overfitting or underfitting tendencies in the models.

# Methods

## Data Exploration and Preprocessing
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
   ```
   df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
   ```
2. Our dataset is structured in groups of sensor types, with multiple of the same type of sensor placed around the room. In the data exploration phase our team noticed that different sensors (though the same type) exhibited differences in predictive impact for room occupancy. For example, in the correlation matrix S1_Temp has a 0.7 correlation with room occupancy and S1-3_Temp all have correlations of 0.65+, but S4_temp has a much lower correlation value of 0.53. To make sense of these sort of differences, we sourced a diagram of sensor placement that was associated with the dataset. This helps provide some context to differences in sensor outputs that we saw during data exploration. For example, one explanation for the differences in Temp sensor correlation could be because the placement of sensor S4 is in the upper right corner of the room, far from central node N and by a window.\
![Screenshot 2023-08-24 231529](https://github.com/jordlim/ecs171project/assets/115687850/bbcb0763-c36e-4c92-ba9d-3cb4c5e03cdf)
3.We visualized the data by plotting time series plots of the data for each sensor (e.g. temperature, light, sound, etc.) in order to visualize data trends.

Transformations and Normalization:
1. During data exploration, we noticed that attributes like temperature (in Celsius) changed by the hundredths, while other attributes such as light (in Lux) changed much more drastically on the order of ones and tens. Because the range of the different axis are very different between each sensor type, we felt that the relationship was unclear. Some data points from the pair plots were also skewed or cluttered together with several points being significantly more distinct.
2. Because the sensors from the experiment measured different variables with different units, we decided to normalize the data in order to reduce the rate of change differences between variable measurements for different observations.
3. We use TensorFlow’s ‘to_categorical’ function to convert the room occupancy status (room_occupancy) into one-hot encoded form for multi-class analysis.
4. For standardization, we use MinMaxScaler to standardize all numerical features so that their values fall between 0 and 1. 
5. We use PCA to reduce its dimensionality to 3 principal components. This helps us reduce computational complexity and allows us to visualize the data more easily.
```
pca = PCA(n_components = 3)
reduced_x_train = pca.fit_transform(x_train_scaled)
reduced_x_test = pca.transform(x_test_scaled)
```
6. We also applied random oversampling to our dataset in order to deal with a class imbalance between the number of people observed inside the room at a given time inside our dataset, as observed in the graph below. Random oversampling takes samples from the classes with fewer observations and duplicates them to create a more balanced distribution in the number of people observed inside the room. 
![occupants in room](https://github.com/jordlim/ecs171project/assets/114113303/a08e2b14-aef9-4d6d-bb4f-4a4f31a16c90)




Data Segmentation:
1. Our project aims to gain insight into how specific sensor values correlate with the room occupancy at a discrete point in time, and also how changes in sensor outputs over time can predict increases and decreases in room occupancy. To do this, we felt that it was necessary to process the original dataframe into two separate dataframes, retaining specific attributes for the “discrete” dataframe and processing its data into a secondary “delta” dataframe.
- Discrete Dataframe: From the unmodified dataset, we dropped the “S5_CO2_Slope” attribute because it described changes in “S5_CO2" which we felt aligns better with the delta dataframe.
- Delta Dataframe: Using the discrete dataframe, we use a loop to generate a new dataframe that instead records changes in sensor values in comparison to the previous timestamp. We retained the “S6_PIR” and “S7_PIR” attributes, which are binary values indicating the presence of movement in the room. These two attributes will help reinforce our calculation of how the room occupancy may have changed.
Below is an example diagram of how the discrete dataframe can be processed into a delta dataframe. \
![Screenshot 2023-08-24 231613](https://github.com/jordlim/ecs171project/assets/115687850/6f3f6ed3-5946-4072-8a4f-0457709918f2)

## Model Selection
Using a loop, we built and compared different models:
- We used confusion matrix, classification report, train/test accuracy and train/test mean squared error to evaluate the performance of these models
- They were trained once on the dataset and repeated again on PCA-reduced data (to potentially improve the model’s efficiency and performance)
1. Artificial Neural Networks
- Feed forward neural network that uses stochastic gradient descent and categorical cross entropy since we have 4 target categories: 0, 1, 2, or 3 people in each room occupancy observation
- For original dataset:
    - 3 hidden layers, input dimension of 16 for each feature of the data (excluding date/time)
    - ‘sigmoid’ activation function to perform logistic regression fit
    - Output layer has 4 nodes for each category (0,1,2,3 people)
    - We created an artificial neural network (ANN) model:
```
If name=='Artificial Neural Network ':
Model. add (Dense (units=12, activation='sigmoid ', input_dim=3))
Model. add (Dense (units=3, activation='sigmoid '))
Model. add (Dense (units=4, activation='softmax '))
Adjusted_ Sgd=SGD (learning_rate=0.2)
Model. compile (optimizer=adjusted_sgd, loss='categorical_crosspropy ')
```
      
- For PCA, reduced the number of features from 16 to 3 to compare the performance of a reduced model with fewer variables to a full model
- For both training and testing sets, overall accuracy was very high, resulting in a very low MSE
    - Training MSE and Testing MSE are approximately equal
    - This leads us to believe that we are not overfitting the data, but there is a possibility that we could be underfitting
- Precision and recall were high for the most part except for two cases: much lower precision for the 2 people category and lower recall in the 3 people category. Possible reasons for this could be
    - Observations of the features for 2 people and 3 people are similar
    - Not enough data points for the categories of 1,2, or 3 people 	
2. Logistic Regression
- Uses the logistic function to plot output between 0 and 1
- In our model, the model maintains a high accuracy from the training and testing set while MSE is low for both training and testing set.
- Based on the confusion matrix, we can find some misclassifications on  some classes
3. Decision Tree
- Tree-based structure that acts like a flowchart to cut down the number of viable options in the data until a point can be classified
- Super high accuracy here too
4. K-Nearest Neighbors
- Uses euclidean distance to determine how far different points are from one another before classifying them (looks at the k nearest points from the original point that is being classified)
- Achieved a high accuracy rate of classification for the number of people inside a room
5. Support Vector Machine
- For PCA, we can assume that it is likely not overfitting accuracy or MSE.
- In comparison to the other models that are possibly not overfitting, the overall accuracies are generally higher and MSEs are generally lower. The original dataset has high accuracy and low MSE, however we can’t assume that accuracy is not overfitting as the training is slightly higher.
- Between PCA and our general dataset, our MSE and accuracy are both closer to 0 or 1 respectively in the dataset. However, given that PCA is less likely to be overfitting and the testing and training values are farther apart, we believe that this is a stronger showing.  
6. Random Forest
- UsesRandomForestClassifier
- Builds a deep and split random forest and classifies through a voting mechanism
- Comparing the random forest model and the post-PCA random forest model, both the training and testing set had accuracies close to 1 which potentially indicates overfitting
- On the test set, the random forest model without PCA slightly outperformed the model with PCA. But on the test set, the MSE of the original dataset is lower.
7. Gradient Boosting
- An ensemble learning algorithm that uses gradient descent to minimize overall loss when adding additional weak learners to an ensemble
- Begins with a simple decision tree model, and uses residuals to add new decision trees that improve upon previous prediction errors
- Can be tuned by limiting the number of trees and max depth, thus limiting model complexity and less likely to overfit
- For gradient boosting, we got very similar model results as random forest, which makes sense as both use an aggregation of decision trees to make one final combined model
- Both Gradient Boosting and Random Forest achieved high training accuracy and low MSE, but did not perform as well for PCA which potentially suggest overfitting
8. Naive Bayes
  - A supervised machine learning algorithm that estimates the likelihood of features observed within each class.
  - Assumes features of dataset are conditionally independent of one another (although in our case, that assumption is not true since we have multiple variables that interact with one another to influence the final outcome).
  - We need to calculate the class priors (probability of a room having a certain number of people without accounting for feature behavior), likelihoods for our features, and the posterior probablitiles. The posterior probabilities are calculated using Bayes' Theorem and tell us the probability of there being a certain number of people in a room.
  - This model performed the worst out of all models created, which is partly due to a violation of one of the underlying assumptions of Naive Bayes: features act independently of one another.
  - The PCA and oversampled dataset did a slightly better job at dealing with the class imbalance to make accurate classifications. This can be seen in the precision and recall rates for when there are people being marked as present inside the room. 


Model 1: All 16 features are used
The total test models are Artificial Neural Network, logistic regression, decision tree, K-nearest neighbor, support vector machine, random forest, gradient enhancement.
```
Models={'Artificial Neural Network ': Sequential(),
'Logistic Regression': LogisticRegion(),
'Decision Tree': DecisionTreeClassifier(),
'K-Nearest Neighbors': KNeighborsClassifier(),
'Support Vector Machine': SVC(),
'Random Forest': RandomForestClassifier(),
'Gradient Boosting': XGBClassifier()}
```

Model 2: The data after dimensionality reduction using PCA has only 3 principal components.
The same testing models.

Both model1 and model2 finally use evaluation indicators: confusion matrix, classification report, accuracy, and MSE.
```
confused_mat_train = confusion_matrix(y_true_train, y_labels_train)
class_rep_train = classification_report(y_true_train, y_labels_train)
accuracy_rate_train = accuracy_score(y_true_train, y_labels_train)
train_mse_score = mean_squared_error(y_true_train, y_labels_train)

confused_mat_test = confusion_matrix(y_true_test, y_labels_test)
class_rep_test = classification_report(y_true_test, y_labels_test)
accuracy_rate_test = accuracy_score(y_true_test, y_labels_test)
test_mse_score = mean_squared_error(y_true_test, y_labels_test)
```

# Results
## Model Evaluation
![ECS 171 - Model Results](https://github.com/jordlim/ecs171project/assets/115687850/0bfe7c63-f75f-4e19-8199-d0e5f504e4c1)

![mse_og](https://github.com/jordlim/ecs171project/assets/114113303/9d90d8fd-57bf-4bf9-9a07-d428720f94b9)

![mse_pca](https://github.com/jordlim/ecs171project/assets/114113303/fdaeefdf-6436-4621-a787-ef1f54393e2d)

![mse_ovesampleed](https://github.com/jordlim/ecs171project/assets/114113303/47f9a135-37ba-4801-aac8-4af92b7ce9bb)

![accuracy_og](https://github.com/jordlim/ecs171project/assets/114113303/6c6f15ac-bd2b-4232-b27c-cb15197b3b3f)

![accuracy_pca](https://github.com/jordlim/ecs171project/assets/114113303/e6120c66-d329-41ed-8856-05d6b2682f77)

![accuracy_os](https://github.com/jordlim/ecs171project/assets/114113303/cd0e0dd6-ce04-4c48-a144-74dca2a033ca)

Above is a summary of the train and test accuracy and MSE scores each model reported. The green highlight indicates the model with the best outcome - for accuracy this would be the model that scored closest to 1.0, and for MSE this would be the model that scored closest to 0.0. The orange highlight is the model with the second best outcome. We felt it was necessary to not just consider the best outcome, because sometimes overly high accuracy or overly low MSE can be a result of overfitting or datasets that aren’t complex enough to begin with.

Observations:
- For the normal dataset version (top chart), we saw that the Random Forest model and the Gradient Boosting model had the same exact train and test output as well as confusion matrices. This can be attributed to the underlying structure of RF and GB, which both incorporate decision trees when classifying data. This could also be attributed to improper hyperparameter tuning for the gradient boosting model or a lack of diverse variables in our data. We can determine if either possibility is true by tuning the different hyperparameters in our gradient boosting model.
- PCA is a method that reduces the dimensions of a dataset, which generally should reduce overfitting. Looking at PCA model results, we see that the K-Nearest Neighbors algorithm had the best test accuracy and test MSE, and the second best train accuracy and train MSE. In addition, the test MSE is lower than the train MSE, which further suggests that there is not overfitting.
- On average, the normal dataset performed better in terms of accuracy and MSE than the PCA dataset. This makes sense, as PCA is a method that reduces dimensions by trying to combine multiple original attributes into groups of attributes for simplicity.

## Fit
Based on the training and testing MSEs calculated for each model, we can see that the test and the train errors are low and very similar to one another. This observation supports the claim that there is not a lot of overfitting going on within our dataset, and that our models have achieved an ideal level of complexity as our training and testing MSE values are very similar, and very small. Our claim is further supported by the high accuracy levels achieved for both the training and testing datasets. Considering how high our accuracy, precision, and recall rates are for all models, it is possible that there is some overfitting in the dataset. It is also possible that we achieved a high accuracy, precision, and recall because of the imbalance in our dataset (there are many more observations with 0 people in the room compared to 1,2, or 3 people). This imbalance may also allow us to achieve a higher accuracy without actually having a model that accurately predicts the number of people inside a room. We plan on validating our claim that our models are considered within the “ideal range” complexity by performing cross validation on our dataset to test the accuracy of our original models. 

# Discussion

## Preprocessing
The first step we took was to look through the dataset and decide on any changes that we felt were necessary to make. First, we checked for duplicates or null values, which could potentially affect our data. Since the date is not something that is a good predictor for room occupancy, we decided to combine date and time into a single category to study overall changes of the data over time.  Our group also decided that instead of only focusing on the discrete values of our different factors, we also wanted to discuss how changes in these values can help predict an increase or decrease in room occupancy. Therefore, we decided to have one data frame representing the discrete values, and a second data frame showing the increase/decrease in the values for each datapoint.

One of the big things our group wanted to study was how each of the factors were correlated to one another. To achieve this, we built a correlation matrix consisting of all of the factors in our data. Since our data had multiple sensors for each type (4 for temperature. 4 for light, etc.) each corresponding to different areas of the room, we believed that the sensors for each type would be highly correlated with one another. Instead, it turned out that some sensors were less correlated than others, likely because of the difference in placement location. The next thing we did was visualize the data by plotting time series plots for each data sensor as well as plotting it against the room occupancy count. To do this, we separated the dataset into its different data types, one for each type of sensor, so we could have a clear visualization of each category and its relation.We then plotted each sensor type against the date and time so we could study how the values change throughout the day. 

The last preprocessing we did was standardizing the data as well as encoding outputs. We noticed that each data type had large data distributions, so we decided to normalize our data so that all the data fall between 0 and 1. We also chose to use one-hot encoding for our room occupancy category to allow for multi-class analysis. Setting X as our sensor data and y as the room occupancy count, we decided to use an 80:20 split. Lastly, we set up the data and used PCA to reduce its dimensionality to 3 principal components to reduce complexity and allow for better visualization of the data.



# Conclusion

# Collaboration
Jamie Wu - jmewu@ucdavis.edu - GitHub: jamtoabeat\
- 
Jordan Lim - jflim@ucdavis.edu - GitHub: jordlim \
- 
Rohan Arumugam - rarumugam@ucdavis.edu - GitHub: rohan-arumugam\
- 
Elson Jian - egjian@ucdavis.edu - GitHub: ElsonJian\
- 
Hyunkyong (HK) Boo - hboo@ucdavis.edu - GitHub: hboo0507\
- 
Juntao Wang - jutwang@ucdavis.edu - GitHub: JuWT
- 
