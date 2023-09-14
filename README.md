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
The dataset we chose is called Room Occupancy Estimation and provides a variety of sensor data about a room over time. We thought this dataset was both intruiging and impactful because of the many applications of being able to dynamically and accurately predict room occupancy without physical or camera surveillance. The purpose of this project is to study how different factors such as time/day, light and sound sensors, CO2 sensors, etc. are related and their abilities to predict occupancy in a room over time. Having a strong predictive model for room occupancy is relevant for a variety of use cases. For example, construction workers can adjust working times and minimize noise disturbance on a property based on an estimate of residents present in a more non-invasive manner. Property owners can also use occupancy information to optimize energy usage and prevent the wasting of unecessary cooling/heating resources. In addition, law enforcement and other government agencies that face emergency situations like hostage negotiations or natural disasters can quickly discern the number of people in an area in a more safe, resource minimal way that helps first responders evacuate people safely. Occupancy detection can also be used to monitor isolated but hazardous rooms like storage freezers or plants with toxic gases. Overall, gaining intelligence about occupancy passively through various sensors is one strong method to address various commercial and public service applications.

## Abstract
Using data collected from multiple non-intrusive sensors that collected ambient room measurements such as temperature, sound, and CO2 levels, we created a model that can predict the number of people inside a room. After preprocessing the data, we conducted exploratory data analysis using correlation matrices, box plots, and pair plots to understand the normal tendencies of the room and identify important features for predicting the number of people inside the room. Next, we compared the performance of multiple machine learning techniques, such as logistic regression, decision trees, random forest classification, and support vector machines in order to determine the model that is best suited for predicting the number of people inside a room and make predictions on how fluctuations in sensor output overtime correlated with occupant detection. We ran our selected machine learning techniques on three variations of our data, including the standard data, a dimensionally-reduced PCA version of the data, and an oversampled version of the data. To validate and test our results, we split our data into train and test segments using k-fold cross-validation techniques.

## Dataset Explanation
This dataset is the result of an experiment performed by the International Institute of Information Technology in Hyderabad to determine room occupancy in a non-intrusive way. Over a period of 4 days, the research team tracked sensor data every 30 seconds for 7 different “sensor nodes,” collecting 10,129 complete records and 16 total attributes describing numerical time series data. Attributes range from light in lux to CO2 slope and aim to estimate the occupancy in a room at a specific time which could range from 0 to 3 individuals. The sensors were labeled from S1 to S7, divided based on their function; S1-S4 measured temperature, light and sound sensors; S5 tracked CO2 levels, and S6 and S7 were both passive infrared (PIR) motion sensors. The sensors were arranged in a star configuration. The PIR motion sensors were deployed on the ceiling to maximize their field of view for optimal motion detection. An edge node periodically compiled data from all sensor nodes.


# Figures
------------------ 
NOT SURE ABOUT THIS PART
1. Architectural Diagram of Neural Network Model: A schematic visualization delineating the architecture, inclusive of layers, nodes, and activation functions employed, will provide insights into the neural network's structure.
2. Confusion Matrices Across Models: Visual confusion matrices for each classification algorithm will facilitate a more immediate understanding of class misclassifications.
3. Feature Importance Visualization: For ensemble tree-based models like Random Forest and Gradient Boosting, a bar graph quantifying feature importance could offer additional interpretive dimensions.
4. Principal Component Analysis (PCA) Results: A bar chart elucidating the variance explained by each principal component can validate the dimensionality reduction approach.
5. Performance Metrics Comparison: Utilize bar graphs or line charts to compare performance metrics such as accuracy and MSE across various models for both the original dataset and the PCA-reduced dataset.
6. Hyperparameter Sensitivity Analysis: If hyperparameter tuning was executed, a graphical representation depicting performance variations against different hyperparameters can be presented.
7. Learning Curves for Model Training: Learning curves plotted against epochs or iterations can aid in diagnosing overfitting or underfitting tendencies in the models.
------------------

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

### Transformations and Normalization:
1. We normalized the data using sklearn.preprocessing.MinMaxScaler(), making attribute values fall between 0 and 1. 
2. We used TensorFlow’s ‘to_categorical’ function to convert the room occupancy status (room_occupancy) into one-hot encoded form for multi-class analysis.
3. We used PCA to reduce its dimensionality to 3 principal components.
```
pca = PCA(n_components = 3)
reduced_x_train = pca.fit_transform(x_train_scaled)
reduced_x_test = pca.transform(x_test_scaled)
```
6. We applied random oversampling, which takes samples from classes with fewer observations and duplicates them to create a more balanced distribution.
```
ros = RandomOverSampler(random_state=9)
X_resampled, y_resampled = ros.fit_resample(X_os,y_os)
```

## Model Selection
Using a loop, we built and compared 8 different models for each of the 3 dataset variations:
- For each model, we used confusion matrix, classification report, train/test accuracy and train/test mean squared error to evaluate the performance of these models
```
models = {'Artificial Neural Network': Sequential(),
'Logistic Regression': LogisticRegression(),
'Decision Tree': DecisionTreeClassifier(),
'K-Nearest Neighbors': KNeighborsClassifier(),
'Support Vector Machine': SVC(),
'Random Forest': RandomForestClassifier(),
'Gradient Boosting': XGBClassifier(),
'Naive Bayes': GaussianNB()}
```
### Artificial Neural Network
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
### Logistic Regression
- Uses the logistic function to plot output between 0 and 1
### Decision Tree
- Tree-based structure that acts like a flowchart to cut down the number of viable options in the data until a point can be classified
### K-Nearest Neighbors
- Uses euclidean distance to determine how far different points are from one another before classifying them (looks at the k nearest points from the original point that is being classified)
### Support Vector Machine
- Classifies data points using support vectors
- Tries to identify a hyperplane that separates the data into classes while optimizing the margin between classes
### Random Forest
- UsesRandomForestClassifier
- Builds a deep and split random forest and classifies through a voting mechanism
### Gradient Boosting
- An ensemble learning algorithm that uses gradient descent to minimize overall loss when adding additional weak learners to an ensemble
- Begins with a simple decision tree model, and uses residuals to add new decision trees that improve upon previous prediction errors
- Can be tuned by limiting the number of trees and max depth, thus limiting model complexity and less likely to overfit
### Naive Bayes (Gaussian)
  - A supervised machine learning algorithm that estimates the likelihood of features observed within each class.
  - Assumes features of dataset are conditionally independent of one another (although in our case, that assumption is not true since we have multiple variables that interact with one another to influence the final outcome).
  - We need to calculate the class priors (probability of a room having a certain number of people without accounting for feature behavior), likelihoods for our features, and the posterior probablitiles. The posterior probabilities are calculated using Bayes' Theorem and tell us the probability of there being a certain number of people in a room.

## Dataset Variations
- Each of the 8 selected models was trained on three variations of data - the standard dataset, a PCA-reduced dataset, and an oversampled dataset to investigate any potential improvements to model efficiency and performance.
### Dataset Variation 1 - Standard
```
X = df.iloc[:, :-1] # get all columns except room_occupancy
y = df.iloc[:,-1] # room occupancy
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 2)
```
### Dataset Variation 2 - PCA-reduced
```
pca = PCA(n_components = 3) # reduce number of features in dataset from 17 to 3 with pca
reduced_x = pca.fit_transform(X_scaled_og)
reduced_x_train = pca.fit_transform(x_train_scaled)
reduced_x_test = pca.transform(x_test_scaled)
```
### Dataset Variation 3 - Randomly Oversampled
```
ros = RandomOverSampler(random_state=9)
X_resampled, y_resampled = ros.fit_resample(X_os,y_os)
x_train_os, x_test_os, y_train_os, y_test_os = train_test_split(X_resampled, y_resampled, test_size = 0.2, random_state = 2)
```
# Results
## Confusion Matrices for 3 Model Formats
### Confusion Matrices for Standard Dataset
![ECS 171 - Standard ANN](https://github.com/jordlim/ecs171project/assets/115687850/af0ee647-8185-420f-965a-7f200666abd4)
![ECS 171 - Standard Logistic](https://github.com/jordlim/ecs171project/assets/115687850/f583752a-4a20-4138-a1e4-8a11bbacb3e6)
![ECS 171 - Standard Decision Tree](https://github.com/jordlim/ecs171project/assets/115687850/c406c0c9-6a01-4914-be1a-fb08d09ae6af)
![ECS 171 - Standard KNN](https://github.com/jordlim/ecs171project/assets/115687850/00266157-5a7f-487e-baaa-a20cf9a74be6)
![ECS 171 - Standard SVM](https://github.com/jordlim/ecs171project/assets/115687850/ba7c683c-43b5-494e-8618-8c20bf9e9ab7)
![ECS 171 - Standard RF](https://github.com/jordlim/ecs171project/assets/115687850/31229b39-e532-4cda-9750-b1e6082cc85f)
![ECS 171 - Standard GB](https://github.com/jordlim/ecs171project/assets/115687850/56b37b69-df44-4df9-a823-cb7316810ec1)
![ECS 171 - Standard NB](https://github.com/jordlim/ecs171project/assets/115687850/ba67a57f-9562-4338-b6bc-64b1f9d43924)

### Confusion Matrices for PCA Dataset
![ECS 171 - PCA ANN](https://github.com/jordlim/ecs171project/assets/115687850/1c4e7f1a-ff71-48f3-8ba2-bf89d1365190)
![ECS 171 - PCA Logistic](https://github.com/jordlim/ecs171project/assets/115687850/c962efb7-9264-40c1-9c34-24b2b6f272e4)
![ECS 171 - PCA Decision Tree](https://github.com/jordlim/ecs171project/assets/115687850/13f1bc20-9376-4391-bda5-481c7427731a)
![ECS 171 - PCA KNN](https://github.com/jordlim/ecs171project/assets/115687850/24b4dcfb-6c11-4388-af7c-c17851c0c144)
![ECS 171 - PCA SVM](https://github.com/jordlim/ecs171project/assets/115687850/1a81b6a4-838a-44f2-a658-d69bb7145752)
![ECS 171 - PCA RF](https://github.com/jordlim/ecs171project/assets/115687850/fb23f63a-8058-488c-bf1b-d7f1d2b5d2de)
![ECS 171 - PCA GB](https://github.com/jordlim/ecs171project/assets/115687850/1f4fd4ef-61d3-4db1-84a1-2635a957a394)
![ECS 171 - PCA NB](https://github.com/jordlim/ecs171project/assets/115687850/977f89fb-9ad8-4d0e-8066-6c0d1a7c51ad)

### Confusion Matrices for Randomly Oversampled Dataset
![ECS 171 - Oversample ANN](https://github.com/jordlim/ecs171project/assets/115687850/6650a561-6b65-4716-ae51-f9ed525f90f3)
![ECS 171 - Oversample Logistic](https://github.com/jordlim/ecs171project/assets/115687850/98929c67-d1ae-48d2-a105-0f67f5ab2fb8)
![ECS 171 - Oversample Decision Tree](https://github.com/jordlim/ecs171project/assets/115687850/cff8ca69-da11-4400-8620-7779399af66d)
![ECS 171 - Oversample KNN](https://github.com/jordlim/ecs171project/assets/115687850/3297c559-5b87-4755-9b04-b432e5249b82)
![ECS 171 - Oversample SVM](https://github.com/jordlim/ecs171project/assets/115687850/1bafa7db-5c76-4ff0-80e9-39bb48d0c309)
![ECS 171 - Oversample RF](https://github.com/jordlim/ecs171project/assets/115687850/7b4caa6f-818d-4f8b-9b21-0cc4746c3bb4)
![ECS 171 - Oversample GB](https://github.com/jordlim/ecs171project/assets/115687850/ee0f8abc-5bd9-4207-9226-461752dbbda1)
![ECS 171 - Oversample NB](https://github.com/jordlim/ecs171project/assets/115687850/cf9198f2-da05-4eb0-9cb7-cb685f711543)

## Model Evaluation for 3 Model Formats (MSE and Accuracy)
### Metrics for Standard Dataset
![ECS 171 - Standard MSE](https://github.com/jordlim/ecs171project/assets/115687850/2cac7c95-5000-438c-92cd-b3578e49a5f0)
![ECS 171 - Standard Accuracy](https://github.com/jordlim/ecs171project/assets/115687850/14dfcc7d-45f0-405d-8253-18777a48d28d)

### Metrics for PCA Dataset
![ECS 171 - PCA MSE](https://github.com/jordlim/ecs171project/assets/115687850/af8ed86d-9504-4a97-9dc1-88c33849fb0a)
![ECS 171 - PCA Accuracy](https://github.com/jordlim/ecs171project/assets/115687850/4bead484-1ffc-4570-a488-62067d228596)

### Metrics for Oversampled Dataset
![ECS 171 - Oversampled MSE](https://github.com/jordlim/ecs171project/assets/115687850/87ea0e1f-54ae-440d-94bb-9eda1856e525)
![ECS 171 - Oversampled Accuracy](https://github.com/jordlim/ecs171project/assets/115687850/e0d33a8d-b448-4405-94de-653b01e18782)

### Summary of Results
Below is a summary of the train and test accuracy and MSE scores for each of the 8 models. The green highlight indicates the model with the best outcome - for accuracy this would be the model that scored closest to 1.0, and for MSE this would be the model that scored closest to 0.0. The orange highlight is the model with the second best outcome. We felt it was necessary to not just consider the best outcome, because sometimes overly high accuracy or overly low MSE can be a result of overfitting or datasets that aren’t complex enough to begin with. To compare the average train and test accuracy and MSE scores for the 3 different dataset variations, the blue highlight indicates best outcome and the purple highlight indicates second best outcome.

Summary of Results Table:
![ECS 171 - Model Results - Standard, PCA, Oversampled](https://github.com/jordlim/ecs171project/assets/115687850/af230352-4097-413d-9ac9-0352be23910e)

# Discussion
## Data Exploration
The first step we took for data exploration was to look through the dataset and decide on any changes that we felt were necessary to make. First, we checked for duplicates or null values, which could potentially affect our data. Our goal was to track changes in sensor output over time, so we processed the “Date” and “Time” attributes into numerical values. We used Pandas to.datetime() to turn the two attributes into one attribute to plot overall changes in sensor output over time. 

To visualize how the different attributes in the data influenced one another, we built a correlation matrix consisting of all of the factors in our data. Since our data had multiple sensors for each type (4 for temperature, 4 for light, etc.) each corresponding to different areas of the room, we initially believed that the sensors for each type (ex. each of the four temperature, light, and sound sensors) would be highly correlated with one another. Instead, our team noticed that different sensors (though the same type) exhibited noteable differences in predictive impact for room occupancy. For example, in the correlation matrix S1_Temp has a 0.7 correlation with room occupancy and S1-3_Temp all have correlations of 0.65+, but S4_temp has a much lower correlation value of 0.53. To make sense of these sort of differences, we sourced a diagram of sensor placement that was associated with the dataset. This helps provide some context to differences in sensor outputs that we saw during data exploration. For example, one explanation for the differences in Temp sensor correlation could be because the placement of sensor S4 is in the upper right corner of the room, far from central node N and by a window.

![Sensor Room Placement Diagram](https://github.com/jordlim/ecs171project/assets/115687850/bbcb0763-c36e-4c92-ba9d-3cb4c5e03cdf)

We further visualized the data by plotting time series plots of the data for each sensor (e.g. temperature, light, sound, etc.) in order to visualize data trends. Specifically, we separated the dataset into its different data types, one for each sensor type, and plotted each sensor type against the date and time so we could study how the values change throughout the day. This was able to give us a clear visualization of each attribute and its trends over time. From this, we noticed many peaks in sensor output intensity during the first 3 and last 2 days of the experiment, with more minimal steady growth for the middle 10-14 days. Below are a few examples of the time series plots to demonstrate these peaks, such as for temperature, light, and CO2 concentration.

![ECS 171 - Temperature Over Time](https://github.com/jordlim/ecs171project/assets/115687850/a27597e5-b867-47db-9417-91ab927471db)
![ECS 171 - Light Over Time](https://github.com/jordlim/ecs171project/assets/115687850/dd2e48dc-fade-45d9-9d3b-555656419fea)
![ECS 171 - CO2 Over Time](https://github.com/jordlim/ecs171project/assets/115687850/f186a233-d293-49ea-8533-78de7d305242)

## Data Preprocessing
During data exploration, we noticed that attributes like temperature (in Celsius) changed by the hundredths, while other attributes such as light (in Lux) changed much more drastically on the order of ones and tens. Because the range of the different axis are very different between each sensor type, we felt that the relationship was unclear. Some data points from the pair plots were also skewed or cluttered together with several points being significantly more distinct. Because the sensors from the experiment measured different variables with different units, we decided to normalize the data in order to reduce the rate of change differences between variable measurements for different observations. To standardize the data, we used MinMaxScaler to standardize all numerical features so that their values fall between 0 and 1. We also decided on using TensorFlow’s ‘to_categorical’ function to convert the room occupancy status (room_occupancy) into one-hot encoded form for multi-class analysis.

Upon seeing the dataset, we also noticed that there was a major class imbalances between the number of instances for our room occupancy variable. 

![occupants in room](https://github.com/jordlim/ecs171project/assets/114113303/a08e2b14-aef9-4d6d-bb4f-4a4f31a16c90)

We initially thought this could lead to some weird predictions in our models. We wanted to explore how manipulating the training data in different ways could impact our predictive results. We decided to test each of the 8 machine learning models on 3 different variations of the dataset - the standard data with minimal changes, a PCA-reduced variant, and a randomly oversampled variant. 

We decided to try a PCA-reduced variant to see if there is any extraneous noise coming from the predictor variables in our dataset that were obtained from the sensors. Our data had around 17 attributes, so we decided to use PCA to reduce its dimensionality to 3 principal components, which helps us reduce complexity and allows to visualize the data more easily.
```
pca = PCA(n_components = 3)
reduced_x_train = pca.fit_transform(x_train_scaled)
reduced_x_test = pca.transform(x_test_scaled)
```
Performing PCA on our dataset did not impact the performance drastically, as some of the models were already achieving near perfect accuracy on the standard dataset. For example, Gradient Boosting, Random Forest, and Decision Trees had accuracies of 1.0. Achieving near perfect results is bizarre, so to make sure that our models were not overfitting the data, we performed cross-validation across all of our data transformations (original, PCA, and random oversampling). We first utilized StratifiedKFold Cross-Validation for our 3-fold neural network in order to get a better idea of how our model handles unknown data. Since each fold in StratifiedKFold Cross-Validation can contain a random balance of room occupants, we can see how our model truly performs against data the model hasn't seen yet. Cross-validating our dataset revealed that the models are not being overfitted as they also have a high accuracy and a low MSE for both the training and testing set. However, looking at the "Summary of Results" table we can see that see that the PCA overall performed worse than our other 2 dataset variants (standard and oversampled). This makes sense to us, because dimensional reduction can lead to some information loss and subsequent decreases in accuracy.

Because of the major class imbalances, we also wanted to apply random oversampling to our dataset in order to deal with a class imbalance between the number of people observed inside the room at a given time inside our dataset. Random oversampling takes samples from the classes with fewer observations and duplicates them to create a more balanced distribution in the number of people observed inside the room. 

## Model Selections
After creating our 3 dataset variants (standard, PCA-reduced, and oversampled), we wanted to try a variety of models. The 8 models that we chose were ANN, Logistic Regression, Decision Tree, K-Nearest Neighbors, Support Vector Machine, Random Forest, Gradient Boosting, and Naive Bayes (Gaussian). Most of the models we tried came from models we learned about in class, and we also added Random Forest and Gradient Boosting as they are more augmented variations of Decision Trees. 

## Analysis
The "Summary of Results Table" is copied here from the Results section for easy viewing.
![ECS 171 - Model Results - Standard, PCA, Oversampled](https://github.com/jordlim/ecs171project/assets/115687850/15a6ee14-a0c9-4ade-a5fc-8217f35ba1d5)

Looking at the blue-highlighted average scores, we can see that across all 8 selected models the oversampled variation of the dataset performed the best out of the three variations we tried (standard, PCA-reduced, and oversampled) in terms of MSE. It's accuracy was a little lower than the standard variation of the dataset, which can be attributed to how oversampling can sometimes augment the inaccuracies that occur due to outliers by duplicating them in the dataset. 

Random Forest and Gradient Boosting performed very similarly overall, which makes sense as both use an aggregation of decision trees to make one final combined model. Overall the Decision Tree and K-Nearest Neighbors models performed well, achiving the second best if not best results for testing MSE and accuracy. 

For the standard dataset variation, we saw that the Random Forest model and the Gradient Boosting model had the same exact train and test output as well as confusion matrices, which we thought was strange. Looking into the two models further, we feel that the similarity can be attributed to the underlying structure of RF and GB, which both incorporate decision trees when classifying data. This could also be attributed to improper hyperparameter tuning for the gradient boosting model or a lack of diverse variables in our data. While we did not get the opportunity to do so for this project, one area of further interest is to tune the hyperparameters for the gradient boosting model to see if the outputs will vary.

ALso, PCA is a method that reduces the dimensions of a dataset which generally should reduce overfitting. Looking at PCA model results, we see that the K-Nearest Neighbors algorithm had the best test accuracy and test MSE, and the second best train accuracy and train MSE. In addition, the test MSE is lower than the train MSE, which further suggests that there is not overfitting.

Breaking down our observations for each of the 8 models further:

### Artificial Neural Network
- For both training and testing sets, overall accuracy was very high, resulting in a very low MSE
    - Training MSE and Testing MSE are approximately equal
    - This leads us to believe that we are not overfitting the data, but there is a possibility that we could be underfitting
- Precision and recall were high for the most part except for two cases: much lower precision for the 2 people category and lower recall in the 3 people category. Possible reasons for this could be
    - Observations of the features for 2 people and 3 people are similar
    - Not enough data points for the categories of 1,2, or 3 people 	
### Logistic Regression
- In our model, the model maintains a high accuracy from the training and testing set while MSE is low for both training and testing set
### Decision Tree
- Super high accuracy here too
### K-Nearest Neighbors
- Achieved a high accuracy rate of classification for the number of people inside a room
### Support Vector Machine
- For PCA, we can assume that it is likely not overfitting accuracy or MSE.
- In comparison to the other models that are possibly not overfitting, the overall accuracies are generally higher and MSEs are generally lower. The original dataset has high accuracy and low MSE, however we can’t assume that accuracy is not overfitting as the training is slightly higher.
- Between PCA and our general dataset, our MSE and accuracy are both closer to 0 or 1 respectively in the dataset. However, given that PCA is less likely to be overfitting and the testing and training values are farther apart, we believe that this is a stronger showing.  
### Random Forest
- Comparing the random forest model and the post-PCA random forest model, both the training and testing set had accuracies close to 1 which potentially indicates overfitting
- On the test set, the random forest model without PCA slightly outperformed the model with PCA. But on the test set, the MSE of the original dataset is lower.
### Gradient Boosting
- For gradient boosting, we got very similar model results as random forest, which makes sense as both use an aggregation of decision trees to make one final combined model
- Both Gradient Boosting and Random Forest achieved high training accuracy and low MSE, but did not perform as well for PCA which potentially suggest overfitting
### Naive Bayes (Gaussian)
  - This model performed the worst out of all models created, which is partly due to a violation of one of the underlying assumptions of Naive Bayes: features act independently of one another.
  - The PCA and oversampled dataset did a slightly better job at dealing with the class imbalance to make accurate classifications. This can be seen in the precision and recall rates for when there are people being marked as present inside the room. 

## Fit
Based on the training and testing MSEs calculated for each model, we can see that the test and the train errors are low and very similar to one another. This observation supports the claim that there is not a lot of overfitting going on within our dataset, and that our models have achieved an ideal level of complexity as our training and testing MSE values are very similar, and very small. Our claim is further supported by the high accuracy levels achieved for both the training and testing datasets. Considering how high our accuracy, precision, and recall rates are for all models, it is possible that there is some overfitting in the dataset. It is also possible that we achieved a high accuracy, precision, and recall because of the imbalance in our dataset (there are many more observations with 0 people in the room compared to 1,2, or 3 people). This imbalance may also allow us to achieve a higher accuracy without actually having a model that accurately predicts the number of people inside a room. We plan on validating our claim that our models are considered within the “ideal range” complexity by performing cross validation on our dataset to test the accuracy of our original models. 

# Conclusion
In our project, we conducted in-depth research on the prediction of the number of people in the room from different perspectives. We adopted different preprocessing techniques and multiple machine learning models, including neural networks, logistic regression, decision trees, and support vector machines, to address this issue. It is particularly noteworthy that we addressed the issue of category imbalance in the dataset and balanced it using random oversampling techniques. In addition, we also attempted dimensionality reduction methods, such as principal component analysis (PCA), to investigate whether there are insignificant characteristics. We also know that different sensors have different effects on room occupancy, which may be due to the placement of the sensors. This discovery is valuable for deeper research and practical applications. Moreover, through cross validation and multiple experiments, we confirmed that the model did not overfit and performed well in predicting the number of people in the room, approaching 100% on the Average Score. The project successfully solved the problem of predicting the number of people in a room and demonstrated various data processing and model training methods.
Future directions:
This project is not only expected to play a role in the business environment, but also has important applications in emergency services and public safety. When facing emergency situations such as hostage negotiations or natural disasters, use models to quickly and accurately identify the number of people in hazardous areas. Potential risks can also be prevented through models. Helps to more effectively help people avoid danger.

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
