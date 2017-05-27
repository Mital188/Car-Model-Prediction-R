"
Authors:
Mital Suresh Modha (msm160530)
Shivangi Gaur      (sxg163230)

The following R code predicts the acceptability of a car model
based on the following factors.
1. Maintenability            4. Buying Price
2. Number of Doors            5. Person Capacity
3. Size of luggauge boot      6. Safety

The cars are classified into: 
1. Unacceptable (unacc)       3. Good (good)
2. acceptable (acc)           4. Very Good (vgood)

The following algorithms are used and compared:
1. K-Nearest Neighbor         4. Gradient Boosting Machine
2. Support Vector Machine     5. Learning Vector Quantization
3. Random Forest              

"
#Loading all the required libraries
install.packages("caret")
install.packages("lattice")
install.packages("ggplot2")
install.packages("MASS")
install.packages("RWeka")
install.packages("randomForest")
install.packages("e1071")
install.packages("kernlab")
install.packages("gbm")
install.packages("plyr")

library(lattice)
library(ggplot2)
library(caret)
library(MASS)
library(e1071)
library(randomForest)
library(kernlab)
library(gbm)
library(plyr)
#reading the data from a text file and storing it as a table
dataset <- read.csv("car.data.txt", sep = ",", header = F)
#Adding the column names
colnames(dataset) <- c("Buying","Maintenance","Doors","Persons","Luggage_boot","Safety","Class")

#Writing the data as a CSV file
write.csv(dataset, file = "car.csv",row.names = F)

#Splitting the data into Training set and validation set
# create a list of 80% of the rows in the original dataset we can use for training
validation_index <- createDataPartition(dataset$Class, p=0.80, list=FALSE)
# select 20% of the data for validation
validation <- dataset[-validation_index,]
# use the remaining 80% of data to training and testing the models
dataset <- dataset[validation_index,]

#Prints the data type of the data, here all are "factors"
sapply(dataset, class)

#Shows the classes for classification
levels(dataset$Class)

#Shows the distribution of each class type
percentage <- prop.table(table(dataset$Class)) * 100
cbind(freq=table(dataset$Class), percentage=percentage)

# Provides a summary of the dataset
summary(dataset)

# Draw the graph of the dataset and shows its distribution
plot(dataset)
plot(dataset[1:6], col = as.numeric(dataset$Class))

x <- dataset[,1:6]
y <- dataset[7]

#bargraph of each class type
par(mfrow=c(1,1))
plot(dataset$Class, ylim = c(0,1000))

# the Follwing shows the bar graph for each attributes and shows
# how they are divided amongst the class
#Mantainance
ggplot(dataset, aes(Maintenance, fill = Class)) + geom_bar()
#Buying Price
ggplot(dataset, aes(Buying, fill = Class)) + geom_bar()
#Number of Doors
ggplot(dataset, aes(Doors, fill = Class)) + geom_bar()
#Person Capacity
ggplot(dataset, aes(Persons, fill = Class)) + geom_bar()
# Luguage Boot
ggplot(dataset, aes(Luggage_boot, fill = Class)) + geom_bar()
#Safety
ggplot(dataset, aes(Safety, fill = Class)) + geom_bar()


#Number of fold = 10
#The following is used in the training function for computational purposes.

control <- trainControl(method="cv", number=10)
metric <- "Accuracy"

#Machine Learning Algorithms

# kNN (k Nearest Neighbors)
set.seed(7)
fit.knn <- train(Class~., data=dataset, method="knn", metric=metric, trControl=control)
plot(fit.knn)
# estimate skill of KNN on the dataset
predictions_train <- predict(fit.knn, dataset)
confusionMatrix(predictions_train, dataset$Class)
predictions_test <- predict(fit.knn, validation)
confusionMatrix(predictions_test, validation$Class)
# Create submission data frame
submission <- data.frame(validation$Class)
submission$validation.Class<- predictions_test
#trying to compare the values of train and test data
table(submission$validation.Class)
table(validation$Class)

# Predict using the test values
for(i in 1:nrow(validation)) {
  predictions <- predict(fit.knn,validation[i,])
  print(predictions)
  print(validation[i,])
}



# SVM

set.seed(7)
fit.svm <- train(Class~., data=dataset, method="svmRadial", metric=metric, trControl=control)
plot(fit.svm)
#estimating the skills of svm on the dataset
predictions_train <- predict(fit.svm, dataset)
confusionMatrix(predictions_train, dataset$Class)
predictions_test <- predict(fit.svm, validation)
confusionMatrix(predictions_test, validation$Class)
# Create submission data frame
submission <- data.frame(validation$Class)
submission$validation.Class<- predictions_test
#trying to compare the values of train and test data
table(submission$validation.Class)
table(validation$Class)

# Predict using the test values
for(i in 1:nrow(validation)) {
  predictions <- predict(fit.svm,validation[i,])
  print(predictions)
  print(validation[i,])
}




# Random Forest

set.seed(7)
fit.rf <- randomForest(as.factor(Class) ~ .,data = dataset, importance = TRUE,ntree =100, nodesize = 2)
plot(fit.rf)
# Predict using the training values
predictions_train <- predict(fit.rf, dataset)
#importance(predictions_train) no applicable method for 'importance' applied to an object of class "factor"
confusionMatrix(predictions_train, dataset$Class)
predictions_test <- predict(fit.rf, validation)
confusionMatrix(predictions_test, validation$Class)
# Create submission data frame
submission <- data.frame(validation$Class)
submission$validation.Class<- predictions_test
#trying to compare the values of train and test data
table(submission$validation.Class)
table(validation$Class)

# Predict using the test values
for(i in 1:nrow(validation)) {
  predictions <- predict(fit.rf,validation[i,])
  print(predictions)
  print(validation[i,])
}

#gbm
set.seed(7)
fit.gbm <- train(Class~., data=dataset, method="gbm", metric=metric, trControl=control,verbose = FALSE)
plot(fit.gbm)
# Predict using the training values
predictions_train <- predict(fit.gbm, dataset)
confusionMatrix(predictions_train, dataset$Class)
predictions_test <- predict(fit.gbm, validation)
confusionMatrix(predictions_test, validation$Class)
# Create submission data frame
submission <- data.frame(validation$Class)
submission$validation.Class<- predictions_test
#trying to compare the values of train and test data
table(submission$validation.Class)
table(validation$Class)

# Predict using the test values
for(i in 1:nrow(validation)) {
  predictions <- predict(fit.gbm,validation[i,])
  print(predictions)
  print(validation[i,])
}


#lvq
set.seed(7)
fit.lvq <- train(Class~., data=dataset, method ="lvq",trControl=control)
plot(fit.lvq)
# Predict using the training values
predictions_train <- predict(fit.lvq, dataset)
confusionMatrix(predictions_train, dataset$Class)
predictions_test <- predict(fit.lvq, validation)
confusionMatrix(predictions_test, validation$Class)
# Create submission data frame
submission <- data.frame(validation$Class)
submission$validation.Class<- predictions_test
#trying to compare the values of train and test data
table(submission$validation.Class)
table(validation$Class)

# Predict using the test values
for(i in 1:nrow(validation)) {
  predictions <- predict(fit.lvq,validation[i,])
  print(predictions)
  print(validation[i,])
}

#Compares the result of all the algorithm used on the dataset
results <- resamples(list(knn=fit.knn, svm=fit.svm, gbm=fit.gbm, lvq=fit.lvq))
summary(results)

# compare accuracy of models
dotplot(results)

#Thr following part prints the all the parameters and the corresponding
#accuracy
#K nearest neighbor
print(fit.knn)
#Support Vector Machines
print(fit.svm)
#Random Forest
print(fit.rf)
#Gradient Boosting
print(fit.gbm)
#Learning Vector Quantization 
print(fit.lvq)