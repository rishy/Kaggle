setwd("~/Documents/development/DataScience/Projects/Kaggle/TFI Restaurant Revenue")

library(caret)
library(ggplot2)
library(dplyr)

train <- read.csv('train.csv')
test <- read.csv('test.csv')
sampleCsv <- read.csv('sampleSubmission.csv')

# Describe the train data
head(train)
summary(train)
str(train)

# Get the year out of Open.Date
train$Open.Period <- ((as.Date("2015-01-01") - 
                         as.Date(train$Open.Date, "%m/%d/%Y")) * 12)/365

test$Open.Period <- ((as.Date("2015-01-01") - 
                        as.Date(test$Open.Date, "%m/%d/%Y")) * 12)/365

# Remove unwanted columns
train_set <- subset(train, select = -c(Id, Open.Date, City))
test_set <- subset(test, select = -c(Id, Open.Date, City))

# Creates Dummy Variables
createDummyVars <- function(df){
  
  # Create Dummy Variables for City.Group and Type features
  df$City.Group.Big <- ifelse(df$City.Group == "Big Cities", 1, 0)
  df$City.Group.Other <- ifelse(df$City.Group == "Other", 1, 0)
  df$Type.DT <- ifelse(df$Type == "DT", 1, 0)
  df$Type.FC <- ifelse(df$Type == "FC", 1, 0)
  df$Type.IL <- ifelse(df$Type == "IL", 1, 0)
  
  # Remove factor columns
  df$City.Group <- NULL
  df$Type <- NULL
  
  # Return new data frame
  df
}

# Create final training and test set with dummy variables
train_data <- createDummyVars(train_set)
test_data <- createDummyVars(test_set)

# A trainControl model for lasso fit
fitControl <- trainControl(method = 'repeatedcv', repeats = 10, number = 10,
                           verboseIter = T)

lasso.fit <- train(revenue ~ ., method = 'lasso', data = train_data,
                   trControl = fitControl, tuneLength = 5, preProcess = c('scale', 'center'))

lasso.predict <- predict(lasso.fit, test_data)

result = data.frame(cbind(test$Id, lasso.predict))
names(result) = names(sampleCsv)

write.csv(result, file = 'first_lasso.csv', row.names = F)

###################### Random Forest ####################

fitControl <- trainControl(method = 'repeatedcv', repeats = 10, number = 10,
                           verboseIter = T)

rf.fit <- train(revenue ~ ., method = 'rf', data = train_data,
                   trControl = fitControl, tuneLength = 5, importance = T)

rf.predict <- predict(rf.fit, test_data)

result = data.frame(cbind(test$Id, rf.predict))
names(result) = names(sampleCsv)

write.csv(result, file = 'first_rf.csv', row.names = F)

################## Ridge Regression ###################

# A trainControl model for lasso fit
fitControl <- trainControl(method = 'repeatedcv', repeats = 10, number = 10,
                           verboseIter = T)

ridge.fit <- train(revenue ~ ., method = 'ridge', data = train_data,
                   trControl = fitControl, tuneLength = 5, preProcess = c('scale', 'center'))

ridge.predict <- predict(ridge.fit, test_data)

result = data.frame(cbind(test$Id, ridge.predict))
names(result) = names(sampleCsv)

write.csv(result, file = 'first_ridge.csv', row.names = F)


######################## SVM with Radial Kernel ########################

radialSVM.fit <- train(revenue ~ ., method = 'svmRadial', data = train_data,
                       trControl = fitControl, tuneLength = 5, preProcess = c('scale', 'center'))

radialSVM.predict <- predict(radialSVM.fit, test_data)

result = data.frame(cbind(test$Id, radialSVM.predict))
names(result) = names(sampleCsv)

write.csv(result, file = 'radialSVM.csv', row.names = F)


