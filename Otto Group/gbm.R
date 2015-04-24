library(caret)
library(doMC)
registerDoMC(cores = 4)

setwd("~/Documents/development/Data Science/Projects/Kaggle/Otto Group")

train <- read.csv('train.csv', header = T)
test <- read.csv('test.csv', header = T)

# Partition data in training, validation and test sets
inTrain <- createDataPartition(train$target, p = 0.75, list = F)
training <- train[inTrain, -1]
validation <- train[-inTrain, - 1]
testing <- test[,-1]

fitControl = trainControl(method = 'repeatedcv', number = 10, repeats = 3,
 classProbs = T, verboseIter = TRUE, allowParallel = TRUE)

formula = as.formula(paste("target", '~', paste(names(training[, -94]), collapse = "+")))

#tuneParams <- data.frame("n.trees" = c(rep(150, 4), rep(200, 4), rep(500, 4), rep(1000, 4)), "interaction.depth" = c(1, 3, 5, 9, 1, 3, 5, 9, 1, 3, 5, 9, 1, 3, 5, 9), shrinkage = c(0.01, 0.1, 1, 10, 0.01, 0.1, 1, 10, 0.01, 0.1, 1, 10, 0.01, 0.1, 1, 10))
#tuneParams <- data.frame("n.trees" = rep(500, 3), "interaction.depth" = c(1, 3, 5), shrinkage = rep(0.01, 3))

gbm.fit <- train(form = formula, data = training, method = "gbm", trControl = fitControl, 
                 tuneLength = 5)

gbm.pred <- predict(gbm.fit, newdata = validation[, -94])

gbm.pred <- predict(gbm.fit, newdata = testing, type = "prob")

sol <- as.data.frame(cbind('id' = test$id, gbm.pred))

sol$id <- as.integer(sol$id)

write.csv(sol, "firstGbm.csv", row.names = F)

