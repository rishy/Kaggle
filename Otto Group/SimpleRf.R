# library(caret)
library(randomForest)
#library(doMC)
#registerDoMC()

setwd("~/Documents/development/Data Science/Projects/Kaggle/Otto Group")

train <- read.csv('train.csv', header = T)
test <- read.csv('test.csv', header = T)
sample <- read.csv('sampleSubmission.csv', header = T)

# Partition data in training, validation and test sets
inTrain <- createDataPartition(train$target, p = 0.75, list = F)
training <- train[inTrain, -1]
validation <- train[-inTrain, - 1]
testing <- test[,-1]

#fitControl = trainControl(method = "cv", number = 5, classProbs = T, verboseIter = TRUE, allowParallel = TRUE)

formula = as.formula(paste("target", '~', paste(names(training[, -94]), collapse = "+")))

tmp <- as.vector(table(training$target)); 
num_clases <- length(tmp); 
min_size <- tmp[order(tmp,decreasing=FALSE)[1]]; 
vector_for_sampsize <- rep(min_size,num_clases); 

#rf.fit <- train(form = formula, data = training, method = "rf", trControl = fitControl)
rf.fit <- randomForest(formula = formula, data = training, xtest = validation[, -94],
                       ytest = validation$target, mtry = 5, ntree = 200, do.trace = T, 
                       keep.forest = T, sampsize = vector_for_sampsize)

rf.pred <- predict(rf.fit, newdata = testing, type = "prob")

sol <- as.data.frame(cbind('id' = test$id, rf.pred))

write.csv(sol, "SampledRf.csv", row.names = F)
