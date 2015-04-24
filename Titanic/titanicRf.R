# R script as a solution to Kaggle Titanic problem with Random Forests

library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
library(randomForest)
library(party)

# Set working directory and import train and test data
setwd("~/Documents/development/learn/vids/Data Science/Projects/Kaggle/Titanic")

train <- read.csv("~/Documents/development/learn/vids/Data Science/Projects/Kaggle/Titanic/train.csv")
test <- read.csv("~/Documents/development/learn/vids/Data Science/Projects/Kaggle/Titanic/test.csv")

test$Survived <- NA
combi <- rbind(train, test)
combi$Name <- as.character(combi$Name)

# Get the title from names
combi$Title <- sapply(combi$Name, FUN=function(x) {strsplit(x, split='[.,]')[[1]][2]})
combi$Title <- sub(' ', '', combi$Title)
table(combi$Title)

# Merge the insignificant ones
combi$Title[combi$Title %in% c('Mme', 'Mlle')]  <- 'Mlle'
combi$Title[combi$Title %in% c('Capt', 'Don', 'Major', 'Sir')] <- 'Sir'
combi$Title[combi$Title %in% c('Dona', 'Lady', 'the Countess', 'Jonkheer')] <- 'Lady'

# Convert it again into factor
combi$Title <- factor(combi$Title)

# Add the number of family members
combi$FamilySize <- combi$SibSp + combi$Parch + 1

combi$Surname <- sapply(combi$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][1]})

# Get the new family name variable by combining the surname to familySize
combi$FamilyID <- paste(as.character(combi$FamilySize), combi$Surname, sep="")

combi$FamilyID[combi$FamilySize <= 2] <- 'Small'

famIDs <- data.frame(table(combi$FamilyID))

famIDs <- famIDs[famIDs$Freq <= 2,]

combi$FamilyID[combi$FamilyID %in% famIDs$Var1] <- 'Small'
combi$FamilyID <- factor(combi$FamilyID)
combi$Age[is.na(combi$Age)] <- predict(Agefit, combi[is.na(combi$Age),])

Agefit <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Title + FamilySize,
                data=combi[!is.na(combi$Age),], method="anova")
combi$Age[is.na(combi$Age)] <- predict(Agefit, combi[is.na(combi$Age),])

summary(combi)

combi$Embarked[which(combi$Embarked == '')] = "S"

combi$Embarked <- factor(combi$Embarked)

combi$Fare[which(is.na(combi$Fare))] = median(combi$Fare, na.rm=TRUE)

# Redduce the no. of factors in FamilyID to be less than 32
combi$FamilyID2 <- combi$FamilyID
combi$FamilyID2 <- as.character(combi$FamilyID2)
combi$FamilyID2[combi$FamilySize <= 3] <- 'Small'
combi$FamilyID2 <- factor(combi$FamilyID2)

# We can now split the data back to training and test data
train <- combi[1:891,]
test <- combi[892:1309,]

# To maintain the consistency in the randomness
set.seed(415)

fit <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilyID2, data=train, importance=TRUE, ntree=2000)

# Look at the important variables
varImpPlot(fit)

prediction <- predict(fit, test)
submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(submit, file = "firstforest.csv", row.names = FALSE)

# Let's try conditional inference trees
set.seed(415)

fit <- cforest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize, data = train, controls=cforest_unbiased(ntree=2000, mtry=3))

Prediction <- predict(fit, test, OOB=TRUE, type = "response")
submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(submit, file = "secondforest.csv", row.names = FALSE)