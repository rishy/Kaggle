# R script as a solution to Kaggle Titanic problem

library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)

# Set working directory and import train and test data
setwd("~/Documents/development/learn/vids/Data Science/Projects/Kaggle/Titanic")

train <- read.csv("~/Documents/development/learn/vids/Data Science/Projects/Kaggle/Titanic/train.csv")
test <- read.csv("~/Documents/development/learn/vids/Data Science/Projects/Kaggle/Titanic/test.csv")

# rbind both test and train data
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

# We can now split the data back to training and test data
train <- combi[1:891,]
test <- combi[892:1309,]

fit <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilyID,
             data=train, method="class")

#  Here’s another drawback with decision trees that I didn’t mention last time: they are biased to favour factors with many levels. 
Prediction <- predict(fit, test, type = "class")

# Save the result in csv format
submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)

write.csv(submit, file = "myfeaturedtree.csv", row.names = FALSE)
