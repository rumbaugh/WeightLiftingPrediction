---
title: "Prediction Assignment"
output:
  html_document: default
  pdf_document: default
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
#setwd('/home/rumbaugh/WeightLiftingPrediction/')
library(caret)
```

Our goal is to predict whether exercises were performed 'correctly' or not. Our data consist of the output of devices worn while individuals performed excercises using dumbbells in various manners. 

```{r}
initdf = read.csv('pml-training.csv')
```

A summary of the data shows there are 160 columns. Part of this summary is shown in the Appendix. The column we are most interested in is *classe*, which details the manner in which an exercise was performed. It is a factor variable, taking five different values. 

There are several issues with the dataset. First, the summary reveals some large outliers for some of the variables. Investigation shows that a single row is responsible for many of these outliers, so we remove it. 

```{r}
traindf = initdf[initdf$gyros_dumbbell_y<30,]
```

The data in some of the columns also appears to be incomplete or not meaningful. In the summary in the Appendix, these columns have 19216 rows that are blank, with the rest having numeric values or "#DIV/0!". Both the difficulty of understanding these columns and the large percentage of identical values mean these will not be useful for prediction, so we remove them

The first 7 columns also do not contain useful features. These include columns such as the row number, the user name, and the time stamp. We remove these columns as well. 

In addition, some columns mostly contain missing values, so we remove those columns. 

```{r}
bad_features = c('new_window', 'kurtosis_roll_belt', 'kurtosis_picth_belt', 'kurtosis_yaw_belt', 'skewness_roll_belt', 'skewness_roll_belt.1', 'skewness_yaw_belt', 'max_yaw_belt', 'min_yaw_belt', 'amplitude_yaw_belt', 'kurtosis_roll_arm', 'kurtosis_picth_arm', 'kurtosis_yaw_arm', 'skewness_roll_arm', 'skewness_pitch_arm', 'skewness_yaw_arm', 'kurtosis_roll_dumbbell', 'kurtosis_picth_dumbbell', 'kurtosis_yaw_dumbbell', 'skewness_roll_dumbbell', 'skewness_pitch_dumbbell', 'skewness_yaw_dumbbell', 'max_yaw_dumbbell', 'amplitude_yaw_dumbbell', 'kurtosis_roll_forearm', 'kurtosis_picth_forearm', 'kurtosis_yaw_forearm', 'skewness_roll_forearm', 'skewness_pitch_forearm', 'skewness_yaw_forearm', 'max_yaw_forearm', 'min_yaw_forearm', 'amplitude_yaw_forearm','min_yaw_dumbbell')
not_predictors = c('X', 'user_name', 'raw_timestamp_part_1', 'raw_timestamp_part_2', 'cvtd_timestamp', 'num_window')
traindf = traindf[,!(names(traindf) %in% c(bad_features, not_predictors))]
traindf = traindf[,colSums(is.na(traindf)) == 0]
```

After paring down the dataset, we are left with 53 columns and 19621 rows. We break this up into a training and testing dataset.

``` {r}
set.seed(234256)
inTrain = createDataPartition(traindf$classe, p = 0.6, list = F)
testdf = traindf[-inTrain,]
traindf = traindf[inTrain,]
```

We fit three mdoels to the training set: Random Forest, Stochastic Gradient Boosting, and Linear Discrimant Analysis. Our goal is to stack the results of these models to create an ensemble method.
``` {r eval=F}
modFit1 = train(classe~., method = 'rf', data = prelim_traindf, preProcess = c('scale', 'center'))
modFit2 = train(classe~., method = 'gbm', data = prelim_traindf, preProcess = c('center', 'scale'), verbose=F)
modFit3 = train(classe~., method = 'lda', data = prelim_traindf, preProcess = c('scale', 'center'))
pred1 = predict(modFit1, testdf)
pred2 = predict(modFit2, testdf)
pred3 = predict(modFit3, testdf)
```
``` {r echo=F}
load('/home/rumbaugh/WeightLiftingPrediction/models.RDS')
```

After training the models on our training set, we find the predictions for the models on the test set. Using these predictions, we can create an ensemble method by having the three models vote on each prediction. We implement this below, and display the accuracies of the three models and the stacked model.
```{r echo=F}
Mode <- function(x, returnNull = T, customNull = NULL) {
  # Function for finding mode of a vector. Includes option
  # to return NULL if no single value is the mode (i.e., 
  # there's a tie). Can return something else if customNull is set.
  ux <- unique(x)
  tmpsort = sort(tabulate(match(x, ux)), decreasing = T)
  if (length(tmpsort) == 1) tmpsort = c(tmpsort, 0)
  if (returnNull & tmpsort[1] == tmpsort[2]) {
     out = customNull
  } else {
     out = ux[which.max(tabulate(match(x, ux)))]
  }
  if (class(out) == "data.frame") {
     out[1,1]
  } else {
     out
  }
}
```
```{r}
get_acc <- function(pred) sum(pred == testdf$classe)/length(pred)
acc1 = get_acc(pred1); acc2 = get_acc(pred2); acc3 = get_acc(pred3)
sprintf("RF Acc.: %.4f, GBM Acc.: %.4f, LDF Acc.: %.4f", acc1, acc2, acc3)
most_accurate = 1
if (acc2 > acc1) most_accurate = 2
if (acc3 > acc1 & acc3 > acc2) most_accurate = 3
# Perform stacking, taking most popular prediction for each row, or
# prediction of most accurate model if none agree
stackdf = data.frame(pred1, pred2, pred3)
stack_pred = sapply(1:nrow(stackdf), function(i) Mode(stackdf[i,], returnNull = T, customNull = stackdf[[i, most_accurate]]))
sprintf('Stacked Acc.: %.4f', sum(stack_pred == testdf$classe)/length(stack_pred))
```
With over 99% accuracy, the random forest method is the most accurate of the three. While stacking creates better predictions that two out of the three models, it actually performs worse than the Random Forest model alone, which is therefore our best model. The 99% accuracy was evaluated on the testing set, separate from where the model was trained, so this represents our estimate of the out-of-sample error rate. 

## Appendix
```{r}
summary(initdf[,c(1:8,12:13,150:154,159:160)])
```
