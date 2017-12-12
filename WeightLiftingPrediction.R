
if (!exists('traindf')) traindf = read.csv(paste(datapath, 'pml-training.csv', sep = '/'))

traindf = traindf[traindf$gyros_dumbbell_y<30,] # Gets rid of one highly anomalous row

## The following features are correlated with new_window.
## When new_window is 'no', they are all empty. When it is 'yes',
## they are 0, #DIV/0!, or some other values. Because of both the
## difficulty to interpret these and the fact that ~98% of the values
## are exactly identical, these are thrown out.

bad_features = c('new_window', 'kurtosis_roll_belt', 'kurtosis_picth_belt', 'kurtosis_yaw_belt', 'skewness_roll_belt', 'skewness_roll_belt.1', 'skewness_yaw_belt', 'max_yaw_belt', 'min_yaw_belt', 'amplitude_yaw_belt', 'kurtosis_roll_arm', 'kurtosis_picth_arm', 'kurtosis_yaw_arm', 'skewness_roll_arm', 'skewness_pitch_arm', 'skewness_yaw_arm', 'kurtosis_roll_dumbbell', 'kurtosis_picth_dumbbell', 'kurtosis_yaw_dumbbell', 'skewness_roll_dumbbell', 'skewness_pitch_dumbbell', 'skewness_yaw_dumbbell', 'max_yaw_dumbbell', 'amplitude_yaw_dumbbell', 'kurtosis_roll_forearm', 'kurtosis_picth_forearm', 'kurtosis_yaw_forearm', 'skewness_roll_forearm', 'skewness_pitch_forearm', 'skewness_yaw_forearm', 'max_yaw_forearm', 'min_yaw_forearm', 'amplitude_yaw_forearm','min_yaw_dumbbell')

## The following features aren't really predictors. They are the row
## number, the user_name, and the timestamp
not_predictors = c('X', 'user_name', 'raw_timestamp_part_1', 'raw_timestamp_part_2', 'cvtd_timestamp', 'num_window')

traindf = traindf[,!(names(traindf) %in% c(bad_features, not_predictors))]

# There are a number of columns that are mostly N/A. This removes them.
traindf = traindf[,colSums(is.na(traindf)) == 0]

# Set up a preliminary training set, a second dataset for training
# the stacked predictors, and a final test set. Split it 60-30-10.
inTrain = createDataPartition(traindf$classe, p = 0.6, list = F)
testdf = traindf[-inTrain,]
traindf = traindf[inTrain,]

# Fit there different models
modFit1 = train(classe~., method = 'rf', data = prelim_traindf, preProcess = c('scale', 'center'))
modFit2 = train(classe~., method = 'gbm', data = prelim_traindf, preProcess = c('center', 'scale'), verbose=F)
modFit3 = train(classe~., method = 'lda', data = prelim_traindf, preProcess = c('scale', 'center'))

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

stack_models <- function(testdf) {
    # Create vectors of predicted values
    pred1 = predict(modFit1, testdf)
    pred2 = predict(modFit2, testdf)
    pred3 = predict(modFit3, testdf)
    
    get_acc <- function(pred) sum(pred == testdf$classe)/length(pred)
    acc1 = get_acc(pred1); acc2 = get_acc(pred2); acc3 = get_acc(pred3)
    most_accurate = 1
    if (acc2 > acc1) most_accurate = 2
    if (acc3 > acc1 & acc3 > acc2) most_accurate = 3
    
    # Perform stacking
    stackdf = data.frame(pred1, pred2, pred3)
    stack_pred = sapply(1:nrow(stackdf), function(i) Mode(stackdf[i,], returnNull = T, customNull = stackdf[[i, most_accurate]]))
}
    
save(modFit1, modFit2, modFit3, pred1, pred2, pred3, file = '/home/rumbaugh/WeightLiftingPrediction/models.RDS')
load('models.RDS')