data<-read.csv('bballdata.csv')
str(data)

# Separate Predictors and Response from non-used variables
temp <- data[,-c(1,2,3,4,5,6,7,8,14,20,21,27)]

# Coerce Predictors to be Numeric.
temp$FG3_PCT_home_bench <- as.character(temp$FG3_PCT_home_bench)
temp$FT_PCT_away_bench <- as.character(temp$FT_PCT_away_bench)
temp$FG3_PCT_away_bench <- as.character(temp$FG3_PCT_away_bench)
temp$FG3_PCT_home_bench <- as.numeric(temp$FG3_PCT_home_bench)
temp$FT_PCT_away_bench <- as.numeric(temp$FT_PCT_away_bench)
temp$FG3_PCT_away_bench <- as.numeric(temp$FG3_PCT_away_bench)
str(temp)

# Remove missing observations
row.has.na <- apply(temp, 1, function(x){any(is.na(x))})
sum(row.has.na)
temp <- temp[!row.has.na,]
table(is.na(temp)) #Verify removal of rows containing NA

# Separate Predictors from Response
pred <- temp[,-21]
response <- factor(temp[,21])
levels(response) = c('no', 'yes')
par(mfrow=c(1,2))
plot(response, main="Original Response Data") #slightly unbalanced

# Histograms
library(psych) 
multi.hist(pred, density=FALSE); #Histograms
library(e1071)
print(skewValues <- apply(pred, 2, skewness)) #Skewness Values

# Outliers
par(mfrow=c(2,5))
boxplot(pred$FG_PCT_home, main="FG%Home")
boxplot(pred$FT_PCT_home, main="FT%Home")
boxplot(pred$FG3_PCT_home, main="FG3%Home")
boxplot(pred$AST_home, main="ASTHome")
boxplot(pred$REB_home, main="REBHome")
boxplot(pred$FG_PCT_home_bench, main="FG%HomeB") ## Removed correlation
boxplot(pred$FT_PCT_home_bench, main="FT%HomeB")
boxplot(pred$FG3_PCT_home_bench, main="FG3%HomeB")
boxplot(pred$AST_home_bench, main="ASTHomeB")
boxplot(pred$REB_home_bench, main="REBHomeB")
boxplot(pred$FG_PCT_away, main="FG%Away")
boxplot(pred$FT_PCT_away, main="FT%Away")
boxplot(pred$FG3_PCT_away, main="FG3%Away")
boxplot(pred$AST_away, main="ASTAway")
boxplot(pred$REB_away, main="REBAway")
boxplot(pred$FG_PCT_away_bench, main="FG%AwayB") ## Removed correlation
boxplot(pred$FT_PCT_away_bench, main="FT%AwayB")
boxplot(pred$FG3_PCT_away_bench, main="FG3%AwayB")
boxplot(pred$AST_away_bench, main="ASTAwayB")
boxplot(pred$REB_away_bench, main="REBAwayB")

# Correlation - Remove predictors with very high correlation
library(corrplot)
library(caret)
par(mfrow=c(1,1))
correlations <- cor(pred)
corrplot(correlations, order = 'hclust') #Correlation Plot
print(highCorr <- findCorrelation(correlations, cutoff = .85))
pred <- pred[,-highCorr] #removed two predictors
corrplot(cor(pred))

# Apply Transformations
cleanup <- preProcess(pred,method=c("BoxCox","center","scale", "spatialSign")) 
predictors <- predict(cleanup,pred)

# Histograms After Transformations
library(psych)
multi.hist(predictors, density=FALSE); #Histograms
library(e1071)
print(skewValues <- apply(predictors, 2, skewness)) #Skewness Values

# Outliers After Transformations
par(mfrow=c(2,5))
boxplot(predictors$FG_PCT_home, main="FG%Home")
boxplot(predictors$FT_PCT_home, main="FT%Home")
boxplot(predictors$FG3_PCT_home, main="FG3%Home")
boxplot(predictors$AST_home, main="ASTHome")
boxplot(predictors$REB_home, main="REBHome")
boxplot(predictors$FG_PCT_home_bench, main="FG%HomeB") ## Removed for correlation
boxplot(predictors$FT_PCT_home_bench, main="FT%HomeB")
boxplot(predictors$FG3_PCT_home_bench, main="FG3%HomeB")
boxplot(predictors$AST_home_bench, main="ASTHomeB")
boxplot(predictors$REB_home_bench, main="REBHomeB")
boxplot(predictors$FG_PCT_away, main="FG%Away")
boxplot(predictors$FT_PCT_away, main="FT%Away")
boxplot(predictors$FG3_PCT_away, main="FG3%Away")
boxplot(predictors$AST_away, main="ASTAway")
boxplot(predictors$REB_away, main="REBAway")
boxplot(predictors$FG_PCT_away_bench, main="FG%AwayB") ## Removed for correlation
boxplot(predictors$FT_PCT_away_bench, main="FT%AwayB")
boxplot(predictors$FG3_PCT_away_bench, main="FG3%AwayB")
boxplot(predictors$AST_away_bench, main="ASTAwayB")
boxplot(predictors$REB_away_bench, main="REBAwayB")

set.seed(314)
trainingRows <- createDataPartition(response, p = .80, list= FALSE) 
#separate the data by injury with 80% training and 20% testing.
trP <- predictors[trainingRows, ]
trR <- response[trainingRows]
teP <- predictors[-trainingRows, ]
teR <- response[-trainingRows]
plot(trR, main="Training Set Response")

##################### Linear Methods #####################

ctrl <- trainControl(method = "LGOCV",
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE)

###### Linear Discriminant Analysis ######
set.seed(314)
LDAFull <- train(trP, trR ,
                 method = "lda",
                 metric = "ROC",
                 trControl = ctrl)
LDAFull

## LDA Test Confusion
pred = predict(LDAFull, teP)
test.conf = confusionMatrix(pred, teR)
print(test.conf)

## LDA Test ROC
pred = predict(LDAFull, teP, type = "prob")
test.roc = roc(response = teR,
               predictor = pred$yes,
               levels = rev(levels(teR)))
plot(test.roc, legacy.axes = TRUE)
print(auc(test.roc))

###### Logistic Regression ######
set.seed(314)
logistic <- train(trP, trR ,
                  method = "glm",
                  metric = "ROC",
                  trControl = ctrl)
logistic

## Logistic Test Confusion
pred = predict(logistic, teP)
test.conf = confusionMatrix(pred, teR)
print(test.conf)

## Logistic Test ROC
pred = predict(logistic, teP, type = "prob")
test.roc = roc(response = teR,
               predictor = pred$yes,
               levels = rev(levels(teR)))
plot(test.roc, legacy.axes = TRUE)
print(auc(test.roc))

###### Partial Least Squares Discriminant Analysis ######
set.seed(314)
plsFit2 <- train(trP, trR ,
                 method = "pls",
                 tuneGrid = expand.grid(.ncomp = 1:10),
                 preProc = c("center","scale"),
                 metric = "ROC",
                 trControl = ctrl)

plsFit2
plot(plsFit2)

## PLS Test Confusion
pred = predict(plsFit2, teP)
test.conf = confusionMatrix(pred, teR)
print(test.conf)

## PLS Test ROC
pred = predict(plsFit2, teP, type = "prob")
test.roc = roc(response = teR,
               predictor = pred$yes,
               levels = rev(levels(teR)))
plot(test.roc, legacy.axes = TRUE)
print(auc(test.roc))

###### Penalized Model ######
glmnGrid <- expand.grid(.alpha = c(0, .1, .2, .4, .6, .8, 1),
                        .lambda = seq(.01, .2, length = 10))
set.seed(314)
glmnTuned <- train(trP,trR,
                   method = "glmnet",
                   tuneGrid = glmnGrid,
                   preProc = c("center", "scale"),
                   metric = "ROC",
                   trControl = ctrl)
glmnTuned
plot(glmnTuned, plotType = "level")

## GLMN Test Confusion
pred = predict(glmnTuned, teP)
test.conf = confusionMatrix(pred, teR)
print(test.conf)

## GLMN Test ROC
pred = predict(glmnTuned, teP, type = "prob")
test.roc = roc(response = teR,
               predictor = pred$yes,
               levels = rev(levels(teR)))
plot(test.roc, legacy.axes = TRUE)
print(auc(test.roc))

##################### Non-Linear Methods #####################

ctrl <- trainControl(method = "LGOCV",
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE)

###### FDA Method ###### 
marsGrid <- expand.grid(.degree = 1:2, .nprune = 2:20)
set.seed(314)
fdaTuned <- train(x = trP, 
                  y = trR,
                  method = "fda",
                  metric = "ROC",
                  tuneGrid = marsGrid,
                  trControl = ctrl)

fdaTuned
plot(fdaTuned)

## FDA Test Confusion
pred = predict(fdaTuned, teP)
test.conf = confusionMatrix(pred, teR)
print(test.conf)

## FDA Test ROC
pred = predict(fdaTuned, teP, type = "prob")
test.roc = roc(response = teR,
               predictor = pred$yes,
               levels = rev(levels(teR)))
plot(test.roc, legacy.axes = TRUE)
print(auc(test.roc))

###### KNN Method ######
library(caret)
set.seed(314)
knnFit <- train(x = trP, 
                y = trR,
                method = "knn",
                metric = "ROC",
                preProc = c("center", "scale"),
                tuneGrid = data.frame(.k = 78),
                trControl = ctrl)
knnFit
plot(knnFit)

## KNN Test Confusion
pred = predict(knnFit, teP)
test.conf = confusionMatrix(pred, teR)
print(test.conf)

## KNN Test ROC
pred = predict(knnFit, teP, type = "prob")
test.roc = roc(response = teR,
               predictor = pred$yes,
               levels = rev(levels(teR)))
par(mfrow=c(1,1))
plot(test.roc, legacy.axes = TRUE)
print(auc(test.roc))

###### MDA Method ######
set.seed(314)
mdaFit <- train(x = trP, 
                y = trR,
                method = "mda",
                metric = "ROC",
                tuneGrid = expand.grid(.subclasses = 1:3),
                trControl = ctrl)
mdaFit

## MDA Test Confusion
pred = predict(mdaFit, teP)
test.conf = confusionMatrix(pred, teR)
print(test.conf)

## MDA Test ROC 
pred = predict(mdaFit, teP, type = "prob")
test.roc = roc(response = teR,
               predictor = pred$yes,
               levels = rev(levels(teR)))
plot(test.roc, legacy.axes = TRUE)
print(auc(test.roc))

####### Naive Bayes Method ####### 
library(klaR)
set.seed(314)
nbFit <- train( x = trP, 
                y = trR,
                method = "nb",
                metric = "ROC",
                tuneGrid = data.frame(.fL = 2,.usekernel = TRUE,.adjust = TRUE),
                trControl = ctrl)

nbFit

## NB Test Confusion
pred = predict(nbFit, teP)
test.conf = confusionMatrix(pred, teR)
print(test.conf)

## NB Test ROC
pred = predict(nbFit, teP, type = "prob")
test.roc = roc(response = teR,
               predictor = pred$yes,
               levels = rev(levels(teR)))
plot(test.roc, legacy.axes = TRUE)
print(auc(test.roc))

###### Neural Net Method ######
nnetGrid <- expand.grid(.size = 1:10, .decay = c(0, .1, .3, .5, 1))
maxSize <- max(nnetGrid$.size)
numWts <- (maxSize * (18 + 1) + (maxSize+1)*2)
set.seed(314)
nnetFit <- train(x = trP, 
                 y = trR,
                 method = "nnet",
                 metric = "ROC",
                 tuneGrid = nnetGrid,
                 trace = FALSE,
                 maxit = 2000,
                 MaxNWts = numWts,
                 trControl = ctrl)
nnetFit
plot(nnetFit)

## NNet Test Confusion
pred = predict(nnetTune, teP)
test.conf = confusionMatrix(pred, teR)
print(test.conf)

## NNet Test ROC
pred = predict(nnetTune, teP, type = "prob")
test.roc = roc(response = teR,
               predictor = pred$yes,
               levels = rev(levels(teR)))
plot(test.roc, legacy.axes = TRUE)
print(auc(test.roc))


###### QDA Method ###### ************RERUN
set.seed(314)
qdaFit <- train(x = trP, 
                y = trR,
                method = "qda",
                metric = "ROC",
                trControl = ctrl)
qdaFit

## QDA Test Confusion
pred = predict(qdaFit, teP)
test.conf = confusionMatrix(pred, teR)
print(test.conf)

## QDA Test ROC
pred = predict(qdaFit, teP, type = "prob")
test.roc = roc(response = teR,
               predictor = pred$yes,
               levels = rev(levels(teR)))
plot(test.roc, legacy.axes = TRUE)
print(auc(test.roc))

###### RDA Method ######
set.seed(314)
rdaGrid <- expand.grid(.gamma = seq(0, 1, length = 11),
                       .lambda = seq(0, 1, length = 11))
rdaFit <- train(x = trP, 
                y = trR,
                method = "rda",
                metric = "ROC",
                tuneGrid = rdaGrid,
                trControl = ctrl)
rdaFit
plot(rdaFit)

## RDA Test Confusion
pred = predict(rdaFit, teP)
test.conf = confusionMatrix(pred, teR)
print(test.conf)

## RDA Test ROC
pred = predict(rdaFit, teP, type = "prob")
test.roc = roc(response = teR,
               predictor = pred$yes,
               levels = rev(levels(teR)))
plot(test.roc, legacy.axes = TRUE)
print(auc(test.roc))


###### SVMR Method ###### 
library(kernlab)
library(caret)
sigmaRangeReduced <- sigest(as.matrix(trP))
svmRGridReduced <- expand.grid(.sigma = sigmaRangeReduced[1],
                               .C = 2^(seq(-4, 6)))
set.seed(314)
svmRModel <- train(x = trP, 
                   y = trR,
                   method = "svmRadial",
                   metric = "ROC",
                   preProc = c("center", "scale"),
                   tuneGrid = svmRGridReduced,
                   fit = FALSE,
                   trControl = ctrl)
svmRModel
plot(svmRModel)

## SVMR Test Confusion
pred = predict(svmRModel, teP)
test.conf = confusionMatrix(pred, teR)
print(test.conf)

## SVMR Test ROC
library(pROC)
pred = predict(svmRModel, teP, type = "prob")
test.roc = roc(response = teR,
               predictor = pred$yes,
               levels = rev(levels(teR)))
plot(test.roc, legacy.axes = TRUE)
print(auc(test.roc))

library(caret)
plot(varImp(svmRModel), 5, main="PLS-Importance")
plot(varImp(glmnTuned), 5, main="PLS-Importance")
