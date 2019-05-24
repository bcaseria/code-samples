#### Question 1 ####

library(boot)
library(stats)
gpa.data = read.csv("~/Classes Spring 2019/STAT 6340/Mini Projects/Project 4/gpa.csv")

attach(gpa.data)

## Part a: plotting the data
plot(gpa ~ act)

## Part b: correlation bootstrap function
cor.fn = function(data, index) {
  result = cor(data$gpa[index],data$act[index])
  return(result)
}

# Obtaining bootstrap estimate, bias, and standard error
set.seed(6174)
cor.boot = boot(gpa.data, cor.fn, R = 1000)
cor.boot

# Computing bootstrap confidence interval
boot.ci(boot.out = cor.boot, type = "perc")

## Part c: Linear model (coefficients, standard errors, and confidence intervals)
gpa.lm = lm(gpa ~ act, data = gpa.data)
summary(gpa.lm)
confint(gpa.lm)

# Diagnostic check
plot(gpa.lm) # Gets QQ plot and Residual vs. Fitted plot
plot(resid(gpa.lm), type="l", ylab = "Residuals", main = "Time Series Plot") #Time Series plot
abline(h=0)

## Part d: Bootstrap lm stuff

# Function for lm bootstrap
lm.fn = function (data,index) {
  result = coef(lm(gpa ~ act, data = data, subset = index))
  return(result)
}

# lm bootstrap
set.seed(6174)
lm.boot = boot(gpa.data, lm.fn, R = 1000)
lm.boot

# Obtaining percentile based confidence limits
lm.boot.sorted = apply(lm.boot$t,2,sort)
t(lm.boot.sorted[c(25,975),])

#### Question 2 ####

library(ISLR)
library(crossval)

attach(OJ)

## Part a: Comparison of STORE, Store7, and StoreID
unique(STORE)
unique(StoreID)
all(STORE == (StoreID %% 7)) # This means all 0's map to 7's, and the rest are equal

unique(Store7)
all((Store7 == 'Yes') == (StoreID == 7)) # This means StoreID = 7 iff Store7 = 'Yes'

## Part b: 
OJ = OJ[-c(14,18)] # Removing redundant store label columns
OJ$StoreID = as.factor(OJ$StoreID) # Making StoreID categorical
# Other redundant columns removed

# Fitting the logistic regression
set.seed(6174)
lr.fit = glm(Purchase ~ . , family = "binomial", data = OJ)

# Getting predictions from training set
lr.pred = predict(lr.fit, OJ[,2:12])

# Computing confusion matrix, with sensitivity, specificity, and accuracy
t = (Purchase == "MM")
p = (lr.pred >= 0)
caret::confusionMatrix(as.factor(p),as.factor(t), positive = "TRUE")

# ROC curve
lr.roc = roc(Purchase, lr.pred, levels = c("MM","CH"))
plot(lr.roc, legacy.axes = TRUE, main = "ROC: Logistic")

# 10-fold cross validation
set.seed(6174)
cv.glm(OJ, lr.fit, K = 10)$delta

## Part c: LDA

# Fitting the LDA
set.seed(6174)
lda.fit = lda(Purchase ~ . , data = OJ)

# Getting predictions from training set
lda.pred = predict(lda.fit, OJ[,2:12])

# Computing confusion matrix, with sensitivity, specificity, and accuracy
t = (Purchase == "MM")
p = (lda.pred$class == "MM")
caret::confusionMatrix(as.factor(p),as.factor(t), positive = "TRUE")

# ROC curve
lda.roc = roc(Purchase, lda.pred$posterior[,2], levels = c("MM","CH"))
plot(lda.roc, legacy.axes = TRUE, main = "ROC: LDA")

# 10-fold cross validation
predfun.lda = function(train.x, train.y, test.x, test.y, negative)
{
  lda.fit = lda(train.x, grouping=train.y)
  ynew = predict(lda.fit, test.x)$class
  out = crossval::confusionMatrix(test.y, ynew, negative=negative)
  return( (out[1]+out[4])/(sum(out)) )
}

X = data.matrix(OJ[,2:12])
Y = OJ[,1]
set.seed(6174)
crossval(predfun.lda, X, Y, K = 10, B = 20, negative = "CH")

## Part d: QDA

# Fitting the QDA
set.seed(6174)
qda.fit = qda(Purchase ~ . , data = OJ)

# Getting predictions from training set
qda.pred = predict(qda.fit, OJ[,2:12])

# Computing confusion matrix, with sensitivity, specificity, and accuracy
t = (Purchase == "MM")
p = (qda.pred$class == "MM")
caret::confusionMatrix(as.factor(p),as.factor(t), positive = "TRUE")

# ROC curve
qda.roc = roc(Purchase, qda.pred$posterior[,2], levels = c("MM","CH"))
plot(qda.roc, legacy.axes = TRUE, main = "ROC: QDA")

# 10-fold cross validation
predfun.qda = function(train.x, train.y, test.x, test.y, negative)
{
  qda.fit = qda(train.x, grouping=train.y)
  ynew = predict(qda.fit, test.x)$class
  out = crossval::confusionMatrix(test.y, ynew, negative=negative)
  return( (out[1]+out[4])/(sum(out)) )
}

X = data.matrix(OJ[,2:12])
Y = OJ[,1]
set.seed(6174)
crossval(predfun.qda, X, Y, K = 10, B = 20, negative = "CH")

## Part e: KNN

K = c(seq(1,70,by = 1))
errs = matrix(0,nrow = length(K),ncol = 1)

# 10-fold cross validation to determine optimal K
predfun.knn = function(train.x, train.y, test.x, test.y, k, negative)
{
  set.seed(6174)
  knn.fit = knn(train.x, test.x, train.y, k = k) # Optimal K = 9
  out = crossval::confusionMatrix(test.y, knn.fit, negative=negative)
  return( (out[1]+out[4])/(sum(out)) )
}

X = data.matrix(OJ[,2:12])
Y = OJ[,1]
for (i in 1:length(K)) {
errs[i] = crossval(predfun.knn, X, Y, K = 10, B = 20, k = K[i], negative = "CH")$stat
}

xyplot(errs ~ K, pch = 19, main = "Comparison of Errors w.r.t K values", ylab = "Error Rate")

# Fitting the KNN with K = 9
K.opt = 9
set.seed(6174)
knn.fit = knn(X,X,Y,k = K.opt, prob = T)

# Computing confusion matrix, with sensitivity, specificity, and accuracy
t = (Purchase == "MM")
p = (knn.fit == "MM")
caret::confusionMatrix(as.factor(p),as.factor(t), positive = "TRUE")

# ROC curve
knn.prob = attr(knn.fit, "prob")
knn.prob[knn.fit == "CH"] = 1 - knn.prob[knn.fit == "CH"]
knn.roc = roc(Purchase, knn.prob, levels = c("MM","CH"))
plot(knn.roc, legacy.axes = TRUE, main = "ROC: KNN (K = 9)")

#### Question 3 ####

attach(Auto)

## Part a: Exploratory Analysis

plot(mpg ~ displacement, pch = 19)
plot(mpg ~ horsepower, pch = 19)
plot(mpg ~ weight, pch = 19)
plot(mpg ~ acceleration, pch = 19)
plot(mpg ~ year, pch = 19)
plot(mpg ~ cylinders, pch = 19)
boxplot(mpg ~ origin, pch = 19, xlab = "Origin", ylab = "MPG")

## Part b: Multiple Least Squares
auto.data = Auto[,1:8]
auto.data[,8] = as.factor(auto.data[,8])
auto.lm = lm(mpg ~ ., data = auto.data)
summary(auto.lm)

## Part c: Best-subset selection
library(leaps)

# Get best model for each size
auto.summary = summary(regsubsets(mpg ~ ., auto.data, nvmax = 8))
plot(auto.summary$adjr2[2:8] ~ c(2:8), pch = 19, xlab = "Number of Variables",
     ylab = "Adjusted R Squared", main = "Adjusted R Squared Comparison - Best Subset" )
auto.summary
which.max(auto.summary$adjr2)

## Part d: Forward Selection

auto.fwd = summary(regsubsets(mpg ~ ., data = auto.data, nvmax = 8, method = "forward"))
plot(auto.fwd$adjr2[2:8] ~ c(2:8), pch = 19, xlab = "Number of Variables",
     ylab = "Adjusted R Squared", main = "Adjusted R Squared Comparison - Forward Selection" )
auto.fwd
which.max(auto.fwd$adjr2)

## Part e: Backward Selection

auto.bkd = summary(regsubsets(mpg ~ ., data = auto.data, nvmax = 8, method = "backward"))
plot(auto.bkd$adjr2[2:8] ~ c(2:8), pch = 19, xlab = "Number of Variables",
     ylab = "Adjusted R Squared", main = "Adjusted R Squared Comparison - Backward Selection" )
auto.bkd
which.max(auto.bkd$adjr2)


