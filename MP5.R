library("ISLR")
library("MASS")
library("caret")
library("e1071")
library("glmnet")
library("stats")
library("ElemStatLearn")
library("leaps")
library("pls")
library("boot")

#### Question 1 ####

attach(Caravan)
x.center = as.data.frame(scale(Caravan[,-86]))
data.1 = data.frame(x.center,Purchase)

idx.test = 1:1000
x.train = x.center[-idx.test,]
x.test = x.center[idx.test,]
y.train = data.1$Purchase[-idx.test]
y.test = data.1$Purchase[idx.test]

## Logistic Regression ##

lr.1 = glm(Purchase ~ ., data = data.1[-idx.test,], family = "binomial")
lr.res = predict(lr.1, data.1[idx.test,], type = "response")
lr.pred = ifelse(lr.res >= .2, "Yes", "No")

confusionMatrix(as.factor(lr.pred), as.factor(data.1$Purchase[idx.test]), positive = "Yes")

## Ridge Regression ##

x.train = model.matrix(Purchase ~ ., data = data.1[-idx.test,])[,-1]
y.train = data.1$Purchase[-idx.test]
x.test = model.matrix(Purchase ~ ., data = data.1[idx.test,])[,-1]
y.test = data.1$Purchase[idx.test]

grid = 10^seq(4, -10, length = 100)
lr.ridge = glmnet(x.train, y.train, alpha = 0, lambda = grid, family = "binomial")
plot(lr.ridge, xvar = "lambda")

set.seed(6174)
cv.ridge = cv.glmnet(x.train, y.train, alpha = 0, family = "binomial")
plot(cv.ridge)
lam.ridge = cv.ridge$lambda.min

ridge.prob = predict(lr.ridge, s = lam.ridge, newx = x.test, type = "response")
ridge.pred = ifelse(ridge.prob >= .2, "Yes", "No")

confusionMatrix(as.factor(ridge.pred), as.factor(data.1$Purchase[idx.test]), positive = "Yes")

as.vector(predict(lr.ridge, s = lam.ridge, type = "coefficients"))

## Lasso Regression ##

grid = 10^seq(4, -10, length = 100)
lr.lasso = glmnet(x.train, y.train, alpha = 1, lambda = grid, family = "binomial")
plot(lr.lasso, xvar = "lambda")

set.seed(6174)
cv.lasso = cv.glmnet(x.train, y.train, alpha = 1, family = "binomial")
plot(cv.lasso)
lam.lasso = cv.lasso$lambda.min

lasso.prob = predict(lr.lasso, s = lam.lasso, newx = x.test, type = "response")
lasso.pred = ifelse(lasso.prob >= .2, "Yes", "No")

confusionMatrix(as.factor(lasso.pred), as.factor(data.1$Purchase[idx.test]), positive = "Yes")

as.vector(predict(lr.lasso, s = lam.lasso, type = "coefficients"))

#### Question 2 ####

mali = read.csv("~/Classes/STAT 6340/mali.csv")
attach(mali)

plot(DistRD ~ Family, data = mali, main = "Nearest Road Distance vs. Cattle")
plot(Cattle ~ DistRD, data = mali, main = "Number of Individuals in Household vs Nearest Road Distance")
# Outliers: 25 (Family), 69 (DistRD), 72 (DistRD), 34 (Cattle)
mali = mali[-c(25,34,69,72),]

# Standardize!
apply(mali, 2, mean)
apply(mali, 2, sd)

mali.center = as.data.frame(scale(mali))
apply(mali.center, 2, mean)
apply(mali.center, 2, sd)

## Run the PCA
pca = prcomp(mali.center)

## Loadings and Eigenvalues
load = pca$rotation
eigen = pca$sdev**2

## Scree Plot
plot(eigen, pch = 19, xlab = "PC Index", ylab = "Explained Variance", main = "Elbow Plot - PCA")

## Calculating cumulative explained variance
cvar = cumsum(eigen)
cvar = cvar/cvar[length(cvar)]

## Displaying relevant things
eigen # Eigenvalues/Variances
cvar # Cumulative Variance
head(pca$x) # Scores
load # Loadings
biplot(pca, scale=0) # Biplot with scores and loadings
cor(pca$x[,1:2],mali.center) # correlation between first two PCs and standardized data

#### Question 3 ####

attach(Auto)
Auto = Auto[,1:8]
Auto$origin = as.factor(Auto$origin)

## Least-squares ##

auto.lm = glm(mpg ~ ., data = Auto)
auto.lm.pred = predict(auto.lm, Auto)

## Best-subset ##

auto.subs = summary(regsubsets(mpg ~ ., data = Auto))
which.max(auto.subs$adjr2)
which.min(auto.subs$cp)
which.min(auto.subs$bic)

auto.sub.lm = glm(mpg ~ ., data = Auto[c(1:5,7:8)]) # Ignoring acceleration
auto.sub.pred = predict(auto.sub.lm,Auto)

## Ridge Regression ##

x = model.matrix(mpg ~ ., data = Auto)[,-1]
y = Auto$mpg

grid = 10^seq(6, -6, length = 100)
auto.ridge = glmnet(x, y, alpha = 0, lambda = grid)
plot(auto.ridge, xvar = "lambda")

set.seed(6174)
auto.cv.ridge = cv.glmnet(x, y, alpha = 0, lambda = grid) # 10-fold cross validation
plot(auto.cv.ridge)
auto.lam.ridge = auto.cv.ridge$lambda.min

auto.ridge.opt = glmnet(x, y, alpha = 0, lambda = auto.lam.ridge)
auto.cv.ridge.opt = cv.glmnet(x, y, alpha = 0, lambda = c(auto.lam.ridge,1))
auto.ridge.pred = predict(auto.ridge, s = auto.lam.ridge, newx = x, type = "response")

## Lasso Regression ##

grid = 10^seq(6, -6, length = 100)
auto.lasso = glmnet(x, y, alpha = 1, lambda = grid)
plot(auto.lasso, xvar = "lambda")

set.seed(6174)
auto.cv.lasso = cv.glmnet(x, y, alpha = 1, lambda = grid) # 10-fold cross validation
plot(auto.cv.lasso)
auto.lam.lasso = auto.cv.lasso$lambda.min

auto.lasso.opt = glmnet(x, y, alpha = 1, lambda = auto.lam.lasso)
auto.cv.lasso.opt = cv.glmnet(x, y, alpha = 1, lambda = c(auto.lam.lasso,1))
auto.lasso.pred = predict(auto.lasso, s = auto.lam.lasso, newx = x, type = "response")

## PCR ##

set.seed(6174)
auto.pcr = pcr(mpg ~ ., data = Auto, scale = TRUE, validation = "CV", segments = 10)

MSEP(auto.pcr)
sqrt(MSEP(auto.pcr)$val[1, 1,])
which.min(MSEP(auto.pcr)$val[1, 1,])
validationplot(auto.pcr, val.type = "MSEP", main = "PCR MSEP")

auto.pcr.opt = pcr(mpg ~ ., data = Auto, scale = TRUE, ncomp = 8)
auto.pcr.pred = predict(auto.pcr, x, ncomp = 8)

## PLS ##

set.seed(6174)
auto.pls = plsr(mpg ~ ., data = Auto, scale = TRUE, validation = "CV", segments = 10)

MSEP(auto.pls)
sqrt(MSEP(auto.pls)$val[1, 1,])
which.min(MSEP(auto.pls)$val[1, 1,])
validationplot(auto.pls, val.type = "MSEP", main = "PLS MSEP")

auto.pls.opt = plsr(mpg ~ ., data = Auto, scale = TRUE, ncomp = 6)
auto.pls.pred = predict(auto.pls, x, ncomp = 6)

## Summaries ##

coeflist = list()
coeflist[[1]] = coef(auto.lm)
coeflist[[2]] = coef(auto.sub.lm)
coeflist[[3]] = coef(auto.ridge.opt)
coeflist[[4]] = coef(auto.lasso.opt)
coeflist[[5]] = coef(auto.pcr.opt, ncomp = 8, intercept = TRUE)
coeflist[[6]] = coef(auto.pls.opt, ncomp = 6, intercept = TRUE)

for (i in 2:7) {
  coeflist[[5]][i] = coeflist[[5]][i]/sd(Auto[,i])
  coeflist[[6]][i] = coeflist[[6]][i]/sd(Auto[,i])
}

num2 = (Auto$origin == 2)
num3 = (Auto$origin == 3)

coeflist[[5]][8] = coeflist[[5]][8]/sd(num2)
coeflist[[6]][8] = coeflist[[6]][8]/sd(num2)
coeflist[[5]][9] = coeflist[[5]][9]/sd(num3)
coeflist[[6]][9] = coeflist[[6]][9]/sd(num3)

set.seed(6174)
cv.glm(Auto, auto.lm, K = 10)$delta[1] # MSE for LinReg
set.seed(6174)
cv.glm(Auto[c(1:5,7:8)], auto.sub.lm, K = 10)$delta[1] # MSE for subset LinReg
set.seed(6174)
auto.cv.ridge.opt$cvm[2] # MSE for ridge
set.seed(6174)
auto.cv.lasso.opt$cvm[2] # MSE for lasso
set.seed(6174)
MSEP(auto.pcr)$val[1, 1,][which.min(MSEP(auto.pcr)$val[1, 1,])] # MSE for PCR
set.seed(6174)
MSEP(auto.pls)$val[1, 1,][which.min(MSEP(auto.pls)$val[1, 1,])] # MSE for PLS




