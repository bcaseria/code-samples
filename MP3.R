library("MASS")
library("caret")
library("e1071")
library("class")
library("stats")
library("pROC")
admission = read.csv("admission.csv")
admission = admission[,1:3]

#### Problem 1: Exploratory Analysis ####

# Scatterplot color coded by group
plot(GMAT ~ GPA, data = admission, col = colors()[admission$Group*30], pch = 19, main = "Groups w.r.t. GPA and GMAT")
legend("topleft", legend = c("Group 1","Group 2","Group 3"), col = colors()[30*(1:3)], pch = 19)

# Box plot for each group based on each explanatory variable
boxplot(GPA ~ Group, data = admission, xlab = "Group", ylab = "GPA", main = "Boxplots by Group: GPA")
boxplot(GMAT ~ Group, data = admission, xlab = "Group", ylab = "GMAT", main = "Boxplots by Group: GMAT")

#### LDA ####

attach(admission)
train = rep(TRUE,85)
train[c(27:31,55:59,81:85)] = FALSE
train.X = cbind(GPA,GMAT)[train,]
test.X = cbind(GPA,GMAT)[!train,]
train.Y = Group[train]
test.Y = Group[!train]

# Training the LDA algorithm
lda.fit <- lda(Group ~ GPA + GMAT, data = admission, subset = train)

# Drawing the contours
n.grid <- 50
x1.grid <- seq(f = min(test.X[, 1]), t = max(test.X[, 1]), l = n.grid)
x2.grid <- seq(f = min(test.X[, 2]), t = max(test.X[, 2]), l = n.grid)
grid <- expand.grid(x1.grid, x2.grid)
colnames(grid) <- colnames(test.X)

pred.grid <- predict(lda.fit, grid)
colnames(pred.grid$posterior) = c("Group 1","Group 2","Group 3")

prob1 <- matrix(pred.grid$posterior[, "Group 1"], nrow = n.grid, ncol = n.grid, byrow = F)
prob2 <- matrix(pred.grid$posterior[, "Group 2"], nrow = n.grid, ncol = n.grid, byrow = F)
prob3 <- matrix(pred.grid$posterior[, "Group 3"], nrow = n.grid, ncol = n.grid, byrow = F)

plot(admission[,1:2], col = ifelse(admission[,3] == 1, colors()[30], ifelse(admission[,3] == 2, colors()[60], colors()[90])),
     pch = 19, main = "Classification with LDA")

dif.total = matrix(2,nrow = 50,ncol = 50)
dif.useful = array(2, dim = c(50,50,2))
prob.total = array(cbind(prob1,prob2,prob3), dim = c(50,50,3))

for (i in 1:50) {for (j in 1:50) {
  dif.useful[i,j,] = prob.total[i,j,prob.total[i,j,] > min(prob.total[i,j,])]
  dif.total[i,j] = dif.useful[i,j,1] - dif.useful[i,j,2]
}}

contour(x1.grid, x2.grid, dif.total, levels = 0, labels = "", xlab = "", ylab = "", 
        main = "", add = T)

# Confusion Matricies and Misclassification
pred.train = predict(lda.fit, as.data.frame(train.X))
confusionMatrix(as.factor(pred.train$class),as.factor(admission$Group[train]))

pred.test = predict(lda.fit, as.data.frame(test.X))
confusionMatrix(as.factor(pred.test$class),as.factor(admission$Group[!train]))

#### QDA ####

# Training the QDA algorithm
qda.fit <- qda(Group ~ GPA + GMAT, data = admission, subset = train)

# Drawing the contours
n.grid <- 50
x1.grid <- seq(f = min(test.X[, 1]), t = max(test.X[, 1]), l = n.grid)
x2.grid <- seq(f = min(test.X[, 2]), t = max(test.X[, 2]), l = n.grid)
grid <- expand.grid(x1.grid, x2.grid)
colnames(grid) <- colnames(test.X)

pred.grid <- predict(qda.fit, grid)
colnames(pred.grid$posterior) = c("Group 1","Group 2","Group 3")

prob1 <- matrix(pred.grid$posterior[, "Group 1"], nrow = n.grid, ncol = n.grid, byrow = F)
prob2 <- matrix(pred.grid$posterior[, "Group 2"], nrow = n.grid, ncol = n.grid, byrow = F)
prob3 <- matrix(pred.grid$posterior[, "Group 3"], nrow = n.grid, ncol = n.grid, byrow = F)

plot(admission[,1:2], col = ifelse(admission[,3] == 1, colors()[30], ifelse(admission[,3] == 2, colors()[60], colors()[90])),
     pch = 19, main = "Classification with QDA")

dif.total = matrix(2,nrow = 50,ncol = 50)
dif.useful = array(2, dim = c(50,50,2))
prob.total = array(cbind(prob1,prob2,prob3), dim = c(50,50,3))

for (i in 1:50) {for (j in 1:50) {
  dif.useful[i,j,] = prob.total[i,j,prob.total[i,j,] > min(prob.total[i,j,])]
  dif.total[i,j] = dif.useful[i,j,1] - dif.useful[i,j,2]
}}

contour(x1.grid, x2.grid, dif.total, levels = 0, labels = "", xlab = "", ylab = "", 
        main = "", add = T)

# Confusion Matricies and Misclassification
pred.train = predict(qda.fit, as.data.frame(train.X))
confusionMatrix(as.factor(pred.train$class),as.factor(admission$Group[train]))

pred.test = predict(qda.fit, as.data.frame(test.X))
confusionMatrix(as.factor(pred.test$class),as.factor(admission$Group[!train]))

#### KNN ####

K = c(seq(1,70,by = 1))
errs = matrix(0,nrow = length(K),ncol = 2)

for (i in 1:length(K)) {
  set.seed(6174)
  res = knn(train.X,train.X,Group[train],K[i])
  errs[i,1] = 1 - sum(res == Group[train])/length(res)
  set.seed(6174)
  res = knn(train.X,test.X,Group[train],K[i])
  errs[i,2] = 1 - sum(res == Group[!train])/length(res)
}

xyplot(errs[,1] + errs[,2] ~ K, pch = 19, main = "Comparison of Errors w.r.t K values", ylab = "Error Rate",
       col = c("red","blue"), key = list(text = list(c("Training Error","Test Error")),
       space="right", points=list(col = c("red","blue"), pch=19)))
# Minimum Test Error: 20% @ K = 16 (Training error = 40%)

k.opt = 16
set.seed(6174)
res.opt.train = knn(train.X,train.X,Group[train], k = k.opt)
set.seed(6174)
res.opt.test = knn(train.X,test.X,Group[train], k = k.opt)

cm.train = matrix(0, nrow = 3, ncol = 3)
cm.test = matrix(0, nrow = 3, ncol = 3)

for (i in 1:70) {
  cm.train[Group[train][i],res.opt.train[i]] = cm.train[Group[train][i],res.opt.train[i]] + 1
}

for (i in 1:15) {
  cm.test[Group[!train][i],res.opt.test[i]] = cm.test[Group[!train][i],res.opt.test[i]] + 1
}

#### Problem 2: Exploratory Analysis ####

bankruptcy = read.csv("bankruptcy.csv")
bankruptcy = bankruptcy[,1:5]

attach(bankruptcy)

boxplot(X1 ~ Group, data = bankruptcy, xlab = "Group", ylab = "X1", main = "Boxplots by Group: X1") # Useful
boxplot(X2 ~ Group, data = bankruptcy, xlab = "Group", ylab = "X2", main = "Boxplots by Group: X2") # Useful
boxplot(X3 ~ Group, data = bankruptcy, xlab = "Group", ylab = "X3", main = "Boxplots by Group: X3") # Useful
boxplot(X4 ~ Group, data = bankruptcy, xlab = "Group", ylab = "X4", main = "Boxplots by Group: X4") # Not that useful

#### Logistic Regression ####

logreg1 = glm(Group ~ X1 + X3, family = binomial, data = bankruptcy) # Model with lowest AIC
summary(logreg1)

#### Problem 3 ####

pred.mod1 = predict(logreg1,bankruptcy[c("X1","X3")])

t = (Group == 0)
p = (pred.mod1 < 0)
confusionMatrix(as.factor(p),as.factor(t), positive = "TRUE")

roc1 = roc(Group, pred.mod1, levels = c("0","1"))
plot(roc1, legacy.axes = TRUE, main = "ROC: Reduced Logistic")

# Logistic Model with all predictors

logreg2 = glm(Group ~ X1 + X2 + X3 + X4, family = binomial, data = bankruptcy)
summary(logreg2)

pred.mod2 = predict(logreg2, bankruptcy[c("X1","X2","X3","X4")])

t = (Group == 0)
p = (pred.mod2 < 0)
confusionMatrix(as.factor(p),as.factor(t), positive = "TRUE")

roc2 = roc(Group, pred.mod2, levels = c("0","1"))
plot(roc2, legacy.axes = TRUE, main = "ROC: Full Logistic")

# LDA

lda3 = lda(Group ~ X1 + X2 + X3 + X4, data = bankruptcy)

pred.mod3 = predict(lda3, bankruptcy[c("X1","X2","X3","X4")])

t = (Group == 0)
p = (pred.mod3$posterior[,1] >= 0.5)
confusionMatrix(as.factor(p),as.factor(t), positive = "TRUE")

roc3 = roc(Group, pred.mod3$posterior[,1], levels = c("0","1"))
plot(roc3, legacy.axes = TRUE, main = "ROC: LDA")

# QDA

qda4 = qda(Group ~ X1 + X2 + X3 + X4, data = bankruptcy)

pred.mod4 = predict(qda4, bankruptcy[c("X1","X2","X3","X4")])

t = (Group == 0)
p = (pred.mod4$posterior[,1] >= 0.5)
confusionMatrix(as.factor(p),as.factor(t), positive = "TRUE")

roc4 = roc(Group, pred.mod4$posterior[,1], levels = c("0","1"))
plot(roc4, legacy.axes = TRUE, main = "ROC: QDA")

# All ROC curves

plot(roc1, legacy.axes = TRUE, main = "All ROC Curves")
plot(roc2, add = TRUE, col = "blue")
plot(roc3, add = TRUE, col = "red")
plot(roc4, add = TRUE, col = "green")
legend("bottomright", legend = c("Reduced Logistic","Full Logistic","LDA","QDA"), 
       col = c("black","blue","red","green"), pch = 19)

