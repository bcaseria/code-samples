## Initialization of variables (K, training and testing explanatory variables, and K-wise errors) ##

K = c(seq(1,30,by = 1),seq(35,100,by = 5))
train.x = train[,1:2]
test.x = test[,1:2]
errs = matrix(0,nrow = length(K),ncol = 2)

#### Gathering training and testing trrors for each K value ####

for (i in 1:length(K)) {
  set.seed(6174)
  res = knn(train.x,train.x,train$y,K[i])
  errs[i,1] = 1 - sum(res == train$y)/length(res)
  set.seed(6174)
  res = knn(train.x,test.x,test$y,K[i])
  errs[i,2] = 1 - sum(res == test$y)/length(res)
}

## Plotting errors with respect to K values ##

xyplot(errs[,1] + errs[,2] ~ K, pch = 19, main = "Comparison of Errors w.r.t K values", ylab = "Error Rate",
       col = c("red","blue"), key = list(text = list(c("Training Error","Test Error")),
       space="right", points=list(col = c("red","blue"), pch=19)))

## Obtaining optimal K ##

opt.ind = which(errs[,2] == min(errs[,2])) # yields opt.ind = 40 -> K = 80
errs[opt.ind,] # Training error = 0.1600, Testing error = 0.1665

#### Plotting training data with decision boundary ####

## Acquiring grid for boundary plot ##

n.grid = 50
x1.grid = seq(f = min(train[,1]), t = max(train[,1]), l = n.grid)
x2.grid = seq(f = min(train[,2]), t = max(train[,2]), l = n.grid)
grid = expand.grid(x1.grid, x2.grid)

## Obtaining probability estimates for each grid point ##

opt.K = K[opt.ind]
set.seed(6174)
opt.res = knn(train.x, grid, train$y, k = opt.K, prob = T)
prob = attr(opt.res, "prob")
prob = ifelse(opt.res == 'yes', prob, 1-prob)
prob = matrix(prob, n.grid, n.grid)

## Plotting 50/50 contour on top of training data ##

plot(train.x, col = ifelse(train$y == 'yes','red','blue'), main = "Training Data with Decision Boundary (yes = red)")
contour(x=x1.grid,y= x2.grid, prob, levels = .5, labels = "", xlab = "", ylab = "", main = "", add = T)
