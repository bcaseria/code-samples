library(e1071)
library(ISLR)
library(readr)
library(splines)

#### Problem 1 ####

Caravan.train = Caravan[1001:5822,]
Caravan.test = Caravan[1:1000,]

## SVM
set.seed(6174)
tune.linear = tune(svm, Purchase ~ ., data = Caravan.train, kernel = "linear", scale = TRUE,
                ranges = list(cost = c(0.001, 0.01, 0.1, 1, 10, 100)))

summary(tune.linear)
model.linear = tune.linear$best.model
linear.pred = predict(model.linear, Caravan.test)
res1a = table(predict = linear.pred, truth = Caravan.test$Purchase)
res1a.out = capture.output(res1a)
cat("Caravan SVM - Linear", res1a.out, file = "1a.txt", sep = "\n", append = TRUE)

## SVM with quadratic kernel
set.seed(6174)
tune.poly = tune(svm, Purchase ~ ., data = Caravan.train, kernel = "polynomial", scale = TRUE, degree = 2,
                   ranges = list(cost = c(0.001, 0.01, 0.1, 1, 10, 100)))

summary(tune.poly)
model.poly = tune.poly$best.model
poly.pred = predict(model.poly, Caravan.test)
res1b = table(predict = poly.pred, truth = Caravan.test$Purchase)
res1b.out = capture.output(res1b)
cat("Caravan SVM - Polynomial", res1b.out, file = "1b.txt", sep = "\n", append = TRUE)

## SVM with radial kernel
set.seed(6174)
tune.radial = tune(svm, Purchase ~ ., data = Caravan.train, kernel = "radial", scale = TRUE,
                 ranges = list(cost = c(0.1, 1, 10, 100, 1000), gamma = c(0.5, 1, 2, 3, 4)))

summary(tune.radial)
model.radial = tune.radial$best.model
radial.pred = predict(model.radial, Caravan.test)
res1c = table(predict = radial.pred, truth = Caravan.test$Purchase)
res1c.out = capture.output(res1c)
cat("Caravan SVM - Radial", res1c.out, file = "1c.txt", sep = "\n", append = TRUE)

#### Problem 2 ####

admission = read_csv("Classes/STAT 6340/admission.csv")
admission$Group = as.factor(admission$Group)
idx.test = c(27:31,55:59,81:85)
admission.train = admission[-idx.test,]
admission.test = admission[idx.test,]

## SVM
set.seed(6174)
tune.linear2 = tune(svm, Group ~ ., data = admission.train, kernel = "linear", scale = TRUE,
                   ranges = list(cost = c(0.01, 0.05, 0.1, 0.5, 1, 5, 10, 100)))

summary(tune.linear2)
model.linear2 = tune.linear2$best.model
linear2.pred = predict(model.linear2, admission.test)
res2a = table(predict = linear2.pred, truth = admission.test$Group)
res2a.out = capture.output(res2a)
cat("Admission SVM - Linear", res2a.out, file = "2a.txt", sep = "\n", append = TRUE)

## SVM with quadratic kernel
set.seed(6174)
tune.poly2 = tune(svm, Group ~ ., data = admission.train, kernel = "polynomial", scale = TRUE, degree = 2,
                 ranges = list(cost = c(1, 5, 10, 50, 100, 500, 1000)))

summary(tune.poly2)
model.poly2 = tune.poly2$best.model
poly2.pred = predict(model.poly2, admission.test)
res2b = table(predict = poly2.pred, truth = admission.test$Group)
res2b.out = capture.output(res2b)
cat("Admission SVM - Polynomial", res2b.out, file = "2b.txt", sep = "\n", append = TRUE)

## SVM with radial kernel
set.seed(6174)
tune.radial2 = tune(svm, Group ~ ., data = admission.train, kernel = "radial", scale = TRUE,
                   ranges = list(cost = c(0.1, 0.5, 1, 10, 100, 1000), gamma = c(0.1, 0.5, 1, 2, 3, 4)))

summary(tune.radial2)
model.radial2 = tune.radial2$best.model
radial2.pred = predict(model.radial2, admission.test)
res2c = table(predict = radial2.pred, truth = admission.test$Group)
res2c.out = capture.output(res2c)
cat("Admission SVM - Radial", res2c.out, file = "2c.txt", sep = "\n", append = TRUE)

## Best model: Radial -> plot with boundaries
png("2cPlot.png", width = 1000, height = 1000)
plot(model.radial2,admission.test, ylim = c(min(admission.test$GPA) - .02,max(admission.test$GPA + .02)),
     xlim = c(min(admission.test$GMAT)-5,max(admission.test$GMAT)+5))
dev.off()

#### Problem 3 ####

lidar = read_csv("Classes/STAT 6340/lidar.csv")

## Variable scatterplot
png("3a.png", width = 1000, height = 1000)
plot(logratio ~ range, data = lidar, pch = 19, main = "Relationship between Range and LogRatio")
dev.off()

## Quartic fit
quartic = lm(logratio ~ poly(range, 4), data = lidar)
domain <- range(lidar$range)
range.grid <- seq(from = domain[1], to = domain[2])
pred.grid <- predict(quartic, newdata = list(range = range.grid), se = TRUE)

png("3b.png", width = 1000, height = 1000)
plot(logratio ~ range, xlim = domain, cex = 0.5, col = "black", pch = 19, data = lidar, main = "Quartic Fit")
lines(range.grid, pred.grid$fit, lwd = 2, col = "blue")
dev.off()

## Cubic Regression Spline
crs = lm(logratio ~ bs(range, df = 4), data = lidar)
attr(bs(lidar$range, df = 4), "knots") # One knot @ Median
pred.grid.crs = predict(crs, newdata = list(range = range.grid), se = TRUE)

png("3c.png", width = 1000, height = 1000)
plot(logratio ~ range, xlim = domain, cex = 0.5, col = "black", pch = 19, data = lidar, main = "Cubic Regression Spline")
lines(range.grid, pred.grid.crs$fit, lwd = 2, col = "blue")
dev.off()

## Cubic Natural Regression Spline
cnrs = lm(logratio ~ ns(range, df = 4), data = lidar)
attr(ns(lidar$range, df = 4), "knots") # Three knots @ Q1, Median, Q3
pred.grid.cnrs = predict(cnrs, newdata = list(range = range.grid), se = TRUE)

png("3d.png", width = 1000, height = 1000)
plot(logratio ~ range, xlim = domain, cex = 0.5, col = "black", pch = 19, data = lidar, main = "Cubic Natural Regression Spline")
lines(range.grid, pred.grid.cnrs$fit, lwd = 2, col = "blue")
dev.off()

## Smoothing Spline
ss = smooth.spline(lidar$range, lidar$logratio, df = 4)
ss$fit$knot

png("3e.png", width = 1000, height = 1000)
plot(logratio ~ range, xlim = domain, cex = 0.5, col = "black", pch = 19, data = lidar, main = "Smoothing Spline")
lines(ss, col = "blue", lwd = 2)
dev.off()

## Combined
png("3f1.png", width = 1000, height = 1000)
plot(logratio ~ range, xlim = domain, cex = 0.5, col = "black", pch = 19, data = lidar, main = "All Fits")
lines(range.grid, pred.grid$fit, lwd = 2, col = "blue")
lines(range.grid, pred.grid.crs$fit, lwd = 2, col = "red")
lines(range.grid, pred.grid.cnrs$fit, lwd = 2, col = "green") # Best
lines(ss, col = "black", lwd = 2)
legend("bottomleft", legend=c("Quartic","Cubic Regression Spline","Cubic Natural Regression Spline","Smoothing Spline"),
       col=c("blue","red","green","black"), lty = 1, cex=.8)
dev.off()

## CNRS with 2 SE bands
png("3f2.png", width = 1000, height = 1000)
plot(logratio ~ range, xlim = domain, cex = 0.5, col = "black", pch = 19, data = lidar, main = "Cubic Natural Regression Spline")
lines(range.grid, pred.grid.cnrs$fit, lwd = 2, col = "blue")
lines(range.grid, pred.grid.cnrs$fit + 2 * pred.grid.cnrs$se, col = "red", lty = "dashed")
lines(range.grid, pred.grid.cnrs$fit - 2 * pred.grid.cnrs$se, col = "red", lty = "dashed")
dev.off()





