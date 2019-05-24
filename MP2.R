## prostate_cancer variable imported with Import Wizard ##

prostate_cancer$vesinv = as.factor(prostate_cancer$vesinv)

fit = lm(psa ~ cancervol + weight + age + benpros + vesinv + capspen + gleason, data = prostate_cancer)

#### Exploratory Analysis ####

plot(fit) # Gets QQ plot and Residual vs. Fitted plot
plot(resid(fit), type="l", ylab = "Residuals", main = "Time Series Plot") #Time Series plot
abline(h=0)

#### Transformation of Response ####

prostate_cancer$psalog = log(prostate_cancer$psa)

fitlog = lm(psalog ~ cancervol + weight + age + benpros + vesinv + capspen + gleason, data = prostate_cancer)

plot(fitlog)
plot(resid(fitlog), type="l", ylab = "Residuals", main = "Time Series Plot with Log Transform") #Time Series plot
abline(h=0)

#### LMs of psalog with each predictor individually ####

list_lm = list()

for (i in 1:7) {
  list_lm[[i]] = lm(prostate_cancer$psalog ~ prostate_cancer[,(i+2)])
}

for (i in 1:7){
  x = prostate_cancer[,(i+2)]
  plot(psalog ~ x, data = prostate_cancer, xlab = colnames(prostate_cancer)[(i+2)])
  abline(list_lm[[i]])
  grDevices::devAskNewPage(ask = TRUE)
}

for (i in 1:7){
  summary(list_lm[[i]])
}

#### Code utilizing full model ####

summary(fitlog)

## Reduction of variables ##

GoodData = prostate_cancer[,c(3,7,8,9,10)]
GoodFit = lm(psalog ~ cancervol + vesinv + capspen + gleason, data = GoodData)
plot(GoodFit) # Gets QQ plot and Residual vs. Fitted plot
plot(resid(GoodFit), type="l", ylab = "Residuals", main = "Time Series Plot") #Time Series plot
abline(h=0)

## Prediction using means/mode ##

GoodData$vesinv = as.numeric(GoodData$vesinv) - 1
new = apply(GoodData[,1:4],2,mean)
new[2] = as.factor(round(new[2])) # mode of binary string = rounded mean

sum(GoodFit$coefficients*c(1,new)) # 3.032413 -> e^3.032413 = 20.74724
