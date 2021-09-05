source("image_compression_lib.R");

detach("package:caret", unload = TRUE)

ip = data.frame(installed.packages())$Package
#install.packages("FactoMineR")
library(FactoMineR)
#install.packages("factoextra")
library(factoextra)
#install.packages("ggplot2")
library(ggplot2)
#install.packages("rpart")
library(rpart)
#install.packages("rattle")
library(rattle)
#install.packages("rpart.plot")
library(rpart.plot)
#install.packages("RColorBrewer")
library(RColorBrewer)
#install.packages("e1071")
library(e1071)
#install.packages("rgl")
library(rgl)
#install.packages("C50")
library(C50)
#install.packages(”caret")
library(caret)
#install.packages("ROCR")
library(ROCR)
if(! "pROC" %in% ip){
  install.packages("pROC")
}
library(pROC)

remove.packages("pROC")

#install.packages("neuralnet")
library(neuralnet)

#install.packages("RSNNS")
library(RSNNS)

# Domande
# 1. La PCA va fatta solo sul training set ??
# 2. Per confrontare i modelli va bene anche se hanno due librerie diverse ?

#---------------------------- import and preprocess ----------------------------
image_compression=load("image_compression_labeled_total2.csv");
dim(image_compression)
image_compression= unique(image_compression);
dim(image_compression)
image_compression=preprocess(image_compression);
dim(image_compression)
#---------------------------- visualization-------------------------------------
sapply(image_compression,class)
levels(image_compression$Quality)
summary(image_compression)
colSums(is.na(image_compression))

image_compression.features=get_features(image_compression)
image_compression.target=get_target(image_compression)

featurePlot(image_compression.features, image_compression.target,
            plot="density",
            scales=list(x=list(relation="free"),
                        y=list(relation="free")), auto.key=list(columns=3))
pie(table(image_compression.target))

par(mfrow=c(1,3))
for(i in 1:3) {
  boxplot(image_compression.features[,i],
          main=names(image_compression.features)[i]) }
for(i in 4:6) {
  boxplot(image_compression.features[,i],
          main=names(image_compression.features)[i]) }
boxplot(image_compression.features[,7],
        main=names(image_compression.features)[7])

par(mfrow=c(1,1))
boxplot(Threshold ~ Quality, data = image_compression)

featurePlot(x=image_compression.features, y=image_compression.target,
            plot="pairs", auto.key=list(columns=3))
#---------------------------- PCA ----------------------------------------------
image_compression.pca=PCA(image_compression.features, ncp=3, scale.unit = TRUE)

image_compression.pca.eig = get_eigenvalue(image_compression.pca)
image_compression.pca.ind = get_pca_ind(image_compression.pca)
image_compression.pca.var = get_pca_var(image_compression.pca)

fviz_eig(image_compression.pca, addlabels = TRUE, ylim = c(0, 50))
image_compression.pca.var$coord
fviz_pca_var(image_compression.pca, col.var = "black")

fviz_pca_biplot(image_compression.pca,
                select.var=list(contrib=6),ggtheme=theme_minimal())
fviz_pca_ind(image_compression.pca, 
             col.ind = "cos2", gradient.cols=c("#00AFBB","#E7B800","#FC4E07"),
             repel=TRUE)

image_compression.reduced = data.frame(image_compression.pca$ind$coord)
image_compression.reduced$target = image_compression.target
levels(image_compression.reduced$target) = c("high_quality","low_quality")

# -------------------------- trainset/testset ----------------------------------
allset = split.data(image_compression.reduced, p = 0.7)
trainset = allset$train
testset = allset$test

#---------------------------- decision tree ------------------------------------
decisionTree = rpart(target ~ ., data=trainset, method="class")
fancyRpartPlot(decisionTree)
testset$Prediction <- predict(decisionTree, testset, type = "class")
confusion.matrix = table(testset$target, testset$Prediction)
sum(diag(confusion.matrix))/sum(confusion.matrix) 
plotcp(decisionTree)

cp.decided = 0.023
prunedDecisionTree = prune(decisionTree, cp=cp.decided)
plotcp(prunedDecisionTree)
fancyRpartPlot(prunedDecisionTree)
prunedDecisionTree.pred <- predict(prunedDecisionTree, testset, type = "class")
confusion.matrix = table(prunedDecisionTree.pred, testset$target)

sum(diag(confusion.matrix))/sum(confusion.matrix)
precision(confusion.matrix, relevant=levels(testset$target)[1])
precision(confusion.matrix, relevant=levels(testset$target)[2])
recall(confusion.matrix, relevant=levels(testset$target)[1])
recall(confusion.matrix, relevant=levels(testset$target)[2])
F_meas(confusion.matrix, relevant=levels(testset$target)[1])
F_meas(confusion.matrix, relevant=levels(testset$target)[2])

#--------------------------- SVM -----------------------------------------------
control = trainControl(method = "repeatedcv", number = 10,repeats = 3,
                       classProbs = TRUE, summaryFunction = twoClassSummary)
svm.model =  train(target ~ ., data=trainset, method = "svmLinear",
                   metric = "ROC",
                   tuneGrid = expand.grid(C = seq(0, 2, length = 20)),
                   trControl = control)
plot(svm.model)
print(svm.model)
print(svm.model$finalModel@param)

coefs <- svm.model$finalModel@coef[[1]]
mat <- svm.model$finalModel@xmatrix[[1]]
w <- coefs %*% mat
detalization <- 100                                                                                                                                                                 
grid <- expand.grid(seq(from=min(trainset$Dim.1),to=max(trainset$Dim.1),
                        length.out=detalization),                                                                                                         
                    seq(from=min(trainset$Dim.2),to=max(trainset$Dim.2),
                        length.out=detalization))                                                                                                         
z <- (svm.model$finalModel@b - w[1,1]*grid[,1] - w[1,2]*grid[,2]) / w[1,3]
plot3d(grid[,1],grid[,2],z)  # this will draw plane.
points3d(trainset$Dim.1[which(trainset$target=='low_quality')],
         trainset$Dim.2[which(trainset$target=='low_quality')],
         trainset$Dim.3[which(trainset$target=='low_quality')], col='red')
points3d(trainset$Dim.1[which(trainset$target=='high_quality')],
         trainset$Dim.2[which(trainset$target=='high_quality')],
         trainset$Dim.3[which(trainset$target=='high_quality')], col='blue')

svm.pred = predict(svm.model, testset)
confusion.matrix = table(svm.pred, testset$target)
sum(diag(confusion.matrix))/sum(confusion.matrix)
precision(confusion.matrix, relevant=levels(testset$target)[1])
precision(confusion.matrix, relevant=levels(testset$target)[2])
recall(confusion.matrix, relevant=levels(testset$target)[1])
recall(confusion.matrix, relevant=levels(testset$target)[2])
F_meas(confusion.matrix, relevant=levels(testset$target)[1])
F_meas(confusion.matrix, relevant=levels(testset$target)[2])

#--------------------------- Nnet ----------------------------------------------
#ind = sample(2, nrow(image_compression.reduced), replace = TRUE, prob=c(0.7, 0.3))
#trainset = image_compression.reduced[ind == 1,]
#testset = image_compression.reduced[ind == 2,]
#trainset$low_quality = trainset$target == "low_quality"
#trainset$high_quality = trainset$target == "high_quality"
#network = neuralnet(low_quality + high_quality~ Dim.1 + Dim.2 + Dim.3, trainset, hidden=5)
#plot(network)
#net.predict = compute(network, testset[c("Dim.1","Dim.2","Dim.3")])$net.result
#net.prediction = c("low_quality", "high_quality")[apply(net.predict, 1, which.max)]
#predict.table = table(testset$target, net.prediction)

control = trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = twoClassSummary)
nnet.model = train(target ~ ., data=trainset, method = "nnet", metric = "ROC", trControl = control)
nnet.model$confusion.matrix = confusionMatrix(nnet.model, norm = "none")$table
nnet.model$confusion.matrix

#-------------------------- measure 10-cross fold validation  ------------------

#pred.rocr = prediction(c(svm.pred), testset$target) 
#perf.rocr = performance(pred.rocr, measure = "auc", x.measure = "cutoff")
#perf.tpr.rocr = performance(pred.rocr, "tpr","fpr")
#opt.cut = function(perf, pred){
#  cut.ind = mapply(FUN=function(x, y, p){
#    d = (x - 0)^2 + (y-1)^2
#    ind = which(d == min(d))
#    c(sensitivity = y[[ind]], specificity = 1-x[[ind]],
#      cutoff = p[[ind]])
#  }, perf@x.values, perf@y.values, pred@cutoffs)
#}
#print(opt.cut(perf.tpr.rocr, pred.rocr))
#acc.perf = performance(pred.rocr, measure = "acc")
#plot(acc.perf)
#ind = which.max( slot(acc.perf, "y.values")[[1]] )
#acc = slot(acc.perf, "y.values")[[1]][ind]
#cutoff = slot(acc.perf, "x.values")[[1]][ind]
#print(c(accuracy= acc, cutoff = cutoff))

control = trainControl(method = "cv", number = 10,
                       classProbs = TRUE, summaryFunction = twoClassSummary)
rpart.model = train(target ~ ., data=trainset, method = "rpart", metric = "ROC",
                    trControl = control)
fancyRpartPlot(rpart.model$finalModel)
rpart.model$confusion.matrix = confusionMatrix(rpart.model, norm = "none")$table

svm.model =  train(target ~ ., data=trainset, method = "svmLinear",
                   metric = "ROC", trControl = control)
svm.model$confusion.matrix = confusionMatrix(svm.model, norm = "none")$table

svm.probs = predict(svm.model, testset[,! names(testset) %in% c("target")],
                    type = "prob")
rpart.probs = predict(rpart.model, testset[,! names(testset) %in% c("target")],
                      type = "prob")
nnet.probs = predict(nnet.model, testset[,! names(testset) %in% c("target")],
                     type = "prob")

svm.ROC = roc(response = testset$target, predictor = svm.probs$high_quality,
              levels = levels(testset[,c("target")]))
plot(svm.ROC,type="S", col="green")

rpart.ROC = roc(response = testset[,c("target")],
                predictor =rpart.probs$high_quality,
                levels = levels(testset[,c("target")]))
plot(rpart.ROC, add=TRUE, col="blue")

nnet.ROC = roc(response = testset$target, predictor = nnet.probs$high_quality,
              levels = levels(testset[,c("target")]))
plot(nnet.ROC, add=TRUE, col="red")

svm.ROC
rpart.ROC
nnet.ROC
cv.values = resamples(list(svm=svm.model, rpart = rpart.model, nnet = nnet.model))
summary(cv.values)
dotplot(cv.values, metric = "ROC") 
bwplot(cv.values, layout = c(3, 1))
splom(cv.values,metric="ROC")
cv.values$timings # get the train times for both models

rpart.model$confusion.matrix
plt <- as.data.frame(rpart.model$confusion.matrix)
plt$Prediction <- factor(plt$Prediction, levels=rev(levels(plt$Prediction)))
ggplot(plt, aes(Reference,Prediction, fill= Freq)) +
  geom_tile() + geom_text(aes(label=Freq)) +
  scale_fill_gradient(low="white", high="#009194") +
  labs(x = "Reference",y = "Prediction") +
  scale_x_discrete(labels=c("High quality","Low Quality")) +
  scale_y_discrete(labels=c("Low Quality","High quality"))

sum(diag(rpart.model$confusion.matrix))/sum(rpart.model$confusion.matrix)
precision(rpart.model$confusion.matrix, relevant=levels(trainset$target)[1])
precision(rpart.model$confusion.matrix, relevant=levels(trainset$target)[2])
recall(rpart.model$confusion.matrix, relevant=levels(trainset$target)[1])
recall(rpart.model$confusion.matrix, relevant=levels(trainset$target)[2])
F_meas(rpart.model$confusion.matrix, relevant=levels(trainset$target)[1])
F_meas(rpart.model$confusion.matrix, relevant=levels(trainset$target)[2])

svm.model$confusion.matrix
plt <- as.data.frame(svm.model$confusion.matrix)
plt$Prediction <- factor(plt$Prediction, levels=rev(levels(plt$Prediction)))
ggplot(plt, aes(Reference,Prediction, fill= Freq)) +
  geom_tile() + geom_text(aes(label=Freq)) +
  scale_fill_gradient(low="white", high="#009194") +
  labs(x = "Reference",y = "Prediction") +
  scale_x_discrete(labels=c("High quality","Low Quality")) +
  scale_y_discrete(labels=c("Low Quality","High quality"))

sum(diag(svm.model$confusion.matrix))/sum(svm.model$confusion.matrix)
precision(svm.model$confusion.matrix, relevant=levels(trainset$target)[1])
precision(svm.model$confusion.matrix, relevant=levels(trainset$target)[2])
recall(svm.model$confusion.matrix, relevant=levels(trainset$target)[1])
recall(svm.model$confusion.matrix, relevant=levels(trainset$target)[2])
F_meas(svm.model$confusion.matrix, relevant=levels(trainset$target)[1])
F_meas(svm.model$confusion.matrix, relevant=levels(trainset$target)[2])

nnet.model$confusion.matrix
plt <- as.data.frame(nnet.model$confusion.matrix)
plt$Prediction <- factor(plt$Prediction, levels=rev(levels(plt$Prediction)))
ggplot(plt, aes(Reference,Prediction, fill= Freq)) +
  geom_tile() + geom_text(aes(label=Freq)) +
  scale_fill_gradient(low="white", high="#009194") +
  labs(x = "Reference",y = "Prediction") +
  scale_x_discrete(labels=c("High quality","Low Quality")) +
  scale_y_discrete(labels=c("Low Quality","High quality"))

sum(diag(nnet.model$confusion.matrix))/sum(nnet.model$confusion.matrix)
precision(nnet.model$confusion.matrix, relevant=levels(trainset$target)[1])
precision(nnet.model$confusion.matrix, relevant=levels(trainset$target)[2])
recall(nnet.model$confusion.matrix, relevant=levels(trainset$target)[1])
recall(nnet.model$confusion.matrix, relevant=levels(trainset$target)[2])
F_meas(nnet.model$confusion.matrix, relevant=levels(trainset$target)[1])
F_meas(nnet.model$confusion.matrix, relevant=levels(trainset$target)[2])

