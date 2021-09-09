#---------------------------- libraries -------------------------------------------------
source("image_compression_lib.R");

installed_packages = data.frame(installed.packages())$Package
needed_packages = c("testit", "FactoMineR", "factoextra", "ggplot2",
                    "rpart", "rattle", "rpart.plot", "RColorBrewer", 
                    "e1071", "rgl", "C50", "caret", "ROCR", "pROC")
for(p in needed_packages){
  if(! p %in% installed_packages){
    install.packages(p)
  }
  library(p, character.only = TRUE)
}

#---------------------------- import and preprocess ----------------------------
image_compression = load("image_compression_labeled_total2.csv");
image_compression = unique(image_compression);
image_compression = preprocess(image_compression);

image_compression.features = get_features(image_compression)
image_compression.target = get_target(image_compression)
image_compression = data.frame(image_compression.features,
                               image_compression.target)
names(image_compression)[names(image_compression)
                         == 'image_compression.target'] <- 'target'

#---------------------------- visualization-------------------------------------
dim(image_compression.features)
summary(image_compression)
sapply(image_compression.features,class)
levels(image_compression.target)
colSums(is.na(image_compression))

featurePlot(image_compression.features, image_compression.target,
            plot="density",
            scales=list(x=list(relation="free"),
                        y=list(relation="free")), auto.key=list(columns=3))

x = data.frame(table(image_compression.target))
pie(x$Freq, labels = x$Freq, main = "City pie chart",col = rainbow(length(x)))
legend("topright", c("high_quality", "low_quality"), cex = 0.8,
       fill = rainbow(length(x)))

par(mfrow=c(1,3))
for(i in 1:3) {
  boxplot(image_compression.features[,i],
          main=names(image_compression.features)[i]) }
for(i in 4:6) {
  boxplot(image_compression.features[,i],
          main=names(image_compression.features)[i]) }

par(mfrow=c(1,1))
boxplot(Threshold ~ target, data = image_compression)
boxplot(Mean ~ target, data = image_compression)
boxplot(Median ~ target, data = image_compression)
boxplot(Rms.contrast ~ target, data = image_compression)
boxplot(M.contrast ~ target, data = image_compression)
boxplot(Entropy ~ target, data = image_compression)

featurePlot(x=image_compression.features, y=image_compression.target,
            plot="pairs", auto.key=list(columns=3))
#---------------------------- PCA ----------------------------------------------
image_compression.pca=PCA(image_compression.features, ncp=3, scale.unit = TRUE)

image_compression.pca.eig = get_eigenvalue(image_compression.pca)
image_compression.pca.ind = get_pca_ind(image_compression.pca)
image_compression.pca.var = get_pca_var(image_compression.pca)

get_eigenvalue(image_compression.pca)
fviz_eig(image_compression.pca, addlabels = TRUE, ylim = c(0, 50))
image_compression.pca.var$coord

fviz_pca_var(image_compression.pca, col.var = "black")

#fviz_pca_biplot(image_compression.pca,
#                select.var=list(contrib=6),ggtheme=theme_minimal())
#fviz_pca_ind(image_compression.pca, 
#            col.ind = "cos2", gradient.cols=c("#00AFBB","#E7B800","#FC4E07"),
#             repel=TRUE)

image_compression.pca$var$contrib

image_compression.reduced = data.frame(image_compression.pca$ind$coord)
image_compression.reduced$target = image_compression.target

# --------------------------- utils ----------------------------------
target.levels = levels(image_compression.reduced$target)

n_high_quality = as.numeric(table(image_compression.reduced$target)[1])
n_low_quality = as.numeric(table(image_compression.reduced$target)[2])
n_individuals = n_high_quality + n_low_quality

reps = 10

control = trainControl(method = "repeatedcv", number = 10,  repeats = reps,
                       classProbs = TRUE, summaryFunction = twoClassSummary)

#---------------------------- decision tree ------------------------------------
#decisionTree = rpart(target ~ ., data=image_compression.reduced, method="class")
#plotcp(decisionTree)
#cp.decided = 0.01494565
#prunedDecisionTree = prune(decisionTree, cp=cp.decided)
#plotcp(prunedDecisionTree)
#fancyRpartPlot(prunedDecisionTree)

rpart.model = train(target ~ ., data=image_compression.reduced, method = "rpart", metric = "ROC",
                    trControl = control)
fancyRpartPlot(rpart.model$finalModel)

rpart.model$confusion.matrix = confusionMatrix(rpart.model, norm = "none")$table
plot_confusion_matrix(rpart.model$confusion.matrix)

best_tune = as.numeric(rpart.model$bestTune)
folds = rpart.model$resampledCM[rpart.model$resampledCM$cp == best_tune, ]

repeats_confusion_matrixes = get_confusion_matrixes_stratified_10_fold(folds, reps)

TP_sd = sd(repeats_confusion_matrixes$cell1)
FP_sd = sd(repeats_confusion_matrixes$cell2)
FN_sd = sd(repeats_confusion_matrixes$cell3)
TN_sd = sd(repeats_confusion_matrixes$cell4)

rpart.model$confusion.matrix = round(rpart.model$confusion.matrix/reps)

assert("La somma degli elementi della matrice non è uguale al numero
       di individui", {
         dim(image_compression.reduced)[1] == sum(rpart.model$confusion.matrix)
       })

accuracy_mean = sum(diag(rpart.model$confusion.matrix))/sum(rpart.model$confusion.matrix)
accuracy_sd = get_accuracy_sd(repeats_confusion_matrixes, accuracy_mean, reps)

# Calculate the mean and standard error
l.model <- lm(cell1 ~ 1, repeats_confusion_matrixes)

# Calculate the confidence interval
confint(l.model, level=0.95)

p1 = precision(rpart.model$confusion.matrix, relevant=target.levels[1])
p2 = precision(rpart.model$confusion.matrix, relevant=target.levels[2])
precision_macro_avg = (p1 + p2) / 2
precision_micro_avg = (p1 * n_high_quality/n_individuals) + (p2 * n_low_quality/n_individuals)

r1 = recall(rpart.model$confusion.matrix, relevant=target.levels[1])
r2 = recall(rpart.model$confusion.matrix, relevant=target.levels[2])
recall_macro_avg = (r1 + r2) / 2
recall_micro_avg = (r1 * n_high_quality/n_individuals) + (r2 * n_low_quality/n_individuals)

F_meas(rpart.model$confusion.matrix, relevant=target.levels[1])
F_meas(rpart.model$confusion.matrix, relevant=target.levels[2])

#---------------------------- SVM -----------------------------------------------
svm.model =  train(target ~ ., data=image_compression.reduced, method = "svmLinear",
                   metric = "ROC",
                   trControl = control)
plot(svm.model)
print(svm.model)
print(svm.model$finalModel@param)

plot_svm_3d_graph(svm.model)

svm.model$confusion.matrix = confusionMatrix(svm.model, norm = "none")$table
plot_confusion_matrix(svm.model$confusion.matrix)
folds = svm.model$resampledCM

repeats_confusion_matrixes = get_confusion_matrixes_stratified_10_fold(folds, reps)

TP_sd = sd(repeats_confusion_matrixes$cell1)
FP_sd = sd(repeats_confusion_matrixes$cell2)
FN_sd = sd(repeats_confusion_matrixes$cell3)
TN_sd = sd(repeats_confusion_matrixes$cell4)

svm.model$confusion.matrix = round(svm.model$confusion.matrix/reps)

assert("La somma degli elementi della matrice non è uguale al numero
       di individui", {
         dim(image_compression.reduced)[1] == sum(svm.model$confusion.matrix)
       })

accuracy_mean = sum(diag(svm.model$confusion.matrix))/sum(svm.model$confusion.matrix)
accuracy_sd = get_accuracy_sd(repeats_confusion_matrixes, accuracy_mean, reps)


p1 = precision(svm.model$confusion.matrix, relevant=target.levels[1])
p2 = precision(svm.model$confusion.matrix, relevant=target.levels[2])
precision_macro_avg = (p1 + p2) / 2
precision_micro_avg = (p1 * n_high_quality/n_individuals) + (p2 * n_low_quality/n_individuals)

r1 = recall(svm.model$confusion.matrix, relevant=target.levels[1])
r2 = recall(svm.model$confusion.matrix, relevant=target.levels[2])
recall_macro_avg = (r1 + r2) / 2
recall_micro_avg = (r1 * n_high_quality/n_individuals) + (r2 * n_low_quality/n_individuals)

F_meas(svm.model$confusion.matrix, relevant=target.levels[1])
F_meas(svm.model$confusion.matrix, relevant=target.levels[2]) 
#---------------------------- Nnet ----------------------------------------------
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

nnet.model = train(target ~ ., data=image_compression.reduced, method = "nnet", metric = "ROC", trControl = control)

nnet.model$confusion.matrix = confusionMatrix(nnet.model, norm = "none")$table
plot_confusion_matrix(nnet.model$confusion.matrix)

best_tune = nnet.model$bestTune
best_size = as.numeric(nnet.model$bestTune[1])
best_decay = as.numeric(nnet.model$bestTune[2])
folds = nnet.model$resampledCM[nnet.model$resampledCM$size == best_size 
                               & nnet.model$resampledCM$decay == best_decay, ]

repeats_confusion_matrixes = get_confusion_matrixes_stratified_10_fold(folds, reps)

TP_sd = sd(repeats_confusion_matrixes$cell1)
FP_sd = sd(repeats_confusion_matrixes$cell2)
FN_sd = sd(repeats_confusion_matrixes$cell3)
TN_sd = sd(repeats_confusion_matrixes$cell4)

nnet.model$confusion.matrix = round(nnet.model$confusion.matrix/reps)

assert("La somma degli elementi della matrice non è uguale al numero
       di individui", {
         dim(image_compression.reduced)[1] == sum(nnet.model$confusion.matrix)
       })

accuracy_mean = sum(diag(nnet.model$confusion.matrix))/sum(nnet.model$confusion.matrix)
accuracy_sd = get_accuracy_sd(repeats_confusion_matrixes, accuracy_mean, reps)

p1 = precision(nnet.model$confusion.matrix, relevant=target.levels[1])
p2 = precision(nnet.model$confusion.matrix, relevant=target.levels[2])
precision_macro_avg = (p1 + p2) / 2
precision_micro_avg = (p1 * n_high_quality/n_individuals) + (p2 * n_low_quality/n_individuals)

r1 = recall(nnet.model$confusion.matrix, relevant=target.levels[1])
r2 = recall(nnet.model$confusion.matrix, relevant=target.levels[2])
recall_macro_avg = (r1 + r2) / 2
recall_micro_avg = (r1 * n_high_quality/n_individuals) + (r2 * n_low_quality/n_individuals)

F_meas(nnet.model$confusion.matrix, relevant=target.levels[1])
F_meas(nnet.model$confusion.matrix, relevant=target.levels[2])

#---------------------------- measure 10-cross fold validation  ------------------
svm.probs = predict(svm.model, image_compression.reduced[,! names(image_compression.reduced) %in% c("target")],
                    type = "prob")
rpart.probs = predict(rpart.model, image_compression.reduced[,! names(image_compression.reduced) %in% c("target")],
                      type = "prob")
nnet.probs = predict(nnet.model, image_compression.reduced[,! names(image_compression.reduced) %in% c("target")],
                     type = "prob")

svm.ROC = roc(response = image_compression.reduced$target, predictor = svm.probs$high_quality,
              levels = levels(image_compression.reduced[,c("target")]))
plot(svm.ROC,type="S", col="green")

rpart.ROC = roc(response = image_compression.reduced[,c("target")],
                predictor =rpart.probs$high_quality,
                levels = levels(image_compression.reduced[,c("target")]))
plot(rpart.ROC, add=TRUE, col="blue")

nnet.ROC = roc(response = image_compression.reduced$target, predictor = nnet.probs$high_quality,
               levels = levels(image_compression.reduced[,c("target")]))
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
