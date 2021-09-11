#---------------------------- libraries -------------------------------------------------
source("image_compression_lib.R");

installed_packages = data.frame(installed.packages())$Package
needed_packages = c("testit", "FactoMineR", "factoextra", "ggplot2",
                    "rpart", "rattle", "rpart.plot", "RColorBrewer", 
                    "e1071", "rgl", "C50", "caret", "ROCR", "pROC","plyr")
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

control = trainControl(method = "repeatedcv", number = 10,  repeats = reps,returnData = TRUE,
                       returnResamp = "final", savePredictions = "final",
                       classProbs = TRUE, summaryFunction = twoClassSummary)

#---------------------------- decision tree ------------------------------------
rpart.model = train(target ~ ., data=image_compression.reduced, method = "rpart", metric = "ROC",
                    trControl = control)
fancyRpartPlot(rpart.model$finalModel)

#-----rpart1------
rpart1.model = train(target ~ ., data=image_compression.reduced, method = "rpart", metric = "ROC",
                     trControl = control)
fancyRpartPlot(rpart1.model$finalModel)

rpart1.model$confusion.matrix = confusionMatrix(rpart1.model, norm = "none")$table

folds = rpart1.model$resampledCM

repeats_confusion_matrixes = get_confusion_matrixes_stratified_10_fold(folds, reps)

TP_sd = sd(repeats_confusion_matrixes$cell1)
FP_sd = sd(repeats_confusion_matrixes$cell2)
FN_sd = sd(repeats_confusion_matrixes$cell3)
TN_sd = sd(repeats_confusion_matrixes$cell4)

rpart1.model$confusion.matrix = round(rpart1.model$confusion.matrix/reps)
plot_confusion_matrix(rpart1.model$confusion.matrix)

#-----rpart2------
rpart2.model = train(target ~ ., data=image_compression.reduced, method = "rpart", metric = "Spec",
                     trControl = control)
fancyRpartPlot(rpart2.model$finalModel)

rpart2.model$confusion.matrix = confusionMatrix(rpart2.model, norm = "none")$table

folds = rpart2.model$resampledCM

repeats_confusion_matrixes = get_confusion_matrixes_stratified_10_fold(folds, reps)

TP_sd = sd(repeats_confusion_matrixes$cell1)
FP_sd = sd(repeats_confusion_matrixes$cell2)
FN_sd = sd(repeats_confusion_matrixes$cell3)
TN_sd = sd(repeats_confusion_matrixes$cell4)

rpart2.model$confusion.matrix = round(rpart2.model$confusion.matrix/reps)
plot_confusion_matrix(rpart2.model$confusion.matrix)
#-----rpart3------
rpart3.model = train(target ~ ., data=image_compression.reduced, method = "rpart", metric = "Sens",
                     trControl = control)

rpart3.model$confusion.matrix = confusionMatrix(rpart3.model, norm = "none")$table

folds = rpart3.model$resampledCM

repeats_confusion_matrixes = get_confusion_matrixes_stratified_10_fold(folds, reps)

TP_sd = sd(repeats_confusion_matrixes$cell1)
FP_sd = sd(repeats_confusion_matrixes$cell2)
FN_sd = sd(repeats_confusion_matrixes$cell3)
TN_sd = sd(repeats_confusion_matrixes$cell4)

rpart3.model$confusion.matrix = round(rpart3.model$confusion.matrix/reps)
plot_confusion_matrix(rpart3.model$confusion.matrix)
#-----TRoc--------

roc_with_ci(rpart1.model, "red")
roc_with_ci(rpart2.model, "purple")
roc_with_ci(rpart3.model, "green")

plot_roc(rpart1.model, "red")
plot_roc(rpart2.model, "purple", add=TRUE)
plot_roc(rpart3.model, "green", add=TRUE)

legend(x = "topright",          # Position
       legend = c("rpart1", "rpart2", "rpart3"),  # Legend texts
       #lty = c(1, 2, 3),           # Line types
       col = c("red", "purple", "green"),           # Line colors
       lwd = 2)                 # Line width

cv.values = resamples(list(rpart1=rpart1.model, rpart2 = rpart2.model, rpart3 = rpart3.model))
summary(cv.values)
dotplot(cv.values, metric = "ROC") 
bwplot(cv.values, layout = c(3, 1))
splom(cv.values,metric="ROC")
cv.values$timings

#-----------
rpart.model = rpart1.model

rpart.model$confusion.matrix = confusionMatrix(rpart.model, norm = "none")$table

best_tune = as.numeric(rpart.model$bestTune)
folds = rpart.model$resampledCM[rpart.model$resampledCM$cp == best_tune, ]

repeats_confusion_matrixes = get_confusion_matrixes_stratified_10_fold(folds, reps)

TP_sd = sd(repeats_confusion_matrixes$cell1)
FP_sd = sd(repeats_confusion_matrixes$cell2)
FN_sd = sd(repeats_confusion_matrixes$cell3)
TN_sd = sd(repeats_confusion_matrixes$cell4)

rpart.model$confusion.matrix = round(rpart.model$confusion.matrix/reps)
rpart.model$confusion.matrix.sd = rpart.model$confusion.matrix
rpart.model$confusion.matrix.sd[1,1] = TP_sd
rpart.model$confusion.matrix.sd[2,1] = FN_sd
rpart.model$confusion.matrix.sd[1,2] = FP_sd
rpart.model$confusion.matrix.sd[2,2] = TN_sd

assert("La somma degli elementi della matrice non è uguale al numero
       di individui", {
         dim(image_compression.reduced)[1] == sum(rpart.model$confusion.matrix)
       })
plot_confusion_matrix(rpart.model$confusion.matrix)
plot_confusion_matrix.sd(rpart.model$confusion.matrix.sd)

accuracy_mean = sum(diag(rpart.model$confusion.matrix))/sum(rpart.model$confusion.matrix)
accuracies = get_accuracies(repeats_confusion_matrixes, reps)
summary(accuracies)
boxplot(accuracies, main="rpart accuracy")

p1 = precision(rpart.model$confusion.matrix, relevant=target.levels[1])
p2 = precision(rpart.model$confusion.matrix, relevant=target.levels[2])
precision_macro_avg = (p1 + p2) / 2
precision_micro_avg = (p1 * n_high_quality/n_individuals) + (p2 * n_low_quality/n_individuals)

rpart.precisions1 = get_precisions(repeats_confusion_matrixes, reps, relevant="0")
rpart.precisions2 = get_precisions(repeats_confusion_matrixes, reps, relevant="1")

r1 = recall(rpart.model$confusion.matrix, relevant=target.levels[1])
r2 = recall(rpart.model$confusion.matrix, relevant=target.levels[2])
recall_macro_avg = (r1 + r2) / 2
recall_micro_avg = (r1 * n_high_quality/n_individuals) + (r2 * n_low_quality/n_individuals)

rpart.recalls1 = get_recalls(repeats_confusion_matrixes, reps, relevant="0")
rpart.recalls2 = get_recalls(repeats_confusion_matrixes, reps, relevant="1")

F_meas(rpart.model$confusion.matrix, beta = 0.5)
F_meas(rpart.model$confusion.matrix, beta = 1)
F_meas(rpart.model$confusion.matrix, beta = 2)

rpart.model$resample
summary(rpart.model$resample)[,1:3]
boxplot(rpart.model$resample$ROC, main="rpart ROC")
boxplot(rpart.model$resample$Sens, main="rpart Sensitivity")
boxplot(rpart.model$resample$Spec, main="rpart Specificity")
rpart.model$times
#---------------------------- SVM -----------------------------------------------
tune.gridxgb <- expand.grid(C=c(0.002,0.5,1))
svm.model =  train(target ~ ., data=image_compression.reduced, method = "svmLinear",
                   metric = "ROC",tuneLength=3, tuneGrid=tune.gridxgb,
                   trControl = control)
plot(svm.model)
print(svm.model)
print(svm.model$finalModel@param)

plot_svm_3d_graph(svm.model)

#-----SVM05------
tune.gridxgb <- expand.grid(C=c(0.5))
svm05.model =  train(target ~ ., data=image_compression.reduced, method = "svmLinear",
                     metric = "ROC", tuneGrid =tune.gridxgb,
                     trControl = control)
plot(svm05.model)
print(svm05.model)
print(svm05.model$finalModel@param)

plot_svm_3d_graph(svm05.model)

svm05.model$confusion.matrix = confusionMatrix(svm05.model, norm = "none")$table

folds = svm05.model$resampledCM

repeats_confusion_matrixes = get_confusion_matrixes_stratified_10_fold(folds, reps)

TP_sd = sd(repeats_confusion_matrixes$cell1)
FP_sd = sd(repeats_confusion_matrixes$cell2)
FN_sd = sd(repeats_confusion_matrixes$cell3)
TN_sd = sd(repeats_confusion_matrixes$cell4)

svm05.model$confusion.matrix = round(svm05.model$confusion.matrix/reps)
plot_confusion_matrix(svm05.model$confusion.matrix)
#-----SVM1-------
tune.gridxgb <- expand.grid(C=c(1))
svm1.model =  train(target ~ ., data=image_compression.reduced, method = "svmLinear",
                    metric = "ROC", tuneGrid =tune.gridxgb,
                    trControl = control)
plot(svm1.model)
print(svm1.model)
print(svm1.model$finalModel@param)

plot_svm_3d_graph(svm1.model)

svm1.model$confusion.matrix = confusionMatrix(svm1.model, norm = "none")$table

folds = svm1.model$resampledCM

repeats_confusion_matrixes = get_confusion_matrixes_stratified_10_fold(folds, reps)

TP_sd = sd(repeats_confusion_matrixes$cell1)
FP_sd = sd(repeats_confusion_matrixes$cell2)
FN_sd = sd(repeats_confusion_matrixes$cell3)
TN_sd = sd(repeats_confusion_matrixes$cell4)

svm1.model$confusion.matrix = round(svm1.model$confusion.matrix/reps)
plot_confusion_matrix(svm1.model$confusion.matrix)
#-----SVM0002-------
tune.gridxgb <- expand.grid(C=c(0.002))
svm0002.model =  train(target ~ ., data=image_compression.reduced, method = "svmLinear",
                       metric = "ROC", tuneGrid =tune.gridxgb,
                       trControl = control)
plot(svm0002.model)
print(svm0002.model)
print(svm0002.model$finalModel@param)

plot_svm_3d_graph(svm0002.model)

svm0002.model$confusion.matrix = confusionMatrix(svm0002.model, norm = "none")$table

folds = svm0002.model$resampledCM

repeats_confusion_matrixes = get_confusion_matrixes_stratified_10_fold(folds, reps)

TP_sd = sd(repeats_confusion_matrixes$cell1)
FP_sd = sd(repeats_confusion_matrixes$cell2)
FN_sd = sd(repeats_confusion_matrixes$cell3)
TN_sd = sd(repeats_confusion_matrixes$cell4)

svm0002.model$confusion.matrix = round(svm0002.model$confusion.matrix/reps)
plot_confusion_matrix(svm0002.model$confusion.matrix)
#-----SVMroc-------
roc_with_ci(svm8.model, "red")
roc_with_ci(svm05.model, "purple")
roc_with_ci(svm1.model, "green")

plot_roc(svm8.model, "red")
plot_roc(svm05.model, "purple", add=TRUE)
plot_roc(svm1.model, "green", add=TRUE)

legend(x = "topright",          # Position
       legend = c("svm0002", "svm05", "svm1"),  # Legend texts
       #lty = c(1, 2, 3),           # Line types
       col = c("red", "purple", "green"),           # Line colors
       lwd = 2)                 # Line width

cv.values = resamples(list(svm002=svm8.model, svm05 = svm05.model, svm1 = svm1.model))
summary(cv.values)
dotplot(cv.values, metric = "ROC") 
bwplot(cv.values, layout = c(3, 1))
splom(cv.values,metric="ROC")
cv.values$timings
#------------------
svm.model = svm05.model

svm.model$confusion.matrix = confusionMatrix(svm.model, norm = "none")$table

folds = svm.model$resampledCM

repeats_confusion_matrixes = get_confusion_matrixes_stratified_10_fold(folds, reps)

TP_sd = sd(repeats_confusion_matrixes$cell1)
FP_sd = sd(repeats_confusion_matrixes$cell2)
FN_sd = sd(repeats_confusion_matrixes$cell3)
TN_sd = sd(repeats_confusion_matrixes$cell4)

svm.model$confusion.matrix = round(svm.model$confusion.matrix/reps)
svm.model$confusion.matrix.sd = svm.model$confusion.matrix
svm.model$confusion.matrix.sd[1,1] = TP_sd
svm.model$confusion.matrix.sd[2,1] = FP_sd
svm.model$confusion.matrix.sd[1,2] = FN_sd
svm.model$confusion.matrix.sd[2,2] = TN_sd

assert("La somma degli elementi della matrice non è uguale al numero
       di individui", {
         dim(image_compression.reduced)[1] == sum(svm.model$confusion.matrix)
       })
plot_confusion_matrix(svm.model$confusion.matrix)
plot_confusion_matrix.sd(svm.model$confusion.matrix.sd)

accuracy_mean = sum(diag(svm.model$confusion.matrix))/sum(svm.model$confusion.matrix)
accuracies = get_accuracies(repeats_confusion_matrixes, reps)
summary(accuracies)
boxplot(accuracies, main="svm accuracy")


p1 = precision(svm.model$confusion.matrix, relevant=target.levels[1])
p2 = precision(svm.model$confusion.matrix, relevant=target.levels[2])
precision_macro_avg = (p1 + p2) / 2
precision_micro_avg = (p1 * n_high_quality/n_individuals) + (p2 * n_low_quality/n_individuals)

svm.precisions1 = get_precisions(repeats_confusion_matrixes, reps, relevant="0")
svm.precisions2 = get_precisions(repeats_confusion_matrixes, reps, relevant="1")

r1 = recall(svm.model$confusion.matrix, relevant=target.levels[1])
r2 = recall(svm.model$confusion.matrix, relevant=target.levels[2])
recall_macro_avg = (r1 + r2) / 2
recall_micro_avg = (r1 * n_high_quality/n_individuals) + (r2 * n_low_quality/n_individuals)

svm.recalls1 = get_recalls(repeats_confusion_matrixes, reps, relevant="0")
svm.recalls2 = get_recalls(repeats_confusion_matrixes, reps, relevant="1")

F_meas(svm.model$confusion.matrix, beta = 0.5)
F_meas(svm.model$confusion.matrix, beta = 1)
F_meas(svm.model$confusion.matrix, beta = 2)

svm.model$resample
summary(svm.model$resample)[,1:3]
boxplot(svm.model$resample$ROC, main="svm ROC")
boxplot(svm.model$resample$Sens, main="svm Sensitivity")
boxplot(svm.model$resample$Spec, main="svm Specificity")
svm.model$times
#---------------------------- Nnet ----------------------------------------------
#ind = sample(2, nrow(image_compression.reduced), replace = TRUE, prob=c(0.7, 0.3))
#trainset = image_compression.reduced[ind == 1,]
#testset = image_compression.reduced[ind == 2,]
#trainset$low_quality = trainset$target == "low_quality"
#trainset$high_quality = trainset$target == "high_quality"
#library(neuralnet)
#network = neuralnet(target ~ ., image_compression.reduced, hidden=5)
#plot(network)
#network$weights
#net.predict = compute(network, testset[c("Dim.1","Dim.2","Dim.3")])$net.result
#net.prediction = c("low_quality", "high_quality")[apply(net.predict, 1, which.max)]
#predict.table = table(testset$target, net.prediction)

tune.gridxgb <- expand.grid(size=c(1,2,3,4,5), decay=c(0.1))
nnet.model = train(target ~ ., data=image_compression.reduced, method = "nnet", 
                   metric = "ROC", trControl = control, tuneGrid=tune.gridxgb)

plot(nnet.model,col="dark green")
print(nnet.model)
print(nnet.model$finalModel@param)

nnet.model$confusion.matrix = confusionMatrix(nnet.model, norm = "none")$table
#------Nnet1------
tune.gridxgb <- expand.grid(size=c(1), decay=c(0.1))
nnet1.model = train(target ~ ., data=image_compression.reduced, method = "nnet", metric = "ROC", trControl = control, tuneGrid = tune.gridxgb)

nnet1.model$confusion.matrix = confusionMatrix(nnet1.model, norm = "none")$table

folds = nnet1.model$resampledCM

repeats_confusion_matrixes = get_confusion_matrixes_stratified_10_fold(folds, reps)

TP_sd = sd(repeats_confusion_matrixes$cell1)
FP_sd = sd(repeats_confusion_matrixes$cell2)
FN_sd = sd(repeats_confusion_matrixes$cell3)
TN_sd = sd(repeats_confusion_matrixes$cell4)

nnet1.model$confusion.matrix = round(nnet1.model$confusion.matrix/reps)
plot_confusion_matrix(nnet1.model$confusion.matrix)
#------Nnet3------
tune.gridxgb <- expand.grid(size=c(3), decay=c(0.1))
nnet3.model = train(target ~ ., data=image_compression.reduced, method = "nnet", metric = "ROC", trControl = control, tuneGrid = tune.gridxgb)

nnet3.model$confusion.matrix = confusionMatrix(nnet3.model, norm = "none")$table

folds = nnet3.model$resampledCM

repeats_confusion_matrixes = get_confusion_matrixes_stratified_10_fold(folds, reps)

TP_sd = sd(repeats_confusion_matrixes$cell1)
FP_sd = sd(repeats_confusion_matrixes$cell2)
FN_sd = sd(repeats_confusion_matrixes$cell3)
TN_sd = sd(repeats_confusion_matrixes$cell4)

nnet3.model$confusion.matrix = round(nnet3.model$confusion.matrix/reps)
plot_confusion_matrix(nnet3.model$confusion.matrix)

#------Nnet5------
tune.gridxgb <- expand.grid(size=c(5), decay=c(0.1))
nnet5.model = train(target ~ ., data=image_compression.reduced, method = "nnet", metric = "ROC", trControl = control, tuneGrid = tune.gridxgb)

nnet5.model$confusion.matrix = confusionMatrix(nnet5.model, norm = "none")$table

folds = nnet5.model$resampledCM

repeats_confusion_matrixes = get_confusion_matrixes_stratified_10_fold(folds, reps)

TP_sd = sd(repeats_confusion_matrixes$cell1)
FP_sd = sd(repeats_confusion_matrixes$cell2)
FN_sd = sd(repeats_confusion_matrixes$cell3)
TN_sd = sd(repeats_confusion_matrixes$cell4)

nnet5.model$confusion.matrix = round(nnet5.model$confusion.matrix/reps)
plot_confusion_matrix(nnet5.model$confusion.matrix)

#------NnetROC-------

roc_with_ci(nnet1.model, "red")
roc_with_ci(nnet3.model, "purple")
roc_with_ci(nnet5.model, "green")

plot(roc(predictor = nnet1.model$pred$high_quality, response = nnet1.model$pred$obs), col="red")
plot(roc(predictor = nnet3.model$pred$high_quality, response = nnet3.model$pred$obs), col="purple", add = TRUE)
plot(roc(predictor = nnet5.model$pred$high_quality, response = nnet5.model$pred$obs), col="green", add = TRUE)

legend(x = "topright",          # Position
       legend = c("nnet1", "nnet3", "nnet5"),  # Legend texts
       #lty = c(1, 2, 3),           # Line types
       col = c("red", "purple", "green"),           # Line colors
       lwd = 2)                 # Line width

cv.values = resamples(list(nnet1=nnet1.model, nnet3 = nnet3.model, nnet5 = nnet5.model))
summary(cv.values)
dotplot(cv.values, metric = "ROC") 
bwplot(cv.values, layout = c(3, 1))
splom(cv.values, metric="ROC")
cv.values$timings
#--------------------
nnet.model=nnet3.model

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
nnet.model$confusion.matrix.sd = nnet.model$confusion.matrix
nnet.model$confusion.matrix.sd[1,1] = TP_sd
nnet.model$confusion.matrix.sd[2,1] = FP_sd
nnet.model$confusion.matrix.sd[1,2] = FN_sd
nnet.model$confusion.matrix.sd[2,2] = TN_sd

assert("La somma degli elementi della matrice non è uguale al numero
       di individui", {
         dim(image_compression.reduced)[1] == sum(nnet.model$confusion.matrix)
       })
plot_confusion_matrix(nnet.model$confusion.matrix)
plot_confusion_matrix.sd(nnet.model$confusion.matrix.sd)

accuracy_mean = sum(diag(nnet.model$confusion.matrix))/sum(nnet.model$confusion.matrix)
accuracies = get_accuracies(repeats_confusion_matrixes, reps)
summary(accuracies)
boxplot(accuracies, main="nnet accuracy")

p1 = precision(nnet.model$confusion.matrix, relevant=target.levels[1])
p2 = precision(nnet.model$confusion.matrix, relevant=target.levels[2])
precision_macro_avg = (p1 + p2) / 2
precision_micro_avg = (p1 * n_high_quality/n_individuals) + (p2 * n_low_quality/n_individuals)

nnet.precisions1 = get_precisions(repeats_confusion_matrixes, reps, relevant="0")
nnet.precisions2 = get_precisions(repeats_confusion_matrixes, reps, relevant="1")

r1 = recall(nnet.model$confusion.matrix, relevant=target.levels[1])
r2 = recall(nnet.model$confusion.matrix, relevant=target.levels[2])
recall_macro_avg = (r1 + r2) / 2
recall_micro_avg = (r1 * n_high_quality/n_individuals) + (r2 * n_low_quality/n_individuals)


nnet.recalls1 = get_recalls(repeats_confusion_matrixes, reps, relevant="0")
nnet.recalls2 = get_recalls(repeats_confusion_matrixes, reps, relevant="1")

F_meas(nnet.model$confusion.matrix, beta = 0.5)
F_meas(nnet.model$confusion.matrix, beta = 1)
F_meas(nnet.model$confusion.matrix, beta = 2)

nnet.model$resample
summary(nnet.model$resample)[,1:3]
boxplot(nnet.model$resample$ROC, main="nnet ROC")
boxplot(nnet.model$resample$Sens, main="nnet Sensitivity")
boxplot(nnet.model$resample$Spec, main="nnet Specificity")
nnet.model$times

#---------------------------- measure 10-cross fold validation  ------------------

roc_with_ci(rpart.model,"red")
roc_with_ci(svm.model,"green")
roc_with_ci(nnet.model,"blue")

plot_roc(rpart.model,"red")
plot_roc(svm.model,"green",add=TRUE)
plot_roc(nnet.model,"blue",add=TRUE)

legend(x = "topright",          # Position
       legend = c("rpart", "svm", "nnet"),  # Legend texts
       #lty = c(1, 2, 3),           # Line types
       col = c("red", "green", "blue"),           # Line colors
       lwd = 2)

cv.values = resamples(list(svm=svm.model, rpart = rpart.model, nnet = nnet.model))
summary(cv.values)
dotplot(cv.values, metric = "ROC") 
bwplot(cv.values, layout = c(3, 1))
splom(cv.values, metric="ROC")
cv.values$timings

boxplot(rpart.precisions1, svm.precisions1, nnet.precisions1, names = c("rpart", "svm", "nnet"), main="precision - high_q")
boxplot(rpart.precisions2, svm.precisions2, nnet.precisions2, names = c("rpart", "svm", "nnet"), main="precision - low_q")

boxplot(rpart.recalls1, svm.recalls1, nnet.recalls1, names = c("rpart", "svm", "nnet"), main="recall - high_q")
boxplot(rpart.recalls2, svm.recalls2, nnet.recalls2, names = c("rpart", "svm", "nnet"), main="recall - low_q")

