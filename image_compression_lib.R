#Funzione per caricare un file csv
load <- function(csv_filename) {
  read.csv(csv_filename);
}

get_features <- function(dataset) {
  dataset[,c('Threshold','Mean','Median','Rms.contrast','M.contrast','Entropy')];
}

get_target <- function(dataset) {
  dataset[,'Quality'];
}

#Funzione per il preprocessing dei dati del dataset di image_compression
preprocess <- function(dataset) {
  
  #Conversione delle stringhe relative ai chunk in array di strighe
  string_chunks = strsplit(dataset[,'Bmp.image']," ");
  
  for (chunk_index in 1:length(string_chunks)){
    #Conversione dei singoli chunk in array di interi
    chunk = as.integer(string_chunks[[chunk_index]]);
    
    #Calcolo indici statistici sul chunk
    dataset[chunk_index,'Mean']=mean(chunk);
    dataset[chunk_index,'Median']=median(chunk);
    #dataset[chunk_index,'Variance']=var(chunk);
    dataset[chunk_index,'Rms.contrast']=sd(chunk);
    dataset[chunk_index,'M.contrast']=(max(chunk)-min(chunk))/(max(chunk)+min(chunk));
    dataset[chunk_index,'Entropy']=shannon.entropy(chunk);
    #dataset[chunk_index,'W.contrast']=sd(chunk);
  }
  
  dataset$Quality[dataset$Quality == "low quality"] = "low_quality";
  dataset$Quality[dataset$Quality == "high quality"] = "high_quality";
  dataset[,'Quality']=factor(dataset[,'Quality']);
  dataset;
}

shannon.entropy <- function(p) {
  if (min(p) < 0 || sum(p) <= 0)
    return(NA)
  p.norm <- p[p>0]/sum(p)
  -sum(log(p.norm,8)*p.norm)
}

split.data <- function(data, p = 0.7, s = 1) {
  set.seed(s)
  index = sample(1:dim(data)[1])
  train = data[index[1:floor(dim(data)[1] * p)], ]
  test = data[index[((ceiling(dim(data)[1] * p)) + 1):dim(data)[1]], ]
  return(list(train=train, test=test)) 
} 

plot_confusion_matrix <- function(confusion.matrix, color = "#009194") {
  plt <- as.data.frame(confusion.matrix)
  plt$Prediction <- factor(plt$Prediction, levels=rev(levels(plt$Prediction)))
  ggplot(plt, aes(Reference,Prediction, fill= Freq)) +
    geom_tile() + geom_text(aes(label=Freq)) +
    scale_fill_gradient(low="white", high=color) +
    labs(x = "Reference",y = "Prediction") +
    scale_x_discrete(labels=c("High quality","Low Quality")) +
    scale_y_discrete(labels=c("Low Quality","High quality"))
}

plot_confusion_matrix.sd <- function(confusion.matrix) {
  plot_confusion_matrix(confusion.matrix, "#4682B4")
}


plot_svm_3d_graph <- function(svm.model) {
  coefs <- svm.model$finalModel@coef[[1]]
  mat <- svm.model$finalModel@xmatrix[[1]]
  w <- coefs %*% mat
  detalization <- 100                                                                                                                                                                 
  grid <- expand.grid(seq(from=min(image_compression.reduced$Dim.1),to=max(image_compression.reduced$Dim.1),
                          length.out=detalization),                                                                                                         
                      seq(from=min(image_compression.reduced$Dim.2),to=max(image_compression.reduced$Dim.2),
                          length.out=detalization))                                                                                                         
  z <- (svm.model$finalModel@b - w[1,1]*grid[,1] - w[1,2]*grid[,2]) / w[1,3]
  plot3d(grid[,1],grid[,2],z, col='gray')  # this will draw plane.
  points3d(image_compression.reduced$Dim.1[which(image_compression.reduced$target=='low_quality')],
           image_compression.reduced$Dim.2[which(image_compression.reduced$target=='low_quality')],
           image_compression.reduced$Dim.3[which(image_compression.reduced$target=='low_quality')], col='red')
  points3d(image_compression.reduced$Dim.1[which(image_compression.reduced$target=='high_quality')],
           image_compression.reduced$Dim.2[which(image_compression.reduced$target=='high_quality')],
           image_compression.reduced$Dim.3[which(image_compression.reduced$target=='high_quality')], col='blue')
  
}

get_confusion_matrixes_stratified_10_fold <- function(folds, reps) {
  repeats_confusion_matrixes = data.frame(cell1=1:reps, cell2=1:reps, cell3=1:reps, cell4=1:reps)
  
  for(i in 1:reps) {
    
    rep_i = if (i<10)  paste(c("Rep0", i), collapse = "") else paste(c("Rep", i), collapse = "")
    
    repeat_i = dplyr::filter(folds, grepl(rep_i, Resample))
    repeat_i <- subset( repeat_i, select = c(cell1, cell2, cell3, cell4) )
    
    repeats_confusion_matrixes[i,] = repeat_i %>% dplyr::summarise_all(sum)
  }
  repeats_confusion_matrixes
}

get_accuracies <- function(repeats_confusion_matrixes, reps){
  accuracies = numeric(reps)
  for(i in 1:reps) {
    confusion_matrix = repeats_confusion_matrixes[i, ]
    accuracies[i] = (confusion_matrix$cell1 + 
                       confusion_matrix$cell4)/sum(confusion_matrix)
  }
  accuracies
}

get_empty_confusion_matrix = function(){
  x = as.factor(c("0","1"))
  empty_cm = confusionMatrix(x, x)$table
  empty_cm[1,1] = empty_cm[1,2] = empty_cm[2,1] = empty_cm[2,2] = 0
  empty_cm
}

get_precisions <- function(repeats_confusion_matrixes, reps, relevant="0"){
  precisions = numeric(reps)
  for(i in 1:reps) {
    confusion_matrix_row = repeats_confusion_matrixes[i, ]
    confusion_matrix = get_confusion_matrix_from_row(confusion_matrix_row)
    precisions[i] = precision(confusion_matrix, relevant=relevant)
  }
  
  precisions
}

get_recalls <- function(repeats_confusion_matrixes, reps, relevant="0"){
  recalls = numeric(reps)
  
  for(i in 1:reps) {
    confusion_matrix_row = repeats_confusion_matrixes[i, ]
    confusion_matrix = get_confusion_matrix_from_row(confusion_matrix_row)
    recalls[i] = recall(confusion_matrix, relevant=relevant)
  }
  
  recalls
}

get_confusion_matrix_from_row = function(confusion_matrix_row){
  confusion_matrix = get_empty_confusion_matrix()
  confusion_matrix[1,1] = confusion_matrix_row$cell1
  confusion_matrix[2,1] = confusion_matrix_row$cell2
  confusion_matrix[1,2] = confusion_matrix_row$cell3
  confusion_matrix[2,2] = confusion_matrix_row$cell4
  
  confusion_matrix
}

roc_with_ci <- function(rpart.model, color) {
  #rpart.model$pred$Resample <- substr(rpart.model$pred$Resample, 8, 12)
  rpart.model$pred$Resample
  plot(roc(predictor = rpart.model$pred$high_quality, response = rpart.model$pred$obs), col=color)
  l_ply(split(rpart.model$pred, rpart.model$pred$Resample), function(d) {
    plot(roc(predictor = d$high_quality, response = d$obs), col="grey", add=TRUE)
  })
  plot(roc(predictor = rpart.model$pred$high_quality, response = rpart.model$pred$obs), col=color, add = TRUE)
  
} 

plot_roc <- function(rpart.model, color, add=FALSE) {
  rpart.ROC = roc(predictor = rpart.model$pred$high_quality, response = rpart.model$pred$obs)
  #levels = levels(image_compression.reduced[,c("target")]))
  plot(rpart.ROC,type="S", col=color, add=add)
} 

