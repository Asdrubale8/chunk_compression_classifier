#Funzione per caricare un file csv
load <- function(csv_filename) {
  read.csv(csv_filename);
}

get_features <- function(dataset) {
  dataset[,c('Threshold','Mean','Median','Variance','Rms.constrast','M.contrast','Entropy')];
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
    dataset[chunk_index,'Variance']=var(chunk);
    dataset[chunk_index,'Rms.constrast']=sd(chunk);
    dataset[chunk_index,'M.contrast']=(max(chunk)-min(chunk))/(max(chunk)+min(chunk));
    dataset[chunk_index,'Entropy']=shannon.entropy(chunk);
    #dataset[chunk_index,'W.contrast']=sd(chunk);
  }
  
  dataset[,'Quality']=factor(dataset[,'Quality']);
  dataset;
}

shannon.entropy <- function(p)
{
  if (min(p) < 0 || sum(p) <= 0)
    return(NA)
  p.norm <- p[p>0]/sum(p)
  -sum(log(p.norm,8)*p.norm)
}

split.data = function(data, p = 0.7, s = 1){
  set.seed(s)
  index = sample(1:dim(data)[1])
  train = data[index[1:floor(dim(data)[1] * p)], ]
  test = data[index[((ceiling(dim(data)[1] * p)) + 1):dim(data)[1]], ]
  return(list(train=train, test=test)) } 