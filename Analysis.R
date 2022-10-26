rm(list=ls())
library(gtools)
#################################################################

setwd("/home/aidan/grape/results/nguyen/Corr/")
Nguyen_Corr = c()

for (a in 1:length(list.files())){
  
  file = read.csv2(list.files()[a], sep = "\t", stringsAsFactors = FALSE)
  as.numeric(min(file$min_train[51]))
  
  Nguyen_Corr = c(Nguyen_Corr, as.numeric(min(file$min_train[51])))
  
}

setwd("/home/aidan/grape/results/nguyen/LS/")
Nguyen_LS = c()

for (a in 1:length(list.files())){
  
  file = read.csv2(list.files()[a], sep = "\t", stringsAsFactors = FALSE)
  min(file$min_train[51])
  
  Nguyen_LS = c(Nguyen_LS, as.numeric(min(file$min_train[51])))
  
}

setwd("/home/aidan/grape/results/nguyen/MSE/")
Nguyen_MSE = c()

for (a in 1:length(list.files())){
  
  file = read.csv2(list.files()[a], sep = "\t", stringsAsFactors = FALSE)
  min(file$min_train[51])
  
  Nguyen_MSE = c(Nguyen_MSE, as.numeric(min(file$min_train[51])))
  
}


#############################################################

setwd("/home/aidan/grape/results/paige1/Corr/")
paige1_Corr = c()

for (a in 1:length(list.files())){
  
  file = read.csv2(list.files()[a], sep = "\t", stringsAsFactors = FALSE)
  min(file$min_train[51])
  
  paige1_Corr = c(paige1_Corr, as.numeric(min(file$min_train[51])))
  
}

setwd("/home/aidan/grape/results/paige1/LS/")
paige1_LS = c()

for (a in 1:length(list.files())){
  
  file = read.csv2(list.files()[a], sep = "\t", stringsAsFactors = FALSE)
  min(file$min_train[51])
  
  paige1_LS = c(paige1_LS, as.numeric(min(file$min_train[51])))
  
}

setwd("/home/aidan/grape/results/paige1/MSE/")
paige1_MSE = c()

for (a in 1:length(list.files())){
  
  file = read.csv2(list.files()[a], sep = "\t", stringsAsFactors = FALSE)
  min(file$min_train[51])
  
  paige1_MSE = c(paige1_MSE, as.numeric(min(file$min_train[51])))
  
}



#############################################################

setwd("/home/aidan/grape/results/vlad//Corr/")
vlad_Corr = c()

for (a in 1:length(list.files())){
  
  file = read.csv2(list.files()[a], sep = "\t", stringsAsFactors = FALSE)
  min(file$min_train[51])
  
  vlad_Corr = c(vlad_Corr, as.numeric(min(file$min_train[51])))
  
}

setwd("/home/aidan/grape/results/vlad/LS/")
vlad_LS = c()

for (a in 1:length(list.files())){
  
  file = read.csv2(list.files()[a], sep = "\t", stringsAsFactors = FALSE)
  min(file$min_train[51])
  
  vlad_LS = c(vlad_LS, as.numeric(min(file$min_train[51])))
  
}

setwd("/home/aidan/grape/results/vlad/MSE/")
vlad_MSE = c()

for (a in 1:length(list.files())){
  
  file = read.csv2(list.files()[a], sep = "\t", stringsAsFactors = FALSE)
  min(file$min_train[30]
  
  vlad_MSE = c(vlad_MSE, as.numeric(min(file$min_train[51])))
  
}

