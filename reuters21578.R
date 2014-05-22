# Data Mining - Assignment 8
# Reuters21578 Text Mining and Topic Models

install.packages("tm")
install.packages("/dcs/pg11/phil/reuters/tm.corpus.Reuters21578/", repos = NULL, type="source")
install.packages("SnowballC")
require(tm)
require(tm.corpus.Reuters21578) # to load data

# other libraries
library(slam)
install.packages("topicmodels")
require("topicmodels")
install.packages("RTextTools")
library(RTextTools)
install.packages("ggplot2")
library(ggplot2)
install.packages("fpc")
library(fpc) #dbscan
library(stats) #kmeans

# Set directory
setwd("~/workspace/reuters21578")





# Pre-processing
#***************

# Load data
data(Reuters21578)
rt<-Reuters21578



# Split data into training and test set based on LEWISSPLIT
#
#      Documents on LEWISSPLIT
#      -----------------------
#          Train    14668
#          Test      6188
#      +   Not-Used   722
#      ------------------
#          Sum      21578  
#
queryIfTrain <- "LEWISSPLIT == 'TRAIN'"
rtTrain = tm_filter(rt, FUN=sFilter, queryIfTrain)
rm(queryIfTrain)

queryIfTest <- "LEWISSPLIT == 'TEST'"
rtTest = tm_filter(rt, FUN=sFilter, queryIfTest)
rm(queryIfTest)



# Removing noise (docs with no topics) from the training set 
#
#      Documents according to Topics tag
#      ---------------------------------
#          Topic Exists  11367
#      +   No Topic      10211 
#      -----------------------
#          Sum           21578  
#
queryIfTopicExist <- "Topics != 'character(0)'"
rtTrain = tm_filter(rtTrain, FUN=sFilter, queryIfTopicExist)



# Create training document term matrix 
dtmTrain <- DocumentTermMatrix(rtTrain, control=list(removePunctuation=TRUE, removeNumbers=TRUE,stopwords=TRUE,stemming=TRUE,minWordLength=3))



# Feature Selection: based on word sparsity and tf-idf
dtmTrain <- removeSparseTerms(dtmTrain,0.995)
summary(col_sums(dtmTrain))
term_tfidf <- tapply(dtmTrain$v/row_sums(dtmTrain)[dtmTrain$i],dtmTrain$j,mean)*log2(nDocs(dtmTrain)/col_sums(dtmTrain>0))
summary(term_tfidf)
dtmTrain <- dtmTrain[,term_tfidf >= 0.09] # only include terms which have a tf-idf of at least 0.09 which is a bit more than the median
dtmTrain <- dtmTrain[row_sums(dtmTrain)>0,] # row check (omit zero rows)



# Create test document term matrix using training dictionary
dict = colnames(dtmTrain) #should return an array of terms as the dictionary
rtTest <- tm_map(rtTest, stripWhitespace)
rtTest <- tm_map(rtTest, tolower)
rtTest <- tm_map(rtTest, removePunctuation)
rtTest <- tm_map(rtTest, removeNumbers)
rtTest <- tm_map(rtTest, stemDocument)
dtmTest <- DocumentTermMatrix(rtTest,list(dictionary=dict) )



# Create main document term matrix
dtm = rbind(dtmTrain,dtmTest)



# Function:
# Get topic label as binary occurrence from corpus
# using 'Topics' metadata tag for a single topic of interest
tExist <- function(topic,corpus){
  # initialize topicExist as zero vector for each document
  topicExist = rep(0,length(corpus))  
  # populate topicExist over documents
  for(i in 1:length(corpus)){
    if(topic %in% meta(corpus[[i]], tag='Topics')){
      topicExist[i]=1
    }
  }   
  return(topicExist)
}



# Create topic class labels for each document 
docLabels = data.frame(tExist("earn",rt),
                       tExist("acq",rt),
                       tExist("money-fx",rt),
                       tExist("grain",rt),
                       tExist("crude",rt),
                       tExist("trade",rt),
                       tExist("interest",rt),
                       tExist("ship",rt),
                       tExist("wheat",rt),
                       tExist("corn",rt))
colnames(docLabels) <- c("earn","acq","money-fx","grain","crude","trade","interest","ship","wheat","corn")
docLabels[docLabels[,1]==1,11]="earn"
docLabels[docLabels[,2]==1,11]="acq"
docLabels[docLabels[,3]==1,11]="money-fx"
docLabels[docLabels[,4]==1,11]="grain"
docLabels[docLabels[,5]==1,11]="crude"
docLabels[docLabels[,6]==1,11]="trade"
docLabels[docLabels[,7]==1,11]="interest"
docLabels[docLabels[,8]==1,11]="ship"
docLabels[docLabels[,9]==1,11]="wheat"
docLabels[docLabels[,10]==1,11]="corn"



# Pre-processing Outputs:
#   dtmTrain
#   dtmTest
#   dtm
#   docLabels
#
#
#      Document Term Matrix  #Documents
#      --------------------------------
#          dtmTrain             7068   
#      +   dtmTest              6188         
#      --------------------------------
#          dtm                 13256 (with 655 terms)
#
#
#      Label Matrix 
#      ------------   
#         docLabels (with dimensions: 21578 Document, 11 Topic Columns (10 binary + 1 multiple))
#
#         NOTE: docLabels's ten frequent topics of interest are:
#               "earn","acq","money-fx","grain","crude","trade","interest","ship","wheat","corn"   
#               - First 10 columns are used for binary occurrence of each topic 
#               - The 11th column is for all 10 topics (label for the most frequent topic)
#





# Topic Modelling
#****************

# Build topic model
TM<-LDA(dtmTrain, 30, method = "VEM")



# Assign topics to training data frame
topicsFromTM = topics(TM) # column with topic labels from topic models
dfTrain= data.frame(inspect(dtmTrain)) 
dfTrain[,656]= topicsFromTM



# Prepare set of features coming from topic model for 10 topics
terms = list()
for(i in 1:10){
  
  # Get document IDs having topic i from docLabels
  ID = rownames(docLabels[docLabels[i]==1,])
  # ID = rownames(docLabels[docLabels[i]==1 & row_sums(docLabels[1:10])==1,])
  
  # Get topic model topics assigned for topic i
  labelsTM <- dfTrain[rownames(dfTrain)%in%ID,656]
  
  # Assign topic model terms for assigned topic model topics
  terms[[i]] = as.vector(terms(TM,15)[1:6,unique(labelsTM)])
}
feaCluster = unique(as.vector(c(terms[[1]],terms[[2]],terms[[3]],terms[[4]],terms[[5]],terms[[6]],terms[[7]],terms[[8]],terms[[9]],terms[[10]])))



# Topic Modelling Outputs:
#   TM
#   terms
#   feaCluster (as overall features for clustering)





# Classification
#***************

# 1ST CLASSIFICATION : 10 binary classification for 10 topics class (BASELINE)
container1 = list()
analytics1 = list()
for(t in 1:10){
  
  # prepare data
  container1[[t]] = create_container(dtm, docLabels[rownames(dtm),t], trainSize=1:7068, testSize=7069:13256,virgin=FALSE)
  
  # train
  SVM <- train_model(container1[[t]],"SVM")
  RF <- train_model(container1[[t]],"RF")
  TREE <- train_model(container1[[t]],"TREE") 
  
  # classify
  SVM_CL <- classify_model(container1[[t]],SVM)
  RF_CL <- classify_model(container1[[t]],RF)
  TREE_CL <- classify_model(container1[[t]],TREE)
  
  # evaluation
  analytics1[[t]] <- create_analytics(container1[[t]], cbind(SVM_CL,RF_CL,TREE_CL))
}



# 2ND CLASSIFICATION : 10 binary classification for 10 topics class using TM
container2 = list()
analytics2 = list()
for(t in 1:10){
  
  #prepare data
  dtmR = dtm[,colnames(dtm)%in%terms[[t]]] # Reduced dtm in columns (features are reduced according to Topic Model)
  container2[[t]] = create_container(dtmR, docLabels[rownames(dtmR),t], trainSize=1:7068, testSize=7069:13256,virgin=FALSE)
 
  # train
  SVM <- train_model(container2[[t]],"SVM")
  RF <- train_model(container2[[t]],"RF")
  TREE <- train_model(container2[[t]],"TREE") 
  
  # classify
  SVM_CL <- classify_model(container2[[t]],SVM)
  RF_CL <- classify_model(container2[[t]],RF)
  TREE_CL <- classify_model(container2[[t]],TREE)
  
  # evaluation
  analytics2[[t]] <- create_analytics(container2[[t]], cbind(SVM_CL,RF_CL,TREE_CL))
}



# Prepare performance results for the first classification
Precision = c(); Recall = c(); F1 = c()
Topics =  c("earn","acq","money-fx","grain","crude","trade","interest","ship","wheat","corn")

# SVM
for(i in 1:10){
  Precision[i]= summary(analytics1[[i]])[1]
  Recall[i]= summary(analytics1[[i]])[2]
  F1[i]= summary(analytics1[[i]])[3]
}
scoreSVM1 = data.frame(Topics,Precision, Recall, F1)

# RF
for(i in 1:10){
  Precision[i]= summary(analytics1[[i]])[4]
  Recall[i]= summary(analytics1[[i]])[5]
  F1[i]= summary(analytics1[[i]])[6]
}
scoreRF1 = data.frame(Topics,Precision, Recall, F1)

# TREE
for(i in 1:10){
  Precision[i]= summary(analytics1[[i]])[7]
  Recall[i]= summary(analytics1[[i]])[8]
  F1[i]= summary(analytics1[[i]])[9]
}
scoreTREE1 = data.frame(Topics,Precision, Recall, F1)

# Prepare plotting data for the first classification
plotSVM1 = data.frame(scoreSVM1[1],stack(scoreSVM1[2:4])) # for plotting only
plotRF1 = data.frame(scoreRF1[1],stack(scoreRF1[2:4])) # for plotting only
plotTREE1 = data.frame(scoreTREE1[1],stack(scoreTREE1[2:4])) # for plotting only
colnames(plotSVM1)= c("Topics","Measure","Type")
colnames(plotRF1)= c("Topics","Measure","Type")
colnames(plotTREE1)= c("Topics","Measure","Type")

# Plotting for the first classification
p1_SVM <- ggplot(data=plotSVM1, aes(x=Topics, y=Measure, group=Type, color=Type))+geom_line()+geom_point()+ ylim(0.6, 1)+ ggtitle("SVM Classification Baseline Performance")
p1_RF <- ggplot(data=plotRF1, aes(x=Topics, y=Measure, group=Type, color=Type))+geom_line()+geom_point()+ ylim(0.6, 1)+ ggtitle("RF Classification Baseline Performance")
p1_TREE <- ggplot(data=plotTREE1, aes(x=Topics, y=Measure, group=Type, color=Type))+geom_line()+geom_point()+ ylim(0.6, 1)+ ggtitle("TREE Classification Baseline Performance")



# Prepare performance results for the second classification
Precision = c(); Recall = c(); F1 = c()
Topics =  c("earn","acq","money-fx","grain","crude","trade","interest","ship","wheat","corn")

# SVM
for(i in 1:10){
  Precision[i]= summary(analytics2[[i]])[1]
  Recall[i]= summary(analytics2[[i]])[2]
  F1[i]= summary(analytics2[[i]])[3]
}
scoreSVM2 = data.frame(Topics,Precision, Recall, F1)

# RF
for(i in 1:10){
  Precision[i]= summary(analytics2[[i]])[4]
  Recall[i]= summary(analytics2[[i]])[5]
  F1[i]= summary(analytics2[[i]])[6]
}
scoreRF2 = data.frame(Topics,Precision, Recall, F1)

# TREE
for(i in 1:10){
  Precision[i]= summary(analytics2[[i]])[7]
  Recall[i]= summary(analytics2[[i]])[8]
  F1[i]= summary(analytics2[[i]])[9]
}
scoreTREE2 = data.frame(Topics,Precision, Recall, F1)

# Prepare plotting data for the second classification
plotSVM2 = data.frame(scoreSVM2[1],stack(scoreSVM2[2:4])) # for plotting only
plotRF2 = data.frame(scoreRF2[1],stack(scoreRF2[2:4])) # for plotting only
plotTREE2 = data.frame(scoreTREE2[1],stack(scoreTREE2[2:4])) # for plotting only
colnames(plotSVM2)= c("Topics","Measure","Type")
colnames(plotRF2)= c("Topics","Measure","Type")
colnames(plotTREE2)= c("Topics","Measure","Type")

# Plotting
p2_SVM <- ggplot(data=plotSVM2, aes(x=Topics, y=Measure, group=Type, color=Type))+geom_line()+geom_point()+ylim(0.6, 1)+ ggtitle("SVM Classification with TM Performance")
p2_RF <- ggplot(data=plotRF2, aes(x=Topics, y=Measure, group=Type, color=Type))+geom_line()+geom_point()+ylim(0.6, 1)+ ggtitle("RF Classification with TM Performance")
p2_TREE <- ggplot(data=plotTREE2, aes(x=Topics, y=Measure, group=Type, color=Type))+geom_line()+geom_point()+ylim(0.6, 1)+ ggtitle("TREE Classification with TM Performance")



# Comparison of Results
p1_SVM # SVM baseline
p1_RF # RF baseline
p1_TREE # TREE baseline

p2_SVM # SVM with topic model features
p2_RF # RF with topic model features
p2_TREE # TREE with topic model features



# Output the results 
write.csv(scoreSVM1, "Classification1_score1_SVM.csv")
write.csv(scoreRF1, "Classification1_score2_RF.csv")
write.csv(scoreTREE1, "Classification1_score3_TREE.csv")
write.csv(scoreSVM2, "Classification2_score1_SVM.csv")
write.csv(scoreRF2, "Classification2_score2_RF.csv")
write.csv(scoreTREE2, "Classification2_score3_TREE.csv")





# Clustering : create ten clusters using k-means, DBSCAN, k-medoids
#************

# Prepare data frame dfR
df = rbind(data.frame(inspect(dtmTrain)),data.frame(inspect(dtmTest)))
dfR = df[,colnames(df)%in%feaCluster] # select features with feaCluster
label=ncol(dfR)+1
dfR[,label]=docLabels[rownames(dfR),11] # assign topic labels to the label col
dfR = na.omit(dfR) # omit rows with NA topic labels
dim(dfR)



# Distribution of labels in the data
table(dfR[,label])



# k-means clustering
kmeans.results <- kmeans(dfR[-label],10)
resKMEANS <- table(dfR[,label], kmeans.results$cluster)



# DBSCAN clustering
ds <- dbscan(dfR[,-label],eps=0.42,MinPts=5)
resDBSCAN <- table(ds$cluster,dfR[,label])



# k-medoids clustering
pamk.result = pamk(dfR[,-label])
pamk.result$nc
resKMEDOIDS <- table(pamk.result$pamobject$clustering,dfR[,label])



# output results
write.csv(resKMEANS, "Clustering_result1_KMEANS.csv")
write.csv(resDBSCAN, "Clustering_result2_DBSCAN.csv")
write.csv(resKMEDOIDS, "Clustering_result3_KMEDOIDS.csv")





