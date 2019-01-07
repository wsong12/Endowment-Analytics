#Bi-plot
file<- read_excel("Desktop/endowment_2018_corrected_version.xls")
X<-file[,-1]
y<-file[,1]
X<-cbind(Row.Names=y,X)
data<-X[,-1]
row.names(data)<-make.names(X[,1],unique=TRUE)
school.pca=prcomp(data,center=TRUE,scale.=TRUE)
biplot(school.pca,choice=c(1,2),cex=0.7,xlab="PC1",ylab="PC2",col=c("blue","red"))

# Tri-plot
library(pca3d)
library(rgl)
scores<-school.pca$x
pca3d(scores,radius=1,col='red')
text3d(scores,texts=rownames(data),cex=0.7,col='blue')