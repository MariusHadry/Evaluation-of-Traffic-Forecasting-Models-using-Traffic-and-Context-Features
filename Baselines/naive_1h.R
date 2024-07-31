library(forecast)

repair <- function(tmp){
  
  x <- data.frame(timestamp=c("2022-03-09 09:00:00", paste("2022-03-09 ", 10:23, ":00:00", sep=""), paste("2022-03-10 0", 0:9, ":00:00", sep=""), paste("2022-03-10 ", 10:13, ":00:00", sep="")),
                  speed_percentage=(rep(tmp[(418-24):417,2], length.out=29) + rep(tmp[437:(437+23),2], length.out=29))/2, 
                  olr_code_id=rep(tmp[1,3],29))
  
  
  y <- data.frame(timestamp=paste("2022-06-01 0", 3:6, ":00:00", sep=""),
                  speed_percentage=(tmp[(2399-24):(2398-20),2] + tmp[(2399+20):(2398+24),2])/2, 
                  olr_code_id=rep(tmp[1,3],4))
  
  
  temp <- tmp[1:417,]
  temp <- rbind(temp, x, tmp[418:2398,], y, tmp[2399:2727,])
  
  return(temp)
  
}

data <- read.csv("data/data_1H_25p.csv",sep=";")



for(j in min(data$olr_code_id):max(data$olr_code_id)){
  
  fc <- c()
  test <- c()
  
  tmp <- data[which(data$olr_code_id==j),]
  
  #tmp <- repair(tmp)
  
  hist.end <- 24*99
  
  for(i in (hist.end+1):(nrow(tmp)-4)){
    hist <- ts(tmp[1:i,2],frequency = 7*24)
    
    test <- c(test, tmp[(i+1):(i+4),2])
    
    fc <- c(fc, as.vector(snaive(hist,h=4)$mean))
    
    
  }
  
  
  
  write.table(cbind(test,fc),file=paste("results_1h_25p/snaive/snaive_",sprintf("%02d",j),".csv",sep=""),sep=";", row.names = F)
  
  
  
}

