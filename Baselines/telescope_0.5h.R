
library(forecast)

repair <- function(tmp){
  
  time_1 <- c()
  
  for (i in 9:23){
    for (j in c("00", "30")){
      time_1 <- c(time_1, paste("2022-03-09 ", sprintf("%02d",i), ":", j, ":00", sep=""))
    }
  }
  
  for (i in 0:13){
    for (j in c("00", "30")){
      time_1 <- c(time_1, paste("2022-03-10 ", sprintf("%02d",i), ":", j, ":00", sep=""))
    }
  }
  
  time_2 <- c()
  
  for (i in 2:6){
    for (j in c("00", "30")){
      time_2 <- c(time_2, paste("2022-06-01 ", sprintf("%02d",i), ":", j, ":00", sep=""))
    }
  }
  
  time_2 <- time_2[-1]
  
  
  # 834
  x <- data.frame(timestamp=time_1,
                  speed_percentage=(rep(tmp[(835-48):834,2], length.out=58) + rep(tmp[873:(873+47),2], length.out=58))/2, 
                  olr_code_id=rep(tmp[1,3],58))
  
  
  y <- data.frame(timestamp=time_2,
                  speed_percentage=(tmp[(4796-48):(4796-40),2] + tmp[(4796+39):(4796+47),2])/2, 
                  olr_code_id=rep(tmp[1,3],9))
  
  
  temp <- tmp[1:834,]
  temp <- rbind(temp, x, tmp[835:4795,], y, tmp[4796:5453,])
  
  return(temp)
  
}

data <- read.csv("data/data_0.5H_25p.csv",sep=";")



for(j in min(data$olr_code_id):max(data$olr_code_id)){
  
  fc <- c()
  test <- c()
  
  tmp <- data[which(data$olr_code_id==j),]
  
  #tmp <- repair(tmp)
  
  hist.end <- 48*99
  
  for(i in (hist.end+1):(nrow(tmp)-8)){
    hist <- ts(tmp[1:i,2],frequency = 7*48)
    
    
    
    
    a <- tryCatch(
      {
        as.vector(telescope.forecast(hist,h=8,plot = F)$mean)
      },
      error=function(cond) {
        rep(NA,8)
      }
    )
    
    fc <- c(fc, a)
    
    
    test <- c(test, tmp[(i+1):(i+8),2])
    
  }
  
  write.table(cbind(test,fc),file=paste("results_0.5h_25p/telescope/telescope_",sprintf("%02d",j),".csv",sep=""),sep=";", row.names = F)
  
  
}
j