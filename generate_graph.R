library(utils)
library(readtext)

# ===================================== #

n_logs <- 15 # the number of experiments performed/logs to be read
results <- data.frame(matrix(ncol = 4, nrow = n_logs))

colnames(results) <- c("code_size", "original_set_dim", "average_forest_error", "average_nn_error")

# we assume all of the result files are from the same experiment
for (i in 1:n_logs) {
  # read logs to get data
  test_log <- read.csv(paste0("./graph_data/log", i, ".csv"), header=TRUE, sep=" ")
  # add data to result table
  results[i,] <- test_log[1,]
}

# plot the forest results
plot(results$code_size, 
     results$average_forest_error, 
     type="l",
     main="Forest autoencoder reconstruction error by code size",
     xlab="code size",
     ylab="average reconsturction error")
