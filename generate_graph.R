library(utils)
library(readtext)
library(sfsmisc)

# ===================================== #

n_logs <- 15 # the number of experiments performed/logs to be read
results <- data.frame(matrix(ncol = 6, nrow = n_logs))

colnames(results) <- c("code_size", "original_set_dim", "average_forest_error", "forest_std", "average_nn_error", "nn_std")

# we assume all of the result files are from the same experiment
for (i in 1:n_logs) {
  # read logs to get data
  test_log <- read.csv(paste0("./graph_data/log", i, ".csv"), header=TRUE, sep=" ")
  # add data to result table
  results[i,] <- test_log[1,]
}

# TODO: calculate lines for upper and lower bound of standard deviation

# plot the forest results
plot(results$code_size, 
     results$average_forest_error, 
     type="l",
     main="Forest autoencoder reconstruction error by code size",
     xlab="code size",
     ylab="average reconsturction error")
with (
  data = results
  , expr = errbar(code_size, average_forest_error, average_forest_error+forest_std, average_forest_error-forest_std, add=T, pch=1, cap=.1)
)

# ====================================== #
# - - - > TEST < - - - #

# za shranjevanje slik
# cairo_pdf("graphs/test.pdf", height=8, width=18, family="CM", pointsize=24)

# par(mar=c(4,4,1,1))

plot(results$code_size, 
     results$average_forest_error,
     # sims[[which.min(bs$errs[is])]] ~ bs$dates, # fix dis
     # log="y",
     ylim=c(0.47, 0.6),
     #ylim=c(0, 50000),
     #xlab="", ylab="active cases",
     xlab="", ylab="",
     # col=rgb(0, 0.1, 1),
     type="l",
     lwd=3
)

# narisemo se zgornjo in spodnjo mejo s crtami
lines(results$code_size,
      results$average_forest_error - results$forest_std,
      type="c",
      lwd=2)

lines(results$code_size,
      results$average_forest_error + results$forest_std,
      type="c",
      lwd=2)


# dodamo barvni poligon
polygon(c(results$code_size, rev(results$code_size)), 
        c(results$average_forest_error + results$forest_std,
          rev(results$average_forest_error - results$forest_std)),
        border=NA, col=rgb(0.8, 0, 0.2, 0.5)
)

# printanje zaenkrat ni vazno
# print(results)

#for(i in 1:length(results)) {
  # lines(c(bs$dates[i], bs$dates[i]), c(1e-5, slo_data$I[i]), col=rgb(0, 0.1, 1), lwd=2)
  # lines(c(bs$dates[i], bs$dates[i]), c(log(1e-5), log(slo_data$I[i])), col=rgb(0, 0.1, 1), lwd=2)
  # lines(c(bs$dates[i], bs$dates[i]), c(1e-2, 1e+6), col=rgb(0, 0.1, 1), lwd=2)
#}
# lines(slo_data$I ~ dates[1:length(slo_data$I)])

dev.off() # WAT is dis?

