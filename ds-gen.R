library(gtools) # permutations
library(hitandrun) # samples


# ===== BINARY DATA SETS ===== #

# complete data set with .d. binary variables
complete.bin.ds = function(d) {
  return(permutations(2, d, v=0:1, repeats.allowed = T))
}


# ===== NUMERIC DATA SETS ===== #

# sample .n. points from a hyperplane in .d. dimensions
sample.random.hyperplane = function(n, d) {
  # sample the .d+1. hyporplane parameters w0 + w x = 0
  w = hypersphere.sample(d + 1, 1)

  # sample .n. points in .d-1. dimensions
  dsr = matrix(runif((d - 1) * n), nrow = n, ncol = d - 1)
  # calculate the .d. dimension taking into account the hyperplane
  ds = cbind(dsr, - cbind(rep(1, n), dsr) %*% w[1:d] / w[d + 1])

  return(ds)
}

# sample .n. points from .nhps. hyperplanes in .d. dimensions
sample.hyperplanes.dataset = function(n, d, nhps) {
  if (nhps == 1) return(sample.random.hyperplane(n, d))
  ds = sample.random.hyperplane(n - (n %/% nhps) * (nhps - 1), d)
  for (hp in 2:nhps) {
    ds = rbind(ds, sample.random.hyperplane(n %/% nhps, d))
  }
  return(ds)
}


# ===== PCA dimensionality of a DATA SET ===== #

# minimal number of the leftmost elements of .x. the sum of which is more than .th.
sum.larger.than = function(x, th) {
  sum = 0
  for (i in 1:length(x)) {
    sum = sum + x[i]
    if (sum > th) return(i)
  }
  return(-1)
}

# number of principal components that preserve more than .th. of the total .ds. variance
pca.dim = function(ds, th = 0.95) {
  pca = prcomp(ds)
  scree = pca$sdev / sum(pca$sdev)
  print(scree)
  return(sum.larger.than(scree, th))
}


# === generate binary data sets === #

# dimensionality of the complete data set
ds = complete.bin.ds(10)
ds1 = complete.bin.ds(10)
ds2 = complete.bin.ds(10)
ds3 = complete.bin.ds(10)
print(pca.dim(ds))

# select rows from the complete data set using a logical formula
ds = ds[ds[, 1] & ds[, 2] & ds[, 3] & ! ds[, 4], ]
ds1 = ds1[ds1[, 1] & ds1[, 2] & ! ds1[, 3] & ds1[, 10], ]
ds2 = ds2[ds2[, 1] & ds2[, 2] & ! ds2[, 3] & ds2[, 9], ]
ds3 = ds3[ds3[, 1] & ds3[, 2] & ds3[, 3] & ! ds3[, 4], ]
# we select multiple formulas and merge the subsets together
DS = rbind(ds, ds1, ds2, ds3)

print(pca.dim(DS))


# === generate numeric data sets === #

ds = sample.hyperplanes.dataset(1024, 10, nhps=1)
print(pca.dim(ds))

ds = sample.hyperplanes.dataset(1024, 10, nhps=2)
print(pca.dim(ds))

# ===== PRINTING GENERATED DATA TO FILE ===== #

write.table(DS, file="generated_set.csv", row.names=FALSE, col.names=FALSE)
