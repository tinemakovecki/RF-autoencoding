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


# ======================================== #
#             GENERATING DATA              #
# ======================================== #

# === generate binary data sets === #

generate.binary.sets = function(n, d, n_parts, formulas) {
  # we generate n different data sets
  for (m in 1:n){
    # we generate the set by parts and merge them together
    formula = formulas[m]
    k = n_parts[m]
    
    for (i in 1:k) {
      # we save the conditions for the current part
      conditions = formula[i]
      ds = complete.bin.ds(d)
      
      # we convert conditions into a vector
      condition_vector = unlist(conditions, use.names=FALSE)
      
      # we have to use the appropriate formula to select a subset, then merge it with others
      for (cond in condition_vector) {
        # check if condition is a negation
        if (cond > 0){
          # cond is positive, the variable is true
          ds = ds[ds[,cond]]
        } else {
          # cond is negative, we take the negation of variable: abs(cond)
          ! ds = ds[ds[,-cond]]
        }
      }
      
      # merge the different set parts
      if (i == 1) {
        data_set = ds
      } else {
        data_set = rbind(data_set, ds)
      }
    }
    
    
    # we save the generated set
    base_path = "C:/Users/Tine/Desktop/RF-autoencoding/generated_data/generated_set_"
    path = paste(base_path, string(m), ".csv", sep="")
    write.table(DS, file=path, row.names=FALSE, col.names=FALSE)
  }
}


# EXAMPLE: generating one multi-part binary dataset

# dimensionality of the complete data set
ds = complete.bin.ds(10)
ds1 = complete.bin.ds(10)
ds2 = complete.bin.ds(10)
ds3 = complete.bin.ds(10)
print(pca.dim(ds))

# select rows from the complete data set using a logical formula
ds = ds[ds[, 1] & ds[, 2] & ds[, 3] & ! ds[, 4], ]
ds1 = ds1[ds1[, 1] & ds1[, 2] & ! ds1[, 3] & ds1[, 4], ]
ds2 = ds2[ds2[, 1] & ds2[, 2] & ! ds2[, 3] & ds2[, 9] & ds2[, 8] & ds2[, 7], ]
ds3 = ds3[ds3[, 1] & ds3[, 2] & ds3[, 3] & ! ds3[, 4], ]
# we select multiple formulas and merge the subsets together
DS = rbind(ds, ds1, ds2, ds3)

print(pca.dim(DS))


# === generate numeric data sets === #

ds = sample.hyperplanes.dataset(1024, 10, nhps=1)
print(pca.dim(ds))

ds = sample.hyperplanes.dataset(1024, 10, nhps=2)
print(pca.dim(ds))


# ======================================== #
#     PRINTING GENERATED DATA TO FILE      #
# ======================================== #

write.table(DS, file="C:/Users/Tine/Desktop/RF-autoencoding/generated_data/generated_set.csv", row.names=FALSE, col.names=FALSE)
