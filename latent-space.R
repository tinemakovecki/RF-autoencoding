library(utils)

set.seed(42)

# dimensionality of the original space
p = 10

# dimensionality of the latent space, should be less than p
q = 5

# Upper bound for number of ones per row/column (see below)
qs = 3

# Number of examples
#  n = qC1 + qC2 + ... + qCqs
#  nCk is a binomial coefficient
# Note: p should be at most n
n = 0
for (qs.c in 1:qs) {
  n = n + choose(q, qs.c)
}
print(n)

# Data set matrix X = Z * W (n x p)
#  Z is the matrix of the data set represented in latent space (n x q)
#  W is the definition of the latent space (q x p)
# Illustrative instance:
#  Examples correspond to users
#  Variables of the original space are movies
#  Latent variables are movie genres


# Generate a random matrix Z
# Assumption: each row has at most qs ones, i.e.,
#  each user watches/likes movies from at most qs genres
#  each example "belongs" to at most qs latent subspace(s)
Z = matrix(0, nrow=n, ncol=q)
i = 1
for (qs.c in 1:qs) {
  c.qs.c = combn(1:q, qs.c)
  for (j in 1:ncol(c.qs.c)) {
    Z[i, c.qs.c[, j]] = 1
    i = i + 1
  }
}
print(Z)


# Generate a matrix W
# Assumption: each column has at most qs ones, i.e.,
#  each movie belongs to at most qs genres
#  each variable in the original space "corresponds" to at most qs latent variable(s)
W = t(Z)
W = W[, sample(1:nrow(Z), p)]
print(W)


X = Z %*% W
# turn X into binary/Boolean matrix
X[X != 0] = 1

print(X)
#print(max(X))

write.table(X, file="C:/Users/Tine/Desktop/RF-autoencoding/latent-space.csv", row.names=FALSE, col.names=FALSE)
