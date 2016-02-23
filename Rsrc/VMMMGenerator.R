require(circular)

path2Save <- "/home/angel/Escritorio/GaussianMixture.txt";

# Prior of mixture
pi <- c(0.377961, 0.40247, 0.219569);


# Centers for the Von mises distros
mu1 <- atan2(0.0118872, -0.999929)   
mu2 <- atan2(0.064114, 0.997943 );
mu3 <- atan2(-0.0269417, 0.999637);

mu <- c(mu1,mu2,mu3)

# concentration parameters
sigma <- c(3.47231, 1.34253, 106.296)


# Number of examples
N <- 1000

X <- matrix(nrow=N, ncol=1);

for(i in 1:N)
{
  # Chose the size of the sequence
  k <- sample( x = 1:3, size=1, prob=pi);
  
  X[i,]<-rvonmises(1, mu[k], sigma[k])
  
}

plot.circular(X, stack=TRUE, shrink=2)

#hist(X,breaks=50)


