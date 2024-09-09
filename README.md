Positive-definite covariance regression (PCR)
================
Jie He, Yumou Qiu and Xiao-hua Zhou

# PCR

<!-- badges: start -->
<!-- badges: end -->

The main goal of package PCR is to estimate coefficient matrices in
high-dimensional positive-definite covariance regression model.

The primary function for parameter estimation is `PCR_est`, which
directly return a list with each element be the parameter estimation
coefficient matrices in the positive-definite covariance regression
model. For illustration, we also add an example with simulated fMRI data
for 100 individuals.

## Installation

The package can be installed using devtools. Packages devtools, Rcpp,
RcppArmadillo are needed to be installed before installing PCR. 
```r
install.packages("devtools")
install.packages("Rcpp")
install.packages("RcppArmadillo")
```

Then you can install the PCR like so:

``` r
devtools::install_github("jiehe2/PCR")
library(PCR)
```

The package is now installed, and you may load it and use all functions
whenever you want.

## Example

This is a basic example which shows you how to solve a common problem:

``` r
library(PCR)
data(example_data)
Y = example_data$Cov
Z = example_data$cov_all
B_0 = example_data$B_0
Lambda_0 = example_data$Lambda_0
Theta_0 = example_data$Theta_0
mu = example_data$mu
combn_mat = example_data$combn_mat
lambda_mat = example_data$lambda_mat
lambda_0 = c(0.01,0.006,0.01,0.004)
lambda_final = lambda_mat
for(i in 1:dim(lambda_mat)[3]){
  lambda_final[ , ,i] = lambda_0[i]*lambda_mat[ , ,i]
}
results = PCR_est(Theta_0,B_0, Lambda_0, mu, lambda_final, 1e-3, Z, Y, 1000, 1e-5, combn_mat)
B_est = results$B_update
```

### Citations

If you end up using `PCR()` in a publication, please cite our paper, for
which this package was created: Jie He, Yumou Qiu & Xiao-hua Zhou (2024).
Positive-Definite Covariance on Scalar Regression Positive-Definite
Regularized Estimation for High Dimensional Covariance on Scalar
Regression.
