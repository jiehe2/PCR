#' A simulated fMRI dataset
#'
#' The data set consists of a simulated fMRI data and observations of three covariates for 100 individuals. 
#'
#' @format A data list with data and initial values:
#' \describe{
#'   \item{Y}{a response array with dimension T x P x n,where T represents the number of time points, 
#'             p refers to the number of brain regions, and n is the number of individuals} 
#'   \item{Cov}{the corresponding covariance matrix of Y for different individuals, which are obtained
#'              based on T observations.}
#'   \item{cov_all}{the observations of K dimensional covariates of 100 individuals, where continuous variables has 
#'                  already been scaled to [0,1], and the first column represents the intercept term in
#'                  the PCR model.}
#'   \item{B_0}{the initial value of matrices Bs in the model, which is an pxpxK array.}   
#'   \item{Theta_0}{the initial value of matrices Thetas in PCR model.}
#'   \item{Lambda_0}{the initial value of matrices Lambdas in the PCR model.}      
#' }
"example_data"

