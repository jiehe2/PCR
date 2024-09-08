#
# Some useful keyboard shortcuts for package authoring:
#
#   Install Package:           'Cmd + Shift + B'
#   Check Package:             'Cmd + Shift + E'
#   Test Package:              'Cmd + Shift + T'


#'
#'
#'Positive-definite covariance regression (PCR)
#'
#'@importFrom Matrix sparseMatrix bdiag
#'@importFrom stats rbeta runif
#'@importFrom methods as
#'@importFrom utils combn
#'@importFrom MASS mvrnorm


#'@param n an integer represents the sample size
#'@param p an integer represents the dimension of the response
#'@param T_0 an iterger represents the replicated observations for the response
#'@param rho_1 correlation in matrix \eqn{B_1}
#'@param rho_2 correlation in matrix \eqn{B_2}
#'@param rho_0 the dependence between different observations of the response
#'@return a list of three variables, the covariates matrix, the response array and the true coefficient matirces \eqn{B_1} and \eqn{B_2}
#'\item{Z}{a n x p matrix refers to observations of the covariates}
#'\item{Y}{a T_0 x p x n array refers to the observations of the response}
#'\item{B_list}{a p x p x 2 array with each element be the coefficient matrix in the regression model}
#'@export
data_generation = function(n, p, T_0 , rho_1 = NULL, rho_2 = NULL, rho_0 = NULL){
  Z = matrix(1,n,2)
  Z[ ,2] = runif(n, min = 1e-3, max = 1-1e-3)
  B_1 = block_diag_CS(p, blockSize = p/10, rho = rho_1)
  group_list = list()
  for(i in 1:10){
    group_list[[i]] = c(((i-1)*(p/10)+1):(i*(p/10)))
  }
  group_pair = matrix(c(1,1,1,3,3,3,2,2,2,5,5,5,7,7,7,10,10,10),9,2,byrow = TRUE)
  B_2 = matrix(0,p,p)
  for(i in 1:nrow(group_pair)){
    pair_temp = group_pair[i, ]
    B_2[group_list[[pair_temp[1]]],group_list[[pair_temp[2]]]] = B_2[group_list[[pair_temp[2]]],group_list[[pair_temp[1]]]] = rho_2
  }
  Y = array(0,dim = c(T_0,p,n))
  for (i in 1:n){
    Y_temp = matrix(0,T_0,p)
    Sigma_temp = Z[i,1]*B_1 + Z[i,2]*B_2
    epsilon_temp = t(MASS::mvrnorm(T_0,mu = rep(0,p), Sigma = Sigma_temp))#T_0 time p
    for(j in 1:T_0){
      if(j == 1){
        Y_temp[j, ] = epsilon_temp[j, ]
      }else{
        Y_temp[j, ] = rho_0*Y_temp[j-1, ] + sqrt((1-rho_0^2)) * epsilon_temp[j, ]
      }
    }
    Y[ , ,i] = Y_temp
  }
  B_list = array(0,dim = c(p,p,2))
  B_list[, ,1] = B_1
  B_list[, ,2] = B_2
  return(list(Z = Z,
              Y = Y,
              B_list = B_list))
}


constraint_mat = function(q){
  mat_con = NULL
  num_0 = 0
  index_list = list()
  for(i in 1:q){
    comb_temp = t(combn(q,i))
    num_temp = nrow(comb_temp)
    for(j in 1:num_temp){
      index_list[[num_0+j]]=comb_temp[j, ]
    }
    num_0 = num_0 + num_temp
  }
  comb_mat = matrix(0,num_0,q)
  for(i in 1:num_0){
    index_0 = index_list[[i]]
    comb_mat[i,index_0] = 1
  }
  return(comb_mat)
}



block_diag_CS = function(p, blockSize = NULL, rho = NULL){
  m = floor(p/blockSize)
  blockCovMat = (1-rho) * diag(blockSize) + rho * matrix(1,blockSize,blockSize)
  List_mat = list(m)
  for (i in 1:m){
    List_mat[[i]] = blockCovMat
  }
  Covmat = bdiag(List_mat)
  Covmat = as.matrix(Covmat)
  return(Covmat)
}
