#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#include <Rcpp.h>
//#include <R.h>
//#include <cmath>

//using namespace std;
using namespace Rcpp;
using namespace arma;
//using namespace stats;


//'@importFrom Rcpp evalCpp
//'@useDynLib PCR
//'@export PCR_est



arma::mat S_function(arma::mat& X, arma::mat& tau){
  int p = X.n_rows;
  arma::mat S = zeros<mat>(p,p);
  for (int i = 0; i < p; i++){
    for (int j = 0; j < p; j++){
      if (i == j){
        S(i,j) = X(i,j);
      }else{
        double value_temp = abs(X(i,j))-tau(i,j);
        if (value_temp > 0){
          S(i,j) = sign(X(i,j))*value_temp;
        }else{
          S(i,j) = 0;
        }
      }
    }
  }
  return S;
}




arma::mat S_function_all(arma::mat& X, arma::mat& tau){
  int p = X.n_rows;
  arma::mat S = zeros<mat>(p,p);
  for (int i = 0; i < p; i++){
    for (int j = 0; j < p; j++){
      double value_temp = abs(X(i,j))-tau(i,j);
      if (value_temp > 0){
        S(i,j) = sign(X(i,j))*value_temp;
      }else{
        S(i,j) = 0;
      }
    }
  }
  return S;
}



arma::mat eigen_proj(arma::mat& X, double epsilon){
  vec eigval;
  arma::mat eigvec;
  arma::mat X_0 = 0.5*(X + X.t());
  eig_sym(eigval,eigvec,X_0);
  int p = eigval.n_elem;
  for (int i = 0; i < p; i++){
    if (eigval[i] < epsilon) eigval[i] = epsilon;
  }
  arma::mat X_update =  eigvec * diagmat(eigval) * eigvec.t();
  return X_update;
}



int neg_eigen_num(mat& X){
  vec eigval;
  arma::mat eigvec;
  eig_sym(eigval,eigvec,X);
  uvec index = find(eigval < 0);
  int count_num = index.n_elem;
  return count_num;
}



double L_1_norm_offdiag(arma::mat& A){
  int p = A.n_cols;
  double norm_A = 0.0;
  for (int i = 0; i < p; i++){
    for (int j = 0; j < p; j++){
      if (i != j){
        norm_A += abs(A(i,j));
      }
    }
  }
  return norm_A;
}



double L_1_norm_all(arma::mat& A){
  int p = A.n_cols;
  double norm_A = 0.0;
  for (int i = 0; i < p; i++){
    for (int j = 0; j < p; j++){
      norm_A += abs(A(i,j));
    }
  }
  return norm_A;
}



double inner_prod(arma::mat& A, arma::mat& B){
  int p = A.n_cols;
  double prod_final = 0.0;
  for (int i = 0; i < p; i++){
    for (int j = 0; j < p; j++){
      prod_final += A(i,j) * B(i,j);
    }
  }
  return prod_final;
}




arma::mat mulnorm_gen(int n, arma::mat K){
  int p = K.n_rows;
  vec mu = zeros<vec>(p);
  arma::mat X = mvnrnd(mu, K, n);
  return X;
}




double bias_criterion(arma::mat& A, arma::mat& B){
  arma::mat C = A - B;
  double norm = trace(C.t() * C);
  return pow(norm,0.5);
}



double F_norm(mat& A){
  double norm = trace(A.t() * A);
  return pow(norm,0.5);
}




arma::uvec intersect_cpp(arma::uvec& a, arma::uvec& b){
  int p_1 = a.size();
  int p_2 = b.size();
  arma::uvec results;
  arma::uvec results_0 = zeros<uvec>(p_1);
  for (int i = 0; i < p_1; i++){
    for (int j = 0; j < p_2; j++){
      results_0[i] += (b[j] == a[i]);
    }
  }
  arma::uvec index = find(results_0 == 1);
  int l = index.size();
  if (l == 0){
    results.reset();
  }else{
    results = a.elem(index);
  }
  return results;
}



double prediction_error(arma::cube& Y_test, arma::mat& Z_test, arma::cube& B_est){
  int n_test = Y_test.n_slices;
  int K = Z_test.n_cols;
  int p = Y_test.n_cols;
  double error = 0.0;
  for (int i = 0; i < n_test; i++){
    arma::mat sample_covariance = Y_test.slice(i);
    arma::mat sigma_pred = zeros<mat>(p,p);
    for (int j = 0; j < K; j++){
      arma::mat B_temp = B_est.slice(j);
      sigma_pred += Z_test(i,j) * B_temp;
    }
    error += (1.0/n_test) * bias_criterion(sample_covariance,sigma_pred);
  }
  return error;
}



Rcpp::List initial_B(arma::cube& Y, arma::mat& Z){
  arma::mat y_0 = Y.slice(0);
  int K = Z.n_cols;
  int p = y_0.n_cols;
  int n = Z.n_rows;
  Rcpp::List results(K);
  for (int k = 0; k < K; k++){
    results[k] = zeros<mat>(p,p);
  }
  arma::mat Z_0 = Z.t() * Z;
  Z_0 = 0.5*(Z_0 + Z_0.t());
  arma::mat coef = inv(Z_0 + 1e-5 * eye(K,K)) * Z.t();
  for (int i = 0; i < p; i++){
    for (int j = i; j < p; j++){
      arma::vec y_temp = zeros<vec>(n);
      for (int t = 0; t < n; t++){
        arma::mat y_mat_temp = Y.slice(t);
        y_temp[t] = y_mat_temp(i,j);
      }
      arma::vec result_temp = coef * y_temp;
      for (int k = 0; k < K; k++){
        arma::mat r_temp = zeros<mat>(p,p);
        r_temp(i,j) = result_temp[k];
        arma::mat r_temp_0 = results[k];
        results[k] = r_temp_0 + r_temp;
      }
    }
  }
  for (int i = 0; i < K; i++){
    arma::mat r_temp_1 = results[i];
    arma::mat r_temp_2 = r_temp_1.t();
    r_temp_2.diag() = zeros<vec>(p);
    r_temp_1 += r_temp_2;
    results[i] =  r_temp_1;
  }
  return results;
}




double spectral_norm(arma::mat& X){
  arma::vec eigval;
  arma::mat eigvec;
  arma::mat X_0 = X.t() * X;
  eig_sym(eigval,eigvec,X_0);
  int p = eigval.n_elem;
  double s_norm = eigval[0];
  for (int i = 1; i < p; i++){
    if (eigval[i] > s_norm){
      s_norm = eigval[i];
    }
  }
  return pow(s_norm,0.5);
}




Rcpp::List error_rate(arma::cube& B, arma::mat& Z, arma::cube& B_est){
  arma::mat B_00 = B.slice(0);
  int p = B_00.n_cols;
  int K = Z.n_cols;
  int n = Z.n_rows;
  arma::vec error = zeros<vec>(n);
  arma::vec sp_norm = zeros<vec>(n);
  arma::vec neg_eigen = zeros<vec>(n);
  arma::mat error_mat = zeros<mat>(p,p);
  for (int i = 0; i < n; i++){
    arma::mat sigma_true = zeros<mat>(p,p);
    arma::mat sigma_est = zeros<mat>(p,p);
    for (int k = 0; k < K; k++){
      arma::mat B_true_temp = B.slice(k);
      arma::mat B_est_temp = B_est.slice(k);
      sigma_true += Z(i,k) * B_true_temp;
      sigma_est += Z(i,k) * B_est_temp;
    }
    error_mat += (1.0/n) * (sigma_est - sigma_true);
    error[i] = bias_criterion(sigma_true,sigma_est)/F_norm(sigma_true);
    neg_eigen[i] = neg_eigen_num(sigma_est);
    mat differ = sigma_true - sigma_est;
    sp_norm[i] = (1.0*spectral_norm(differ))/spectral_norm(sigma_true);
  }
  return Rcpp::List::create(Named("error") = error,
                            Named("sp_norm") = sp_norm,
                            Named("error_mat") = error_mat,
                            Named("neg_eigen") = neg_eigen);
}




arma::mat mat_A_0(arma::mat& z, arma::cube& B, arma::cube& y){
  arma::mat y_0 = y.slice(0);
  int n = z.n_rows;
  int p = y_0.n_cols;
  int K = z.n_cols;
  arma::mat A_mat = zeros<mat>(p,p);
  for (int i = 0; i < n; i++){
    arma::mat y_temp = y.slice(i);
    arma::vec coef_temp = z.row(i).t();
    arma::mat B_comb = zeros<mat>(p,p);
    for (int j = 1; j < K; j++){
      arma::mat B_temp = B.slice(j);
      B_comb += coef_temp[j] * B_temp;
    }
    A_mat += y_temp - B_comb;
  }
  A_mat = (1.0/n) * A_mat;
  return A_mat;
}



arma::mat mat_A_k(arma::mat& z, arma::cube& B, int k, arma::cube& y){
  arma::mat y_0 = y.slice(0);
  int n = z.n_rows;
  int p = y_0.n_cols;
  int K = z.n_cols;
  arma::mat A_mat = zeros<mat>(p,p);
  arma::vec z_temp = z.col(k);
  for (int i = 0; i < n; i++){
    arma::mat y_temp = y.slice(i);
    arma::vec coef_temp = z.row(i).t();
    coef_temp[k] = 0.0;
    arma::mat B_comb = zeros<mat>(p,p);
    for (int j = 0; j < K; j++){
      mat B_temp = B.slice(j);
      B_comb += coef_temp[j] * B_temp;
    }
    A_mat += z_temp[i] * (y_temp - B_comb);
  }
  A_mat = (1.0/n) * A_mat;
  return A_mat;
}



arma::mat Theta_B_Lambda(arma::cube& Theta, arma::cube& B, arma::cube& Lambda, arma::vec& mu, arma::mat& combn_mat){
  int K = B.n_slices;
  int con_num = combn_mat.n_rows;
  arma::mat Theta_temp = Theta.slice(0);
  int p = Theta_temp.n_rows;
  arma::mat Lambda_final = Lambda.slice(0);
  double coef_0_temp = 1.0/mu[0];
  arma::mat Theta_final = Theta.slice(0);
  Theta_final = coef_0_temp * Theta_final;
  for(int i = 0; i < con_num; i++){
    double mu_temp = 1.0/mu[i+1];
    Theta_temp = Theta.slice(i+1);
    arma::mat Lambda_temp = Lambda.slice(i+1);
    arma::rowvec comb_temp = combn_mat.row(i);
    arma::mat comb_B = zeros<mat>(p,p);
    for(int j = 1; j < K; j++){
      arma::mat B_temp = B.slice(j);
      comb_B += comb_temp[j-1] * B_temp;
    }
    Theta_final += mu_temp * (Theta_temp - comb_B);
    Lambda_final += Lambda_temp;
  }
  arma::mat results = Theta_final - Lambda_final;
  return results;
}




Rcpp::List Theta_B_Lambda_k(arma::cube& Theta, arma::cube& B, arma::cube& Lambda, arma::vec& mu, arma::mat& combn_mat, arma::vec& z_temp, int index){
  int K = B.n_slices;
  int p = B.n_cols;
  int con_num = combn_mat.n_rows;
  arma::vec coef_vec = zeros<vec>(con_num);
  arma::mat Theta_temp = Theta.slice(0);
  double mu_temp = 0.0;
  arma::mat comb_B = B.slice(0);
  arma::mat Lambda_final = zeros<mat>(p,p);
  arma::mat Lambda_temp = zeros<mat>(p,p);
  arma::mat Theta_final = zeros<mat>(p,p);
  for(int i = 0; i < con_num; i++){
    mu_temp = 1.0/mu[i+1];
    Theta_temp = Theta.slice(i+1);
    Lambda_temp = Lambda.slice(i+1);
    arma::rowvec comb_temp = combn_mat.row(i);
    double a_temp = comb_temp[index-1];
    comb_temp[index-1] = 0.0;
    comb_B = B.slice(0);
    for(int j = 1; j < K; j++){
      arma::mat B_temp = B.slice(j);
      comb_B += comb_temp[j-1] * B_temp;
    }
    Theta_final += (mu_temp * a_temp) * (Theta_temp - comb_B);
    Lambda_final += a_temp * Lambda_temp;
    coef_vec[i] = (mu_temp*a_temp);
  }
  arma::mat results = Theta_final - Lambda_final;
  double coef_sum = sum(coef_vec);

  return Rcpp::List::create(Named("results") = results,
                            Named("coef_sum") = coef_sum);
}



Rcpp::List update_function(arma::cube& Theta, arma::cube& B, arma::cube& Lambda, arma::vec& mu, arma::cube& lambda, double epsilon,
                           arma::mat& Z, arma::cube& Y, arma::mat& combn_mat){
  int K = Z.n_cols;
  int con_num = combn_mat.n_rows;
  arma::cube Theta_1 = Theta;
  arma::cube B_1 = B;
  arma::cube Lambda_1 = Lambda;
  arma::mat B_temp = B_1.slice(0);
  int p = B_temp.n_rows;
  arma::mat Theta_update = zeros<mat>(p,p);
  arma::mat B_update = zeros<mat>(p,p);
  arma::mat Lambda_update = zeros<mat>(p,p);
  arma::vec mu_1 = mu;
  arma::mat B_0 = B_1.slice(0);
  double mu_temp_0 = mu[0];
  arma::mat Theta_0 = Theta_1.slice(0);
  arma::mat Lambda_0 = Lambda_1.slice(0);
  mu_1[0] = 1.0/(mu_temp_0);
  Theta_update = B_0 + (mu_temp_0 * Lambda_0);
  Theta_update = eigen_proj(Theta_update,epsilon);
  Theta_1.slice(0) = Theta_update;
  double mu_temp = 0.0;
  for (int i = 0; i < con_num; i++){
    B_0 = B_1.slice(0);
    mu_temp = mu[i+1];
    mu_1[i+1] = 1.0/mu_temp;
    Theta_0 = Theta_1.slice(i+1);
    Lambda_0 = Lambda_1.slice(i+1);
    arma::rowvec coef_temp_0 = combn_mat.row(i);
    for(int j = 1; j < K; j++){
      arma::mat B_T_temp = B_1.slice(j);
      B_0 += coef_temp_0[j-1] * B_T_temp;
    }
    Theta_update = B_0 + mu_temp * Lambda_0 ;
    Theta_update = eigen_proj(Theta_update,epsilon);
    Theta_1.slice(i+1) = Theta_update;
  }
  for(int i = 0; i < K; i++){
    arma::mat lambda_0 = lambda.slice(i);
    if(i == 0){
      double coef_0 = 1 + sum(mu_1);
      arma::mat A_temp_0 = mat_A_0(Z,B_1,Y);
      arma::mat S_matrix_0 = Theta_B_Lambda(Theta_1,B_1,Lambda_1,mu,combn_mat);
      S_matrix_0 += A_temp_0;
      B_update = (1.0/coef_0)*S_function(S_matrix_0,lambda_0);
    }else{
      arma::vec z_temp = Z.col(i);
      double z_0 = mean(z_temp % z_temp);
      Rcpp::List B_coef_list = Theta_B_Lambda_k(Theta_1,B_1,Lambda_1,mu,combn_mat,z_temp,i);
      double mu_star_0 = B_coef_list["coef_sum"];
      double mu_star = z_0 + mu_star_0;
      lambda_0 = (1.0/mu_star) * lambda_0;
      arma::mat A_0 = mat_A_k(Z,B,i,Y);
      arma::mat comb_A_temp = B_coef_list["results"];
      comb_A_temp += A_0;
      arma::mat S_matrix = (1.0/mu_star) * comb_A_temp;
      B_update = S_function_all(S_matrix,lambda_0);
    }
    B_1.slice(i) = B_update;
  }
  Lambda_0 = Lambda_1.slice(0);
  Theta_update = Theta_1.slice(0);
  B_0 = B_1.slice(0);
  mu_temp = mu[0];
  Lambda_update = Lambda_0 - (1.0/mu_temp)*(Theta_update - B_0);
  Lambda_1.slice(0) = Lambda_update;
  for (int i = 0; i < con_num; i++){
    B_0 = B_1.slice(0);
    mu_temp = mu[i+1];
    Lambda_0 = Lambda_1.slice(i+1);
    Theta_update = Theta_1.slice(i+1);
    rowvec coef_temp_1 = combn_mat.row(i);
    for(int j = 1; j < K; j++){
      B_temp = B_1.slice(j);
      B_0 += coef_temp_1[j-1] * B_temp;
    }
    Lambda_update = Lambda_0 - (1.0/mu_temp) *(Theta_update - B_0);
    Lambda_1.slice(i+1) = Lambda_update;
  }

  return Rcpp::List::create(Named("B") = B_1,
                            Named("Theta") = Theta_1,
                            Named("Lambda") = Lambda_1);
}




double objective_function(arma::cube& Theta, arma::cube& B, arma::cube& Lambda, arma::cube& Y, arma::mat& Z, arma::cube& lambda, arma::vec& mu, arma::mat& combn_mat){
  arma::mat y_0 = Y.slice(0);
  int n = Z.n_rows;
  int K = Z.n_cols;
  int com_num = combn_mat.n_rows;
  int p = y_0.n_cols;
  arma::mat B_temp = B.slice(0);
  double F_norm_1 = 0.0;
  for (int i = 0; i < n; i++){
    arma::mat comb_temp = zeros<mat>(p,p);
    arma::mat Y_temp = Y.slice(i);
    arma::vec z_temp = Z.row(i).t();
    for (int j = 0; j < K; j++){
      B_temp = B.slice(j);
      comb_temp += z_temp[j] * B_temp;
    }
    arma::mat diff = comb_temp - Y_temp;
    double F_norm_1_temp = F_norm(diff);
    F_norm_1 += pow(F_norm_1_temp,2);
  }
  double B_norm = 0.0;
  for (int k = 0; k < K; k++){
    B_temp = B.slice(k);
    arma::mat lambda_0 = lambda.slice(k);
    lambda_0 = lambda_0 % B_temp;
    if(k == 0){
      B_norm += L_1_norm_offdiag(lambda_0);
    }else{
      B_norm += L_1_norm_all(lambda_0);
    }
  }
  double inner_prod_TB = 0.0;
  double T_B_norm = 0.0;
  arma::mat B_0_0 = B.slice(0);
  arma::mat T_B_temp = zeros<mat>(p,p);
  arma::mat Lambda_0 = Lambda.slice(0);
  arma::mat Theta_0 = Theta.slice(0);
  T_B_temp = Theta_0 - B_0_0;
  inner_prod_TB += inner_prod(Lambda_0,T_B_temp);
  double T_B_norm_0 = F_norm(T_B_temp);
  T_B_norm += (1.0/(2*mu[0])) * pow(T_B_norm_0,2);
  for (int i = 0; i < com_num; i++){
    B_0_0 = B.slice(0);
    Lambda_0 = Lambda.slice(i+1);
    Theta_0 = Theta.slice(i+1);
    rowvec coef_temp = combn_mat.row(i);
    for(int j = 1; j < K; j++){
      B_temp = B.slice(j);
      B_0_0 += coef_temp[j-1] * B_temp;
    }
    T_B_temp = Theta_0 - B_0_0;
    inner_prod_TB += inner_prod(Lambda_0,T_B_temp);
    T_B_norm_0 = F_norm(T_B_temp);
    T_B_norm += (1.0/(2*mu[i+1])) * pow(T_B_norm_0,2);
  }
  double obj_func = 1.0/(2*n) * F_norm_1 + B_norm - inner_prod_TB + T_B_norm;
  return obj_func;
}

//'Parameter estimation for positive-definite covariance regression model (PCR)
//'@param Theta_0 p x p x q array.
//'@param B_0  p x p x K coefficients array.
//'@param Lambda_0 p x p x q array for Lagrange multipliers.
//'@param mu K x 1 vector.
//'@param lambda p x p x q thresholding array.
//'@param epsilon positive number for positive-definite projection.
//'@param Z n x q covariate matrix.
//'@param Y p x p x n array, each element represent the sample covariance matrix of an individual.
//'@param iter_col an integer represents the maximize iteration time for convergence.
//'@param error_col a positive number refers to the tolerence value for algorithm convergence.
//'@param combn_mat a \eqn{2^q-1} x (q-1) matrix with each row be the coefficients of \eqn{B_1},...,\eqn{B_q} in constraints.
//'@return a list of six variables
//'\describe{
//' \item{B_update}{a p x p x K array refers to the estimation of coefficient matrices B}
//' \item{Theta_update}{a p x p x K array represents the estimation of Theta}
//' \item{Lambda_update}{a p x p x K array represents the estimation of Lambda}
//' \item{iter_num}{a integer represents the iteration time}
//' \item{convergence}{convergence indicator, the value is 1 if the algorithm convergences, or the value is 0}
//' \item{error}{the absolute value for the difference between the value of objective function at two iterations}
//'}
//'@examples
//'data(example_data)
//'Y = example_data$Cov
//'Z = example_data$cov_all
//'B_0 = example_data$B_0
//'Lambda_0 = example_data$Lambda_0
//'Theta_0 = example_data$Theta_0
//'mu = example_data$mu
//'combn_mat = example_data$combn_mat
//'lambda_mat = example_data$lambda_mat
//'lambda_0 = c(0.01,0.006,0.01,0.004)
//'lambda_final = lambda_mat
//'for(i in 1:dim(lambda_mat)[3]){
//'  lambda_final[ , ,i] = lambda_0[i]*lambda_mat[ , ,i]
//'}
//'results = PCR_est(Theta_0,B_0, Lambda_0, mu, lambda_final, 1e-3, Z, Y, 1000, 1e-5, combn_mat)
//'B_est = results$B_update
//'@export
//[[Rcpp::export]]
Rcpp::List PCR_est(arma::cube& Theta_0, arma::cube& B_0, arma::cube& Lambda_0, arma::vec& mu, arma::cube& lambda, double epsilon,
                    arma::mat& Z, arma::cube& Y, int iter_col, double error_col, arma::mat& combn_mat){
  int iter_num = 0;
  double error = 0.0;
  arma::cube Theta_update = Theta_0;
  arma::cube B_update = B_0;
  arma::cube Lambda_update = Lambda_0;
  arma::cube Theta_1 = Theta_0;
  arma::cube B_1 = B_0;
  arma::cube Lambda_1 = Lambda_0;
  arma::vec function_value = zeros<vec>(1);
  double func_value_0 = objective_function(Theta_1,B_1,Lambda_1,Y,Z,lambda,mu,combn_mat);
  while(iter_num < iter_col){
    iter_num += 1;
    List update_list = update_function(Theta_1,B_1,Lambda_1,mu,lambda,epsilon,Z,Y,combn_mat);
    arma::cube Theta_update_temp = update_list["Theta"];
    arma::cube B_update_temp = update_list["B"];
    arma::cube Lambda_update_temp = update_list["Lambda"];
    double func_value_1 = objective_function(Theta_update_temp,B_update_temp,Lambda_update_temp,Y,Z,lambda,mu,combn_mat);
    error = (1.0*abs(func_value_1-func_value_0))/abs(func_value_0);
    Theta_update = Theta_update_temp;
    B_update = B_update_temp;
    Lambda_update = Lambda_update_temp;
    if(error < error_col){
      break;
    }else{
      Theta_1 = Theta_update;
      B_1 = B_update;
      Lambda_1 = Lambda_update;
      func_value_0 = func_value_1;
    }
  }
  int convergence =  1 * (iter_num < iter_col);

  return Rcpp::List::create(Named("B_update") = B_update,
                            Named("Theta_update") = Theta_update,
                            Named("Lambda_update") = Lambda_update,
                            Named("iter_num") = iter_num,
                            Named("convergence") = convergence,
                            Named("error") = error);
}



double min_cpp(vec& a){
  int p = a.size();
  double min = a[0];
  for (int i = 0; i < p; i++){
    if (a[i] < min) min = a[i];
  }
  return min;
}



arma::vec Lambda_selection_parallel(arma::mat& Z, arma::cube& Y, arma::vec& mu, double epsilon, int iter_col, double error_col,
                                    arma::mat& lambda_set, arma::cube& lambda_zero, arma::vec& group_label, arma::mat& combn_mat){
  arma::vec uni_group = unique(group_label);
  int G = uni_group.size();
  int K = Z.n_cols;
  arma::mat Y_0 = Y.slice(0);
  int p = Y_0.n_cols;
  int num_lam = lambda_set.n_rows;
  int com_num = combn_mat.n_rows;
  arma::vec error = zeros<vec>(num_lam);
  arma::cube lambda_selected = zeros<cube>(p,p,K);
  for(int j = 0; j < num_lam; j++){
    arma::vec lambda_temp = lambda_set.row(j).t();
    for(int u = 0; u < K; u++){
      arma::mat lambda_zero_temp = lambda_zero.slice(u);
      lambda_selected.slice(u) = lambda_temp[u]*lambda_zero_temp;
    }
    arma::vec error_temp = zeros<vec>(G);
    for (int i = 0; i < G; i++){
      arma::uvec index_est = find(group_label != (i+1));
      arma::uvec index_test = find(group_label == (i+1));
      int n_est = index_est.size();
      int n_test = index_test.size();
      arma::cube Y_est = zeros<cube>(p,p,n_est);
      arma::cube Y_test = zeros<cube>(p,p,n_test);
      for(int t = 0; t < n_est; t++){
        int index_1 = index_est[t];
        Y_est.slice(t) = Y.slice(index_1);
      }
      for(int v = 0; v < n_test; v++){
        int index_2 = index_test[v];
        Y_test.slice(v) = Y.slice(index_2);
      }
      arma::mat Z_est = Z.rows(index_est);
      arma::mat Z_test = Z.rows(index_test);
      Rcpp::List initial_list = initial_B(Y_est,Z_est);
      arma::cube B_0 = zeros<cube>(p,p,K);
      for (int s = 0; s < K; s++){
        arma::mat init_temp = initial_list[s];
        B_0.slice(s) = init_temp;
      }
      arma::cube Theta_0 = zeros<cube>(p,p,(com_num+1));
      Theta_0.slice(0) = B_0.slice(0);
      for(int s = 0; s < com_num; s++){
        arma::mat B_0_temp = B_0.slice(0);
        arma::rowvec coef_temp = combn_mat.row(s);
        for(int t = 1; t < K; t++){
          arma::mat B_t_temp = B_0.slice(t);
          B_0_temp += coef_temp[t-1] * B_t_temp;
        }
        Theta_0.slice(s+1) = B_0_temp;
      }
      arma::cube Lambda_0 = zeros<cube>(p,p,(com_num+1));
      Rcpp::List estimation_list = PCR_est(Theta_0,B_0,Lambda_0,mu,lambda_selected,epsilon,Z_est,Y_est,iter_col,error_col,combn_mat);
      cube B_est = estimation_list["B_update"];
      double error_temp_0 = prediction_error(Y_test,Z_test,B_est);
      error_temp[i] = error_temp_0;
    }
    error[j] = mean(error_temp);
  }
  arma::uvec index_0 = find(error == min_cpp(error));
  arma::vec lambda_sel = lambda_set.row(index_0[0]).t();
  arma::vec error_final_vec = error.elem(index_0);
  arma::vec error_vec = zeros<vec>(1);
  error_vec[0] = error_final_vec[0];
  int index_temp = index_0[0];
  index_temp += 1;
  arma::vec index_vec = zeros<vec>(1);
  index_vec[0] = index_temp;
  arma::vec results_final_0 = join_cols(lambda_sel,error_vec);
  arma::vec results_final = join_cols(results_final_0,index_vec);
  return(results_final);
}



arma::mat overlap_bolck_diagonal(int p,Rcpp::List group_list, double rho_1,double rho_2, int num_group){
  arma::mat cov_mat = zeros<mat>(p,p);
  for (int i = 0; i < num_group; i++){
    arma::uvec index_temp = group_list[i];
    int num_temp = index_temp.n_elem;
    arma::mat cov_temp = rho_1*ones<mat>(num_temp,num_temp);
    cov_temp.diag() = ones<vec>(num_temp);
    cov_mat.submat(index_temp-1,index_temp-1) = cov_temp;
    if (i < num_group-1){
      arma::uvec index_temp_1 = zeros<uvec>(1);
      index_temp_1[0] = index_temp[num_temp-1];
      arma::uvec index_temp_2 = group_list[i+1];
      int num_temp_1 = index_temp_2.n_elem;
      arma::mat cov_temp_1 = rho_2*ones<mat>(num_temp_1,1);
      cov_mat.submat(index_temp_1-1,index_temp_2-1) = cov_temp_1.t();
      cov_mat.submat(index_temp_2-1,index_temp_1-1) = cov_temp_1;
    }
  }
  return cov_mat;
}




Rcpp::List error_mat(arma::cube& B, arma::cube& B_est, arma::mat& Z){
  int K = Z.n_cols;
  int n = Z.n_rows;
  arma::mat B_0 = B.slice(0);
  int p = B_0.n_cols;
  arma::mat error_abs = zeros<mat>(p,p);
  arma::mat error = zeros<mat>(p,p);
  for (int i = 0; i < n; i++){
    arma::mat Sigma_true = zeros<mat>(p,p);
    arma::mat Sigma_est = zeros<mat>(p,p);
    for(int j = 0; j < K; j++){
      arma::mat B_temp = B.slice(j);
      arma::mat B_est_temp = B_est.slice(j);
      Sigma_true += Z(i,j) * B_temp;
      Sigma_est += Z(i,j) * B_est_temp;
    }
    error_abs += (1.0/n) * abs(Sigma_true - Sigma_est);
    error += (1.0/n) * (Sigma_true - Sigma_est);
  }
  return Rcpp::List::create(Named("error_abs") = error_abs,
                            Named("error") = error);
}




arma::cube Theta_init(arma::cube& B, arma::mat& combn_mat){
  arma::mat B_0 = B.slice(0);
  int p = B_0.n_rows;
  int K = B.n_slices;
  int con_num = combn_mat.n_rows;
  arma::cube results_final = zeros<cube>(p,p,(con_num+1));
  results_final.slice(0) = B_0;
  for(int i = 0; i < con_num; i++){
    arma::rowvec coef_vec = combn_mat.row(i);
    B_0 = B.slice(0);
    for(int j = 1; j < K; j++){
      arma::mat B_temp = B.slice(j);
      B_0 += coef_vec[j-1] * B_temp;
    }
    results_final.slice(i+1) = B_0;
  }
  return results_final;
}



arma::cube  Var_resid(arma::cube& Y, arma::mat& Z, arma::cube& B){
  arma::mat y_0 = Y.slice(0);
  int p = y_0.n_cols;
  int n = Z.n_rows;
  int K = Z.n_cols;
  arma::cube resid_all = zeros<cube>(p,p,n);
  for(int i = 0; i < n; i++){
    arma::mat Y_temp = Y.slice(i);
    arma::vec Z_temp = Z.row(i).t();
    arma::mat rev = zeros<mat>(p,p);
    for(int j = 0; j < K; j++){
      arma::mat B_temp = B.slice(j);
      rev += Z_temp[j] * B_temp;
    }
    arma::mat r_temp = Y_temp - rev;
    resid_all.slice(i) = r_temp;
  }
  return resid_all;
}


