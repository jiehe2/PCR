// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// PCR_est
Rcpp::List PCR_est(arma::cube& Theta_0, arma::cube& B_0, arma::cube& Lambda_0, arma::vec& mu, arma::cube& lambda, double epsilon, arma::mat& Z, arma::cube& Y, int iter_col, double error_col, arma::mat& combn_mat);
RcppExport SEXP _PCR_PCR_est(SEXP Theta_0SEXP, SEXP B_0SEXP, SEXP Lambda_0SEXP, SEXP muSEXP, SEXP lambdaSEXP, SEXP epsilonSEXP, SEXP ZSEXP, SEXP YSEXP, SEXP iter_colSEXP, SEXP error_colSEXP, SEXP combn_matSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::cube& >::type Theta_0(Theta_0SEXP);
    Rcpp::traits::input_parameter< arma::cube& >::type B_0(B_0SEXP);
    Rcpp::traits::input_parameter< arma::cube& >::type Lambda_0(Lambda_0SEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type mu(muSEXP);
    Rcpp::traits::input_parameter< arma::cube& >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< double >::type epsilon(epsilonSEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< arma::cube& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< int >::type iter_col(iter_colSEXP);
    Rcpp::traits::input_parameter< double >::type error_col(error_colSEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type combn_mat(combn_matSEXP);
    rcpp_result_gen = Rcpp::wrap(PCR_est(Theta_0, B_0, Lambda_0, mu, lambda, epsilon, Z, Y, iter_col, error_col, combn_mat));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_PCR_PCR_est", (DL_FUNC) &_PCR_PCR_est, 11},
    {NULL, NULL, 0}
};

RcppExport void R_init_PCR(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
