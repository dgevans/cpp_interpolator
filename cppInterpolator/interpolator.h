//
//  interpolator.h
//  cppInterpolator
//
//  Created by David Evans on 8/11/13.
//  Copyright (c) 2013 David Evans. All rights reserved.
//

#ifndef __cppInterpolator__interpolator__
#define __cppInterpolator__interpolator__
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/utility.hpp>
#include <vector>
#include <string>

using namespace Eigen;

typedef Matrix<double,Dynamic,Dynamic,RowMajor> rMatrixXd;
typedef Matrix<long,Dynamic,1> VectorXl;


class breakpoints {
    
    //holds the breakpoints
    Eigen::VectorXd v;
    int np;
    friend class SplinePython;
public:
    breakpoints(const Eigen::VectorXd& v_):v(v_){np=v.rows();};
    breakpoints():v(),np(0){};
    
    double operator[](int j) const;
    
    double p() const {return np;} ;
    
    const Eigen::VectorXd& getv() const{return v;};
};

inline double breakpoints::operator[](int j) const
{
    if(j >= np)
        return v[np-1];
    if(j < 0)
        return v[0];
    return v[j];
}

/*
 *All basis functions will be derived from this class
 */
class basis_functions {
    
    virtual std::pair<VectorXd,VectorXl> save_state() const = 0;
    
    virtual void load_state(const VectorXd &state_d, const VectorXl &state_l) = 0;
    
    friend struct interpolator_pickle;
    
public:
    virtual ~basis_functions(){};
    
    virtual basis_functions* clone() const = 0;
    virtual double operator()(double x, int i,int d = 0) const=0;
    virtual int get_n() const=0;
};

class basis_splines : public basis_functions {
    
    breakpoints v;
    
    int k;
    
    double SplineBasis(const breakpoints &v,int k, int j, double x, int der = 0) const;
    
    std::pair<VectorXd,VectorXl> save_state() const;
    
    void load_state(const VectorXd &state_d, const VectorXl &state_l);
    
public:
    
    basis_splines* clone() const{return new basis_splines(*this);};
    
    basis_splines(breakpoints _v, int _k):v(_v),k(_k){};
    basis_splines(){};
    
    double operator()(double x, int i,int d =0) const{return SplineBasis(v, k, i, x,d);};
    
    int get_n()const {return v.p()+k-1;};
};


class basis_hermite : public basis_functions{
    int order;
    
    double x_mean;
    
    double eval(double x, int i, int d) const;
    
    std::pair<VectorXd,VectorXl> save_state() const;
    
    void load_state(const VectorXd &state_d, const VectorXl &state_l);
    
public:
    basis_hermite(int _order,double _x_mean):order(_order),x_mean(_x_mean){};
    basis_hermite(){};
    
    basis_hermite* clone() const{return new basis_hermite(*this);};
    
    double operator()(double x, int i, int d = 0) const {return eval(x,i,d);};
    
    int get_n() const{return order+1;};
};

inline basis_functions* new_clone(basis_functions const& other){
    return other.clone();
}

struct interpolator_INFO {
        
    std::vector<std::string> types;
    
    std::vector<int> order;
    
    std::vector<int> k;//this is just for splines
    
    interpolator_INFO(){};
    
    interpolator_INFO(int N):types(N),order(N),k(N){};
    
};

class interpolator {
    
    interpolator_INFO INFO;
    
    long N;
    
    boost::ptr_vector<basis_functions> bf;
    
    VectorXd c;
    
    int max_poly;
    
    int buildBasisFunctions(const MatrixXd &X);
    
    void fit(const MatrixXd &X, const VectorXd &Y);
    
    int getNumberBasisPolynomials() const;
    
    friend struct interpolator_pickle;
    
public:
    
    interpolator(){};
    
    interpolator(const MatrixXd& X, const VectorXd& Y,const interpolator_INFO &INFO);
    
    interpolator(const MatrixXd& X, const VectorXd& Y,const interpolator_INFO &INFO, int max_poly);
    
    double operator()(const RowVectorXd& X) const;
    
    VectorXd eval(const Map<rMatrixXd> &X) const;
    
    
    double operator()(const RowVectorXd& X, const VectorXl &d) const;
    
    VectorXd eval_der(const Map<rMatrixXd> &X, const VectorXl &d) const;
    
    VectorXd get_c(){return c;};    
};



inline Eigen::MatrixXd kron(const Eigen::MatrixXd &X1, const Eigen::MatrixXd &X2)
{
    long n1 = X1.rows(), n2 = X2.rows(), m1 = X1.cols(), m2 = X2.cols();
    long n = n1*n2;
    long m = m1*m2;
    
    Eigen::MatrixXd X(n,m);
    
    for(long i1 = 0; i1 < n1; i1++)
    {
        for(long i2 = 0; i2< n2; i2++)
        {
            long i = i1*n2+i2;
            for(long j1 = 0; j1 < m1; j1++)
            {
                for(long j2 =0; j2<m2; j2++)
                {
                    long j = j1*m2+j2;
                    X(i,j) = X1(i1,j1)*X2(i2,j2);
                }
            }
        }
    }
    return X;
}

inline Eigen::SparseMatrix<double,Eigen::RowMajor> kron(Eigen::SparseMatrix<double,Eigen::RowMajor> &X1, Eigen::SparseMatrix<double,Eigen::RowMajor> &X2)
{
    int n1 = X1.rows(), n2 = X2.rows(), m1 = X1.cols(), m2 = X2.cols();
    int n = n1*n2;
    int m = m1*m2;
    
    std::vector<Eigen::Triplet<double> > list;
    list.reserve(X1.nonZeros()*X2.nonZeros());
    for (int i1 = 0; i1 < n1; i1++)
    {
        for(int i2 = 0; i2 < n2; i2++)
        {
            int i = i1*n2+i2;
            for(Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it1(X1,i1); it1; ++it1)
            {
                for(Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it2(X2,i2); it2; ++it2)
                {
                    int j = it1.col()*m2+it2.col();
                    double X_ij = it1.value()*it2.value();
                    list.push_back(Eigen::Triplet<double>(i,j,X_ij));
                }
            }
        }
    }
    
    Eigen::SparseMatrix<double,Eigen::RowMajor> X(n,m);
    X.setFromTriplets(list.begin(),list.end());
    return X;
}

#endif /* defined(__cppInterpolator__interpolator__) */
