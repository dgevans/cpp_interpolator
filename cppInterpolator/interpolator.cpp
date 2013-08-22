//
//  interpolator.cpp
//  cppInterpolator
//
//  Created by David Evans on 8/11/13.
//  Copyright (c) 2013 David Evans. All rights reserved.
//

#include "interpolator.h"
#include <boost/math/special_functions/hermite.hpp>
#include <iostream>



double basis_splines::SplineBasis(const breakpoints &v,int k, int j, double x, int der) const{
    if(der <=0 ){
        if (k <= 0)
        {
            return (x >= v[j])*(x<v[j+1])||(j==v.p()-2)*(x>=v[j])||(j==0)*(x < v[j+1]);//allows for extrapolation
        }
        else {
            //if j == n it will be too large for k-1 so drop second term
            if(j==v.p()+k-2)
                return (x-v[j-k])/(v[j]-v[j-k])*SplineBasis(v,k-1,j-1,x);
            else if(j == 0)//drop first term which whould have j = -1
                return (v[j+1]-x)/(v[j+1]-v[j+1-k])*SplineBasis(v,k-1,j,x);
            else
                return (x-v[j-k])/(v[j]-v[j-k])*SplineBasis(v,k-1,j-1,x) + (v[j+1]-x)/(v[j+1]-v[j+1-k])*SplineBasis(v,k-1,j,x);
        }
    }else{
        if (k <= 0) {
            if ( (x == v[j]) || (x==v[j+1])) {
                return std::numeric_limits<double>::quiet_NaN();
            }else
                return 0.0;
        }else
        {
            if(j==v.p()+k-2)
                return k/(v[j]-v[j-k])*SplineBasis(v, k-1, j-1,x, der-1);
            else if(j==0)
                return -k/(v[j+1]-v[j+1-k])*SplineBasis(v,k-1,j,x,der-1);
            else
                return k/(v[j]-v[j-k])*SplineBasis(v, k-1, j-1,x, der-1)-k/(v[j+1]-v[j+1-k])*SplineBasis(v,k-1,j,x,der-1);
        }
        
    }
}


std::pair<VectorXd, VectorXl> basis_splines::save_state() const
{
    VectorXl b(1);
    b(0) = k;
    return std::pair<VectorXd, VectorXl>(v.getv(),b);
}

void basis_splines::load_state(const VectorXd &state_d, const VectorXl &state_l)
{
    v = breakpoints(state_d);
    k = state_l(0);
}

/*
 *Evaluate command to help compute derivative recursively
 */
double basis_hermite::eval(double x, int i,int d) const
{
    if(d == 0)
        return boost::math::hermite<double>(i, x-x_mean);
    else if(d == 1)
        return 2*(x-x_mean)*boost::math::hermite<double>(i, x-x_mean) - boost::math::hermite<double>(i+1, x-x_mean);
    else
        return 2*(x-x_mean)*eval(x,i,d-1) + 2*eval(x,i,d-2)
        -eval(x,i+1,d-1);
}

std::pair<VectorXd, VectorXl> basis_hermite::save_state() const
{
    VectorXd a(1);
    VectorXl b(1);
    a(0) = x_mean;
    b(0) = order;
    return std::pair<VectorXd, VectorXl>(a,b);
}

void basis_hermite::load_state(const VectorXd &state_d, const VectorXl &state_l)
{
    x_mean = state_d(0);
    order = state_l(0);
}

VectorXd unique_sort(const VectorXd &x)
{
    VectorXd x_sorted = x;
    std::sort(&x_sorted(0), &x_sorted(x.rows()-1));
    double *it = std::unique(&x_sorted(0), &x_sorted(x.rows()));
    x_sorted.conservativeResize(it-&x_sorted(0));
    return x_sorted;
}

breakpoints setUpBreakpoints(const VectorXd &x,int n, int k)
{
    /*std::vector<VectorXd> x;
    //for each i store the unique points
    int Nprod = 1;
    int NX = X.rows();
    for(int i =0; i < N; i++)
    {
        x.push_back(X.col(i));
        std::sort(&x[i](0), &x[i](NX-1)+1);
        double *it = std::unique(&x[i](0), &x[i](NX-1)+1);
        x[i].conservativeResize(it-&x[i](0));
        
        Nprod *= x[i].rows();
    }
    double frac;
    bool complete = false;
    if(Nprod != X.rows())
        frac = std::pow(X.rows()/double(Nprod),1.0/N);
    else
    {
        complete = true;
        frac = 1;
    }*/
    //number of breakpoints
    int np = n-k+1;
    VectorXd bp(np);
    VectorXd x_sorted = unique_sort(x);
    int nx = x_sorted.rows();
    bp(0) = x_sorted(0);
    bp(np-1) = x_sorted(nx-1);

    for(int ip = 1; ip <np-1; ip++)
    {
        //proportion of point < ip
        double P = double(ip)/np;
        int i1 = floor(P*nx);
        int i2 = ceil(P*nx);
        bp(ip) =x_sorted(i1) + (P*n-i1)*(x_sorted(i2)-x_sorted(i1));
    }
    return bp;
}

/*
 *Constructor for the interpolator class
 */
interpolator::interpolator(const MatrixXd& X, const VectorXd& Y,const interpolator_INFO &_INFO):INFO(_INFO),max_poly(0)
{
    N = INFO.types.size();
    if (INFO.order.size() != N)
        throw "order size does not match types size in INFO";
    if (X.cols() == N)
    {
        if (X.rows() != Y.rows())
            throw "X and Y do not have the same number of rows";
        max_poly = buildBasisFunctions(X);
        fit(X, Y);
    }
    else if(X.rows() == N)
    {
        if (X.cols() != Y.rows())
            throw "X and Y do not have the same number of rows";
        max_poly = buildBasisFunctions(X.transpose());
        fit(X.transpose(), Y);
    }else
        throw "X must have length N in one dimension";
}

/*
 *Constructor for the interpolator class
 */
interpolator::interpolator(const MatrixXd& X, const VectorXd& Y,const interpolator_INFO &_INFO,int _max_poly):INFO(_INFO),max_poly(_max_poly)
{
    N = INFO.types.size();
    if (INFO.order.size() != N)
        throw "order size does not match types size in INFO";
    if (X.cols() == N)
    {
        if (X.rows() != Y.rows())
            throw "X and Y do not have the same number of rows";
        buildBasisFunctions(X);
        fit(X, Y);
    }
    else if(X.rows() == N)
    {
        if (X.cols() != Y.rows())
            throw "X and Y do not have the same number of rows";
        buildBasisFunctions(X.transpose());
        fit(X.transpose(), Y);
    }else
        throw "X must have length N in one dimension";
}

/*
 *Builds the basis functions to fit
 */
int interpolator::buildBasisFunctions(const MatrixXd &X)
{
    //iterate over dimensions
    bf.clear();
    int _max_poly = 0;
    for (int i = 0; i < N; i++) {
        if(INFO.types[i] == "spline")
        {
            breakpoints v = setUpBreakpoints(X.col(i), INFO.order[i], INFO.k[i]);
            bf.push_back(new basis_splines(v,INFO.k[i]));
            
        }else if (INFO.types[i] == "hermite")
        {
            double xmean = X.col(i).mean();
            bf.push_back(new basis_hermite(INFO.order[i],xmean));
            _max_poly += INFO.order[i];
        }else
        {
            throw "Types not of hermite or Spline";
        }
    }
    return _max_poly;
}

/*
 *Computes the number of basis functions from combining polynomials.
 */
int interpolator::getNumberBasisPolynomials() const
{
    int n_poly =0;
    std::vector<int> _max_poly(N,0);
    for (int i=0; i < N; i++) {
        
    }
}


/*
 *Fits the data using the basis functions
 */
void interpolator::fit(const MatrixXd &X, const VectorXd &Y)
{
    //get number of total number of rows
    int ncols = 1;
    bool no_spline = true;
    for(int i = 0; i < N; i++)
    {
        ncols *= bf[i].get_n();
        if (INFO.types[i] == "spline") {
            no_spline = false;
        }
    }
    
    if( (X.rows()*ncols < 1e+8) || no_spline)
    {
        MatrixXd Phi(X.rows(),ncols);
        //fill each row
        for(int i = 0; i < X.rows(); i++)
        {
            RowVectorXd B(1);
            B(0) = 1;
            for(int j = 0; j < N; j++)
            {
                int n = bf[j].get_n();
                RowVectorXd temp(n);
                for(int ib = 0; ib < n; ib++)
                    temp(ib) = bf[j](X(i,j),ib);
                B = kron(temp,B);
            }
            Phi.row(i) = B;
        }
        //c = Phi.jacobiSvd(ComputeThinU | ComputeThinV).solve(Y);
        if(X.rows() == ncols)
        {
            c = Phi.fullPivLu().solve(Y);
        }else if(X.rows() > ncols){
            //solve least squares problem.
            c = Phi.jacobiSvd( ComputeThinU | ComputeThinV ).solve(Y);
        }else
            throw "Too many basis functions";
        
    }else
    {
        SparseMatrix<double,RowMajor> Phi(X.rows(),ncols);
        //fill each row
        for(int i = 0; i < X.rows(); i++)
        {
            SparseMatrix<double,RowMajor> B(1,1);
            B.coeffRef(0,0) = 1;
            B.makeCompressed();
            for(int j =0; j < N; j++)
            {
                int n = bf[j].get_n();
                SparseMatrix<double,RowMajor> temp(1,n);
                temp.reserve(n);
                for(int ib = 0; ib < n; ib ++)
                {
                    double v_ib = bf[j](X(i,j),ib);
                    if(v_ib != 0)
                        temp.insert(0, ib) = v_ib;
                }
                B = kron(temp,B);
            }
            Phi.row(i) = B;
        }
        SparseMatrix<double> A = (Phi.transpose())*Phi;
        VectorXd b = ( Phi.transpose() ) * Y;
        throw "Have not implemented yet!";
        //CholmodDecomposition<SparseMatrix<double> > solver;
        //UmfPackLU<SparseMatrix<double> > solver;
        //BiCGSTAB<SparseMatrix<double> > solver;
        //SimplicialLDLT<SparseMatrix<double> > solver;
        //Solve the matrix
        //solver.compute(A);
        //c = solver.solve(b);
    }
}
double interpolator::operator()(const RowVectorXd &X) const
{
    //setup
    VectorXd B(1);
    B(0) = 1;
    
    for (int i = 0; i < N; i++) {
        int n =  bf[i].get_n();
        VectorXd temp(n);
        for (int ib = 0; ib < n; ib++) {
            temp(ib) = bf[i](X(i),ib);
        }
        B= kron(temp,B);
    }
    
    return B.dot(c);
}

double interpolator::operator()(const RowVectorXd &X, const VectorXl &d) const
{
    //setup
    VectorXd B(1);
    B(0) = 1;
    
    for (int i = 0; i < N; i++) {
        int n =  bf[i].get_n();
        VectorXd temp(n);
        for (int ib = 0; ib < n; ib++) {
            temp(ib) = bf[i](X(i),ib,d(i));
        }
        B= kron(temp,B);
    }
    
    return B.dot(c);
}


VectorXd interpolator::eval(const Map<rMatrixXd> &X) const
{
    if (X.cols() == N) {
        long nX = X.rows();
        VectorXd Y(nX);
        for (int i =0; i < nX; i++) {
            Y(i) =(*this)(X.row(i));
        }
        return Y;
    }else if (X.rows() == N)
    {
        long nX = X.cols();
        VectorXd Y(nX);
        for (int i =0; i < nX; i++) {
            Y(i) =(*this)(X.transpose().row(i));
        }
        return Y;
    }else
        throw "One dimension of X needs to be length N";
}

VectorXd interpolator::eval_der(const Map<rMatrixXd> &X, const VectorXl &d) const
{
    if (X.cols() == N) {
        long nX = X.rows();
        VectorXd Y(nX);
        for (int i =0; i < nX; i++) {
            Y(i) =(*this)(X.row(i),d);
        }
        return Y;
    }else if (X.rows()==N)
    {
        long nX = X.cols();
        VectorXd Y(nX);
        for (int i =0; i < nX; i++) {
            Y(i) =(*this)(X.transpose().row(i),d);
        }
        return Y;
    }else
        throw "One dimension of X needs to be length N";
}