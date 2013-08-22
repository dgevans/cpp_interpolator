//
//  main.cpp
//  cppInterpolator
//
//  Created by David Evans on 8/11/13.
//  Copyright (c) 2013 David Evans. All rights reserved.
//

#include <iostream>
#include "interpolator.h"

int main(int argc, const char * argv[])
{

    // insert code here...
    std::cout << "Hello, World!\n";
    
    VectorXd X = VectorXd::LinSpaced(20, 0., 1.);
    VectorXd Y(X.rows());
    for (int i =0; i<X.rows(); i++) {
        Y(i) = exp(X(i));
    }
    auto INFO = interpolator_INFO(1);
    INFO.order[0] = 10;
    INFO.types[0] = "hermite";
    INFO.k[0] = 3;
    auto f = interpolator(X,Y,INFO);
    VectorXd Xhat = VectorXd::LinSpaced(55, 0., 1.);
    std::cout<<f(Xhat.row(20))<<std::endl;
    std::cout<<exp(Xhat(20))<<std::endl;
    for (int i = 0; i< Xhat.rows(); i++) {
        std::cout<<f(Xhat.row(i))<<","<<exp(Xhat(i))<<","<<f(Xhat.row(i))-exp(Xhat(i))<<std::endl;
    }
    
    return 0;
}

