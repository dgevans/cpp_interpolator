//
//  cpp_interpolate.cpp
//  cppInterpolator
//
//  Created by David Evans on 8/12/13.
//  Copyright (c) 2013 David Evans. All rights reserved.
//

#include "interpolator.h"
#include <eigen3/Eigen/Dense>
#include <iostream>

#include <Python.h>
#include <numpy/arrayobject.h>
#include <boost/python.hpp>
#include <boost/python/tuple.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/shared_ptr.hpp>

#include "NumpyConverter.hpp"
#include <vector>
#include <string>


namespace bp = boost::python;
using namespace Eigen;
using namespace std;


/*
 *A constructor for interpolator_INFO for Python
 */
static boost::shared_ptr<interpolator_INFO> make_INFO(const bp::list& py_types, const bp::list& py_order, const bp::list& py_k)
{
    
    boost::shared_ptr<interpolator_INFO> INFO(new interpolator_INFO());
    
    if (bp::len(py_types)==bp::len(py_order) && bp::len(py_types) == bp::len(py_k)) {
        long N = bp::len(py_k);
        for(int i = 0; i < N; i++)
        {
            INFO->types.push_back(bp::extract<string>(py_types[i]));
            INFO->order.push_back(bp::extract<int>(py_order[i]));
            INFO->k.push_back(bp::extract<int>(py_k[i]));
        }
    }
    
    return INFO;
}

MatrixXd test(const Map<rMatrixXd> &X)
{
    return X;
}

struct interpolator_pickle : bp::pickle_suite
{
    
    static
    boost::python::tuple
    getinitargs(interpolator const& w)
    {
        return boost::python::make_tuple();
    }
    
    static
    boost::python::tuple
    getstate(interpolator const& interp)
    {
        bp::list py_bf;
        for (int i =0; i < interp.N; i++)
        {
            std::pair<VectorXd,VectorXl> state = interp.bf[i].save_state();
            py_bf.append(bp::make_tuple(state.first, state.second));
            
        }
        
        //finally return tuple of all thes objects
        return bp::make_tuple(interp.INFO,interp.N,py_bf,interp.c);
    }
    
    static
    void
    setstate(interpolator& interp, boost::python::tuple state)
    {
        interp.INFO = bp::extract<interpolator_INFO>(state[0]);
        interp.N = bp::extract<long>(state[1]);
        
        interp.bf.clear();
        bp::list bf_states = bp::extract<bp::list>(state[2]);
        for (int i =0; i < interp.N; i++) {
            bp::tuple bf_state = bp::extract<bp::tuple>(bf_states[i]);
            if (interp.INFO.types[i] == "spline")
                interp.bf.push_back(new basis_splines());
            else if(interp.INFO.types[i] == "hermite")
                interp.bf.push_back(new basis_hermite());
            else
                throw "in pickle set state: unkown type";
            interp.bf[i].load_state(bp::extract<VectorXd>(bf_state[0]), bp::extract<VectorXl>(bf_state[1]));
        }
        interp.c = bp::extract<VectorXd>(state[3]);
    }
};

struct interpolator_INFO_pickle : bp::pickle_suite
{
    
    static
    boost::python::tuple
    getinitargs(interpolator_INFO const& w)
    {
        return boost::python::make_tuple();
    }
    
    static
    boost::python::tuple
    getstate(interpolator_INFO const& INFO)
    {
        bp::list bp_types,bp_order,bp_k;
        for (int i = 0; i < INFO.types.size(); i++) {
            bp_types.append(INFO.types[i]);
            bp_order.append(INFO.order[i]);
            bp_k.append(INFO.k[i]);
        }
        return bp::make_tuple(bp_types,bp_order,bp_k);
    }
    
    static
    void
    setstate(interpolator_INFO& INFO, boost::python::tuple state)
    {
        INFO.types.clear();
        INFO.order.clear();
        INFO.k.clear();
        
        bp::list bp_types = bp::extract<bp::list>(state[0]);
        bp::list bp_order = bp::extract<bp::list>(state[1]);
        bp::list bp_k = bp::extract<bp::list>(state[2]);
        
        for(int i=0; i < bp::len(bp_types);i++)
        {
            INFO.types.push_back(bp::extract<std::string>(bp_types[i]));
            INFO.order.push_back(bp::extract<int>(bp_order[i]));
            INFO.k.push_back(bp::extract<int>(bp_k[i]));
        }
    }
};

using namespace boost::python;

//BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(interpolator_call, interpolator::operator(), 1, 1)

//BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(interpolator_call_der, interpolator::operator(), 2, 2)

BOOST_PYTHON_MODULE(cpp_interpolator)
{
    //first register the Eigen objects
    Register<MatrixXd>();
    Register<rMatrixXd>();
    Register<VectorXd>();
    Register<VectorXl>();
    
    //Now register interpolate_INFO
    class_<interpolator_INFO, boost::shared_ptr<interpolator_INFO> >("interpolate_INFO")
    .def_pickle(interpolator_INFO_pickle())
    .def("__init__", make_constructor(&make_INFO));
    
    //Now register interpolate class
    class_<interpolator>("interpolate",init<const MatrixXd&,const VectorXd&,const interpolator_INFO &>())
    .def(init<>())
    .def(init<const MatrixXd&,const VectorXd&,const interpolator_INFO &,int>())
    .def_pickle(interpolator_pickle())
    .def("__call__",&interpolator::eval)
    .def("__call__",&interpolator::eval_der)
    .def("get_c",&interpolator::get_c);
    
}




