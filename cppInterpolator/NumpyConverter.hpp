//
//  Converter.hpp
//  
//
//  Created by David Evans on 6/14/13.
//
//

#ifndef _Converter_hpp
#define _Converter_hpp
#include <boost/python.hpp>
#include <boost/python/numeric.hpp>
#include <boost/numpy.hpp>
#include <eigen3/Eigen/Dense>


namespace bp = boost::python;
namespace bn = boost::numpy;

template<class matrix_t>
struct ndarray_to_Matrix_converter {
    typedef typename matrix_t::Scalar _Scalar;
    const static int _Rows = matrix_t::RowsAtCompileTime;
    const static int _Cols = matrix_t::ColsAtCompileTime;
    const static int _Options = matrix_t::Options;
    
    ndarray_to_Matrix_converter() {
        bp::converter::registry::push_back(
                                           &convertible,
                                           &construct,
                                           bp::type_id< matrix_t >()
                                           );
    }
    
    /**
     *  Test to see if we can convert this to the desired type; if not return zero.
     *  If we can convert, returned pointer can be used by construct().
     */
    static void * convertible(PyObject * p) {
        try {
            bp::object obj(bp::handle<>(bp::borrowed(p)));
            int Nmin = (_Rows > 1) + (_Cols > 1);
            std::auto_ptr<bn::ndarray> array;
            if (_Options == Eigen::RowMajor)
                array = std::auto_ptr<bn::ndarray>(
                                             new bn::ndarray(
                                                             bn::from_object(obj, bn::dtype::get_builtin<_Scalar>(), Nmin, 2, bn::ndarray::C_CONTIGUOUS)
                                                             )
                                             );
            else
                array = std::auto_ptr<bn::ndarray>(
                                                 new bn::ndarray(
                                                                 bn::from_object(obj, bn::dtype::get_builtin<_Scalar>(), Nmin, 2, bn::ndarray::F_CONTIGUOUS)
                                                                 )
                                                 );
            //scalar case
            if ((array->get_nd() == 0)&&(_Rows >1 ||_Cols>1)) return 0;
            //1d vector case
            if( array->get_nd() == 1)
            {
                if((_Rows >1)&&(_Cols>1)) return 0;
                if((_Rows==1)&&( (_Cols != array->shape(0))&&(_Cols != Eigen::Dynamic))) return 0;
                if((_Cols==1)&&( (_Rows != array->shape(0))&&(_Rows != Eigen::Dynamic))) return 0;
            }
            //2d vector case
            if(array->get_nd() == 2)
            {
                if (( (_Rows != array->shape(0))&&(_Rows != Eigen::Dynamic)) || ( (_Cols != array->shape(1))&&(_Cols != Eigen::Dynamic)))
                {
                    return 0;
                }
            }
            
            return array.release();
        } catch (bp::error_already_set & err) {
            bp::handle_exception();
            return 0;
        }
    }
    
    /**
     *  Finish the conversion by initializing the C++ object into memory prepared by Boost.Python.
     */
    static void construct(PyObject * obj, bp::converter::rvalue_from_python_stage1_data * data) {
        // Extract the array we passed out of the convertible() member function.
        std::auto_ptr<bn::ndarray> array(reinterpret_cast<bn::ndarray*>(data->convertible));
        // Find the memory block Boost.Python has prepared for the result.
        typedef bp::converter::rvalue_from_python_storage<matrix_t> storage_t;
        storage_t * storage = reinterpret_cast<storage_t*>(data);
        
        // Fill the result with the values from the NumPy array.
        int n = 1;
        int m = 1;
        switch (array->get_nd()) {
            case 1:
                if (_Cols != 1) {
                    n = 1;
                    m = array->shape(0);
                }else
                {
                    n = array->shape(0);
                    m = 1;
                }
                break;
            case 2:
                n = array->shape(0);
                m = array->shape(1);
                break;
                
            default:
                break;
        }
        // Use placement new to initialize the result.
        matrix_t * mat = new (storage->storage.bytes) matrix_t();
        Eigen::Map<matrix_t > temp(reinterpret_cast<_Scalar *>(array->get_data()),n,m);
        *mat = temp;
        // Finish up.
        data->convertible = storage->storage.bytes;
    }
};

template<class matrix_t>
struct Matrix_to_ndarray
{
    typedef typename matrix_t::Scalar _Scalar;
    const static int _Rows = matrix_t::RowsAtCompileTime;
    const static int _Cols = matrix_t::ColsAtCompileTime;
    const static int _Options = matrix_t::Options;
    
    static PyObject* convert(const matrix_t &M)
    {
        int n = M.rows();
        int m = M.cols();
        bp::tuple shape;
        bp::tuple stride;
        if (n==1)
        {
            shape = bp::make_tuple(m);
            stride = bp::make_tuple(sizeof(_Scalar));
        }
        else
        {
            shape = bp::make_tuple(n,m);
            if(_Options == Eigen::RowMajor)
                stride = bp::make_tuple(m*sizeof(_Scalar), sizeof(_Scalar));
            else
                stride = bp::make_tuple(sizeof(_Scalar), n*sizeof(_Scalar));
        }
        
        bn::ndarray a = bn::from_data(
                                      &M(0),
                                      bn::dtype::get_builtin<_Scalar>(),
                                      shape,
                                      stride,
                                      bp::object()
                                      );
        return bp::incref(a.copy().ptr());
        
    }
};

template<class matrix_t>
struct ndarray_to_MapMatrix_converter {
    typedef typename matrix_t::Scalar _Scalar;
    const static int _Rows = matrix_t::RowsAtCompileTime;
    const static int _Cols = matrix_t::ColsAtCompileTime;
    const static int _Options = matrix_t::Options;
    ndarray_to_MapMatrix_converter() {
        bp::converter::registry::push_back(
                                           &convertible,
                                           &construct,
                                           bp::type_id< Eigen::Map<matrix_t> >()
                                           );
    }
    
    /**
     *  Test to see if we can convert this to the desired type; if not return zero.
     *  If we can convert, returned pointer can be used by construct().
     */
    static void * convertible(PyObject * p) {
        try {
            bp::object obj(bp::handle<>(bp::borrowed(p)));
            int Nmin = (_Rows > 1) + (_Cols > 1);
            std::auto_ptr<bn::ndarray> array;
            if (_Options == Eigen::RowMajor)
                array = std::auto_ptr<bn::ndarray>(
                                                   new bn::ndarray(
                                                                   bn::from_object(obj, bn::dtype::get_builtin<_Scalar>(), Nmin, 2, bn::ndarray::C_CONTIGUOUS)
                                                                   )
                                                   );
            else
                array = std::auto_ptr<bn::ndarray>(
                                                   new bn::ndarray(
                                                                   bn::from_object(obj, bn::dtype::get_builtin<_Scalar>(), Nmin, 2, bn::ndarray::F_CONTIGUOUS)
                                                                   )
                                                   );
            //scalar case
            if ((array->get_nd() == 0)&&(_Rows >1 ||_Cols>1)) return 0;
            //1d vector case
            if( array->get_nd() == 1)
            {
                if((_Rows >1)&&(_Cols>1)) return 0;
                if((_Rows==1)&&( (_Cols != array->shape(0))&&(_Cols != Eigen::Dynamic))) return 0;
                if((_Cols==1)&&( (_Rows != array->shape(0))&&(_Rows != Eigen::Dynamic))) return 0;
            }
            //2d vector case
            if(array->get_nd() == 2)
            {
                if (( (_Rows != array->shape(0))&&(_Rows != Eigen::Dynamic)) || ( (_Cols != array->shape(1))&&(_Cols != Eigen::Dynamic)))
                {
                    return 0;
                }
            }
            
            return array.release();
        } catch (bp::error_already_set & err) {
            bp::handle_exception();
            return 0;
        }
    }
    
    /**
     *  Finish the conversion by initializing the C++ object into memory prepared by Boost.Python.
     */
    static void construct(PyObject * obj, bp::converter::rvalue_from_python_stage1_data * data) {
        // Extract the array we passed out of the convertible() member function.
        std::auto_ptr<bn::ndarray> array(reinterpret_cast<bn::ndarray*>(data->convertible));
        // Find the memory block Boost.Python has prepared for the result.
        typedef bp::converter::rvalue_from_python_storage< Eigen::Map<matrix_t> > storage_t;
        storage_t * storage = reinterpret_cast<storage_t*>(data);
        
        // Fill the result with the values from the NumPy array.
        int n = 1;
        int m = 1;
        switch (array->get_nd()) {
            case 1:
                if (_Cols != 1) {
                    n = 1;
                    m = array->shape(0);
                }else
                {
                    n = array->shape(0);
                    m = 1;
                }
                break;
            case 2:
                n = array->shape(0);
                m = array->shape(1);
                break;
                
            default:
                break;
        }
        // Use placement new to initialize the result.

        new (storage->storage.bytes) Eigen::Map<matrix_t>(reinterpret_cast<_Scalar *>(array->get_data()),n,m);
        // Finish up.
        data->convertible = storage->storage.bytes;
    }
};

template <class matrix_t>
static void Register() {
    bn::initialize();
    ndarray_to_Matrix_converter< matrix_t >();
    ndarray_to_MapMatrix_converter< matrix_t >();
    bp::to_python_converter<
    matrix_t,
    Matrix_to_ndarray< matrix_t > >();
}


#endif
