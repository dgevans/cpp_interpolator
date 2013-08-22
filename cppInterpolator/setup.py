from distutils.core import setup

#setup.py
from distutils.extension import Extension

import numpy

# define the name of the extension to use
extension_name='cpp_interpolator'
extension_version ='1.0'
# define the directories to search for include files
# to get this to work, you may need to include the path
# to your boost installation. Mine was in
# '/opt/local', hence the corresponding entry.
include_dirs = ['/opt/local/include', '.',numpy.get_include(),'/usr/local/include']
library_dirs = ['/usr/local/lib','/opt/local/lib']
# define the libraries to link with the boost python library
libraries = ['boost_python','boost_numpy']
# define the source files for the extension
source_files = ['interpolator.cpp','cpp_interpolate.cpp']
# create the extension and add it to the python distribution
setup(name='cpp_interpolator',
      version=extension_version,
      ext_modules=[Extension(extension_name, source_files, include_dirs=include_dirs, library_dirs=library_dirs, libraries=libraries)])