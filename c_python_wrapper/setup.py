from distutils.core import setup, Extension
import numpy


TomoAO = Extension('TomoAO',
                    sources = ['./matcov_wrapper.c', './matcov_styc.c'],
                    include_dirs = ['/usr/local/include','/usr/local/include/fann','./src',
                        numpy.get_include()],
                    library_dirs = ['.','/usr/local/lib'],
                    libraries = [],
                    extra_link_args = [],
                    extra_compile_args = []
                    )

setup (name = 'TomoAO',
       version = '0.1',
       description = 'A Wrapper around Erics c covmat code.',
       ext_modules = [TomoAO])
