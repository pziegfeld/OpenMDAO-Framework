import os.path
import setuptools
import sys

from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration

include_dirs_extra = []
COIN_INCLUDE_DIR = os.environ.get('COIN_INCLUDE_DIR', '')
if COIN_INCLUDE_DIR:
    include_dirs_extra.append( COIN_INCLUDE_DIR )
    
lib_dirs_extra = []
COIN_LIB_DIR = os.environ.get('COIN_LIB_DIR', '')
if COIN_LIB_DIR:
    lib_dirs_extra.append( COIN_LIB_DIR )
COIN_THIRDPARTY_LIB_DIR = os.environ.get('COIN_THIRDPARTY_LIB_DIR', '')
if COIN_THIRDPARTY_LIB_DIR:
    lib_dirs_extra.append( COIN_THIRDPARTY_LIB_DIR )
    
if sys.platform == 'win32':
    import openmdao.util.distutils_fix
    sdkdir = os.environ.get('WindowsSdkDir')
    include_dirs = [os.path.join(sdkdir,'Include')]
    include_dirs.extend( include_dirs_extra )
    library_dirs = [os.path.join(sdkdir,'Lib')]
    library_dirs.extend( lib_dirs_extra )

    # make sure we have mt.exe available in path
    path = os.environ['PATH'].split(';')
    path.append(os.path.join(sdkdir,'bin'))
    os.environ['PATH'] = ';'.join(path)
    libraries = [
        'ipopt',
        'coinhsl',
        'coinmumps',
        'coinmetis',
        'm',
        'coinlapack',
        'coinblas',
        'gfortran',
        'pthread'
        ]
else:
    include_dirs = include_dirs_extra
    library_dirs = lib_dirs_extra
    libraries = [
        'ipopt',
        'coinhsl',
        'coinmetis',
        'coinmumps',
        'm',
        'coinlapack',
        'coinblas',
        ]

config = Configuration(name='pyipopt')
config.add_extension('pyipopt',
                     sources=['pyipopt.c',
                              'callback.c'],
                     include_dirs=include_dirs,
                     library_dirs=library_dirs,
                     libraries = libraries )

kwds = {'install_requires':['numpy'],
        'version': '1.0.0',
        'zip_safe': False,
        'license': 'public domain',
   # NOTE: we use 'url' here, but it really translates to 'home-page'
   # in the metadata. Go figure.
        'url': 'http://code.google.com/p/pyipopt/',
        'package_data': {'openmdao.main': ['*.html']},
       }
kwds.update(config.todict())

setup(**kwds)

