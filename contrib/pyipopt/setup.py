import os.path
import setuptools
import sys

from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration

if sys.platform == 'win32':
    import openmdao.util.distutils_fix
    sdkdir = os.environ.get('WindowsSdkDir')
    include_dirs = [os.path.join(sdkdir,'Include')]
    library_dirs = [os.path.join(sdkdir,'Lib')]
    # make sure we have mt.exe available in path
    path = os.environ['PATH'].split(';')
    path.append(os.path.join(sdkdir,'bin'))
    os.environ['PATH'] = ';'.join(path)
else:
    include_dirs = ['/home/hschilli/local/include/coin/',]
    library_dirs = ['/home/hschilli/local/lib/coin',
                    '/home/hschilli/local/lib/coin/ThirdParty',
                    '/home/hschilli/local/lib']
    libraries = [
        'ipopt',
        'coinhsl',
        'coinmetis',
        'coinmumps',
        'm',
        'coinlapack',
        'coinblas',
        ]

config = Configuration(name='ipopt')
config.add_extension('ipopt',
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

