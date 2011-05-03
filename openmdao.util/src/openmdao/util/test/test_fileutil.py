"""
Test File Utility Functions
"""

import os
import stat
import shutil
import logging
import os.path
import sys
import unittest
import tempfile
from fnmatch import fnmatch

from openmdao.util.fileutil import find_in_path, build_directory, find_files, find_exe

structure = {
    'top': {
        'foo/bar.exe': 'some stuff...',
        'blah': {
            'somefile': '# a comment',
            },
        'somedir/dir2': {
                    }
        }
    }

class FileUtilTestCase(unittest.TestCase):

    def setUp(self):
        self.startdir = os.getcwd()
        self.tempdir = tempfile.mkdtemp()
        os.chdir(self.tempdir)
        build_directory(structure)

    def tearDown(self):
        os.chdir(self.startdir)
        #shutil.rmtree(self.tempdir)

    def test_find_in_path(self):
        if sys.platform == 'win32':
            path=r'C:\a\b\c;top;top\blah;top\foo'
        else:
            path = '/a/b/c:top:top/blah:top/foo'
        fname = find_in_path('bar', path)
        self.assertEqual(fname, None)
        # search for a file with an extension
        fname = find_in_path('bar', path, exts=('.exe',))
        self.assertTrue(fname is not None)
        self.assertEqual(os.path.basename(fname), 'bar.exe')
        # search for a file without an extension
        fname = find_in_path('somefile', path)
        self.assertTrue(fname is not None)
        self.assertEqual(os.path.basename(fname), 'somefile')
        # make sure we don't find directories
        fname = find_in_path('blah', path)
        self.assertEqual(fname, None)
        
    def test_find_files(self):
        flist = find_files(self.tempdir)
        self.assertEqual(set([os.path.basename(f) for f in flist]), 
                         set(['bar.exe', 'somefile']))
        flist = find_files(self.tempdir, '*.exe')
        self.assertEqual(set([os.path.basename(f) for f in flist]), 
                         set(['bar.exe']))
        matcher = lambda name: fnmatch(name, '*.exe') or fnmatch(name, '*some*')
        flist = find_files(self.tempdir, matcher)
        self.assertEqual(set([os.path.basename(f) for f in flist]), 
                         set(['bar.exe', 'somefile']))
        flist = find_files(self.tempdir, exclude='*.exe')
        self.assertEqual(set([os.path.basename(f) for f in flist]), 
                         set(['somefile']))
        flist = find_files(self.tempdir, exclude=matcher)
        self.assertEqual(set([os.path.basename(f) for f in flist]), 
                         set([]))
        flist = find_files(self.tempdir, match='*.exe', exclude=matcher)
        self.assertEqual(set([os.path.basename(f) for f in flist]), 
                         set([]))
        
        # /tmp/tmp3QnGmp/
        # /tmp/tmp3QnGmp/top
        # /tmp/tmp3QnGmp/top/somedir
        # /tmp/tmp3QnGmp/top/somedir/dir2
        # /tmp/tmp3QnGmp/top/foo
        # /tmp/tmp3QnGmp/top/foo/bar.exe
        # /tmp/tmp3QnGmp/top/blah
        # /tmp/tmp3QnGmp/top/blah/somefile
        
    def test_find_exe( self ) :
        # Look for file that does exists but is not executable
        test_existing_file_path =  os.path.join( self.tempdir, "top/foo", "bar.exe" )
        found_path = find_exe( test_existing_file_path ) 
        self.assertEqual(  found_path, None )

        # Look for same file but make it executable first
        os.chmod( test_existing_file_path, stat.S_IEXEC )
        found_path = find_exe( test_existing_file_path ) 
        self.assertEqual(  found_path, test_existing_file_path )

        # Look for file that does not exist
        test_nonexisting_file_path =  os.path.join( self.tempdir, "top/not_there", "bar.exe" )
        found_path = find_exe( test_nonexisting_file_path ) 
        self.assertEqual(  found_path, None )

        # Look for file using a relative path but
        #   at first it is not on the PATH
        test_file_path_relative = "bar.exe"
        found_path = find_exe( test_file_path_relative ) 
        self.assertEqual(  found_path, None )

        # Now add to the PATH the directory containing that file
        os.environ["PATH"] += os.pathsep +  os.path.join( self.tempdir, "top/foo" )
        found_path = find_exe( test_file_path_relative ) 
        self.assertEqual(  found_path, test_existing_file_path )

if __name__ == '__main__':
    unittest.main()

