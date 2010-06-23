
# This is a fabric file that constructs an openmdao release and
# deploys it to the openmdao.org site.
#
# Usage: fab release    (this will prompt you for a version id)
#     OR
#        fab release:version_id
#

from fabric.api import run, env, local, put, cd, prompt, hide, hosts
import sys
import os
#from os.path import join,dirname,normpath
import tempfile
import shutil
import fnmatch
import tarfile


class _VersionError(RuntimeError):
    pass

        
        
def _check_version(version):
    with hide('running', 'stdout'):
        result = run('ls ~/downloads')
    lst = [x.strip() for x in result.split('\n')]
    if version in lst:
        raise _VersionError('Version %s already exists. Please specify a different version' % version)
    return version


def _release(version=None, test=False):
    """Creates source distributions, docs, binary eggs, and install script for 
    the current openmdao namespace packages, uploads them to openmdao.org/dists, 
    and updates the index.html file there.
    """
    if version is not None:
        try:
            version = _check_version(version)
        except _VersionError, err:
            print str(err),'\n'
            version = None
        
    if version is None:
        version = prompt('Enter version id:', validate=_check_version)

    dist_dir = os.path.normpath(os.path.join(os.path.dirname(__file__),'..'))
    scripts_dir = os.path.join(dist_dir, 'scripts')
    doc_dir = os.path.join(dist_dir, 'docs')
    util_dir = os.path.join(dist_dir,'openmdao.util','src','openmdao','util')
    tmpdir = tempfile.mkdtemp()
    startdir = os.getcwd()
    try:
        # build the release distrib (docs are built as part of this)
        if test:
            teststr = '--test'
        else:
            teststr = ''
        local(sys.executable+' '+ os.path.join(scripts_dir,'mkrelease.py')+
              ' --version=%s %s -d %s' % (version, teststr, tmpdir), capture=False)
        
        # tar up the docs so we can upload them to the server
        os.chdir(os.path.join(tmpdir, '_build'))
        try:
            archive = tarfile.open(os.path.join(tmpdir,'docs.tar.gz'), 'w:gz')
            archive.add('html')
            archive.close()
        finally:
            os.chdir(startdir)
        
        run('mkdir ~/downloads/%s' % version)
        run('chmod 755 ~/downloads/%s' % version)
        
        # push new distribs up to the server
        for f in os.listdir(tmpdir):
            if f.startswith('openmdao_src'): 
                # upload the repo source tar
                put(os.path.join(tmpdir,f), '~/downloads/%s/%s' % (version, f), 
                    mode=0644)
            elif f.endswith('.tar.gz') and f != 'docs.tar.gz':
                put(os.path.join(tmpdir,f), '~/dists/%s' % f, mode=0644)
            elif f.endswith('.egg'):
                put(os.path.join(tmpdir,f), '~/dists/%s' % f, mode=0644)
        
        # for now, put the go-openmdao script up without the version
        # id in the name
        put(os.path.join(tmpdir, 'go-openmdao-%s.py' % version), 
            '~/downloads/%s/go-openmdao.py' % version,
            mode=0755)

        # put the docs on the server and untar them
        put(os.path.join(tmpdir,'docs.tar.gz'), '~/downloads/%s/docs.tar.gz' % version) 
        with cd('~/downloads/%s' % version):
            run('tar xzf docs.tar.gz')
            run('mv html docs')
            run('rm -f docs.tar.gz')

        # FIXME: change to a single version of mkdlversionindex.py that sits
        # in the downloads dir and takes an arg indicating the destination
        # directory, so we won't have a separate copy of mkdlversionindex.py
        # in every download/<version> directory.
        put(os.path.join(scripts_dir,'mkdlversionindex.py'), 
            '~/downloads/%s/mkdlversionindex.py' % version)
        
        # update the index.html for the version download directory on the server
        with cd('~/downloads/%s' % version):
            run('python2.6 mkdlversionindex.py')

        # update the index.html for the dists directory on the server
        with cd('~/dists'):
            run('python2.6 mkegglistindex.py')

        # update the index.html for the downloads directory on the server
        with cd('~/downloads'):
            run('python2.6 mkdownloadindex.py')

        # if everything went well update the 'latest' link to point to the 
        # most recent version directory
        run('rm -f ~/downloads/latest')
        run('ln -s ~/downloads/%s ~/downloads/latest' % version)
            
    finally:
        shutil.rmtree(tmpdir)

            
@hosts('openmdao@web103.webfaction.com')
def release(version=None, test=False):
    _release(version)
    

@hosts('bnaylor@torpedo.grc.nasa.gov')
def testrelease(version=None):
    _release(version, test=True)
  
    
#Builds and runs tests on a branch on all our development platforms
#Currently can only run remotely on viper and torpedo
#You must run from the top level of your branch.
#If you have uncommitted changes on your branch you will get an error message
# Usage: fab testbranch
#        fab testbranch -u username, if you are not on viper, storm, or torpedo
#        fab testbranch:host=hostname.grc.nasa.gov,  to run on a single host

 
def exportbranch():
    """Creates a tar file by doing a bzr export in the current branch on the current host
    """
    branchdir=os.getcwd()
    #print branchdir
    with cd('%s' % branchdir):
        local('bzr export testbranch.tar.gz')

#Local developer script to build and run tests on a branch on each development platform
def _testbranch():
    """Builds and runs tests on a branch on all our development platforms
    Currently can only run remotely on viper and torpedo
    You must run from the top level of your branch.
    If you have uncommitted changes on your branch you will get an error message
    """
    #export the current branch to a tarfile
    exportbranch()
    winplatforms=["storm.grc.nasa.gov"]
    if env.host in winplatforms:
        devbindir='devenv\Scripts'
	unpacktar="gzip "    #need to find out what it is on windoze
        pyversion="python"
        removeit="rmdir"
    else:
        devbindir='devenv/bin'
	unpacktar="tar xvf"
	pyversion="python2.6"
	removeit="rm -rf"
    branchdir=os.getcwd()
    print('original branch dir on localhost is is %s' % branchdir)
    #remove any previous testbranches
    run('%s testbranch' % removeit)
    #Copy exported branch tarfile to desired test platform
    put(os.path.join(branchdir,'testbranch.tar.gz'), 'testbranch.tar.gz') 
    #unpack the tarfile
    run('%s testbranch.tar.gz' % unpacktar)
    with cd('testbranch'):
        #Make a .bzr directory to fool go-openmdao-dev.py into thinking this is a real repository
        run('mkdir .bzr')    #this also works on windoze
        #build it
        run('%s go-openmdao-dev.py' % pyversion)
        #change to devenv/bin or devenv\Scripts (on windows), activate the envronment, and run tests
        with cd(devbindir):
            print("Please wait while the environment is activated and the tests are run")
            run('source activate && echo $PATH && echo environment activated, please wait while tests run && openmdao_test')
            print('Tests completed') 
    
@hosts('torpedo.grc.nasa.gov', 'viper.grc.nasa.gov')    
def testbranch():
    _testbranch()
