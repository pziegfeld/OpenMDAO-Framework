CC = gcc
CFLAGS = -O3 -fpic -shared -g
#CFLAGS = -g -O0 -fpic -shared
DFLAGS = -fpic -shared
#LDFLAGS = -L$(HOME)/local/lib/coin -L$(HOME)/local/lib/coin/ThirdParty -L$(HOME)/local/lib -lipopt -lcoinhsl -lcoinmetis -lcoinmumps -lm -lcoinlapack -lcoinblas
LDFLAGS = -L$(COIN_LIB) -L$(COIN_LIB)/ThirdParty  -lipopt -lcoinhsl -lcoinmetis -lcoinmumps -lcoinlapack -lcoinblas



# You might also need -lgfortran, I don't have it because I use g77 to compile my ipopt
PY_DIR =  /usr/local/lib/python2.6/site-packages/
PY_DIR =  /usr/local/lib/python2.6/site-packages/
PY_DIR =  $(HOME)/local/lib/python2.6/site-packages/

# Change this to your ipopt include path that includes IpStdCInterface.h 
#IPOPT_INCLUDE = $(HOME)/local/include/coin/


# Change this to your python dir which includes Python.h
# You might need to download the python source code or install python-dev to get
# this header file. Note that Pyipopt needs this as an extend python module. 
PYTHON_INCLUDE = /usr/local/include/python2.6/

# Change this to your numpy include path which contains numpy/arrayobject.h
# If you don't want this and would like to use list. I have a nasty (and unmaintained version) in the package called pyipopt-list.c. You can compile it and use it without numpy. 
# However, numpy is strongly suggested. 

NUMPY_INCLUDE = /usr/local/lib/python2.6/site-packages/numpy-1.3.0-py2.6-linux-x86_64.egg/numpy/core/include/

pyipopt: callback.c pyipopt.c
	$(CC) -o pyipopt.so -I$(PYTHON_INCLUDE) -I$(COIN_INCLUDE) -I$(NUMPY_INCLUDE) $(CFLAGS) $(LDFLAGS) pyipopt.c callback.c

debug: callback.c pyipopt.c
	$(CC) -g -o pyipopt.so -I$(PYTHON_INCLUDE) -I$(COIN_INCLUDE) -I$(NUMPY_INCLUDE) $(DFLAGS) $(LDFLAGS) pyipopt_debug.c callback.c

debug_install: debug
	cp ./pyipopt.so $(PY_DIR)

install: pyipopt
	cp ./pyipopt.so $(PY_DIR)
clean:
	rm pyipopt.so 

splint:
	splint -I$(PYTHON_INCLUDE) -I$(COIN_INCLUDE) -I$(NUMPY_INCLUDE) pyipopt.c

lint:
	gcc -ansi -pedantic -W -Wall -Wundef -Wstrict-prototypes -Wmissing-prototypes \
	    -Wmissing-declarations -Wcast-qual -Wwrite-strings -I$(PYTHON_INCLUDE) -I$(COIN_INCLUDE) -I$(NUMPY_INCLUDE) callback.c