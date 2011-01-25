/* Copyright (c) 2008, Eric You Xu, Washington University
* All rights reserved.
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
*     * Redistributions of source code must retain the above copyright
*       notice, this list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright
*       notice, this list of conditions and the following disclaimer in the
*       documentation and/or other materials provided with the distribution.
*     * Neither the name of the Washington University nor the
*       names of its contributors may be used to endorse or promote products
*       derived from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS "AS IS" AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE REGENTS AND CONTRIBUTORS BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/*  This file contains five call-back functions for IPOPT */
/* For the interface of these five functions, check 
	Ipopt's document: Ipopt C Interface */
/* TODO: change the apply_new interface */
	
#include "hook.h"

#include <unistd.h>


void logger(char* str)
{
  /* 	printf("%s\n", str); */
}


Bool eval_intermediate_callback(Index alg_mod, /* 0 is regular, 1 is resto */
                                Index iter_count, Number obj_value,
                                Number inf_pr, Number inf_du,
                                Number mu, Number d_norm,
                                Number regularization_size,
                                Number alpha_du, Number alpha_pr,
                                Index ls_trials, UserDataPtr data)
{
  logger("[Callback:E]intermediate_callback");
  
  DispatchData *myowndata = (DispatchData*) data;
  UserDataPtr user_data = (UserDataPtr) myowndata->userdata;
  
  long result_as_long ;
  Bool result_as_bool;
  
  PyObject *python_algmod = Py_BuildValue("i", alg_mod);
  PyObject *python_iter_count = Py_BuildValue("i", iter_count);
  PyObject *python_obj_value = Py_BuildValue("d", obj_value);
  PyObject *python_inf_pr = Py_BuildValue("d", inf_pr);
  PyObject *python_inf_du = Py_BuildValue("d", inf_du);
  PyObject *python_mu = Py_BuildValue("d", mu);
  PyObject *python_d_norm = Py_BuildValue("d", d_norm);
  PyObject *python_regularization_size = Py_BuildValue("d", regularization_size);
  PyObject *python_alpha_du = Py_BuildValue("d", alpha_du);
  PyObject *python_alpha_pr = Py_BuildValue("d", alpha_pr);
  PyObject *python_ls_trials = Py_BuildValue("i", ls_trials);
  
  PyObject* arglist = NULL ;
  
  if (user_data != NULL) 
    arglist =  Py_BuildValue("(OOOOOOOOOOOO)", 
                             python_algmod,
                             python_iter_count,
                             python_obj_value,
                             python_inf_pr,
                             python_inf_du,
                             python_mu,
                             python_d_norm,
                             python_regularization_size,
                             python_alpha_du,
                             python_alpha_pr,
                             python_ls_trials,
                             (PyObject*)user_data);
  else 
    arglist =  Py_BuildValue("(OOOOOOOOOOO)", 
                             python_algmod,
                             python_iter_count,
                             python_obj_value,
                             python_inf_pr,
                             python_inf_du,
                             python_mu,
                             python_d_norm,
                             python_regularization_size,
                             python_alpha_du,
                             python_alpha_pr,
                             python_ls_trials
                             );
  
  
  PyObject* result  = PyObject_CallObject (myowndata->eval_intermediate_callback_python ,
                                           arglist);
  
  if (!result) 
    PyErr_Print();

  result_as_long =  PyInt_AsLong(result);
  result_as_bool = (Bool)result_as_long;
  
  Py_DECREF(result);
  Py_CLEAR(arglist);
  logger("[Callback:R] intermediate_callback");
  return result_as_bool ;
}




Bool eval_f(Index n, Number* x, Bool new_x,
            Number* obj_value, UserDataPtr data)
{
  logger("[Callback:E]eval_f");
  /* int dims[1]; */
  npy_intp dims[1];
  dims[0] = n;
  
  DispatchData *myowndata = (DispatchData*) data;
  UserDataPtr user_data = (UserDataPtr) myowndata->userdata;
  
  import_array( )
  PyObject *arrayx = PyArray_SimpleNewFromData(1, dims, PyArray_DOUBLE, (char*) x);
  if (!arrayx) return FALSE;
  
  if (new_x && myowndata->apply_new_python) {
    /* Call the python function to applynew */
    PyObject* arg1;
    arg1 = Py_BuildValue("(O)", arrayx);
    PyObject* tempresult = PyObject_CallObject (myowndata->apply_new_python, arg1);
    if (!tempresult) {
      printf("[Error] Python function apply_new returns a None\n");
      Py_DECREF(arg1);	
      return FALSE;
    }
    Py_DECREF(arg1);
    Py_DECREF(tempresult);
  }
  
  
  PyObject* arglist;
  
  if (user_data != NULL)
    arglist = Py_BuildValue("(OO)", arrayx, (PyObject*)user_data);
  else 
    arglist = Py_BuildValue("(O)", arrayx);
  
  PyObject* result  = PyObject_CallObject (myowndata->eval_f_python ,arglist);
  
  if (!result) 
    PyErr_Print();
  
  if (!PyFloat_Check(result))
    PyErr_Print();
  
  *obj_value =  PyFloat_AsDouble(result);
  Py_DECREF(result);
  Py_DECREF(arrayx);
  Py_CLEAR(arglist);
  logger("[Callback:R] eval_f");
  return TRUE;
}

Bool eval_grad_f(Index n, Number* x, Bool new_x,
                 Number* grad_f, UserDataPtr data)
{
  logger("[Callback:E] eval_grad_f");
  
  DispatchData *myowndata = (DispatchData*) data;
  UserDataPtr user_data = (UserDataPtr) myowndata->userdata;
  
  if (myowndata->eval_grad_f_python == NULL) PyErr_Print();
  
  /* int dims[1]; */
  npy_intp dims[1];
  dims[0] = n;
  import_array( ); 
  
  /* PyObject *arrayx = PyArray_FromDimsAndData(1, dims, PyArray_DOUBLE , (char*) x); */
  PyObject *arrayx = PyArray_SimpleNewFromData(1, dims, PyArray_DOUBLE, (char*) x);
  if (!arrayx) return FALSE;
  
  if (new_x && myowndata->apply_new_python) {
    /* Call the python function to applynew */
    PyObject* arg1 = Py_BuildValue("(O)", arrayx);
    PyObject* tempresult = PyObject_CallObject (myowndata->apply_new_python, arg1);
    if (!tempresult) {
      printf("[Error] Python function apply_new returns a None\n");
      Py_DECREF(arg1);	
      return FALSE;
    }
    Py_DECREF(arg1);
    Py_DECREF(tempresult);
  }	
  
  PyObject* arglist; 
  
  if (user_data != NULL)
    arglist = Py_BuildValue("(OO)", arrayx, (PyObject*)user_data);
  else 
    arglist = Py_BuildValue("(O)", arrayx);
  
  PyArrayObject* result = 
    (PyArrayObject*) PyObject_CallObject 
    (myowndata->eval_grad_f_python, arglist);
  
  if (!result) 
    PyErr_Print();
  
  if (!PyArray_Check(result))
    PyErr_Print();
  
  double *tempdata = (double*)result->data;
  int i;
  for (i = 0; i < n; i++)
    grad_f[i] = tempdata[i];
  
  Py_DECREF(result);
  Py_CLEAR(arrayx);
  Py_CLEAR(arglist);
  logger("[Callback:R] eval_grad_f");	
  return TRUE;
}


Bool eval_g(Index n, Number* x, Bool new_x,
            Index m, Number* g, UserDataPtr data)
{

  logger("[Callback:E] eval_g");
  
  DispatchData *myowndata = (DispatchData*) data;
  UserDataPtr user_data = (UserDataPtr) myowndata->userdata;
  
  if (myowndata->eval_g_python == NULL) 
    PyErr_Print();
  /* int dims[1]; */
  npy_intp dims[1];
  int i;
  double *tempdata;
  
  dims[0] = n;
  import_array( );
  
  /* PyObject *arrayx = PyArray_FromDimsAndData(1, dims, PyArray_DOUBLE , (char*) x); */
  PyObject *arrayx = PyArray_SimpleNewFromData(1, dims, PyArray_DOUBLE, (char*) x);
  if (!arrayx) return FALSE;
  
  if (new_x && myowndata->apply_new_python) {
    /* Call the python function to applynew */
    PyObject* arg1 = Py_BuildValue("(O)", arrayx);
    PyObject* tempresult = PyObject_CallObject (myowndata->apply_new_python, arg1);
    if (!tempresult) {
      printf("[Error] Python function apply_new returns a None\n");
      Py_DECREF(arg1);	
      return FALSE;
    }
    Py_DECREF(arg1);
    Py_DECREF(tempresult);
  }
  
  PyObject* arglist; 
  
  if (user_data != NULL)
    arglist = Py_BuildValue("(OO)", arrayx, (PyObject*)user_data);
  else 
    
    arglist = Py_BuildValue("(O)", arrayx);
  
  PyArrayObject* result = 
    (PyArrayObject*) PyObject_CallObject 
    (myowndata->eval_g_python, arglist);
  
  if (!result) 
    PyErr_Print();
  
  if (!PyArray_Check(result))
    PyErr_Print();
  
  tempdata = (double*)result->data;
  for (i = 0; i < m; i++){
    g[i] = tempdata[i];
  }
  
  Py_DECREF(result);
  Py_CLEAR(arrayx);
  Py_CLEAR(arglist);
  logger("[Callback:R] eval_g");
  return TRUE;
}

Bool eval_jac_g(Index n, Number *x, Bool new_x,
                Index m, Index nele_jac,
                Index *iRow, Index *jCol, Number *values,
                UserDataPtr data)
{

  logger("[Callback:E] eval_jac_g");
  
  DispatchData *myowndata = (DispatchData*) data;
  UserDataPtr user_data = (UserDataPtr) myowndata->userdata;
  
  int i;
  long* rowd = NULL; 
  long* cold = NULL;
  
  /* int dims[1]; */
  npy_intp dims[1];
  dims[0] = n;
  
  double *tempdata;
  
  
  if (myowndata->eval_grad_f_python == NULL) /*  Why??? */
    PyErr_Print();
  
  if (values == NULL) {
    import_array( )
      PyObject *arrayx = PyArray_SimpleNewFromData(1, dims, PyArray_DOUBLE, (char*) x);
    if (!arrayx) return FALSE;
    
    PyObject* arglist; 
    
    if (user_data != NULL)
      arglist = Py_BuildValue("(OOO)", 
                              arrayx, Py_True, (PyObject*)user_data);
    else 
      arglist = Py_BuildValue("(OO)", arrayx, Py_True);	
    
    PyObject* result = PyObject_CallObject (myowndata->eval_jac_g_python, arglist);
    if ( !result )
      {
        
        printf("[PyIPOPT] return from eval_jac_g is null\n");
        /* TODO: need to deal with reference counting here */
        return FALSE;
      }
    if (!PyTuple_Check(result))
      {
        PyErr_Print();
      }
    
    
    PyArrayObject* row = (PyArrayObject*) PyTuple_GetItem(result, 0);
    PyArrayObject* col = (PyArrayObject*) PyTuple_GetItem(result, 1);
    
    if (!row || !col || !PyArray_Check(row) || !PyArray_Check(col))
      {
        fprintf( stderr, "qqq: problems with row and col\n" ) ;
        PyErr_Print();
      }
    
    rowd = (long*) row->data;
    cold = (long*) col->data;
    
    for (i = 0; i < nele_jac; i++) {
      iRow[i] = (Index) rowd[i];
      jCol[i] = (Index) cold[i];
    }
    Py_CLEAR(arrayx);
    Py_DECREF(result);
    Py_CLEAR(arglist);
    logger("[Callback:R] eval_jac_g(1)");	
  }
  
  else {
    PyObject *arrayx = PyArray_SimpleNewFromData(1, dims, PyArray_DOUBLE, (char*) x);
    
    if (!arrayx) return FALSE;
    
    if (new_x && myowndata->apply_new_python) {
      /* Call the python function to applynew */
      PyObject* arg1 = Py_BuildValue("(O)", arrayx);
      PyObject* tempresult = PyObject_CallObject (myowndata->apply_new_python, arg1);
      if (!tempresult) {
        printf("[Error] Python function apply_new returns a None\n");
        Py_DECREF(arg1);	
        return FALSE;
      }
      Py_DECREF(arg1);
      Py_DECREF(tempresult);
    }
    
    PyObject* arglist; 
    if (user_data != NULL)
      arglist = Py_BuildValue("(OOO)", 
                              arrayx, Py_False, (PyObject*)user_data);
    else 
      arglist = Py_BuildValue("(OO)", arrayx, Py_False);	
    
    PyArrayObject* result = (PyArrayObject*) 
      PyObject_CallObject (myowndata->eval_jac_g_python, arglist);
    
    if (!result || !PyArray_Check(result)) 
      PyErr_Print();
    
    /*  Code is buggy here. We assume that result is a double array */
    assert (result->descr->type == 'd');
    tempdata = (double*)result->data;
    
    for (i = 0; i < nele_jac; i++)
      values[i] = tempdata[i];
    
    Py_DECREF(result);
    Py_CLEAR(arrayx);
    Py_CLEAR(arglist);
    logger("[Callback:R] eval_jac_g(2)");
  }
  logger("[Callback:R] eval_jac_g");
  return TRUE;
}


Bool eval_h(Index n, Number *x, Bool new_x, Number obj_factor,
            Index m, Number *lambda, Bool new_lambda,
            Index nele_hess, Index *iRow, Index *jCol,
            Number *values, UserDataPtr data)
{
  logger("[Callback:E] eval_h");
  
  DispatchData *myowndata = (DispatchData*) data;
  UserDataPtr user_data = (UserDataPtr) myowndata->userdata;
  
  
  int i;
  npy_intp dims[1];
  npy_intp dims2[1];
  
  if (myowndata->eval_h_python == NULL)
    {	printf("There is no eval_h assigned");
      return FALSE;
    }
  if (values == NULL) {
    PyObject *newx = Py_True;
    PyObject *objfactor = Py_BuildValue("d", obj_factor);
    PyObject *lagrange = Py_True;
    
    PyObject* arglist;
    
    if (user_data != NULL) 
      arglist =  Py_BuildValue("(OOOOO)", newx, lagrange, objfactor, Py_True, (PyObject*)user_data);
    else 
      arglist =  Py_BuildValue("(OOOO)", newx, lagrange, objfactor, Py_True);
    
    PyObject* result 	= 
      PyObject_CallObject (myowndata->eval_h_python, arglist);
    if (!PyTuple_Check(result))
      PyErr_Print();
    
    PyArrayObject* row = (PyArrayObject*)PyTuple_GetItem(result, 0);
    PyArrayObject* col = (PyArrayObject*)PyTuple_GetItem(result, 1);
    
    long* rdata = (long*)row->data;
    long* cdata = (long*)col->data;
    
    for (i = 0; i < nele_hess; i++) {
      iRow[i] = (Index)rdata[i];
      jCol[i] = (Index)cdata[i];
      /* printf("PyIPOPT_DEBUG %d, %d\n", iRow[i], jCol[i]); */
    }
    
    Py_DECREF(objfactor);
    Py_DECREF(result);
    Py_CLEAR(arglist);
    logger("[Callback:R] eval_h (1)");
  }
  else {	
    PyObject *objfactor = Py_BuildValue("d", obj_factor);
    
    dims[0] = n;
    PyObject *arrayx = PyArray_SimpleNewFromData(1, dims, PyArray_DOUBLE, (char*) x);
    if (!arrayx) return FALSE;
    
    if (new_x && myowndata->apply_new_python) {
      /*  Call the python function to applynew  */
      PyObject* arg1 = Py_BuildValue("(O)", arrayx);
      PyObject* tempresult = 
        PyObject_CallObject (myowndata->apply_new_python, arg1);
      if (!tempresult) {
        /* printf("[Error] Python function apply_new returns a None\n"); */
        Py_DECREF(arg1);	
        return FALSE;
      }
      Py_DECREF(arg1);
      Py_DECREF(tempresult);
    }
    
    dims2[0] = m;
    PyObject *lagrangex = PyArray_SimpleNewFromData(1, dims2, PyArray_DOUBLE, (char*) lambda);
    if (!lagrangex) return FALSE;
    
    PyObject* arglist;
    
    if (user_data != NULL)
      arglist = Py_BuildValue("(OOOO)", arrayx, lagrangex, objfactor, Py_False, (PyObject*)user_data);
    else
      arglist = Py_BuildValue("(OOOOO)", arrayx, lagrangex, objfactor, Py_False);
    PyArrayObject* result = 
      (PyArrayObject*) PyObject_CallObject 
      (myowndata->eval_h_python, arglist);
    
    if (!result) printf("[Error] Python function eval_h returns a None\n");
    
    double* tempdata = (double*)result->data;
    for (i = 0; i < nele_hess; i++)
      {	values[i] = tempdata[i];
        /*  printf("PyDebug %f \n", values[i]); */
      }	
    Py_CLEAR(arrayx);
    Py_CLEAR(lagrangex);
    Py_CLEAR(objfactor);
    Py_DECREF(result);
    Py_CLEAR(arglist);
    logger("[Callback:R] eval_h (2)");
  }	                         
  return TRUE;
}


