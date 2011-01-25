#include "Python.h"
#include "IpStdCInterface.h"
#include <stdio.h>
#include "numpy/arrayobject.h"       /* NumPy header */

#ifndef PY_IPOPT_HOOK
#define PY_IPOPT_HOOK

 Bool eval_f(Index n, Number* x, Bool new_x,
          Number* obj_value, UserDataPtr user_data);
          
 Bool eval_grad_f(Index n, Number* x, Bool new_x,
                 Number* grad_f, UserDataPtr user_data);

 Bool eval_g(Index n, Number* x, Bool new_x,
            Index m, Number* g, UserDataPtr user_data);

 Bool eval_jac_g(Index n, Number *x, Bool new_x,
                Index m, Index nele_jac,
                Index *iRow, Index *jCol, Number *values,
                UserDataPtr user_data);

 Bool eval_h(Index n, Number *x, Bool new_x, Number obj_factor,
            Index m, Number *lambda, Bool new_lambda,
            Index nele_hess, Index *iRow, Index *jCol,
            Number *values, UserDataPtr user_data);


 Bool eval_intermediate_callback(Index alg_mod, /* 0 is regular, 1 is resto */
                            Index iter_count, Number obj_value,
                            Number inf_pr, Number inf_du,
                            Number mu, Number d_norm,
                            Number regularization_size,
                            Number alpha_du, Number alpha_pr,
                            Index ls_trials, UserDataPtr user_data);


typedef struct {
	PyObject *eval_f_python;
	PyObject *eval_grad_f_python; 
	PyObject *eval_g_python;
	PyObject *eval_jac_g_python;
	PyObject *eval_intermediate_callback_python;
	PyObject *eval_h_python;
	PyObject *apply_new_python;
	PyObject* userdata;
} DispatchData;


PyObject *problem_getattr(PyObject* self, char* attrname) ;

// DispatchData myowndata;

// static IpoptProblem nlp = NULL;             /* IpoptProblem */

void logger(char* str);


typedef struct {
	PyObject_HEAD
	IpoptProblem nlp;
	DispatchData* data;
} problem;



#endif
