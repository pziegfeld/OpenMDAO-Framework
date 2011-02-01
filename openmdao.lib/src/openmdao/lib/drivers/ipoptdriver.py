"""
    ipoptdriver.py - Driver for the IPOPT optimizer.
    
    See Appendix B for additional information on the :ref:`IPOPTDriver`.
"""

# pylint: disable-msg=C0103

#public symbols
__all__ = ['IPOPTdriver']


import sys

# pylint: disable-msg=E0611,F0401
from numpy import zeros, ones, array, product, append, array_equal

import pyipopt.pyipopt as pyipopt

from openmdao.main.api import Driver
from openmdao.main.exceptions import RunStopped
from openmdao.lib.datatypes.api import Enum, Float, Int, Dict
from openmdao.main.hasparameters import HasParameters
from openmdao.main.hasconstraints import HasConstraints
from openmdao.main.hasobjective import HasObjective
from openmdao.util.decorators import add_delegate

class IpoptReturnStatus:
    '''A fake enum for the possible values of
    the status variable returned by Ipopt
    '''

    def __init__( self ):
        pass
    
    Solve_Succeeded = 0
    Solved_To_Acceptable_Level = 1
    Infeasible_Problem_Detected = 2
    Search_Direction_Becomes_Too_Small = 3
    Diverging_Iterates = 4
    User_Requested_Stop = 5
    Feasible_Point_Found = 6
    
    Maximum_Iterations_Exceeded = -1
    Restoration_Failed = -2
    Error_In_Step_Computation = -3
    Maximum_CpuTime_Exceeded = -4
    Not_Enough_Degrees_Of_Freedom = -10
    Invalid_Problem_Definition = -11
    Invalid_Option = -12,
    Invalid_Number_Detected = -13
    


# functions for doing derivatives and gradients
def func_mod_one_var(f, x_, i, x, driver):
    ''' f is a function of an array and outputs one value.
    evaluate the function at x except for the i element
    of the array which will be substituted with x_.
    The idea is to turn a function of an array into a function
    of one variable'''
    
    xcopy = x[:]
    xcopy[i] = x_
    return f( xcopy, driver )

def gradient( func, x0, dx, driver, order=2):
    '''Compute the gradient of the given function at the given
    point using finite differences
    '''

    x0_copy = x0.copy()
    n = len(x0_copy)
    grad = zeros( n )
    for i in range( n ) :
        ff = lambda y, driver: func_mod_one_var( func, y, i, x0_copy, driver )
        grad[i] = derivative( ff, x0_copy[i], driver, dx, order=order)
    return grad
    
def jacobian( g, x0, dx, driver, order=2):
    '''g is an array of functions'''

    m = len(g)
    n = len(x0)

    dg = zeros((m, n))

    for j in range(m):
        for i in range(n):
            x0_copy = x0.copy() # TODO: Do i need all these copy commands ??
            ff = lambda y, driver: func_mod_one_var( g[j], y, i,
                                                     x0_copy, driver )
            deriv = derivative( ff, x0_copy[i], driver, dx, order=order)
            dg[j, i] = deriv
            
    return dg

def derivative(func, x0, driver, dx=1.0, args=(), order=2):
    """Given a function, use a central difference formula with spacing dx to
    compute the nth derivative at x0.
    
    order is the number of points to use and must be odd.
    
    Warning: Decreasing the step size too small can result in
    round-off error.
    """

    x0_copy = x0.copy()
    
    n = 1 # only do 
    assert (order+1 >= n+1), \
           "Number of points must be at least the derivative order + 1."
    assert (order % 2 == 0), "Even number of points only."
    
    if order == 2:
        weights = array([-1, 0, 1])/2.0
    elif order == 4:
        weights = array([1, -8, 0, 8, -1])/12.0
    elif order == 6:
        weights = array([-1, 9, -45, 0, 45, -9, 1])/60.0
    elif order == 8:
        weights = array([3, -32, 168, -672, 0, 672, -168, 32, -3])/840.0

    val = 0.0
    ho = ( order + 1 ) >> 1
    for k in range(order+1):
        #val += weights[k]*func(x0+(k-ho)*dx,*args)

        # Do not eval any functions unless needed
        if weights[k] != 0.0 :
            tmp = weights[k]*func(x0_copy+(k-ho)*dx, driver)
            val += tmp

    deriv = val / product((dx,)*n, axis=0)

    return deriv






def eval_f_and_eval_g(x, driver, user_data = None):
    '''
    eval objective function and constraints.
    Cache results
    '''

    # qqq
    driver.last_eval_x = x.copy()

    # update the design variables in the model
    driver.set_parameters(x)
    super(IPOPTdriver, driver).run_iteration()

    return 











def eval_f(x, driver, user_data = None):
    '''evaluate objective function'''

    if not array_equal( x, driver.last_eval_x ):
        eval_f_and_eval_g( x, driver )
    else:
        pass

    obj = driver.eval_objective()

    return obj


def eval_grad_f(x, driver, user_data = None):
    '''
    gradient of the object function.
    Use the gradient function above
    '''

    if not array_equal( x, driver.last_x ):
        eval_grad_f_and_jac_g( x, driver )
    else:
        pass

    return driver.grad_f

def eval_grad_f_and_jac_g(x, driver, user_data = None):
    '''
    gradient of the object function and jacobian of constraints.
    Cache results
    '''


    # qqq
    driver.last_x = x.copy()

    nvar = len( x ) 
    driver.grad_f = zeros(nvar, 'd')
    ncon = len( driver.get_ineq_constraints() ) + \
           len( driver.get_eq_constraints() )
    driver.jac_g = zeros((ncon, nvar), 'd')
    for i in range( nvar ):
        xtmp = x.copy()
        xtmp[i] -= driver.fd_delta
        f1 = eval_f( xtmp, driver ) # this runs the model at xtmp

        update_constraints( driver)
        # While we are at it, get the constraints too
        g1 = driver.constraint_vals.copy()

        xtmp = x.copy()
        xtmp[i] += driver.fd_delta
        f2 = eval_f( xtmp, driver )

        update_constraints( driver)
        g2 = driver.constraint_vals.copy()

        inv_fd_delta = 0.5 / driver.fd_delta
        driver.grad_f[i] = ( f2 - f1 ) * inv_fd_delta


        for j in range( ncon ):
            driver.jac_g[j, i] = ( g2[j] - g1[j] ) * inv_fd_delta


    return 

def eval_g(x, driver, user_data= None):
    '''evaluate constraint functions'''



    if not array_equal( x, driver.last_eval_x ):
        eval_f_and_eval_g( x, driver )
    else:
        pass

    for i, v in enumerate(driver.get_ineq_constraints().values()):
        val = v.evaluate()

        # OpenMDAO accepts inequalities in any form, e.g.
        #    x1 > x2 + 5
        #    x1 + x2 < 22
        #
        # Need to take that kind of equation and put it in a form that
        #  Ipopt wants. The simplest way is to handle this is to
        #  turn those into
        #    x1 -x2 - 5 > 0
        #    x1 + x2 - 22 > 0
        #  
        if '>' in val[2]:
            driver.constraint_vals[i] = -(val[1]-val[0])
        else:
            driver.constraint_vals[i] = -(val[0]-val[1])

    n_ineq_constraints = len( driver.get_ineq_constraints().values())
    for i, val in enumerate(driver.get_eq_constraints().values()):
        val = val.evaluate()
        driver.constraint_vals[i + n_ineq_constraints] = val[0] - val[1]
            
    return driver.constraint_vals

    
def update_constraints( driver):
    '''evaluate constraint functions'''

    for i, v in enumerate(driver.get_ineq_constraints().values()):
        val = v.evaluate()

        if '>' in val[2]:
            driver.constraint_vals[i] = -(val[1]-val[0])
        else:
            driver.constraint_vals[i] = -(val[0]-val[1])

    n_ineq_constraints = len( driver.get_ineq_constraints().values())
    for i, val in enumerate(driver.get_eq_constraints().values()):
        val = val.evaluate()
        driver.constraint_vals[i + n_ineq_constraints] = val[0] - val[1]
            
    return driver.constraint_vals

    
def eval_jac_g(x, flag, driver, user_data = None):
    '''In pyipopt, it sets the flag to indicate whether
    to return iRow and jCol or the actual values of the
    Jacobian
    '''

    # not needed. Just wastes time
    #super(IPOPTdriver, driver).run_iteration()
    
    ncon = len( driver.get_ineq_constraints() ) + \
           len( driver.get_eq_constraints() )
    
    if flag:
        # for 4 variables and 2 constraints, for example,
        #   return (array([0, 0, 0, 0, 1, 1, 1, 1]), 
        #         array([0, 1, 2, 3, 0, 1, 2, 3]))

        nvar = len( x )
        
        irow = array( [[ ]] )
        for i in range( ncon ) :
            newrow = array( [ [i] * nvar ] )
            if irow.shape == (1, 0):
                irow = newrow
            else:
                irow = append( irow , newrow, axis=0 )
 
        jcol = array( [[]] )
        for i in range( ncon ) :
            newrow = array( [ range( nvar ) ] )
            if jcol.shape == (1, 0):
                jcol = newrow
            else:
                jcol = append( jcol , newrow, axis= 0 )
 
        return ( irow, jcol )
                         
    else:
 
        if not array_equal( x, driver.last_x ):
            eval_grad_f_and_jac_g( x, driver )
            
        return driver.jac_g
 
def intermediate_callback(alg_mod, iteration, obj_value, 
                    inf_pr, inf_du,
                    mu, d_norm,
                    regularization_size,
                    alpha_du, alpha_pr,
                    ls_trials,
                    driver,
                    user_data= None, ):
    ''' Ipopt calls back to this function each iteration'''
   
    # Incrementing the count here gets a true value of the iterations
    #   done by Ipopt
    driver.iter_count += 1

    if driver._stop: 
        return False
    else:
        return True

@add_delegate(HasParameters, HasConstraints, HasObjective)
class IPOPTdriver(Driver):
    """ Driver wrapper of C version of IPOPT. 
    """
    # pylint: disable-msg=E1101

    # Control parameters for IPOPT. Specifically
    #    list the most common. Leave the rest for the
    #    dictionary "options"
    print_level = Enum(5, range(13), iotype='in',
                       desc='Print '
                       'information during IPOPT solution. Higher values are '
                       'more verbose. Use 0 for no output')
    
    tol = Float(1.0e-8, iotype='in', low = 0.0,
                desc='convergence tolerance.' + \
                'Algorithm terminates if the scaled NLP error becomes ' + \
                'smaller than this value and if additional conditions ' + \
                '(see Ipopt manual) are met')
    
    max_iter = Int(3000, iotype='in', low=0, 
                   desc='Maximum number of iterations')
    
    max_cpu_time = Float(1.0e6, iotype='in', low = 0.0,
                         desc='limit on CPU seconds' )

    
    constr_viol_tol = Float(0.0001, iotype='in', low = 0.0,
                            desc='absolute tolerance on constraint violation' )
    
    obj_scaling_factor = Float(1.0, iotype='in', 
                               desc='scaling factor for the objective function')
    
    linear_solver = Enum('ma27',
                         [ 'ma27', 'ma57', 'ma77',
                           'pardiso', 'wsmp', 'mumps', 'custom'],
                         iotype='in', desc='linear algebra package used' )
    
    fd_delta = Float(0.001,  iotype='in',
                    desc='The delta used by the finite ' + \
                    'difference code when computing derivatives' )
    
    options = Dict({
        'output_file' : "",
        'file_print_level' : 5,
        'print_user_options' : "no",
        'print_options_documentation' : "no",
        'print_timing_statistics' : "no",
        'option_file_name' : "",
        'replace_bounds' : "no",
        'skip_finalize_solution_call' : "no",
        'print_info_string' : "no",
        's_max' : 100.0,
        'dual_inf_tol' : 1.0,
        'compl_inf_tol' : 0.0001,
        'acceptable_tol' : 1e-06,
        'acceptable_iter' : 15,
        'acceptable_dual_inf_tol' : 1e+10,
        'acceptable_constr_viol_tol' : 0.01,
        'acceptable_compl_inf_tol' : 0.01,
        'acceptable_obj_change_tol' : 1e+20,
        'diverging_iterates_tol' : 1e+20,
        'mu_target' : 0.0,
        'nlp_scaling_method' : "gradient-based",
        'nlp_scaling_max_gradient' : 100.0,
        'nlp_scaling_obj_target_gradient' : 0.0,
        'nlp_scaling_constr_target_gradient' : 0.0,
        'nlp_scaling_min_value' : 1e-08,
        'nlp_lower_bound_inf' : -1e+19,
        'nlp_upper_bound_inf' : 1e+19,
        'fixed_variable_treatment' : "make_parameter",
        'dependency_detector' : "none",
        'dependency_detection_with_rhs' : "no",
        'num_linear_variables' : 0,
        'kappa_d' : 1e-05,
        'bound_relax_factor' : 1e-08,
        'honor_original_bounds' : "yes",
        'check_derivatives_for_naninf' : "no",
        'jac_c_constant' : "no",
        'jac_d_constant' : "no",
        'hessian_constant' : "no",
        'bound_push' : 0.01,
        'bound_frac' : 0.01,
        'slack_bound_push' : 0.01,
        'slack_bound_frac' : 0.01,
        'constr_mult_init_max' : 1000.0,
        'bound_mult_init_val' : 1.0,
        'bound_mult_init_method' : "constant",
        'least_square_init_primal' : "no",
        'least_square_init_duals' : "no",
        'mu_max_fact' : 1000.0,
        'mu_max' : 100000.0,
        'mu_min' : 1e-11,
        'adaptive_mu_globalization' : "obj-constr-filter",
        'adaptive_mu_kkterror_red_iters' : 4,
        'adaptive_mu_kkterror_red_fact' : 0.9999,
        'filter_margin_fact' : 1e-05,
        'filter_max_margin' : 1.0,
        'adaptive_mu_restore_previous_iterate' : "no",
        'adaptive_mu_monotone_init_factor' : 0.8,
        'adaptive_mu_kkt_norm_type' : "2-norm-squared",
        'mu_strategy' : "monotone",
        'mu_oracle' : "quality-function",
        'fixed_mu_oracle' : "average_compl",
        'mu_init' : 0.1,
        'barrier_tol_factor' : 10.0,
        'mu_linear_decrease_factor' : 0.2,
        'mu_superlinear_decrease_power' : 1.5,
        'mu_allow_fast_monotone_decrease' : "yes",
        'tau_min' : 0.99,
        'sigma_max' : 100.0,
        'sigma_min' : 1e-06,
        'quality_function_norm_type' : "2-norm-squared",
        'quality_function_centrality' : "none",
        'quality_function_balancing_term' : "none",
        'quality_function_max_section_steps' : 8,
        'quality_function_section_sigma_tol' : 0.01,
        'quality_function_section_qf_tol' : 0.0,
        'alpha_red_factor' : 0.5,
        'accept_every_trial_step' : "no",
        'accept_after_max_steps' : -1,
        'alpha_for_y' : "primal",
        'alpha_for_y_tol' : 10.0,
        'tiny_step_tol' : 2.22045e-15,
        'tiny_step_y_tol' : 0.01,
        'watchdog_shortened_iter_trigger' : 10,
        'watchdog_trial_iter_max' : 3,
        'theta_max_fact' : 10000.0,
        'theta_min_fact' : 0.0001,
        'eta_phi' : 1e-08,
        'delta' : 1.0,
        's_phi' : 2.3,
        's_theta' : 1.1,
        'gamma_phi' : 1e-08,
        'gamma_theta' : 1e-05,
        'alpha_min_frac' : 0.05,
        'max_soc' : 4,
        'kappa_soc' : 0.99,
        'obj_max_inc' : 5.0,
        'max_filter_resets' : 5,
        'filter_reset_trigger' : 5,
        'corrector_type' : "none",
        'skip_corr_if_neg_curv' : "yes",
        'skip_corr_in_monotone_mode' : "yes",
        'corrector_compl_avrg_red_fact' : 1.0,
        'nu_init' : 1e-06,
        'nu_inc' : 0.0001,
        'rho' : 0.1,
        'kappa_sigma' : 1e+10,
        'recalc_y' : "no",
        'recalc_y_feas_tol' : 1e-06,
        'slack_move' : 1.81899e-12,
        'constraint_violation_norm_type' : "1-norm",
        'warm_start_init_point' : "no",
        'warm_start_same_structure' : "no",
        'warm_start_bound_push' : 0.001,
        'warm_start_bound_frac' : 0.001,
        'warm_start_slack_bound_push' : 0.001,
        'warm_start_slack_bound_frac' : 0.001,
        'warm_start_mult_bound_push' : 0.001,
        'warm_start_mult_init_max' : 1e+06,
        'warm_start_entire_iterate' : "no",
        'linear_system_scaling' : "mc19",
        'linear_scaling_on_demand' : "yes",
        'mehrotra_algorithm' : "no",
        'fast_step_computation' : "no",
        'min_refinement_steps' : 1,
        'max_refinement_steps' : 10,
        'residual_ratio_max' : 1e-10,
        'residual_ratio_singular' : 1e-05,
        'residual_improvement_factor' : 1.0,
        'neg_curv_test_tol' : 1.0,
        'max_hessian_perturbation' : 1e+20,
        'min_hessian_perturbation' : 1e-20,
        'perturb_inc_fact_first' : 100.0,
        'perturb_inc_fact' : 8.0,
        'perturb_dec_fact' : 0.333333,
        'first_hessian_perturbation' : 0.0001,
        'jacobian_regularization_value' : 1e-08,
        'jacobian_regularization_exponent' : 0.25,
        'perturb_always_cd' : "no",
        'expect_infeasible_problem' : "no",
        'expect_infeasible_problem_ctol' : 0.001,
        'expect_infeasible_problem_ytol' : 1e+08,
        'start_with_resto' : "no",
        'soft_resto_pderror_reduction_factor' : 0.9999,
        'max_soft_resto_iters' : 10,
        'required_infeasibility_reduction' : 0.9,
        'max_resto_iter' : 3000000,
        'evaluate_orig_obj_at_resto_trial' : "yes",
        'resto_penalty_parameter' : 1000.0,
        'resto_proximity_weight' : 1.0,
        'bound_mult_reset_threshold' : 1000.0,
        'constr_mult_reset_threshold' : 0.0,
        'resto_failure_feasibility_threshold' : 0.0,
        'derivative_test' : "none",
        'derivative_test_first_index' : -2,
        'derivative_test_perturbation' : 1e-08,
        'derivative_test_tol' : 0.0001,
        'derivative_test_print_all' : "no",
        'jacobian_approximation' : "exact",
        'findiff_perturbation' : 1e-07,
        'point_perturbation_radius' : 10.0,
        'limited_memory_aug_solver' : "sherman-morrison",
        'limited_memory_max_history' : 6,
        'limited_memory_update_type' : "bfgs",
        'limited_memory_initialization' : "scalar1",
        'limited_memory_init_val' : 1.0,
        'limited_memory_init_val_max' : 1e+08,
        'limited_memory_init_val_min' : 1e-08,
        'limited_memory_max_skipping' : 2,
        'limited_memory_special_for_resto' : "no",
        'hessian_approximation' : "exact",
        'hessian_approximation_space' : "nonlinear-variables",
        'ma27_pivtol' : 1e-08,
        'ma27_pivtolmax' : 0.0001,
        'ma27_liw_init_factor' : 5.0,
        'ma27_la_init_factor' : 5.0,
        'ma27_meminc_factor' : 10.0,
        'ma27_skip_inertia_check' : "no",
        'ma27_ignore_singularity' : "no",
        'ma57_pivtol' : 1e-08,
        'ma57_pivtolmax' : 0.0001,
        'ma57_pre_alloc' : 1.05,
        'ma57_pivot_order' : 5,
        'ma57_automatic_scaling' : "yes",
        'ma57_block_size' : 16,
        'ma57_node_amalgamation' : 16,
        'ma57_small_pivot_flag' : 0,
        'pardiso_matching_strategy' : "complete+2x2",
        'pardiso_redo_symbolic_fact_only_if_inertia_wrong' : "no",
        'pardiso_repeated_perturbation_means_singular' : "no",
        'pardiso_out_of_core_power' : 0,
        'pardiso_msglvl' : 0,
        'pardiso_skip_inertia_check' : "no",
        'pardiso_max_iter' : 500,
        'pardiso_iter_relative_tol' : 1e-06,
        'pardiso_iter_coarse_size' : 5000,
        'pardiso_iter_max_levels' : 10000,
        'pardiso_iter_dropping_factor' : 0.5,
        'pardiso_iter_dropping_schur' : 0.1,
        'pardiso_iter_max_row_fill' : 10000000,
        'pardiso_iter_inverse_norm_factor' : 5e+06,
        'pardiso_iterative' : "no",
        'pardiso_max_droptol_corrections' : 4,
        'mumps_pivtol' : 1e-06,
        'mumps_pivtolmax' : 0.1,
        'mumps_mem_percent' : 1000,
        'mumps_permuting_scaling' : 7,
        'mumps_pivot_order' : 7,
        'mumps_scaling' : 77,
        'mumps_dep_tol' : -1.0,
        'ma28_pivtol' : 0.01,
        'warm_start_target_mu' : 0.0,
        }, iotype='in',
                   desc='List of additional optimization parameters' )
    

    def __init__(self, doc=None):
        super(IPOPTdriver, self).__init__( doc)
        
        self.iter_count = 0

        # define the IPOPTdriver's private variables
        # note, these are all resized in config_ipopt
        
        self.design_vals = zeros(0, 'd')
        self.constraint_vals = zeros(0, 'd' )        
        self.nlp = None

        self.status = None
        self.obj = 0.0
        self.zl = None        
        self.zu = None        

        self.last_x = None
        self.last_eval_x = None

    def start_iteration(self):
        """Perform initial setup before iteration loop begins."""
        
        self._config_ipopt()
        #self.iter_count = 0
        
        # get the initial values of the parameters
        for i, val in enumerate(self.get_parameters().values()):
            self.design_vals[i] = val.expreval.evaluate()
            
        # perform an initial run for self-consistency
        #super(IPOPTdriver, self).run_iteration()

        # update constraint value array
        for i, v in enumerate(self.get_ineq_constraints().values()):
            val = v.evaluate()
            if '>' in val[2]:
                self.constraint_vals[i] = val[1]-val[0]
            else:
                self.constraint_vals[i] = val[0]-val[1]
        

        nvar = len( self.get_parameters().values() )
        x_L = array( [ x.low for x in self.get_parameters().values() ] )
        x_U = array( [ x.high for x in self.get_parameters().values() ] )
        n_ineq_con =  len( self.get_ineq_constraints() )
        ncon = n_ineq_con + len( self.get_eq_constraints() )
        # Ipopt treats equality and inequality constraints together.
        # For equality constraints, just set the g_l and g_u to be
        # equal. For this driver, the inequality constraints come
        # first. g_l is set to 0.0 and g_l is set to the largest float.
        # For the equality constraints, both g_l and g_u are set to zero.
        g_L = zeros( ncon, 'd')
        g_U = zeros( ncon, 'd')
        g_U = ones( n_ineq_con, 'd') * sys.float_info.max

        nnzj = nvar * ncon # number of non zeros in Jacobian
                           # of constraints. Assumed to be dense
        nnzh = nvar * ( nvar + 1 ) / 2 # number of non zeros in hessian

        try:
            self.nlp = pyipopt.create(
               nvar, x_L, x_U,
               ncon, g_L, g_U,
               nnzj, nnzh,
               eval_f, eval_grad_f,
               eval_g, eval_jac_g,
               intermediate_callback
               # f2py lets you pass extra args to
               # callback functions
               # http://cens.ioc.ee/projects/f2py2e/usersguide/index.html#call-back-arguments
               # We pass the driver itself to the callbacks
               ### not using them for now
               #             eval_f_extra_args = (self,),
               #             eval_grad_f_extra_args = (self,),
               #             eval_g_extra_args = (self,),
               #             eval_jac_g_extra_args = (self,),
               #             intermediate_cb_extra_args = (self,),
               )
            
            self.nlp.set_intermediate_callback( intermediate_callback )

        except Exception, err:
            self._logger.error(str(err))
            raise

        # Set optimization options
        self.nlp.int_option( 'print_level', self.print_level )
        self.nlp.num_option( 'tol', self.tol )

        self.nlp.int_option( 'max_iter', self.max_iter )
        self.nlp.num_option( 'max_cpu_time', self.max_cpu_time )
        self.nlp.num_option( 'constr_viol_tol', self.constr_viol_tol )
        self.nlp.num_option( 'obj_scaling_factor', self.obj_scaling_factor )
        self.nlp.str_option( 'linear_solver', self.linear_solver )
        
        for option, value in self.options.iteritems():
            if isinstance( value, int ):
                self.nlp.int_option( option, value )
            elif isinstance( value, str ):
                self.nlp.str_option( option, value )
            elif isinstance( value, float ):
                self.nlp.num_option( option, value )
            else:
                self.raise_exception("Cannot handle option '%s' of type '%s'" \
                                   % ( option, type(value)), ValueError)
               
        # does the Hessian calculation so we do not have to
        self.nlp.str_option( "hessian_approximation", "limited-memory" )
        # suppresses the copyright banner and all output
        #self.nlp.str_option( "sb", 'yes' )
        self.nlp.str_option( "suppress_all_output", 'yes' )
        
    def continue_iteration(self):
        """Returns True if iteration should continue.
             Get info from the optimizer to see if it
             is done iterating
        """

        return self.iter_count == 0
    
    def pre_iteration(self):
        """Checks or RunStopped and evaluates objective"""
        
        super(IPOPTdriver, self).pre_iteration()
        if self._stop:
            self.raise_exception('Stop requested', RunStopped)
            
    def run_iteration(self):
        """ The IPOPT driver iteration"""
        
        try:
            ( self.design_vals,
              self.zl, self.zu,
              self.obj,
              self.status ) = self.nlp.solve(self.design_vals, self)

        # so we can check for stops
        except Exception, err:
            self._logger.error(str(err))
            raise

                        
        # update the parameters in the model
        dvals = [float(val) for val in self.design_vals]
        self.set_parameters(dvals)

        # update the model
        super(IPOPTdriver, self).run_iteration()

        
    def post_iteration(self):
        """ Checks IPOPT's return status and writes out cases"""

        super(IPOPTdriver, self).post_iteration()
        
        
    def _config_ipopt(self):
        """Set up arrays, and perform some
        validation and make sure that array sizes are consistent.
        """
        params = self.get_parameters().values()
        
        # size arrays based on number of parameters
        num_dvs = len(params)
        self.design_vals = zeros(num_dvs, 'd')

        if num_dvs < 1:
            self.raise_exception('no parameters specified', RuntimeError)
            
        # size constraint related arrays
        length = len(self.get_ineq_constraints()) + \
                 len(self.get_eq_constraints())
        self.constraint_vals = zeros(length, 'd')
        

