"""
Test the IPOPT optimizer component using a variety of
  problems and special cases.

Problem                     | Objective | nvars | Lin Cons | Nonlin Cons | 
==========================================================================
Rosen Suzuki                | nonlinear |   4   |    0     |      3      | 
IPOPT Manual #1             | linear    |   2   |    2     |      1      | 
Constrained Betts           | nonlinear |   2   |    1     |      0      | 
Paraboloid                  | nonlinear |   2   |    0     |      0      ! 
Paraboloid w lin constraint | nonlinear |   2   |    1     |      0      | 
Paraboloid w lin constraint | nonlinear |   2   |  1 and 3 |      0      | 
Paraboloid w non-lin const  | nonlinear |   2   |    0     |      1      | 

"""


import sys
import os
import unittest

import numpy

# disable complaints about missing doc strings in tests
# pylint: disable-msg=C0111

# disable complaints about .__init__: Use super on an old style class
# pylint: disable-msg=E1002

# disable complaints about Module 'numpy' has no 'array' member
# pylint: disable-msg=E1101 

# Disable complaints about Too few public methods
# pylint: disable-msg=R0903

# Disable complaints Invalid name "setUp" (should match [a-z_][a-z0-9_]{2,30}$)
# pylint: disable-msg=C0103

# Disable complaints Comma not followed by a space
# pylint: disable-msg=C0324

# Disable complaints Used builtin function 'map'
# pylint: disable-msg=W0141

# Disable complaints Too many public methods
# pylint: disable-msg=R0904

# Disable complaints about not being able to import modules that Python
#     really can import
# pylint: disable-msg=F0401,E0611

from openmdao.main.api import Assembly, Component, set_as_top
from openmdao.lib.datatypes.api import Float, Array
from openmdao.lib.drivers.api import IPOPTdriver
from openmdao.util.testutil import assert_rel_error
from openmdao.lib.datatypes.api import TraitError

from openmdao.lib.drivers.ipoptdriver import IpoptReturnStatus


class OptRosenSuzukiComponent(Component):
    """ From the NEWSUMT User's Manual:
    EXAMPLE 2 - CONSTRAINED ROSEN-SUZUKI FUNCTION. NO GRADIENT INFORMATION.
    
         MINIMIZE OBJ = X(1)**2 - 5*X(1) + X(2)**2 - 5*X(2) +
                        2*X(3)**2 - 21*X(3) + X(4)**2 + 7*X(4) + 50
    
         Subject to:
    
              G(1) = X(1)**2 + X(1) + X(2)**2 - X(2) +
                     X(3)**2 + X(3) + X(4)**2 - X(4) - 8   .LE.0
    
              G(2) = X(1)**2 - X(1) + 2*X(2)**2 + X(3)**2 +
                     2*X(4)**2 - X(4) - 10                  .LE.0
    
              G(3) = 2*X(1)**2 + 2*X(1) + X(2)**2 - X(2) +
                     X(3)**2 - X(4) - 5                     .LE.0
                     
    This problem is solved beginning with an initial X-vector of
         X = (1.0, 1.0, 1.0, 1.0)
    The optimum design is known to be
         OBJ = 6.000
    and the corresponding X-vector is
         X = (0.0, 1.0, 2.0, -1.0)
    """
    
    x = Array(iotype='in')
    result = Float(iotype='out')
    
    # pylint: disable-msg=C0103
    def __init__(self, doc=None):
        """Initialize"""
        
        super(OptRosenSuzukiComponent, self).__init__(doc)
        # Initial guess
        self.x = numpy.array([1., 1., 1., 1.], dtype=float)
        self.result = 0.
        
        self.opt_objective = 6.
        self.opt_design_vars = [0., 1., 2., -1.]

    def execute(self):
        """calculate the new objective value"""

        
        self.result = (self.x[0]**2 - 5.*self.x[0] + 
                       self.x[1]**2 - 5.*self.x[1] +
                       2.*self.x[2]**2 - 21.*self.x[2] + 
                       self.x[3]**2 + 7.*self.x[3] + 50)

class Example1FromManualComponent(Component):
    """ From the NEWSUMT User's Manual:

         EXAMPLE 1
    
         MINIMIZE OBJ = 10.0 * X(1) + X(2)
    
         Subject to:
    
              G(1) = 2.0 * X(1) - X(2) - 1.0 > 0
              G(2) = X(1) - 2.0 * X(2) + 1.0 > 0    
              G(3) = - X(1)**2 + 2.0 * ( X(1) + X(2) ) - 1.0 > 0
                     
    This problem is solved beginning with an initial X-vector of
         X = (2.0, 1.0)
    The optimum design is known to be
         OBJ = 5.5917
    and the corresponding X-vector is
         X = (0.5515, 0.1006)
    """
    
    x = Array(iotype='in')
    result = Float(iotype='out')
    
    # pylint: disable-msg=C0103
    def __init__(self, doc=None):
        """Initialize"""
        
        super(Example1FromManualComponent, self).__init__(doc)
        # Initial guess
        self.x = numpy.array([2.0, 1.0], dtype=float)

        self.result = 0.0
        
        self.opt_objective = 5.5917
        self.opt_design_vars = [0.5515, 0.1006]

    def execute(self):
        """calculate the new objective value"""
        
        self.result = (10.0 * self.x[0] + self.x[1] )



class ParaboloidComponent(Component):
    """     
         MINIMIZE OBJ = ( X(1) - 2.0 ) ** 2 +  ( X(2) - 3.0 ) **2
    """
    
    x = Array(iotype='in')
    result = Float(iotype='out')
    
    # pylint: disable-msg=C0103
    def __init__(self, doc=None):
        super(ParaboloidComponent, self).__init__(doc)
        self.x = numpy.array([10., 10.], dtype=float) # initial guess

        self.result = 0.
        
        self.opt_objective = 0.
        self.opt_design_vars = [2., 3.]

    def execute(self):
        """calculate the new objective value"""
        self.result = (self.x[0] - 2.0) ** 2 + (self.x[1] - 3.0) ** 2


class QuadParaboloidComponent(Component):
    """     
         MINIMIZE OBJ = ( X(1) - 2.0 ) ** 4 +  ( X(2) - 3.0 ) **4
    """
    
    x = Array(iotype='in')
    result = Float(iotype='out')
    
    # pylint: disable-msg=C0103
    def __init__(self, doc=None):
        super(QuadParaboloidComponent, self).__init__(doc)
        self.x = numpy.array([10., 10.], dtype=float) # initial guess

        self.result = 0.
        
        self.opt_objective = 0.
        self.opt_design_vars = [2., 3.]

    def execute(self):
        """calculate the new objective value"""
        self.result = (self.x[0] - 2.0) ** 4 + (self.x[1] - 3.0) ** 4


class ConstrainedBettsComponent(Component):
    """     
         MINIMIZE OBJ = 0.01 * x(1) **2 + x(2) ** 2 - 100.0
    
         Subject to:

              2 <= x(1) <= 50
            -50 <= x(2) <= 50
    
              10 * x(1) - x(2) >= 10.0

                  or

              10.0 - 10.0 * x(1) + x(2) <= 0.0

                  or

              - 10.0 + 10.0 * x(1) - x(2) >= 0.0
                     
    This problem is solved beginning with an initial X-vector of
         X = (-1.0, - 1.0 )
    The optimum design is known to be
         OBJ = - 99.96 
    and the corresponding X-vector is
         X = (2.0, 0.0 )
    """
    
    x = Array(iotype='in')
    result = Float(iotype='out')
    
    # pylint: disable-msg=C0103
    def __init__(self, doc=None):
        super(ConstrainedBettsComponent, self).__init__(doc)
        self.x = numpy.array([-1.0, -1.0], dtype=float) # initial guess
        self.result = 0.
        
        self.opt_objective = -99.96
        self.opt_design_vars = [2.0, 0.0]

    def execute(self):
        """calculate the new objective value"""

        self.result = 0.01 * self.x[0] ** 2 + self.x[1] ** 2 - 100.0



class IPOPTdriverParaboloidTestCase(unittest.TestCase):
    """test IPOPT optimizer component using an unconstrained
    paraboloid function"""

    def setUp(self):
        '''setup'''
        self.top = set_as_top(Assembly())
        self.top.add('comp', ParaboloidComponent())
        self.top.add('driver', IPOPTdriver())
        self.top.driver.workflow.add('comp')


        self.top.driver.print_level = 0
        
        map(self.top.driver.add_constraint,[ ])    
        
    def tearDown(self):
        '''tear down'''
        
        self.top = None

    def test_opt1(self):
        
        self.top.driver.add_objective('comp.result')

        self.top.driver.add_parameters( [
            ('comp.x[0]', -100.0, 100.0),
            ('comp.x[1]', -100.0, 100.0),
            ] )
        
        self.top.run()
        self.assertAlmostEqual(self.top.comp.opt_objective, 
                               self.top.driver.eval_objective(), places=2)
        self.assertAlmostEqual(self.top.comp.opt_design_vars[0], 
                               self.top.comp.x[0], places=2)
        self.assertAlmostEqual(self.top.comp.opt_design_vars[1], 
                               self.top.comp.x[1], places=2)
        self.assertEqual( IpoptReturnStatus.Solve_Succeeded, self.top.driver.status )

    def test_invalid_parameter_name(self):
        
        self.top.driver.add_objective('comp.result')

        self.top.driver.add_parameters( [
            ('comp.x[0]', -100.0, 100.0),
            ('comp.x[1]', -100.0, 100.0),
            ] )
        
        self.top.driver.options = {'invalid_parameter_name':0,}
        try:
            self.top.run()
        except ValueError, err:
            self.assertEqual(str(err), "invalid_parameter_name is not a valid int option" )
        else:
            self.fail('ValueError expected')

        self.assertEqual( None, self.top.driver.status )

    
    def test_invalid_parameter_type(self):
        # Check to see if we pass in a dictionary
        #   as one of the Ipopt options
        
        self.top.driver.add_objective('comp.result')

        self.top.driver.add_parameters( [
            ('comp.x[0]', -100.0, 100.0),
            ('comp.x[1]', -100.0, 100.0),
            ] )
        
        self.top.driver.options = {'invalid_parameter_type':{},}
        try:
            self.top.run()
        except ValueError, err:
            self.assertEqual(str(err), "driver: Cannot handle " + \
                             "option 'invalid_parameter_type' of type '<type 'dict'>'" )
        else:
            self.fail('ValueError expected')
        self.assertEqual( None, self.top.driver.status )

    def test_invalid_parameter_value(self):
        
        self.top.driver.add_objective('comp.result')

        self.top.driver.add_parameters( [
            ('comp.x[0]', -100.0, 100.0),
            ('comp.x[1]', -100.0, 100.0),
            ] )

        self.print_level = 0
        self.top.driver.options = {'max_iter':-99}
        try:
            self.top.run()
        except ValueError, err:
            # This is the exception message that is thrown. It is misleading
            #   since it really is a valid name for an option, it just
            #   has an invalid value
            self.assertEqual(str(err), "max_iter is not a valid int option" )
        else:
            self.fail('ValueError expected')
        self.assertEqual( None, self.top.driver.status )
    

    def test_max_iteration_too_low(self):

        self.top.driver.add_objective('comp.result')

        self.top.driver.add_parameters( [
            ('comp.x[0]', -100.0, 100.0),
            ('comp.x[1]', -100.0, 100.0),
            ] )

        self.top.driver.print_level = 0
        self.top.driver.max_iter = 1
        
        self.top.run()
        self.assertEqual( IpoptReturnStatus.Maximum_Iterations_Exceeded, self.top.driver.status )

    def test_using_max_cpu_time_option(self):

        self.top.driver.add_objective('comp.result')

        self.top.driver.add_parameters( [
            ('comp.x[0]', -100.0, 100.0),
            ('comp.x[1]', -100.0, 100.0),
            ] )

        self.top.driver.print_level = 0 
        self.top.driver.max_cpu_time = 0.001
        
        self.top.run()
        self.assertEqual( IpoptReturnStatus.Maximum_CpuTime_Exceeded, self.top.driver.status )
    
    def test_using_mumps_linear_solver(self):

        self.top.driver.add_objective('comp.result')

        self.top.driver.add_parameters( [
            ('comp.x[0]', -100.0, 100.0),
            ('comp.x[1]', -100.0, 100.0),
            ] )

        self.top.driver.print_level = 0
        self.top.driver.linear_solver = 'mumps'
        
        self.top.run()
        self.assertAlmostEqual(self.top.comp.opt_objective, 
                               self.top.driver.eval_objective(), places=2)
        self.assertAlmostEqual(self.top.comp.opt_design_vars[0], 
                               self.top.comp.x[0], places=2)
        self.assertAlmostEqual(self.top.comp.opt_design_vars[1], 
                               self.top.comp.x[1], places=2)
        self.assertEqual( IpoptReturnStatus.Solve_Succeeded, self.top.driver.status )
   
    def test_setting_obj_scaling_factor(self):

        self.top.driver.add_objective('comp.result')

        self.top.driver.add_parameters( [
            ('comp.x[0]', -100.0, 100.0),
            ('comp.x[1]', -100.0, 100.0),
            ] )

        self.top.driver.print_level = 0
        self.top.driver.obj_scaling_factor = 1.0+6
        
        self.top.run()
        self.assertAlmostEqual(self.top.comp.opt_objective, 
                               self.top.driver.eval_objective(), places=2)
        self.assertAlmostEqual(self.top.comp.opt_design_vars[0], 
                               self.top.comp.x[0], places=2)
        self.assertAlmostEqual(self.top.comp.opt_design_vars[1], 
                               self.top.comp.x[1], places=2)
        self.assertEqual( IpoptReturnStatus.Solve_Succeeded, self.top.driver.status )

    def test_no_feasible_solution(self):

        self.top.driver.add_objective('comp.result')

        self.top.driver.add_parameters( [
            ('comp.x[0]', -100.0, 100.0),
            ('comp.x[1]', -100.0, 100.0),
            ] )

        map(self.top.driver.add_constraint,[
            'comp.x[0] - 4.0 > 0.0',
            'comp.x[0] - 3.0 < 0.0',
            ] )
        self.top.driver.print_level = 0
        
        self.top.run()

        self.assertEqual( IpoptReturnStatus.Infeasible_Problem_Detected, self.top.driver.status )

    def test_invalid_linear_solver(self):
        
        self.top.driver.add_objective('comp.result')

        self.top.driver.add_parameters( [
            ('comp.x[0]', -100.0, 100.0),
            ('comp.x[1]', -100.0, 100.0),
            ] )

        try:
            self.top.driver.linear_solver = 'invalid_solver'
        except TraitError, err:
            self.assertEqual(str(err),
                             "driver: Trait 'linear_solver' must be " + \
                             "in ['ma27', 'ma57', 'ma77', 'pardiso', " + \
                             "'wsmp', 'mumps', 'custom'], but a value " + \
                             "of invalid_solver <type 'str'> was specified." )
            
        else:
            self.fail('TraitError expected')
        self.assertEqual( None, self.top.driver.status )

    def test_add_option_wrappers(self):
        
        self.top.driver.add_objective('comp.result')

        self.top.driver.add_parameters( [
            ('comp.x[0]', -100.0, 100.0),
            ('comp.x[1]', -100.0, 100.0),
            ] )

        self.top.run() # need to do this so we get nlp set

        try:
            self.top.driver.nlp.str_option( 'linear_solver', 44 )
        except TypeError, err:
            self.assertEqual(str(err),
                             "str_option() argument 2 must be string, not int" ) 
        else:
            self.fail('TypeError expected')

        try:
            self.top.driver.nlp.str_option( 'linear_solver', 'invalid_solver' )
        except ValueError, err:
            self.assertEqual(str(err),
                             "linear_solver is not a valid string option" )
        else:
            self.fail('ValueError expected')

        try:
            self.top.driver.nlp.num_option( 'max_cpu_time', 'string' )
        except TypeError, err:
            self.assertEqual(str(err), "a float is required" ) 
        else:
            self.fail('TypeError expected')

        try:
            self.top.driver.nlp.int_option( 'max_iter', 'string' )
        except TypeError, err:
            self.assertEqual(str(err), "an integer is required" ) 
        else:
            self.fail('TypeError expected')

        self.assertEqual( 0, self.top.driver.status )

    
class IPOPTdriverParaboloidWithLinearConstraintTestCase(unittest.TestCase):
    """test IPOPT optimizer component using a
    paraboloid function constrained by a linear constraint"""

    def setUp(self):
        '''setUp'''
        
        self.top = set_as_top(Assembly())
        self.top.add('comp', ParaboloidComponent())
        self.top.add('driver', IPOPTdriver())
        self.top.driver.workflow.add('comp')
        self.top.driver.print_level = 0
        
    def tearDown(self):
        ''' tear down'''

        self.top = None

    def test_opt1(self):

        self.top.driver.add_objective( 'comp.result' )
        
        self.top.driver.add_parameters( [
            ('comp.x[0]', -100.0, 100.0),
            ('comp.x[1]', -100.0, 100.0),
            ] )

        map(self.top.driver.add_constraint,[
            'comp.x[0] - 4.0 > 0.0',
            ] )
        self.top.run()
        self.assertAlmostEqual(4.0, 
                               self.top.driver.eval_objective(), places=2)
        self.assertAlmostEqual(4.0,
                               self.top.comp.x[0], places=2)
        self.assertAlmostEqual(3.0,
                               self.top.comp.x[1], places=2)


    def test_opt1_with_three_constraints(self):

        self.top.driver.add_objective( 'comp.result' )
        
        self.top.driver.add_parameters( [
            ('comp.x[0]', -100.0, 100.0),
            ('comp.x[1]', -100.0, 100.0),
            ] )

        map(self.top.driver.add_constraint,[
            'comp.x[0] - 4.0 > 0.0',
            'comp.x[0] - 3.0 > 0.0',
            ] )
        self.top.run()
        self.assertAlmostEqual(4.0, 
                               self.top.driver.eval_objective(), places=2)
        self.assertAlmostEqual(4.0,
                               self.top.comp.x[0], places=2)
        self.assertAlmostEqual(3.0,
                               self.top.comp.x[1], places=2)




class IPOPTdriverParaboloidWithLinearEqualityTestCase(unittest.TestCase):
    """test IPOPT optimizer component using a
    paraboloid function constrained by a linear equality constraint"""

    def setUp(self):
        '''setUp'''
        
        self.top = set_as_top(Assembly())
        self.top.add('comp', ParaboloidComponent())
        self.top.add('driver', IPOPTdriver())
        self.top.driver.workflow.add('comp')
        self.top.driver.print_level = 0
        
    def tearDown(self):
        ''' tear down'''

        self.top = None

    def test_opt1(self):

        self.top.driver.add_objective( 'comp.result' )
        
        self.top.driver.add_parameters( [
            ('comp.x[0]', -100.0, 100.0),
            ('comp.x[1]', -100.0, 100.0),
            ] )

        map(self.top.driver.add_constraint,[
            'comp.x[0] - 4.0 = 0.0',
            ] )
        self.top.run()
        self.assertAlmostEqual(4.0, 
                               self.top.driver.eval_objective(), places=2)
        self.assertAlmostEqual(4.0,
                               self.top.comp.x[0], places=2)
        self.assertAlmostEqual(3.0,
                               self.top.comp.x[1], places=2)



class IPOPTdriverParaboloidWithNonLinearConstraintTestCase(unittest.TestCase):
    """test IPOPT optimizer component using a
    paraboloid function constrained by a nonlinear constraint"""

    def setUp(self):
        '''setup test'''
        self.top = set_as_top(Assembly())
        self.top.add('comp', ParaboloidComponent())
        self.top.add('driver', IPOPTdriver())
        self.top.driver.workflow.add('comp')
        self.top.driver.print_level = 0
        
    def tearDown(self):
        '''tear down'''
        self.top = None

    def test_opt1(self):

        self.top.driver.add_objective( 'comp.result' )

        self.top.driver.add_parameters( [
            ('comp.x[0]', -100.0, 100.0),
            ('comp.x[1]', -100.0, 100.0),
            ] )

        map(self.top.driver.add_constraint,[ '- comp.x[0]**2 - ( comp.x[1] - 3.0 )**2 + 1.0 > 0.0' ] )
        self.top.run()

        self.assertAlmostEqual(1.0, 
                               self.top.driver.eval_objective(), places=2)
        self.assertAlmostEqual(1.0,
                               self.top.comp.x[0], places=2)
        self.assertAlmostEqual(3.0,
                               self.top.comp.x[1], places=2)

    def test_fd_delta_set_large(self):

        self.top.driver.add_objective( 'comp.result' )

        self.top.driver.add_parameters( [
            ('comp.x[0]', -100.0, 100.0),
            ('comp.x[1]', -100.0, 100.0),
            ] )

        self.top.driver.fd_delta = 9999.0

        map(self.top.driver.add_constraint,[ '- comp.x[0]**2 - ( comp.x[1] - 3.0 )**2 + 1.0 > 0.0' ] )
        self.top.run()

        self.assertAlmostEqual(1.0, 
                               self.top.driver.eval_objective(), places=2)
        self.assertAlmostEqual(1.0,
                               self.top.comp.x[0], places=2)
        self.assertAlmostEqual(3.0,
                               self.top.comp.x[1], places=2)


class IPOPTdriverRosenSuzukiTestCase(unittest.TestCase):
    """test IPOPT optimizer component using the Rosen Suzuki problem"""

    def setUp(self):
        '''setup test'''
        self.top = set_as_top(Assembly())
        self.top.add('comp', OptRosenSuzukiComponent())
        self.top.add('driver', IPOPTdriver())
        self.top.driver.workflow.add('comp')
        self.top.driver.options = {'print_level':0,}
        
    def tearDown(self):
        '''tear down'''
        self.top = None
        
    def test_opt1(self):

        self.top.driver.add_objective('comp.result')

        self.top.driver.add_parameters( [
            ('comp.x[0]', -10.0, 99.0),
            ('comp.x[1]', -10.0, 99.0),
            ('comp.x[2]', -10.0, 99.0),
            ('comp.x[3]', -10.0, 99.0),
            ] )

        map(self.top.driver.add_constraint,[
            'comp.x[0]**2+comp.x[0]+comp.x[1]**2-comp.x[1]+comp.x[2]**2+comp.x[2]+comp.x[3]**2-comp.x[3] < 8',
            'comp.x[0]**2-comp.x[0]+2*comp.x[1]**2+comp.x[2]**2+2*comp.x[3]**2-comp.x[3] < 10',
            '2*comp.x[0]**2+2*comp.x[0]+comp.x[1]**2-comp.x[1]+comp.x[2]**2-comp.x[3] < 5'])        

        self.top.run()

        self.assertAlmostEqual(self.top.comp.opt_objective, 
                               self.top.driver.eval_objective(), places=2)
        self.assertAlmostEqual(self.top.comp.opt_design_vars[0], 
                               self.top.comp.x[0], places=1)
        self.assertAlmostEqual(self.top.comp.opt_design_vars[1], 
                               self.top.comp.x[1], places=2)
        self.assertAlmostEqual(self.top.comp.opt_design_vars[2], 
                               self.top.comp.x[2], places=2)
        self.assertAlmostEqual(self.top.comp.opt_design_vars[3], 
                               self.top.comp.x[3], places=1)

class IPOPTdriverExample1FromManualTestCase(unittest.TestCase):
    """
      Example 1 from the NEWSUMT manual
    """

    def setUp(self):
        '''setup test'''
        self.top = set_as_top(Assembly())
        self.top.add('comp', Example1FromManualComponent())
        self.top.add('driver', IPOPTdriver())
        self.top.driver.workflow.add('comp')
        self.top.driver.print_level = 0
        
    def tearDown(self):
        '''tear down'''
        self.top = None
        
    def test_opt1(self):

        self.top.driver.add_objective('comp.result')
        
        self.top.driver.add_parameters( [
            ('comp.x[0]', 0.0, 100.0),
            ('comp.x[1]', 0.0, 100.0),
            ] )

        map(self.top.driver.add_constraint,[
            '2.0 * comp.x[0] - comp.x[1] - 1.0 > 0.0',
            'comp.x[0] - 2.0 * comp.x[1] + 1.0 > 0.0',
            '- comp.x[0]**2 + 2.0 * ( comp.x[0] + comp.x[1]) - 1.0 > 0.0'
            ])

        self.top.run()

        assert_rel_error(self,
                         self.top.driver.eval_objective(),
                         self.top.comp.opt_objective, 
                         0.005)
        self.assertAlmostEqual(self.top.comp.opt_design_vars[0], 
                               self.top.comp.x[0], places=2)
        self.assertAlmostEqual(self.top.comp.opt_design_vars[1], 
                               self.top.comp.x[1], places=2)




class IPOPTdriverConstrainedBettsTestCase(unittest.TestCase):
    """test IPOPT optimizer component for the Constrained Betts problem
    
    """

    def setUp(self):
        '''setup test'''
        self.top = set_as_top(Assembly())
        self.top.add('comp', ConstrainedBettsComponent())
        self.top.add('driver', IPOPTdriver())
        self.top.driver.workflow.add('comp')
        self.top.driver.options = {'print_level':0,}
        
    def tearDown(self):
        '''tear down'''
        self.top = None

    def test_opt1(self):

        self.top.driver.add_objective( 'comp.result' )

        self.top.driver.add_parameters( [
            ('comp.x[0]', 2.0, 50.0),
            ('comp.x[1]', -50.0, 50.0),
            ] )
        
        map(self.top.driver.add_constraint,[ '-10.0 + 10.0 * comp.x[0] - comp.x[1] > 0.0' ] )

        self.top.run()

        assert_rel_error(self,
                         self.top.comp.opt_objective, 
                         self.top.driver.eval_objective(),
                         0.001)
        self.assertAlmostEqual(self.top.comp.opt_design_vars[0], 
                               self.top.comp.x[0], places=2)
        self.assertAlmostEqual(self.top.comp.opt_design_vars[1], 
                               self.top.comp.x[1], places=2)

    def test_setting_optional_options(self):

        self.top.driver.options = {'derivative_test':'first-order',}
        self.top.driver.add_objective( 'comp.result' )

        self.top.driver.add_parameters( [
            ('comp.x[0]', 2.0, 50.0),
            ('comp.x[1]', -50.0, 50.0),
            ] )
        
        map(self.top.driver.add_constraint,[ '-10.0 + 10.0 * comp.x[0] - comp.x[1] > 0.0' ] )

        self.top.run()

        assert_rel_error(self,
                         self.top.comp.opt_objective, 
                         self.top.driver.eval_objective(),
                         0.001)
        self.assertAlmostEqual(self.top.comp.opt_design_vars[0], 
                               self.top.comp.x[0], places=2)
        self.assertAlmostEqual(self.top.comp.opt_design_vars[1], 
                               self.top.comp.x[1], places=2)

    def test_fd_delta_set_too_large(self):

        self.top.driver.add_objective( 'comp.result' )

        self.top.driver.add_parameters( [
            ('comp.x[0]', 2.0, 50.0),
            ('comp.x[1]', -50.0, 50.0),
            ] )
        
        self.top.driver.fd_delta = 99999999999999999999999999999999999999999.0

        map(self.top.driver.add_constraint,[ '-10.0 + 10.0 * comp.x[0] - comp.x[1] > 0.0' ] )

        self.top.run()

        actual = self.top.driver.eval_objective()
        desired =  self.top.comp.opt_objective
        tolerance = 0.001
        error = (actual - desired) / desired
        if abs(error) < tolerance:
            self.fail('actual %s, desired %s; error %s should be > tolerance %s'
                           % (actual, desired, error, tolerance))


if __name__ == "__main__":

    suite = unittest.TestSuite()

    suite.addTest(unittest.makeSuite(IPOPTdriverParaboloidTestCase))
    suite.addTest(unittest.makeSuite(IPOPTdriverParaboloidWithLinearConstraintTestCase))
    suite.addTest(unittest.makeSuite(IPOPTdriverParaboloidWithNonLinearConstraintTestCase))
    suite.addTest(unittest.makeSuite(IPOPTdriverConstrainedBettsTestCase))
    suite.addTest(unittest.makeSuite(IPOPTdriverRosenSuzukiTestCase))
    suite.addTest(unittest.makeSuite(IPOPTdriverExample1FromManualTestCase))

    results = unittest.TextTestRunner(verbosity=2).run(suite)

