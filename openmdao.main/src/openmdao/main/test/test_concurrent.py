"""
Test various control flow patterns.
"""

import time
import traceback

#import router
#import selector

from enthought.traits.api import Int, Str, Bool
from openmdao.main.api import Container, Component, Assembly
from openmdao.main.exceptions import CircularDependencyError
from openmdao.main.container import state_strings

import unittest
import logging


# pylint: disable-msg=E1101
# "Instance of <class> has no <attr> member"

VERBOSE = True


class Sleepy(Component):
    """ Set each output to sum of inputs. """

    in1 = Str(io_direction='in')
    in2 = Str(io_direction='in')
    out1 = Str(io_direction='out')
    
    def __init__(self, delay=0):
        Component.__init__(self)
        self.delay = delay

    def execute(self):
        """ Delay, then set outputs to sum of inputs. """
        time.sleep(self.delay)
        self.out1 = self.in1 + self.in2

class Integrator(Sleepy):
    """ Sum outputs from each run. """

    integral = Str(io_direction='out')
    
    def __init__(self, delay=0):
        Sleepy.__init__(self, delay=delay)

    def execute(self):
        """ Delay, then set outputs to sum/integral of inputs. """
        time.sleep(self.delay)
        self.out1 = self.in1 + self.in2
        self.integral = self.integral + self.in1 + self.in2


class Compressor(Component):
    """ Set 'extrapolated' output if inflow > map. """

    inflow = Int(io_direction='in')
    mymap = Int(io_direction='in')
    outflow = Int(io_direction='out')
    extrapolated = Bool(io_direction='out')
    
    def __init__(self, name, parent, delay=0):
        Component.__init__(self, name, parent)
        self.delay = delay

    def execute(self):
        """
        Delay, then set outflow to sum of inflow and mymap.
        Set extrapolated True if inflow > mymap.
        """
        logging.debug('             %s running...' % self.name)
        time.sleep(self.delay)
        self.outflow = self.inflow + self.mymap
        self.extrapolated = self.inflow > self.mymap


class MapGenerator(Component):
    """ Generate a 'updated map' by incrementing previous value. """

    inflow = Int(io_direction='in')
    mymap = Int(1, io_direction='out')
    
    def __init__(self, delay=0):
        Component.__init__(self)
        self.delay = delay

    def execute(self):
        """ delay, then increment map value. """
        logging.debug('             %s running...' % self.name)
        time.sleep(self.delay)
        self.mymap += 1

def display(comp, prefix='', suffix=''):
    """ Display state. """
    print prefix, comp.name, suffix
    prefix += '   '
    inputs = {}
    outputs = {}
    containers = {}

    for attr,obj in comp.items():
        if attr != 'parent':
            tattr = comp.trait(attr)
            if tattr.io_direction == 'in':
                inputs[attr] = (obj, 
                                '' if comp.get_valid(attr) else 'invalid',
                                '' if comp.get_enabled(attr) else 'disabled')
            elif tattr.io_direction == 'out':
                outputs[attr] = (obj, 
                                '' if comp.get_valid(attr) else 'invalid',
                                '' if comp.get_enabled(attr) else 'disabled')
            elif isinstance(obj, Container):
                containers[attr] = obj

    for name in sorted(inputs.keys()):
        print prefix, name, inputs[name]
    for name in sorted(containers.keys()):
        if isinstance(containers[name], Component):
            comp = containers[name]
            suffix = state_strings[comp.state] if comp._enabled else 'disabled'
        else:
            suffix = ''
        display(containers[name], prefix, suffix)
        
    for name in sorted(outputs.keys()):
        print prefix, name, outputs[name]
            
class TestRig(Assembly):

    def __init__(self, state_table, data_table):
        Assembly.__init__(self)
        self.workflow.record_states = True
        self.state_table = state_table
        self.data_table = data_table
            
    def run(self, msg=None, component=None):
        """ Run test. """
        if msg and VERBOSE:
            print '\n'+msg

        if VERBOSE:
            print "\nBefore:"
            display(self)
            print
        
        try:
            if component is None:
                super(TestRig,self).run()
            else:
                component.run()
        except Exception, exc:
            print 'Caught', exc
            traceback.print_exc()

        if VERBOSE:
            print '\nAfter:'
            display(self)
            if self.workflow.record_states:
                print '\nDispatch table:'
                for i, row in enumerate(self.workflow.dispatch_table):
                    print '   ', i, sorted(row)

        self.verify_dispatch_table(self.state_table)

        for i, data in enumerate(self.data_table):
            vref, expected = data
            actual = '%s' % self.resolve(vref)
            if actual != expected:
                print 'ERROR: [%d] %s %s vs. %s' \
                      % (i, vref, actual, expected)

    def resolve(self, ref):
        """ Return object referenced by `ref`. """
        if isinstance(ref, basestring):
            names = ref.split('.')
            obj = self
            for name in names[:-1]:
                obj = getattr(obj, name)
            ref = getattr(obj, names[-1])
        return ref

    def step(self, states):
        """ Perform one step of evaluation and verify dispatch table. """
        super(TestRig,self).step()
        self.verify_dispatch_table(states)

    def verify_dispatch_table(self, expected_table):
        """ Verify that dispatch table has expected contents. """
        if len(self.workflow.dispatch_table) != len(expected_table):
            print 'ERROR: %d states vs. expected %d' \
                  % (len(self.workflow.dispatch_table), len(expected_table))
        for i, state in enumerate(expected_table):
            if i >= len(self.workflow.dispatch_table):
                break
            actual = sorted(self.workflow.dispatch_table[i])
            expected = sorted(state)
            if actual != expected:
                print 'ERROR: [%d] %s vs. %s' % (i, actual, expected)


class TestCase(unittest.TestCase):
    #def test_basic(self):
        #"""\
              #Basic functionality:
                   #B ---\\
                 #/       \\
               #A - C \\    F
                 #\     E /
                   #D /
              #"""
        #state_table = [
            #['A'],
            #['B', 'C', 'D'],
            #['E'],
            #['F']
        #]
        #data_table = [
            #['F.out1', '(a1a2b2a1a2c2a1a2d2, valid)'],
        #]
        #rig = TestRig(state_table, data_table)
        #rig.name = 'TestRig'

        #A = Sleepy('A', rig)
        ##A.set('in1', 'a1')
        ##A.set('in2', 'a2')

        #B = Sleepy('B', rig, delay=0.1)
        ##B.set('in2', 'b2')

        #C = Sleepy('C', rig, delay=0.2)
        ##C.set('in2', 'c2')

        #D = Sleepy('D', rig, delay=0.3)
        ##D.set('in2', 'd2')

        #E = Sleepy('E', rig)

        #F = Sleepy('F', rig)

        #rig.connect('A.out1', 'B.in1')
        #rig.connect('A.out1', 'C.in1')
        #rig.connect('A.out1', 'D.in1')
        #rig.connect('B.out1', 'F.in1')
        #rig.connect('C.out1', 'E.in1')
        #rig.connect('D.out1', 'E.in2')
        #rig.connect('E.out1', 'F.in2')

        #rig.run()

        ## Re-run with nothing to do.
        #rig.state_table = []
        #rig.run('Nothing to do.')

        ## Step through from beginning.
        #rig.invalidate()
        #rig.step([['A']])
        #rig.step([['A'], ['B', 'C', 'D']])
        #rig.step([['A'], ['B', 'C', 'D']])
        #rig.step([['A'], ['B', 'C', 'D']])
        #rig.step([['A'], ['B', 'C', 'D'], ['E']])
        #rig.step([['A'], ['B', 'C', 'D'], ['E'], ['F']])
        #rig.step([])
        #if rig.state != VALID:
            #print "ERROR: expected rig 'valid', actual '%s'" % rig.state

        ## Re-run sequentially.
        #rig.invalidate()
        #rig.workflow.sequential = True
        #rig.state_table = [
            #['A'],
            #['B'], ['C'], ['D'],
            #['E'],
            #['F']
        #]
        #rig.run('Sequential')


    def test_sequence(self):
        """\
              Pattern 1 - Sequence:
               A - B - C
              """
        state_table = [
            ['A'],
            ['B'],
            ['C'],
        ]
        data_table = [
            ['C.out1', '(a1a2b2c2, valid)'],
        ]
        rig = TestRig(state_table, data_table)
        rig.name = 'TestRig'

        rig.add_container('A', Sleepy())
        rig.A.in1 = 'a1'
        rig.A.in2 = 'a2'

        rig.add_container('B', Sleepy())
        rig.B.in2 = 'b2'

        rig.add_container('C', Sleepy())
        rig.C.in2 = 'c2'

        rig.connect('A.out1', 'B.in1')
        rig.connect('B.out1', 'C.in1')

        # Normal run.
        rig.run()
        display(rig, suffix='after first run')

        # Re-run with nothing to do.
        rig.state_table = []
        rig.run('Nothing to do.')

        # Set new input on A.
        rig.A.in2 = 'a3'
        rig.state_table = [
            ['A'],
            ['B'],
            ['C'],
        ]
        rig.data_table = [
            ['C.out1', '(a1a3b2c2, valid)'],
        ]
        rig.run('New input on A')

        # Set new input on B.
        rig.B.in2 = 'b2'
        rig.state_table = [
            ['B'],
            ['C'],
        ]
        rig.data_table = [
            ['C.out1', '(a1a3b3c2, valid)'],
        ]
        rig.run('New input on B')

        # Set new input on C.
        rig.C.in2 = 'c3'
        rig.state_table = [
            ['C'],
        ]
        rig.data_table = [
            ['C.out1', '(a1a3b3c3, valid)'],
        ]
        rig.run('New input on C')

        # Try to create a loop.
        dst = 'A.in2'
        try:
            rig.connect('C.out1', dst)
        except CircularDependencyError, exc:
            self.assertEqual(str(exc), 
                "circular dependency (['A', 'B', 'C']) would be created by connecting C.out1 to %s"%dst)
        else:
            self.fail('CircularDependencyError expected when connecting C.out1 to %s'%dst)

        # Run individual components.
        rig.A.in2 = 'a4'
        rig.state_table = [
            ['A'],
            ['B'],
        ]
        rig.data_table = [
            ['C.out1', '(a1a4b3c3, valid)'],
        ]
        rig.run('Specify C.run, auto-run predecessors', rig.C)

        rig.A.in2 = 'a5'
        rig.state_table = [
            ['A'],
        ]
        rig.data_table = [
            ['B.out1', '(a1a5b3, valid-linked)'],
        ]
        rig.run('Specify B.run, auto-run predecessor', rig.B)

        # Remove 'B', causing disconnections.
        rig.remove_container('B')
        rig.invalidate_deps(['A.in1'])
        rig.A.in2 = 'a6'
        rig.C.in2 = 'c4'
        rig.state_table = [
            ['A', 'C'],
        ]
        rig.data_table = [
            ['A.out1', '(a1a6, valid)'],
            ['C.in1', '(a1a4b3, valid)'],
            ['C.out1', '(a1a4b3c4, valid)'],
        ]
        rig.run('Now just A and C')


    #def test_parallel_split(self):
        #"""\
              #Pattern 2 - Parallel Split:
               #A - B
                 #\\ C
              #"""
        #state_table = [
            #['A'],
            #['B', 'C'],
        #]
        #data_table = [
            #['B.out1', '(a1a2b2, valid)'],
            #['C.out1', '(a1a2c2, valid)'],
        #]
        #rig = TestRig(state_table, data_table)
        #rig.name = 'TestRig'

        #A = Sleepy('A', rig)
        #A.set('in1', 'a1')
        #A.set('in2', 'a2')

        #B = Sleepy('B', rig, 0.1)
        #B.set('in2', 'b2')

        #C = Sleepy('C', rig, 0.1)
        #C.set('in2', 'c2')

        #rig.connect('A.out1', 'B.in1')
        #rig.connect('A.out1', 'C.in1')

        #rig.run()


    #def test_synchronization(self):
        #"""\
              #Pattern 3 - Synchronization:
               #A - C
               #B /
              #"""
        #state_table = [
            #['A', 'B'],
            #['C'],
        #]
        #data_table = [
            #['C.out1', '(a1a2b1b2, valid)'],
        #]
        #rig = TestRig(state_table, data_table)
        #rig.name = 'TestRig'

        #A = Sleepy('A', rig)
        #A.set('in1', 'a1')
        #A.set('in2', 'a2')

        #B = Sleepy('B', rig, delay=0.1)
        #B.set('in1', 'b1')
        #B.set('in2', 'b2')

        #C = Sleepy('C', rig)

        #rig.connect('A.out1', 'C.in1')
        #rig.connect('B.out1', 'C.in2')

        #rig.run()


    #def test_exclusive_choice(self):
        #"""\
              #Pattern 4 - Exclusive Choice:
                 #/ (if   choice1) - B
               #A - (elif choice2) - C
                 #\\ (else        ) - D
              #"""
        #state_table = [
            #['A'],
            #['R'],
            #['B'],
        #]
        #data_table = [
            #['B.out1', '(6, valid)'],
            #['C.out1', '(None, disabled)'],
            #['D.out1', '(None, disabled)'],
        #]
        #rig = TestRig(state_table, data_table)
        #rig.name = 'TestRig'

        #A = Sleepy('A', rig)
        #A.set('in1', 1)
        #A.set('in2', 2)

        #R = router.ExclusiveChoice('R', rig)
        #Int('x', R, 'in')
        #R.add_branch('choice1', 'self.x.value == 3')
        #R.add_branch('choice2', 'self.x.value == 4')

        #B = Sleepy('B', rig, delay=0.1)
        #B.set('in2', 3)

        #C = Sleepy('C', rig, delay=0.1)
        #C.set('in2', 4)

        #D = Sleepy('D', rig)
        #D.set('in2', 5)

        #rig.connect('A.out1',      'R.x')
        #rig.connect('R.choice1.x', 'B.in1')
        #rig.connect('R.choice2.x', 'C.in1')
        #rig.connect('R.else.x',    'D.in1')

        ## Choice1, 'x == 3' condition.
        #rig.run()

        ## Choice2, 'x == 4' condition.
        #rig.state_table = [
            #['A'],
            #['R'],
            #['C'],
        #]
        #rig.data_table = [
            #['B.out1', '(6, disabled)'],
            #['C.out1', '(8, valid)'],
            #['D.out1', '(None, disabled)'],
        #]
        #A.set('in1', 2)
        #rig.run("Testing 'x == 4' condition...")

        ## 'else' condition.
        #rig.state_table = [
            #['A'],
            #['R'],
            #['D'],
        #]
        #rig.data_table = [
            #['B.out1', '(6, disabled)'],
            #['C.out1', '(8, disabled)'],
            #['D.out1', '(10, valid)'],
        #]
        #A.set('in1', 3)
        #rig.run("Testing 'else' condition...")


    #def test_simple_merge(self):
        #"""\
              #Pattern 5 - Simple Merge:
               #A ----- M - C
                 #\\ B /
              #"""
        #state_table = [
            #['A'],
            #['M', 'B'],
            #['C'],
            #['M'],
            #['C'],
        #]
        #data_table = [
            #['C.out1', '(a1a2b2c2, valid)'],
            #['C.integral', '(a1a2c2a1a2b2c2, valid)'],
        #]
        #rig = TestRig(state_table, data_table)
        #rig.name = 'TestRig'

        #A = Sleepy('A', rig)
        #A.set('in1', 'a1')
        #A.set('in2', 'a2')

        #B = Sleepy('B', rig, delay=0.1)
        #B.set('in2', 'b2')

        #M = selector.SimpleMerge('M', rig)
        #Int('x', M, 'out')
        #M.add_branch('a')
        #M.add_branch('b')

        #C = Integrator('C', rig)
        #C.set('in2', 'c2')

        #rig.connect('A.out1', 'M.a.x')
        #rig.connect('A.out1', 'B.in1')
        #rig.connect('B.out1', 'M.b.x')
        #rig.connect('M.x',    'C.in1')

        #rig.run()


    #def test_multiple_choice(self):
        #"""\
              #Pattern 6 - Multiple Choice:
                 #/ (if choice1) - B
               #A - (if choice2) - C
                 #\\ (otherwise ) - D
              #"""
        #state_table = [
            #['A'],
            #['R'],
            #['B', 'C'],
        #]
        #data_table = [
            #['B.out1', '(6, valid)'],
            #['C.out1', '(7, valid)'],
            #['D.out1', '(None, disabled)'],
        #]
        #rig = TestRig(state_table, data_table)
        #rig.name = 'TestRig'

        #A = Sleepy('A', rig)
        #A.set('in1', 1)
        #A.set('in2', 2)

        #R = router.MultiChoice('R', rig)
        #Int('x', R, 'in')
        #R.add_branch('choice1', 'self.x.value == 3')
        #R.add_branch('choice2', 'self.x.value <= 4')

        #B = Sleepy('B', rig, delay=0.1)
        #B.set('in2', 3)

        #C = Sleepy('C', rig, delay=0.1)
        #C.set('in2', 4)

        #D = Sleepy('D', rig)
        #D.set('in2', 5)

        #rig.connect('A.out1',        'R.x')
        #rig.connect('R.choice1.x',   'B.in1')
        #rig.connect('R.choice2.x',   'C.in1')
        #rig.connect('R.otherwise.x', 'D.in1')

        ## Choices 1&2, 'x == 3' condition.
        #rig.run()

        ## 'otherwise' condition.
        #rig.state_table = [
            #['A'],
            #['R'],
            #['D'],
        #]
        #rig.data_table = [
            #['B.out1', '(6, disabled)'],
            #['C.out1', '(7, disabled)'],
            #['D.out1', '(10, valid)'],
        #]
        #A.set('in1', 3)
        #rig.run("Testing 'otherwise' condition...")


    #def test_synchronizing_merge(self):
        #"""\
              #Pattern 7 - Synchronizing Merge:
                 #/ (if choice1) - B \\
               #A - (if choice2) - C - M - E
                 #\\ (otherwise ) - D /
              #"""
        #state_table = [
            #['A'],
            #['R'],
            #['B', 'C'],
            #['M'],
            #['E']
        #]
        #data_table = [
            #['E.integral', '(13, valid)'],
        #]
        #rig = TestRig(state_table, data_table)
        #rig.name = 'TestRig'

        #A = Sleepy('A', rig)
        #A.set('in1', 1)
        #A.set('in2', 2)

        #R = router.MultiChoice('R', rig)
        #Int('x', R, 'in')
        #R.add_branch('choice1', 'self.x.value == 3')
        #R.add_branch('choice2', 'self.x.value <= 4')

        #B = Sleepy('B', rig, delay=0.1)
        #B.set('in2', 3)

        #C = Sleepy('C', rig, delay=0.2)
        #C.set('in2', 4)

        #D = Sleepy('D', rig)
        #D.set('in2', 5)

        #M = selector.SynchronizingMerge('M', rig)
        #Int('y', M, 'out')
        #M.add_branch('b')
        #M.add_branch('c')
        #M.add_branch('d')

        #E = Integrator('E', rig)
        #E.set('in2', 6)

        #rig.connect('A.out1',        'R.x')
        #rig.connect('R.choice1.x',   'B.in1')
        #rig.connect('R.choice2.x',   'C.in1')
        #rig.connect('R.otherwise.x', 'D.in1')
        #rig.connect('B.out1',        'M.b.y')
        #rig.connect('C.out1',        'M.c.y')  # Overlap!
        #rig.connect('D.out1',        'M.d.y')
        #rig.connect('M.y',           'E.in1')

        ## Choices 1&2, 'x == 3' condition.
        #rig.run()

        ## 'otherwise' condition.
        #rig.state_table = [
            #['A'],
            #['R'],
            #['D'],
            #['M'],
            #['E']
        #]
        #rig.data_table = [
            #['E.integral', '(16, valid)'],
        #]
        #A.set('in1', 3)
        #E.set('integral', 0)
        #rig.run("Testing 'otherwise' condition...")


    #def test_multi_merge(self):
        #"""\
              #Pattern 8 - Multi-Merge:
                 #/ (if choice1) - B \\
               #A - (if choice2) - C - M - E
                 #\\ (otherwise ) - D /
              #"""
        #state_table = [
            #['A'],
            #['R'],
            #['B', 'C'],
            #['M'],
            #['E'],
            #['M'],
            #['E']
        #]
        #data_table = [
            #['E.integral', '(25, valid)'],
        #]
        #rig = TestRig(state_table, data_table)
        #rig.name = 'TestRig'

        #A = Sleepy('A', rig)
        #A.set('in1', 1)
        #A.set('in2', 2)

        #R = router.MultiChoice('R', rig)
        #Int('x', R, 'in')
        #R.add_branch('choice1', 'self.x.value == 3')
        #R.add_branch('choice2', 'self.x.value <= 4')

        #B = Sleepy('B', rig, delay=0.1)
        #B.set('in2', 3)

        #C = Sleepy('C', rig, delay=0.2)
        #C.set('in2', 4)

        #D = Sleepy('D', rig)
        #D.set('in2', 5)

        #M = selector.SimpleMerge('M', rig)
        #Int('y', M, 'out')
        #M.add_branch('b')
        #M.add_branch('c')
        #M.add_branch('d')

        #E = Integrator('E', rig)
        #E.set('in2', 6)

        #rig.connect('A.out1',        'R.x')
        #rig.connect('R.choice1.x',   'B.in1')
        #rig.connect('R.choice2.x',   'C.in1')
        #rig.connect('R.otherwise.x', 'D.in1')
        #rig.connect('B.out1',        'M.b.y')
        #rig.connect('C.out1',        'M.c.y')  # Overlap!
        #rig.connect('D.out1',        'M.d.y')
        #rig.connect('M.y',           'E.in1')

        ## Choices 1&2, 'x == 3' condition.
        #rig.run()

        ## 'otherwise' condition.
        #rig.state_table = [
            #['A'],
            #['R'],
            #['D'],
            #['M'],
            #['E']
        #]
        #rig.data_table = [
            #['E.integral', '(16, valid)'],
        #]
        #A.set('in1', 3)
        #E.set('integral', 0)
        #rig.run("Testing 'otherwise' condition...")


    #def test_while_do(self):
        #"""\
              #Pattern 21a - Intuctured Loop (while-do):
                   #+-------+
                   #v       |
               #A - M - R - B   C
                         #\\____/
              #"""
        #state_table = [
            #['A'],
            #['M'],
            #['R'],
            #['B'],
            #['M'],
            #['R'],
            #['B'],
            #['M'],
            #['R'],
            #['C'],
        #]
        #data_table = [
            #['C.out1', '(9, valid)']
        #]
        #rig = TestRig(state_table, data_table)
        #rig.name = 'TestRig'

        #A = Sleepy('A', rig)
        #A.set('in1', 1)
        #A.set('in2', 2)

        #M = selector.SimpleMerge('M', rig)
        #Int('integral', M, 'out')
        #M.add_branch('a')
        #M.add_branch('b')

        #R = router.Repeat('R', rig, 'self.integral.value < 5')
        #Int('integral', R, 'in')

        #B = Integrator('B', rig)
        #B.set('in2', 0)

        #C = Sleepy('C', rig)
        #C.set('in2', 3)

        #rig.connect('A.out1',            'M.a.integral')
        #rig.connect('M.integral',        'R.integral')
        #rig.connect('R.repeat.integral', 'B.in1')
        #rig.connect('B.integral',        'M.b.integral')
        #rig.connect('R.exit.integral',   'C.in1')

        #rig.run()

        ## Try to create loops.
        #for dst in ('A.in1', 'M.a.integral', 'B.in2', 'M.b.integral',
                    #'R.integral', 'C.in2'):
            #try:
                #rig.connect('C.out1', dst)
            #except errors.CircularDependencyError, exc:
                #if Int(exc) != 'Link would create a circular dependency':
                    #print 'ERROR: wrong exception for', dst, Int(exc)
            #else:
                #print 'ERROR: loop not detected for', dst


    #def test_do_while(self):
        #"""\
              #Pattern 21b - Intuctured Loop (do-while):
               #A - M - B - R - C
                   #^       |
                   #+-------+
              #"""
        #state_table = [
            #['A'],
            #['M'],
            #['B'],
            #['R'],
            #['M'],
            #['B'],
            #['R'],
            #['C'],
        #]
        #data_table = [
            #['C.out1', '(9, valid)']
        #]
        #rig = TestRig(state_table, data_table)
        #rig.name = 'TestRig'

        #A = Sleepy('A', rig)
        #A.set('in1', 1)  # Make valid.
        #A.set('in2', 2)  # Make valid.

        #M = selector.SimpleMerge('M', rig)
        #Int('val', M, 'out')
        #M.add_branch('a')
        #M.add_branch('r')

        #B = Integrator('B', rig)
        #B.set('in2', 0)  # Make valid.

        #R = router.Repeat('R', rig, 'self.integral.value < 5')
        #Int('integral', R, 'in')

        #C = Sleepy('C', rig)
        #C.set('in2', 3)  # Make valid.

        #rig.connect('A.out1',            'M.a.val')
        #rig.connect('M.val',             'B.in1')
        #rig.connect('B.integral',        'R.integral')
        #rig.connect('R.repeat.integral', 'M.r.val')
        #rig.connect('R.exit.integral',   'C.in1')

        #rig.run()

        ## Try to create loops.
        #for dst in ('A.in1', 'M.a.val', 'B.in2',
                    #'R.integral', 'M.r.val', 'C.in2'):
            #try:
                #rig.connect('C.out1', dst)
            #except errors.CircularDependencyError, exc:
                #if Int(exc) != 'Link would create a circular dependency':
                    #print 'ERROR: wrong exception for', dst, Int(exc)
            #else:
                #print 'ERROR: loop not detected for', dst


    #def test_map_generator(self):
        #"""\
              #Run compressor with guessed map.
              #Repeat with generated map if 'extrapolated'.
              #"""
        #state_table = [
            #['Guess', 'Inlet',],
            #['Select'],
            #['Compressor'],
            #['Repeat'],
            #['MapGenerator'],
            #['Select'],
            #['Compressor'],
            #['Repeat'],
            #['MapGenerator'],
            #['Select'],
            #['Compressor'],
            #['Repeat'],
            #['Burner'],
        #]
        #rig = TestRig(state_table, [])
        #rig.name = 'TestRig'

        #inlet = Sleepy('Inlet', rig)
        #inlet.set('in1', 1)  # Make valid.
        #inlet.set('in2', 2)  # Make valid.

        #compressor = Compressor('Compressor', rig)

        #burner = Sleepy('Burner', rig)
        #burner.set('in2', 3)  # Make valid.

        #map_gen = MapGenerator('MapGenerator', rig)

        #guess = Sleepy('Guess', rig, 0.1)
        #guess.set('in1', 0)  # Make valid.
        #guess.set('in2', 0)  # Make valid.

        #repeat = router.Repeat('Repeat', rig, 'self.extrapolated.value')
        #Int('outflow', repeat, 'in')
        #Int('extrapolated', repeat, 'in')

        #select = selector.SimpleMerge('Select', rig)
        #Int('map', select, 'out')
        #select.add_branch('generated')
        #select.add_branch('guess')

        #rig.connect('Inlet.out1', 'Compressor.inflow')
        #rig.connect('Inlet.out1', 'MapGenerator.inflow')
        #rig.connect('Select.map', 'Compressor.map')
        #rig.connect('Compressor.outflow', 'Repeat.outflow')
        #rig.connect('Compressor.extrapolated', 'Repeat.extrapolated')
        #rig.connect('Repeat.exit.outflow', 'Burner.in1')
        #rig.connect('Repeat.repeat.'+FC_OUT, 'MapGenerator.'+FC_IN)
        #rig.connect('Guess.out1', 'Select.guess.map')
        #rig.connect('MapGenerator.map', 'Select.generated.map')

        #rig.run()


    #def test_subassembly(self):
        #"""
              #Subassembly:
                  #S.......
               #A -+-- C -+- E
                  #. X    .
               #B -+-- D -+- F
                  #........
              #"""
        #state_table = [
            #['A', 'B'],
            #['S'],
            #['E', 'F']
        #]
        #data_table = [
            #['A.out1', '(a1a2, valid-linked)'],
            #['B.out1', '(b1b2, valid-linked)'],
            #['S.C.out1', '(a1a2b1b2, valid-linked)'],
            #['S.D.out1', '(a1a2b1b2, valid-linked)'],
            #['E.out1', '(a1a2b1b2e2, valid)'],
            #['F.out1', '(a1a2b1b2f2, valid)'],
        #]
        #rig = TestRig(state_table, data_table)
        #rig.name = 'TestRig'

        #A = Sleepy('A', rig)
        #A.set('in1', 'a1')
        #A.set('in2', 'a2')

        #B = Sleepy('B', rig)
        #B.set('in1', 'b1')
        #B.set('in2', 'b2')

        #S = Assembly('S', rig)
        #Int('in1', S, 'in')
        #Int('in2', S, 'in')
        #Int('out1', S, 'out')
        #Int('out2', S, 'out')

        #C = Sleepy('C', S)

        #D = Sleepy('D', S)

        #E = Sleepy('E', rig)
        #E.set('in2', 'e2')

        #F = Sleepy('F', rig)
        #F.set('in2', 'f2')

        #rig.connect('A.out1', 'S.in1')
        #rig.connect('B.out1', 'S.in2')
        #S.connect('in1', 'C.in1')
        #S.connect('in2', 'C.in2')
        #S.connect('in1', 'D.in1')
        #S.connect('in2', 'D.in2')
        #S.connect('C.out1', 'out1')
        #S.connect('D.out1', 'out2')
        #rig.connect('S.out1', 'E.in1')
        #rig.connect('S.out2', 'F.in1')

        #rig.run()

        ## Run E, which should require A and C (not B, D, or F).
        #A.set('in2', 'a3')
        #rig.state_table = [
            #['A'],
            #['S'],
        #]
        #rig.data_table = [
            #['A.out1', '(a1a3, valid-linked)'],
            #['B.out1', '(b1b2, valid-linked)'],
            #['S.C.out1', '(a1a3b1b2, valid-linked)'],
            #['S.D.out1', '(a1a2b1b2, invalid-linked)'],
            #['E.out1', '(a1a3b1b2e2, valid)'],
            #['F.out1', '(a1a2b1b2f2, invalid)'],
        #]
        #rig.run('Run E and necessary predecessors', E)

        ## Run D, which should require B (not A, C, E, or F).
        #B.set('in2', 'b3')
        #rig.state_table = [
            #['B'],
        #]
        #rig.data_table = [
            #['A.out1', '(a1a3, valid-linked)'],
            #['B.out1', '(b1b3, valid-linked)'],
            #['S.C.out1', '(a1a3b1b2, invalid-linked)'],
            #['S.D.out1', '(a1a3b1b3, valid-linked)'],
            #['E.out1', '(a1a3b1b2e2, invalid)'],
            #['F.out1', '(a1a2b1b2f2, invalid)'],
        #]
        #rig.run('Run D and necessary predecessors', D)


    #def test_input_selection(self):
        #"""\
              #Input selection scheme:
                      #/ PlugFlow  \\
               #Select - StaticFlow - Merge - ProfileScaler
                 #^    \\ LiveFlow  /
                 #|
               #Choice
              #"""
        #state_table = [
            #['Select'],
            #['PlugFlow'],
            #['Merge'],
            #['ProfileScaler']
        #]
        #data_table = [
            #['ProfileScaler.out1', '(plugflow, valid)'],
        #]
        #rig = TestRig(state_table, data_table)
        #rig.name = 'TestRig'

        #ec = router.ExclusiveChoice('Select', rig)
        #Int('choice', ec, 'in')
        #ec.add_branch('plug', "self.choice.value == 'plug'")
        #ec.add_branch('static', "self.choice.value == 'static'")

        #pf = Sleepy('PlugFlow', rig)
        #pf.set('in1', 'plug')
        #pf.set('in2', 'flow')

        #sf = Sleepy('StaticFlow', rig)
        #sf.set('in1', 'static')
        #sf.set('in2', 'flow')

        #lf = Sleepy('LiveFlow', rig)
        #lf.set('in1', 'live')
        #lf.set('in2', 'flow')

        #merge = selector.SimpleMerge('Merge', rig)
        #Int('profile', merge, 'out')
        #merge.add_branch('plug')
        #merge.add_branch('static')
        #merge.add_branch('live')

        #ps = Sleepy('ProfileScaler', rig)
        #ps.set('in2', '')

        #rig.connect('Select.plug.flow_control_out', 'PlugFlow.flow_control_in')
        #rig.connect('Select.static.flow_control_out', 'StaticFlow.flow_control_in')
        #rig.connect('Select.else.flow_control_out', 'LiveFlow.flow_control_in')
        #rig.connect('PlugFlow.out1', 'Merge.plug.profile')
        #rig.connect('StaticFlow.out1', 'Merge.static.profile')
        #rig.connect('LiveFlow.out1', 'Merge.live.profile')
        #rig.connect('Merge.profile', 'ProfileScaler.in1')

        #ec.set('choice', 'plug')
        #rig.run()

        ## Try static flow.
        #ec.set('choice', 'static')
        #rig.state_table = [
            #['Select'],
            #['StaticFlow'],
            #['Merge'],
            #['ProfileScaler']
        #]
        #rig.data_table = [
            #['ProfileScaler.out1', '(staticflow, valid)'],
        #]
        #rig.run('Static flow')

        ## Try live flow.
        #ec.set('choice', 'live')
        #rig.state_table = [
            #['Select'],
            #['LiveFlow'],
            #['Merge'],
            #['ProfileScaler']
        #]
        #rig.data_table = [
            #['ProfileScaler.out1', '(liveflow, valid)'],
        #]
        #rig.run('Live flow')

        ## Try updated live flow.
        #lf.set('in1', 'updated')
        #rig.state_table = [
            #['LiveFlow'],
            #['Merge'],
            #['ProfileScaler']
        #]
        #rig.data_table = [
            #['ProfileScaler.out1', '(updatedflow, valid)'],
        #]
        #rig.run('Updated flow')


    #def test_remove(self):
        #"""\
              #Remove (linked) variables and branches:
               #A - (choice1  ) - B - M - D - E
                 #\\ (otherwise) - C /
              #"""
        #state_table = [
            #['A'],
            #['R'],
            #['C'],
            #['M'],
            #['D'],
            #['E'],
        #]
        #data_table = [
            #['E.out1', '(a1a2c2d2e2, valid)'],
        #]
        #rig = TestRig(state_table, data_table)
        #rig.name = 'TestRig'

        #A = Sleepy('A', rig)
        #A.set('in1', 'a1')
        #A.set('in2', 'a2')

        #R = router.MultiChoice('R', rig)
        #Int('x', R, 'in')
        #R.add_branch('choice1', 'False')

        #B = Sleepy('B', rig)
        #B.set('in2', 'b2')

        #C = Sleepy('C', rig)
        #C.set('in2', 'c2')

        #D = Sleepy('D', rig)
        #D.set('in2', 'd2')

        #M = selector.SimpleMerge('M', rig)
        #Int('y', M, 'out')
        #M.add_branch('b')
        #M.add_branch('c')

        #E = Sleepy('E', rig)
        #E.set('in2', 'e2')

        #rig.connect('A.out1',             'R.x')
        #rig.connect('A.flow_control_out', 'R.flow_control_in')
        #rig.connect('R.choice1.x',                'B.in1')
        #rig.connect('R.choice1.flow_control_out', 'B.flow_control_in')
        #rig.connect('R.otherwise.x',                'C.in1')
        #rig.connect('R.otherwise.flow_control_out', 'C.flow_control_in')
        #rig.connect('B.out1',             'M.b.y')
        #rig.connect('B.flow_control_out', 'M.b.flow_control_in')
        #rig.connect('C.out1',             'M.c.y')
        #rig.connect('C.flow_control_out', 'M.c.flow_control_in')
        #rig.connect('M.y',                'D.in1')
        #rig.connect('M.flow_control_out', 'D.flow_control_in')
        #rig.connect('D.out1',             'E.in1')
        #rig.connect('D.flow_control_out', 'E.flow_control_in')

        #rig.run()

        #R.remove_variable('x')
        #rig.invalidate()
        #rig.run('R.x removed')

        #M.remove_variable('y')
        #rig.invalidate()
        #rig.run('M.y removed')

        #R.remove_branch('choice1')
        #M.remove_branch('b')
        #rig.invalidate()
        #rig.state_table = [
            #['A', 'B'],
            #['R'],
            #['C'],
            #['M'],
            #['D'],
            #['E'],
        #]
        #rig.run('R.choice1 and M.b removed')


if __name__ == '__main__':
    unittest.main()
#    VERBOSE = True

