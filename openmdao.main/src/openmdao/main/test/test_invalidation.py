# pylint: disable-msg=C0111,C0103

import unittest
import logging

from enthought.traits.api import Int, TraitError

from openmdao.main.api import Assembly, Component, set_as_top

def dump_stuff(obj, **kwargs):
    valid = kwargs.get('valid')
    io = kwargs.get('iotype')
    if io == 'in' or io == 'both':
        for name in obj.list_inputs(valid=valid):
            print '.'.join([obj.get_pathname(),name])
    if io == 'out' or io == 'both':
        for name in obj.list_outputs(valid=valid):
            print '.'.join([obj.get_pathname(),name])
    if kwargs.get('recurse'):
        for val in obj.values():
            if isinstance(val, Component):
                dump_stuff(val, **kwargs)
        
class Simple(Component):
    a = Int(iotype='in')
    b = Int(iotype='in')
    c = Int(iotype='out')
    d = Int(iotype='out')
    
    def __init__(self):
        super(Simple, self).__init__()
        self.a = 1
        self.b = 2
        self.run_count = 0

    def execute(self, required_outputs=None):
        self.run_count += 1
        self.c = self.a + self.b
        self.d = self.a - self.b

class InvalidationTestCase(unittest.TestCase):

    def _setup_comps(self, names, top=None):
        if top is None:
            top = set_as_top(Assembly())
        for name in names:
            comp = top.add_container(name, Simple())
            self.assertEqual(comp.run_count, 0)
            self._check_valids(comp, [True, True, False, False])
        return top

    def _run_disconnected(self, top, names, runcount):
        top.run()
        for name in names:
            comp = getattr(top, name)
            self.assertEqual(comp.run_count, runcount)
            self.assertEqual(comp.c, 3)
            self.assertEqual(comp.d, -1)
            self._check_valids(comp, [True, True, True, True])
        top.run()
        for name in names:
            comp = getattr(top, name)
            self.assertEqual(comp.run_count, runcount) # run count shouldn't change
            self._check_valids(comp, [True, True, True, True])
            
    def _check_valids(self, comp, expected):
        valids = [comp.get_valid(v) for v in ['a','b','c','d']]
        self.assertEqual(valids, expected)
            
    def test_single_component(self):
        top = self._setup_comps(['comp1'])
        self._run_disconnected(top, ['comp1'], 1)
        
        top.set('comp1.a', 5)
        self._check_valids(top.comp1, [True, True, False, False])
        top.run()
        self._check_valids(top.comp1, [True, True, True, True])
        self.assertEqual(top.comp1.run_count, 2)
        self.assertEqual(top.comp1.c, 7)
        self.assertEqual(top.comp1.d, 3)
        top.run()
        self.assertEqual(top.comp1.run_count, 2) # run_count shouldn't change
        self._check_valids(top.comp1, [True, True, True, True])
        
    def test_two_disconnected_components(self):
        compnames = ['comp1','comp2']
        top = self._setup_comps(compnames)
        self._run_disconnected(top, compnames, 1)
        top.comp1.b = 6
        self._check_valids(top.comp1, [True, True, False, False])
        self._check_valids(top.comp2, [True, True, True, True])
        top.run()
        self._check_valids(top.comp1, [True, True, True, True])
        self._check_valids(top.comp2, [True, True, True, True])
        self.assertEqual(top.comp1.c, 7)
        self.assertEqual(top.comp1.d, -5)
        self.assertEqual(top.comp2.c, 3)
        self.assertEqual(top.comp2.d, -1)
        
    def test_two_connected_components(self):
        compnames = ['comp1','comp2']
        top = self._setup_comps(compnames)
        top.connect('comp1.c', 'comp2.a')
        self.assertEqual(top.comp2.run_count, 0)
        self.assertEqual(top.comp2.c, 0)
        self.assertEqual(top.comp2.d, 0)
        self._check_valids(top.comp2, [False, True, False, False])
        top.run()
        self.assertEqual(top.comp1.run_count, 1)
        self.assertEqual(top.comp2.run_count, 1)
        self.assertEqual(top.comp2.c, 5)
        self.assertEqual(top.comp2.d, 1)
        self._check_valids(top.comp2, [True, True, True, True])
        
        # check that invalidation passes into comp2 when we change comp1
        top.comp1.b = 6
        self._check_valids(top.comp1, [True, True, False, False])
        self._check_valids(top.comp2, [False, True, False, False])
        top.run()
        self.assertEqual(top.comp1.run_count, 2)
        self.assertEqual(top.comp2.run_count, 2)
        self.assertEqual(top.comp2.a, 7)
        self.assertEqual(top.comp2.c, 9)
        self.assertEqual(top.comp2.d, 5)
        self._check_valids(top.comp1, [True, True, True, True])
        self._check_valids(top.comp2, [True, True, True, True])
        
        # now try setting a value, then disconnecting
        top.comp1.b = 12
        self._check_valids(top.comp1, [True, True, False, False])
        self._check_valids(top.comp2, [False, True, False, False])
        top.disconnect('comp1.c', 'comp2.a')
        self._check_valids(top.comp1, [True, True, False, False])
        self._check_valids(top.comp2, [True, True, False, False])
        top.run()
        self.assertEqual(top.comp1.run_count, 3)
        self.assertEqual(top.comp2.run_count, 3)
        self.assertEqual(top.comp2.a, 7)
        self.assertEqual(top.comp2.c, 9)
        self.assertEqual(top.comp2.d, 5)
        self._check_valids(top.comp1, [True, True, True, True])
        self._check_valids(top.comp2, [True, True, True, True])
        
        # now reconnect and run
        top.connect('comp1.c', 'comp2.a')
        top.run()
        self.assertEqual(top.comp1.run_count, 3)
        self.assertEqual(top.comp2.run_count, 4)
        self.assertEqual(top.comp2.a, 13)
        self.assertEqual(top.comp2.c, 15)
        self.assertEqual(top.comp2.d, 11)
        self._check_valids(top.comp1, [True, True, True, True])
        self._check_valids(top.comp2, [True, True, True, True])
                
        
if __name__ == "__main__":
    unittest.main()


