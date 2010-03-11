#public symbols
__all__ = ["Driver"]




from enthought.traits.api import implements, List
from enthought.traits.trait_base import not_none
import networkx as nx
from networkx.algorithms.traversal import strongly_connected_components

from openmdao.main.interfaces import IDriver
from openmdao.main.component import Component
from openmdao.main.stringref import StringRef, StringRefArray

class Driver(Component):
    """ A Driver iterates over a collection of Components until some condition
    is met. """
    
    implements(IDriver)
    
    def __init__(self, doc=None):
        super(Driver, self).__init__(doc=doc)
        self._ref_graph = { None: None, 'in': None, 'out': None }
        self._ref_comps = { None: None, 'in': None, 'out': None }
        self.graph_regen_needed()
    
    def graph_regen_needed(self):
        """If called, reset internal graphs to that they will be
        regenerated when they are requested next.
        """
        self._iteration_comps = None
        self._simple_iteration_subgraph = None
        self._simple_iteration_set = None    
        self._driver_tree = None
        
    def _pre_execute (self):
        super(Driver, self)._pre_execute()
        self.is_ready()

    def is_ready (self):
        """Return True if this driver is ready to run.
        """
        if super(Driver, self).is_ready():
            return True

        refnames = self.get_refvar_names(iotype='in')
        
        # TODO: this should really be a callback, any time a StringRef input is changed
        if not all(self.get_valids(refnames)):
            #self._call_execute = True
            ## force regeneration of _ref_graph, _ref_comps, _iteration_comps
            self._ref_graph = { None: None, 'in': None, 'out': None } 
            self._ref_comps = { None: None, 'in': None, 'out': None }
            self.graph_regen_needed()
            
        # force execution of the driver if any of its StringRefs reference
        # invalid Variables
        for name in refnames:
            rv = getattr(self, name)
            if isinstance(rv, list):
                for entry in rv:
                    if not entry.refs_valid():
                        return True
            else:
                if not rv.refs_valid():
                    return True

    def execute(self, required_outputs=None):
        """ Iterate over a collection of Components until some condition
        is met. If you don't want to structure your driver to use pre_iteration,
        post_iteration, etc., just override this function. As a result, none
        of the <start/pre/post/continue>_iteration() functions will be called.
        """
        self.start_iteration()
        while self.continue_iteration():
            self.pre_iteration()
            self.run_iteration(required_outputs)
            self.post_iteration()

    def start_iteration(self):
        """Called just prior to the beginning of an iteration loop. This can 
        be overridden by inherited classes. It can be used to perform any 
        necessary pre-iteration initialization.
        """
        self._continue = True

    def continue_iteration(self):
        """Return False to stop iterating."""
        return self._continue
    
    def pre_iteration(self):
        """Called prior to each iteration."""
        pass
        
    def run_iteration(self, required_outputs=None):
        """Runs a single iteration of the workflow that this driver is associated with."""
        if self.parent:
            self.parent.workflow.run(required_outputs)
        else:
            self.raise_exception('Driver cannot run referenced components without a parent',
                                 RuntimeError)

    def post_iteration(self):
        """Called after each iteration."""
        self._continue = False  # by default, stop after one iteration

    def get_refvar_names(self, iotype=None):
        """Return a list of names of all StringRef and StringRefArray traits
        in this instance.
        """
        if iotype is None:
            checker = not_none
        else:
            checker = iotype
        
        return [n for n,v in self._traits_meta_filter(iotype=checker).items() 
                    if v.is_trait_type(StringRef) or 
                       v.is_trait_type(StringRefArray)]
        
    def get_referenced_comps(self, iotype=None):
        """Return a set of names of Components that we reference based on the 
        contents of our StringRefs and StringRefArrays.  If iotype is
        supplied, return only component names that are referenced by ref
        variables with matching iotype.
        """
        if self._ref_comps[iotype] is None:
            comps = set()
        else:
            return self._ref_comps[iotype]
    
        for name in self.get_refvar_names(iotype):
            obj = getattr(self, name)
            if isinstance(obj, list):
                for entry in obj:
                    comps.update(entry.get_referenced_compnames())
            else:
                comps.update(obj.get_referenced_compnames())
                
        self._ref_comps[iotype] = comps
        return comps
        
    #def get_ref_graph(self, iotype=None):
        #"""Returns the dependency graph for this Driver based on
        #StringRefs and StringRefArrays.
        #"""
        #if self._ref_graph[iotype] is not None:
            #return self._ref_graph[iotype]
        
        #self._ref_graph[iotype] = nx.DiGraph()
        #name = self.name
        
        #if iotype == 'out' or iotype is None:
            #self._ref_graph[iotype].add_edges_from([(name,rv) 
                                  #for rv in self.get_referenced_comps(iotype='out')])
            
        #if iotype == 'in' or iotype is None:
            #self._ref_graph[iotype].add_edges_from([(rv, name) 
                                  #for rv in self.get_referenced_comps(iotype='in')])
        #return self._ref_graph[iotype]
