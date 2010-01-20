
#public symbols
__all__ = ['Assembly']


from enthought.traits.api import Array, List, Instance, TraitError
from enthought.traits.api import TraitType, Undefined
from enthought.traits.trait_base import not_none

import networkx as nx

from openmdao.main.interfaces import IDriver
from openmdao.main.component import Component
from openmdao.main.container import Container, INVALID
from openmdao.main.workflow import Workflow
from openmdao.main.asyncworkflow import AsyncWorkflow
from openmdao.main.dataflow import Dataflow
from openmdao.main.driver import Driver


def _filter_internal_edges(edges):
    """Return a copy of the given list of edges with edges removed that are
    connecting two variables on the same component.
    """
    return [(u,v) for u,v in edges
                          if u.split('.', 1)[0] != v.split('.', 1)[0]]
    
class PassthroughTrait(TraitType):
    """A trait that can use another trait for validation, but otherwise is
    just a trait that lives on an Assembly boundary and can be connected
    to other traits within the Assembly.
    """
    
    def validate(self, obj, name, value):
        if self.validation_trait:
            return self.validation_trait.validate(obj, name, value)
        return value

class Assembly (Component):
    """This is a container of Components. It understands how to connect inputs
    and outputs between its children and how to run a Workflow.
    """
    driver = Instance(Driver)
    workflow = Instance(Workflow)
    
    def __init__(self, doc=None, directory=''):
        self._child_io_graphs = {}
        self._need_child_io_update = True
        
        # A graph of Variable names (local path), 
        # with connections between Variables as directed edges.  
        # Children are queried for dependencies between their inputs and outputs
        # so they can also be represented in the graph. 
        self._var_graph = nx.DiGraph()
        
        super(Assembly, self).__init__(doc=doc, directory=directory)
        
        # add any Variables we may have inherited from our base classes
        # to our _var_graph..
        for v in self.keys(io_direction=not_none):
            if v not in self._var_graph:
                self._var_graph.add_node(v)
        
        self.workflow = AsyncWorkflow(scope=self)

    def get_component_graph(self):
        """Retrieve the dataflow graph of child components."""
        return self.workflow.get_graph()
    
    def get_var_graph(self):
        """Returns the Variable dependency graph, after updating it with child
        info if necessary.
        """
        if self._need_child_io_update:
            vargraph = self._var_graph
            childiographs = self._child_io_graphs
            for childname,val in childiographs.items():
                graph = getattr(self, childname).get_io_graph()
                if graph is not val:  # child io graph has changed
                    if val is not None:  # remove old stuff
                        vargraph.remove_nodes_from(val)
                    childiographs[childname] = graph
                    vargraph.add_nodes_from(graph.nodes_iter())
                    vargraph.add_edges_from(graph.edges_iter())
            self._need_child_io_update = False
        return self._var_graph
        
    #def get_io_graph(self):
        #"""For now, just return our base class version of get_io_graph."""
        ## TODO: make this return an actual graph of inputs to outputs based on 
        ##       the contents of this Assembly instead of a graph where all 
        ##       outputs depend on all inputs
        ## NOTE: if the io_graph changes, this function must return a NEW graph
        ## object instead of modifying the old one, because object identity
        ## is used in the parent assembly to determine of the graph has changed
        #return super(Assembly, self).get_io_graph()
    
    def add_driver(self, name, obj):
        # FIXME: this is only a temporary fix before moving to driverflow
        if not isinstance(obj, Driver):
            self.raise_exception("The object passed to add_driver must be a Driver",
                                 RuntimeError)
        obj.parent = self
        obj.name = name
        setattr(self, name, obj)
        # if this object is already installed in a hierarchy, then go
        # ahead and tell the obj (which will in turn tell all of its
        # children) that its scope tree back to the root is defined.
        if self._call_tree_rooted is False:
            obj.tree_rooted()
        self.driver = obj
        return obj
    
    def add_container(self, name, obj):
        """Update dependency graph and call base class add_container.
        Returns the added Container object.
        """
        if isinstance(obj, Driver):
            self.raise_exception("The object passed to add_container must not be a Driver",
                                 RuntimeError)
        obj = super(Assembly, self).add_container(name, obj)
        if isinstance(obj, Component):
            # since the internals of the given Component can change after it's
            # added to us, wait to collect its io_graph until we need it
            self._child_io_graphs[obj.name] = None
            self._need_child_io_update = True
            self.workflow.add_node(obj.name)
        return obj
        
    def remove_container(self, name):
        """Remove the named object from this container."""
        trait = self.trait(name)
        if trait is not None:
            obj = getattr(self, name)
            # if the named object is a Component, then assume it must
            # be removed from our workflow
            if isinstance(obj, Component):
                start = name + '.'
                edges = [(u,v) for u,v in self._var_graph.edges() 
                                  if (u.startswith(start) or v.startswith(start)) and
                                  u.split('.')[0] != v.split('.')[0]]
                for src,sink in edges:
                    self.disconnect(src, sink)
                
                self.workflow.remove_node(obj.name)
                
                if name in self._child_io_graphs:
                    childgraph = self._child_io_graphs[name]
                    if childgraph is not None:
                        self._var_graph.remove_nodes_from(childgraph)
                    del self._child_io_graphs[name]
            
        return super(Assembly, self).remove_container(name)
    
    def create_passthrough(self, pathname, alias=None):
        """Creates a PassthroughTrait that uses the trait indicated by
        pathname for validation (if it's not a property trait), adds it to
        self, and creates a connection between the two. If alias is None,
        the name of the 'promoted' trait will be the last entry in its
        pathname.  This is different than the create_alias function because
        the new trait is only tied to the specified trait by a connection
        in the Assembly. This means that updates to the new trait value will
        not be reflected in the connected trait until the assembly executes.
        The trait specified by pathname must exist.
        """
        if alias:
            newname = alias
        else:
            parts = pathname.split('.')
            newname = parts[-1]

        oldtrait = self.trait(newname)
        if oldtrait:
            self.raise_exception("a trait named '%s' already exists" %
                                 newname, TraitError)
        trait, val = self._find_trait_and_value(pathname)
        if not trait:
            self.raise_exception("the trait named '%s' can't be found" %
                                 pathname, TraitError)
        io_direction = trait.io_direction
        # the trait.trait_type stuff below is for the case where the trait is actually
        # a ctrait (very common). In that case, trait_type is the actual underlying
        # trait object
        if (getattr(trait,'get') or getattr(trait,'set') or
            getattr(trait.trait_type, 'get') or getattr(trait.trait_type,'set')):
            trait = None  # not sure how to validate using a property
                          # trait without setting it, so just don't use it
        newtrait = PassthroughTrait(io_direction=io_direction, validation_trait=trait)
        self.add_trait(newname, newtrait)
        setattr(self, newname, val)

        if io_direction == 'in':
            self.connect(newname, pathname)
        else:
            self.connect(pathname, newname)

        return newtrait

    def get_dyn_trait(self, pathname, io_direction=None):
        """Retrieves the named trait, attempting to create a Passthrough trait
        on-the-fly if the specified trait doesn't exist.
        """
        trait = self.trait(pathname)
        if trait is None:
            newtrait = self.create_passthrough(pathname)
            if io_direction is not None and io_direction != newtrait.io_direction:
                self.raise_exception(
                    "new trait has io_direction of '%s' but '%s' was expected" %
                    (newtrait.io_direction, io_direction), TraitError)
        return trait

    def split_varpath(self, path):
        """Return a tuple of compname,component,varname given a path
        name of the form 'compname.varname'. If the name is of the form 'varname'
        then compname will be None and comp is self. 
        """
        try:
            compname, varname = path.split('.', 1)
        except ValueError:
            return (None, self, path)
        
        return (compname, getattr(self, compname), varname)

    def connect(self, srcpath, destpath):
        """Connect one src Variable to one destination Variable. This could be
        a normal connection (output to input) or a passthrough connection."""

        srccompname, srccomp, srcvarname = self.split_varpath(srcpath)
        srctrait = srccomp.get_dyn_trait(srcvarname, 'out')
        destcompname, destcomp, destvarname = self.split_varpath(destpath)
        desttrait = destcomp.get_dyn_trait(destvarname, 'in')
        
        if srccompname == destcompname:
            self.raise_exception(
                'Cannot connect %s to %s. Both are on same component.' %
                                 (srcpath, destpath), RuntimeError)
        if srccomp is not self and destcomp is not self:
            # it's not a passthrough, so must connect input to output
            if srctrait.io_direction != 'out':
                self.raise_exception(
                    '.'.join([srccomp.get_pathname(),srcvarname])+
                    ' must be an output variable',
                    RuntimeError)
            if desttrait.io_direction != 'in':
                self.raise_exception(
                    '.'.join([destcomp.get_pathname(),destvarname])+
                    ' must be an input variable',
                    RuntimeError)
                
        if self.is_destination(destpath):
            self.raise_exception(destpath+' is already connected',
                                 RuntimeError)             
            
        # test compatability (raises TraitError on failure)
        if desttrait.validate is not None:
            desttrait.validate(destcomp, destvarname, 
                               getattr(srccomp, srcvarname))
        
        if destcomp is not self:
            if srccomp is not self: # neither var is on boundary
                self.workflow.connect(srcpath, destpath)
                destcomp.set_source(destvarname, srcpath)

        if srccomp is not self:
            srccomp.link_output(srcvarname, destpath)
        
        vgraph = self.get_var_graph()
        vgraph.add_edge(srcpath, destpath)
            
        # invalidate destvar if necessary
        if destcomp is self and desttrait.io_direction == 'out': # boundary output
            if destcomp.get_valid(destvarname) and \
               srccomp.get_valid(srcvarname) is False:
                if self.parent:
                    # tell the parent that anyone connected to our boundary
                    # output is invalid.
                    # Note that it's a dest var in this scope, but a src var in
                    # the parent scope.
                    self.parent.invalidate_deps([destpath], True)
            self.set_valid(destpath, False)
        elif srccomp is self and srctrait.io_direction == 'in': # boundary input
            self.set_valid(srcpath, False)
        else:
            destcomp.set_valid(destvarname, False)
            self.invalidate_deps([destpath])
        
        self._io_graph = None

    def disconnect(self, varpath, varpath2=None):
        """If varpath2 is supplied, remove the connection between varpath and
        varpath2. Otherwise, if varpath is the name of a trait, remove all
        connections to/from varpath in the current scope. If varpath is the
        name of a Component, remove all connections from all of its inputs
        and outputs. 
        """
        vargraph = self.get_var_graph()
        if varpath not in vargraph:  # a boundary variable name
            tup = varpath.split('.', 1)
            if len(tup) == 1 and isinstance(getattr(self, varpath), Component):
                comp = getattr(self, varpath)
                for var in comp.list_inputs():
                    self.disconnect('.'.join([varpath, var]))
                for var in comp.list_outputs():
                    self.disconnect('.'.join([varpath, var]))
            else:
                self.raise_exception("'%s' is not a linkable attribute" %
                                     varpath, RuntimeError)
            return
        
        to_remove = []
        if varpath2 is not None:
            if varpath2 in vargraph[varpath]:
                to_remove.append((varpath, varpath2))
            elif varpath in vargraph[varpath2]:
                to_remove.append((varpath2, varpath))
            else:
                self.raise_exception('%s is not connected to %s' % 
                                     (varpath, varpath2), RuntimeError)
        else:  # remove all connections from the Variable
            to_remove.extend(vargraph.edges(varpath)) # outgoing edges
            to_remove.extend(vargraph.in_edges(varpath)) # incoming
        
        for src,sink in _filter_internal_edges(to_remove):
            utup = src.split('.', 1)
            vtup = sink.split('.', 1)
            if len(vtup) > 1:
                getattr(self, vtup[0]).remove_source(vtup[1])
                # if its a connection between two children 
                # (no boundary connections) then remove a connection 
                # between two components in the component graph
                if len(utup) > 1:
                    self.workflow.disconnect(src, sink)
            if len(utup) > 1:
                getattr(self, utup[0]).unlink_output(utup[1], sink)
        
        vargraph.remove_edges_from(to_remove)
        
        # the io graph has changed, so have to remake it
        self._io_graph = None  


    def is_destination(self, varpath):
        """Return True if the Variable specified by varname is a destination
        according to our graph. This means that either it's an input connected
        to an output, or it's the destination part of a passtru connection.
        """
        tup = varpath.split('.',1)
        preds = self._var_graph.pred.get(varpath, {})
        if len(tup) == 1:
            return len(preds) > 0
        else:
            start = tup[0]+'.'
            for pred in preds:
                if not pred.startswith(start):
                    return True
        return False

    def execute (self):
        """By default, run child components in data flow order."""
        self._update_inputs_from_boundary(self.list_inputs(valid=True))
        if self.driver:
            self.driver.run()
        else:
            self.workflow.run()
        self._update_boundary_outputs()
        
    def _update_boundary_outputs (self):
        """Update output variables on our bounary."""
        invalid_outs = self.list_outputs(valid=False)
        vgraph = self.get_var_graph()
        for out in invalid_outs:
            inedges = vgraph.in_edges(nbunch=out)
            if len(inedges) == 1:
                setattr(self, out, self.get(inedges[0][0]))

    def step(self):
        """ Execute one workflow 'step'. """
        if self.workflow.is_active:
            compnames = None
        else:
            self._update_inputs_from_boundary(self.list_inputs(valid=True))
            compnames = self._compnames
        try:
            self.workflow.step(compnames)
        except StopIteration:
            self.state = VALID
        
    def stop(self):
        """Stop the workflow."""
        if self.driver:
            self.driver.stop()
        else:
            self.workflow.stop()
    
    def list_connections(self, show_passthrough=True):
        """Return a list of tuples of the form (outvarname, invarname).
        """
        if show_passthrough:
            return _filter_internal_edges(self.get_var_graph().edges())
        else:
            return _filter_internal_edges([(outname,inname) for outname,inname in 
                                                self.get_var_graph().edges_iter() 
                                                if '.' in outname and '.' in inname])

    #def update_inputs(self, varnames):
        #"""Transfer input data to input variables on the specified component.
        #The varnames iterator is assumed to contain names that include the
        #component name, for example: ['comp1.a', 'comp1.b'].
        #"""
        #parent = self.parent
        #vargraph = self.get_var_graph()
        #pred = vargraph.pred
        
        #srcvars_needed = {}
        #for vname in varnames:
            #preds = pred.get(vname, ())
            #if len(preds) == 0: 
                #continue
            #srcname = preds.keys()[0]
            #srccompname,srccomp,srcvarname = self.split_varpath(srcname)
            #destcompname,destcomp,destvarname = self.split_varpath(vname)

            #if srccomp.get_valid(srcvarname) is False:  # source is invalid 
                ## need to backtrack to get a valid source value
                #if srccompname is None: # a boundary var
                    #if parent:
                        #parent.update_inputs(['.'.join([self.name, srcname])])
                    #else:
                        #srccomp.set_valid(srcvarname, True) # validate source
                #else:
                    #if srccompname not in srcvars_needed:
                        #srcvars_needed[srccompname] = set()
                    #srcvars_needed[srccompname].add(srcvarname)
                    ##srccomp.update_outputs([srcvarname])

            #try:
                #srcval = srccomp.get_wrapped_attr(srcvarname)
            #except Exception, err:
                #self.raise_exception(
                    #"cannot retrieve value of source attribute '%s'" %
                    #srcname, type(err))
            #try:
                #destcomp.set(destvarname, srcval, srcname=srcname)
            #except Exception, exc:
                #msg = "cannot set '%s' from '%s': %s" % (vname, srcname, exc)
                #self.raise_exception(msg, type(exc))
        
    def make_inputs_valid(self, compname, inputs):
        """ Make inputs for child component valid. """
        # Trace inputs back to source outputs.
        # Add those components to evaluation list.
        trace_inputs = [(compname, inputs)]
        compnames = set()
        need_valid = set()
        while trace_inputs:
            dest, dst_inputs = trace_inputs.pop()
            for vname in dst_inputs:
                destpath = '.'.join([dest, vname])
                srcpath,dpath = self._var_graph.in_edges(nbunch=destpath)[0] # should be only one link to an input
                srcparts = srcpath.split('.')
                if len(srcparts) > 1:
                    srcname = srcparts[0]
                    if srcname not in compnames:
                        compnames.add(srcname)
                        src = getattr(self, srcname)
                        req_inputs = src.list_inputs(valid=False)
                        if req_inputs:
                            trace_inputs.append((srcname, req_inputs))
                else:
                    need_valid.add(srcpath)

        # If any of our inputs are needed and not valid, call parent.
        if need_valid:
            self.parent.make_inputs_valid(self, need_valid)
            self._update_inputs_from_boundary(need_valid)

        # Evaluate selected components.
        self.workflow.run(compnames)
        
        comp = getattr(self, compname)
        for inp in inputs:
            comp.set_valid(inp, True)

    def _update_inputs_from_boundary(self, inputs):
        """ Transfer boundary inputs to internal destinations. """
        for src in inputs:
            for src_path, dst_path in self._var_graph.edges(nbunch=src):
                if get_enabled(src):
                    if get_valid(src):
                        if VERBOSE:
                            print 'Transfer %s to %s' \
                                  % ('.'.join([self.name,src]), dst_path)
                        self.set(dst_path, getattr(self, src))
                else: # disabled
                    if VERBOSE:
                        print 'Disabling %s' % dst_path
                    dstcompname, dstvarname = dst_path.split('.', 1)
                    dst_comp = getattr(self, dstcompname)
                    dst_comp.set_enabled(dstvarname, False)

    def get_valids(self, names):
        """Returns a list of boolean values indicating whether the named
        attributes are valid (True) or invalid (False). Entries in names may
        specify either direct traits of self or those of direct children of
        self, but no deeper in the hierarchy than that.
        """
        valids = []
        for name in names:
            if self.trait(name):
                valids.append(self.get_valid(name))
            else:
                tup = name.split('.', 1)
                if len(tup) > 1:
                    comp = getattr(self, tup[0])
                    valids.append(comp.get_valid(tup[1]))
                else:
                    self.raise_exception("get_valids: unknown variable '%s'" %
                                         name, RuntimeError)
        return valids

    def invalidate_deps(self, varnames, notify_parent=False):
        """Mark all Variables invalid that depend on varnames. 
        
        Returns a list of our newly invalidated boundary outputs.
        """
        self.state = INVALID
        vargraph = self.get_var_graph()
        succ = vargraph.succ  #successor nodes in the graph
        stack = set(varnames)
        outs = []
        while len(stack) > 0:
            name = stack.pop()
            if name in vargraph:
                tup = name.split('.', 1)
                if len(tup)==1:
                    self.set_valid(name, False)
                else:
                    getattr(self, tup[0]).set_valid(tup[1], False)
            else:
                self.raise_exception("%s is not an io trait" % name,
                                     RuntimeError)
            for vname in succ.get(name, []):
                tup = vname.split('.', 1)
                if len(tup) == 1:  #boundary var or Component
                    if self.trait(vname).io_direction == 'out':
                        # it's an output boundary var
                        outs.append(vname)
                else:  # a var from a child component 
                    compname, compvar = tup
                    comp = getattr(self, compname)
                    if comp.get_valid(compvar):  # node is a valid Variable
                        for newvar in comp.invalidate_deps([compvar]):
                            stack.add('.'.join([compname, newvar]))
                        stack.add(vname)
        
        if len(outs) > 0:
            for out in outs:
                self.set_valid(out, False)
            if notify_parent and self.parent:
                self.parent.invalidate_deps(
                    ['.'.join([self.name,n]) for n in outs], 
                    notify_parent)
        return outs

    def disable_deps(self, varnames, notify_parent=False):
        """Mark all Variables disabled that depend on varnames. If
        an input is disabled, the corresponding component will not
        execute.
        
        Returns a list of our newly disabled boundary outputs.
        """
        vargraph = self.get_var_graph()
        succ = vargraph.succ  #successor nodes in the graph
        stack = set(varnames)
        outs = []
        while len(stack) > 0:
            name = stack.pop()
            if name in vargraph:
                tup = name.split('.', 1)
                if len(tup)==1:
                    self.set_enabled(name, False)
                else:
                    getattr(self, tup[0]).set_enabled(tup[1], False)
            else:
                self.raise_exception("%s is not an io trait" % name,
                                     RuntimeError)
            for vname in succ.get(name, []):
                tup = vname.split('.', 1)
                if len(tup) == 1:  #boundary var or Component
                    if self.trait(vname).io_direction == 'out':
                        # it's an output boundary var
                        outs.append(vname)
                else:  # a var from a child component 
                    compname, compvar = tup
                    comp = getattr(self, compname)
                    if comp.get_enabled(compvar):  # node is an enabled Variable
                        for newvar in comp.disable_deps([compvar]):
                            stack.add('.'.join([compname, newvar]))
                        stack.add(vname)
        
        if len(outs) > 0:
            for out in outs:
                self.set_enabled(out, False)
            if notify_parent and self.parent:
                self.parent.disable_deps(
                    ['.'.join([self.name,n]) for n in outs], 
                    notify_parent)
        return outs

