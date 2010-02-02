
#public symbols
__all__ = ['Assembly']


from enthought.traits.api import Array, List, Instance, TraitError
from enthought.traits.api import TraitType, Undefined
from enthought.traits.trait_base import not_none

import networkx as nx

from openmdao.main.interfaces import IDriver
from openmdao.main.component import Component
from openmdao.main.container import Container, READY
from openmdao.main.workflow import Workflow
from openmdao.main.asyncworkflow import AsyncWorkflow
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
        # A graph of Variable names (local path), 
        # with connections between Variables as directed edges.  
        self._var_graph = nx.DiGraph()
        self._child_io_infos = {}
        
        super(Assembly, self).__init__(doc=doc, directory=directory)
        
        self._io_info = (0, None)
        
        # add any Variables we may have inherited from our base classes
        # to our _var_graph.
        for v in self.keys(io_direction=not_none):
            self._var_graph.add_node(v)
        
        self.workflow = AsyncWorkflow(scope=self)

    def get_var_graph(self):
        """Returns the Variable dependency graph, after updating it with child
        info if necessary.
        """
        vargraph = self._var_graph
        childioinfos = self._child_io_infos
        for childname, tup in childioinfos.items():
            old_id, old_graph = tup
            if old_id is None and old_graph is not None: # component io graph never changes
                continue
            new_id, new_graph = getattr(self, childname).get_io_info(old_id)
            if new_graph: # graph has changed
                if old_graph is not None:  # remove old stuff
                    vargraph.remove_nodes_from(old_graph)
                childioinfos[childname] = (new_id, new_graph.copy())
                vargraph.add_nodes_from(new_graph.nodes_iter())
                vargraph.add_edges_from(new_graph.edges_iter())
        return self._var_graph
    
    def get_io_info(self, graph_id):
        """Return a graph connecting our input variables to our output
        variables.
        """
        if self._io_info[1] is None:
            vargraph = self.get_var_graph()
            ins = self.list_inputs()
            outs = self.list_outputs()
            io_graph = nx.DiGraph()
            io_graph.add_nodes_from(ins)
            io_graph.add_nodes_from(outs)
            
            for invar in ins:
                #find a path from in to any outs
                ans = nx.dfs_preorder(vargraph, source=invar)
                
            self._io_info = (self._io_info[0]+1, io_graph)
        if self._io_info[0] != graph_id:
            return self._io_info
        else:
            return (graph_id, None)
    
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
        self._child_io_infos[name] = (None, None)
        if isinstance(obj, Component):
            self.workflow.add_node(obj.name)
        return obj
        
    def remove_container(self, name):
        """Remove the named object from this container."""
        trait = self.trait(name)
        if trait is not None:
            if name in self._child_io_infos:
                del self._child_io_infos[name]
            obj = getattr(self, name)
            # if the named object is a Component, then assume it must
            # be removed from our workflow
            if isinstance(obj, Component):
                self.disconnect_component(name)
                self.workflow.remove_node(obj.name)
            
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
        parts = pathname.split('.')
        if alias:
            newname = alias
        else:
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

    def _split_varpath(self, path):
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

        srccompname, srccomp, srcvarname = self._split_varpath(srcpath)
        srctrait = srccomp.get_dyn_trait(srcvarname, 'out')
        destcompname, destcomp, destvarname = self._split_varpath(destpath)
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
        
        self._io_graph = None  # io graph has changed
        
        # test compatability (raises TraitError on failure)
        srcval = getattr(srccomp, srcvarname)
        if desttrait.validate is not None:
            desttrait.validate(destcomp, destvarname, srcval)
        
        self.workflow.connect(srcpath, destpath)
        if destcomp is not self:
            if srccomp is not self: # neither var is on boundary
                #self.workflow.connect(srcpath, destpath)
                destcomp.set_source(destvarname, srcpath)
                destcomp.invalidate([destvarname])

        if srccomp is not self:
            srccomp.link_output(srcvarname)
        
        self._var_graph.add_edge(srcpath, destpath)
        
        src_valid = srccomp.get_valid(srcvarname)
        dest_valid = destcomp.get_valid(destvarname)
            
        # invalidate destvar if necessary
        if destcomp is self and desttrait.io_direction == 'out': # boundary output
            if dest_valid:
                if self.parent:
                    # tell the parent that anyone connected to our boundary
                    # output is invalid.
                    # Note that it's a dest var in this scope, but a src var in
                    # the parent scope.
                    self.parent.invalidate_dependent_inputs(self.name, [destpath])
                self.set_valid(destpath, False)
        else:
            destcomp.set_valid(destvarname, False)
        
    def disconnect_component(self, compname):
        """Remove all connections to any inputs or outputs of the given component."""
        comp = getattr(self, compname)
        if isinstance(comp, Component):
            vargraph = self.get_var_graph()
            for varname in ['.'.join((compname, name)) for name in comp.list_inputs()]:
                if varname in vargraph:
                    self.disconnect(varname)
            for varname in ['.'.join((compname, name)) for name in comp.list_outputs()]:
                if varname in vargraph:
                    self.disconnect(varname)
        else:
            self.raise_exception("'%s' is not a component" %
                                 compname, RuntimeError)

    def disconnect(self, varpath, varpath2=None):
        """If varpath2 is supplied, remove the connection between varpath and
        varpath2. Otherwise, remove all connections to/from varpath in the 
        current scope.
        """
        vargraph = self.get_var_graph()
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
            to_remove.extend(vargraph.in_edges(varpath)) # incoming edges
        
        self._io_graph = None # io graph has changed, force regen later
        
        for src,sink in _filter_internal_edges(to_remove):
            self.workflow.disconnect(src, sink)
            utup = src.split('.', 1)
            vtup = sink.split('.', 1)
            if len(vtup) > 1:
                getattr(self, vtup[0]).remove_source(vtup[1])
            if len(utup) > 1:
                getattr(self, utup[0]).unlink_output(utup[1])
        
        vargraph.remove_edges_from(to_remove)

    def is_destination(self, varpath):
        """Return True if the Variable specified by varname is a destination
        according to our graph. This means that either it's an input connected
        to an output, or it's the destination part of a passtru connection.
        """
        return len(self._var_graph.in_edges(varpath)) > 0

    def execute(self, required_outputs=None):
        """Run child components"""
        self._update_inputs_from_boundary(self.list_inputs(valid=True))
        if self.driver:
            self.driver.run(required_outputs=required_outputs)
        else:
            if required_outputs is None:
                self.workflow.run()
            else:
                compnames = set()
                for src,dest in self.get_var_graph().in_edges(required_outputs):
                    tup = src.split('.', 1)
                    if len(tup) > 1:
                        compnames.add(tup[0])
                self.workflow.run(compnames)
        self._update_boundary_outputs()

    def _update_boundary_outputs (self):
        """Update output variables on our boundary, if their sources
        are valid.  This does not force execution to make any required
        inputs valid.
        """
        invalid_outs = self.list_outputs(valid=False)
        vgraph = self.get_var_graph()
        for out in invalid_outs:
            inedges = vgraph.in_edges(nbunch=out)
            if inedges:
                src = inedges[0][0]
                srcparts = src.split('.', 1)
                if getattr(self, srcparts[0]).get_valid(srcparts[1]):
                    setattr(self, out, self.get(src))

    def _post_execute (self):
        """Update output variables and anything else needed after execution. 
        Overrides of this function must call this version.
        """
        super(Assembly, self)._post_execute()
        for name in self.list_containers():
            comp = getattr(self, name)
            if isinstance(comp, Component) and not comp.ok():
                self._execute_succeeded = False
                break

    def mark_valid_outputs(self):
        """Assemblies allow partial validation based on connectivity 
        between inputs and outputs.
        """
        # FIXME: we shouldn't be marking all Assembly outputs as valid...
        for name in self.list_outputs(valid=False):
            self.set_valid(name, True)

    def step(self):
        """ Execute one workflow 'step'. """
        nodes = None
        if not self.workflow.is_active:
            self._update_inputs_from_boundary(self.list_inputs(valid=False))
            nodes = self.workflow.nodes()
        try:
            self.workflow.step(nodes)
        except StopIteration:
            pass
        
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

    def add_trait(self, name, *trait):
        """Overrides base definition of add_trait in order to
        update the vargraph.
        """
        super(Assembly, self).add_trait(name, *trait)
        self.get_var_graph().add_node(name)

    def make_inputs_valid(self, compname, inputs):
        """ Make inputs for child component valid. """
        # Trace inputs back to source outputs.
        # Add those components to evaluation list.
        trace_inputs = [(compname, inputs)]
        compnames = set()
        need_valid = set()
        while trace_inputs:
            dest, dst_inputs = trace_inputs.pop()
            print 'making inputs %s valid for comp %s' % (dst_inputs, dest)
            for vname in dst_inputs:
                destpath = ('.'.join([dest, vname]) if dest else vname)
                inedges = self.get_var_graph().in_edges(destpath)
                if inedges:
                    srcpath, dpath = inedges[0] # should be only one link to an input
                else:
                    continue
                srcparts = srcpath.split('.')
                if len(srcparts) > 1:
                    srcname = srcparts[0]
                    if srcname not in compnames:
                        compnames.add(srcname)
                        src = getattr(self, srcname)
                        # FIXME: this should really get only the *required* invalid inputs
                        req_inputs = src.list_inputs(valid=False)
                        if req_inputs:
                            trace_inputs.append((srcname, req_inputs))
                else:
                    need_valid.add(srcpath)

        # If any of our inputs are needed and not valid, call parent.
        if need_valid:
            self.parent.make_inputs_valid(self.name, need_valid)
            self._update_inputs_from_boundary(need_valid)

        # Evaluate selected components.
        self.workflow.run(compnames)

    def _update_inputs_from_boundary(self, inputs):
        """ Transfer boundary inputs to internal destinations. """
        for src in inputs:
            for src_path, dst_path in self.get_var_graph().edges(src):
                dstparts = dst_path.split('.', 1)
                if len(dstparts) == 1:
                    update_needed = not self.get_valid(dst_path)
                else:
                    update_needed = not getattr(self, dstparts[0]).get_valid(dstparts[1])
                if update_needed:
                    if self.get_enabled(src):
                        if self.get_valid(src):
                            self.set(dst_path, getattr(self, src))
                    else: # disabled
                        dstcompname, dstvarname = dst_path.split('.', 1)
                        getattr(self, dstcompname).set_enabled(dstvarname, False)

    def is_ready(self):
        """ Return True if this component is ready (and needs) to run. """
        for name in self.list_inputs():
            if not self.get_valid(name) or not self.get_enabled(name):
                return False  # Not ready -- not all inputs valid.
        if not self._valid:
            return True
        if self.list_outputs(valid=False):
            return True
        for name in self.list_containers():
            comp = getattr(self, name)
            if isinstance(comp, Component) and comp.is_ready():
                return True
        return False
    
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

    def invalidate_dependent_inputs(self, comp_name, sources):
        """ Invalidate inputs connected to now-invalid outputs. """
        invalidated = []
        print '%s.invalidate_dep_inputs(%s, %s)' % (self.name,comp_name,sources)
        for source in sources:
            if comp_name:
                edges = self.get_var_graph().edges('.'.join((comp_name, source)))
            else:
                edges = self.get_var_graph().edges(source)
            for src, dest in edges:
                destparts = dest.split('.', 1)
                if len(destparts) == 1:
                    if self.set_valid(dest, False):
                        print 'dep_inputs: invalidating %s due to change in %s' % (dest,src)
                        invalidated.append(dest)
                else:
                    print 'dep_inputs: invalidating %s due to change in %s' % (dest,src)
                    getattr(self, destparts[0]).set_valid(destparts[1], False)
        if invalidated and self.parent:
            self.parent.invalidate_dependent_inputs(self.name, invalidated)

    def invalidate(self, inputs=None):
        """Invalidate inputs/outputs/components based on specified changed inputs.
        If inputs is None, invalidate all children and dependent components.
        """
        if not inputs:
            inputs = self.list_inputs(valid=True)
            for name in self.list_containers():
                getattr(self, name).invalidate(None)
        self.invalidate_dependent_inputs('', inputs)

    def disable_dependent_inputs(self, comp_name, outputs):
        """ Disable inputs connected to now-disabled outputs. """
        disabled = []
        for source in outputs:
            for src, dest in self.get_var_graph().edges('.'.join((comp_name, source))):
                destparts = dest.split('.', 1)
                if len(destparts) == 1:
                    if self.set_enabled(dest, False):
                        disabled.append(dest)
                else:
                    getattr(self, destparts[0]).set_disabled(destparts[1])
        if disabled and self.parent:
            self.parent.disable_dependent_inputs(self.name, invalidated)
