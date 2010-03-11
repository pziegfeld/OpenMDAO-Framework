
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


def _path_exists(G, source, target):
    """Returns True if there is a path from source to target."""
    # does BFS from both source and target and meets in the middle
    if source is None or target is None:
        raise nx.NetworkXException(\
            "Bidirectional shortest path called without source or target")
    if target == source:  return source in G.predecessors(source)

    Gpred=G.predecessors_iter
    Gsucc=G.successors_iter

    # predecesssor and successors in search
    pred={source:None}
    succ={target:None}

    # initialize fringes, start with forward
    forward_fringe=[source] 
    reverse_fringe=[target]  

    while forward_fringe and reverse_fringe:
        this_level=forward_fringe
        forward_fringe=[]
        for v in this_level:
            for w in Gsucc(v):
                if w not in pred:
                    forward_fringe.append(w)
                    pred[w]=v
                if w in succ:  return True # found path
        this_level=reverse_fringe
        reverse_fringe=[]
        for v in this_level:
            for w in Gpred(v):
                if w not in succ:
                    succ[w]=v
                    reverse_fringe.append(w)
                if w in pred:  return True # found path

    return False  # no path found

class _PassthroughTrait(TraitType):
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
        self._src_infos = {}  # dict of sources required for each destination
        
        super(Assembly, self).__init__(doc=doc, directory=directory)
        
        self._io_info = (0, None)
        
        # add any Variables we may have inherited from our base classes
        # to our _var_graph.
        for v in self.keys(iotype=not_none):
            self._var_graph.add_node(v)
        
        self.workflow = AsyncWorkflow(scope=self)
        
    def get_var_graph(self):
        """Returns the Variable dependency graph."""
        vargraph = self._var_graph
        childioinfos = self._child_io_infos
        for childname, tup in childioinfos.items():
            old_id, old_graph = tup
            if old_id is None and old_graph is not None: # component io graph never changes
                continue                                 # so no need to query it again
            new_id, new_graph = getattr(self, childname).get_io_info(old_id)
            if new_graph: # graph has changed
                self._io_info = (self._io_info[0], None)
                if old_graph is not None:  # remove old stuff
                    vargraph.remove_nodes_from(old_graph)
                childioinfos[childname] = (new_id, new_graph.copy())
                vargraph.add_nodes_from(new_graph.nodes_iter())
                vargraph.add_edges_from(new_graph.edges_iter())
        return vargraph

    
    def get_io_info(self, graph_id=-999):
        """Return a graph connecting our input variables to our output
        variables.
        """
        vargraph = self.get_var_graph() # this will remove the old io graph
                                        # if the var graph has changed
        if self._io_info[1] is None: # need to build a new io graph
            # find all pairs of connected nodes
            ins = self.list_inputs()
            outs = self.list_outputs()
            inpaths = ['.'.join((self.name, name)) for name in ins]
            outpaths = ['.'.join((self.name, name)) for name in outs]
            io_graph = nx.DiGraph()
            io_graph.add_nodes_from(inpaths)
            io_graph.add_nodes_from(outpaths)
            for inp, inpath in zip(ins, inpaths):
                for out, outpath in zip(outs, outpaths):
                    if _path_exists(vargraph, inp, out):
                        io_graph.add_edge(inpath, outpath)
            self._io_info = (self._io_info[0]+1, io_graph)
    
        if self._io_info[0] != graph_id:
            return self._io_info
        else:
            return (graph_id, None)
    
    def _get_required_comp_info(self, G, outputs):
        """Return a dict of component names and their required outputs needed
        to validate the requested outputs.
        """
        neighbors=G.predecessors
    
        compnames = set([n for n in self.list_containers() 
                         if isinstance(getattr(self, n), Component)])
        compinfos = {}
        visited = set()
        stack = []
        stack.extend(outputs)
        while stack:
            v=stack.pop()
            if v not in visited:
                visited.add(v)
                parts = v.split('.', 1)
                if parts[0] not in compinfos and parts[0] in compnames:
                    compinfos[parts[0]] = []
                if len(parts) > 1:
                    compinfos[parts[0]].append(parts[1])
                stack.extend(neighbors(v))
        return compinfos

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
        """Creates a _PassthroughTrait that uses the trait indicated by
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
        iotype = trait.iotype
        # the trait.trait_type stuff below is for the case where the trait is actually
        # a ctrait (very common). In that case, trait_type is the actual underlying
        # trait object
        if (getattr(trait,'get') or getattr(trait,'set') or
            getattr(trait.trait_type, 'get') or getattr(trait.trait_type,'set')):
            trait = None  # not sure how to validate using a property
                          # trait without setting it, so just don't use it
        newtrait = _PassthroughTrait(iotype=iotype, validation_trait=trait)
        self.add_trait(newname, newtrait)

        setattr(self, newname, val)

        if iotype == 'in':
            self.connect(newname, pathname)
        else:
            self.connect(pathname, newname)

        return newtrait

    def get_dyn_trait(self, pathname, iotype=None):
        """Retrieves the named trait, attempting to create a Passthrough trait
        on-the-fly if the specified trait doesn't exist.
        """
        trait = self.trait(pathname)
        if trait is None:
            newtrait = self.create_passthrough(pathname)
            if iotype is not None and iotype != newtrait.iotype:
                self.raise_exception(
                    "new trait has iotype of '%s' but '%s' was expected" %
                    (newtrait.iotype, iotype), TraitError)
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
            if srctrait.iotype != 'out':
                self.raise_exception(
                    '.'.join([srccomp.get_pathname(),srcvarname])+
                    ' must be an output variable',
                    RuntimeError)
            if desttrait.iotype != 'in':
                self.raise_exception(
                    '.'.join([destcomp.get_pathname(),destvarname])+
                    ' must be an input variable',
                    RuntimeError)
                
        if self.is_destination(destpath):
            self.raise_exception(destpath+' is already connected',
                                 RuntimeError)             
        
        # test compatability (raises TraitError on failure)
        srcval = srccomp.get_wrapped_attr(srcvarname)
        if desttrait.validate is not None:
            desttrait.validate(destcomp, destvarname, srcval)
        
        try:
            self.workflow.connect(srcpath, destpath)
            if destcomp is not self:
                if srccomp is not self: # neither var is on boundary
                    destcomp.invalidate([destvarname])
            destcomp.set_source(destvarname, srcpath)
    
            if srccomp is not self:
                srccomp.link_output(srcvarname)
            
            self._var_graph.add_edge(srcpath, destpath)
            
            with destcomp._lock:
                # invalidate destvar if necessary
                if destcomp is self and desttrait.iotype == 'out': # boundary output
                    if destcomp.get_valid(destvarname):
                        if self.parent:
                            # tell the parent that anyone connected to our boundary
                            # output is invalid.
                            # Note that it's a dest var in this scope, but a src var in
                            # the parent scope.
                            self.parent.invalidate_dependent_inputs(self.name, [destpath])
                        self.set_valid(destpath, False)
                else:
                    destcomp.set_valid(destvarname, False)
        finally:
            self._io_info = (self._io_info[0], None)  # io graph has changed
        
        
    def disconnect_component(self, compname):
        """Remove all connections to any inputs or outputs of the given component."""
        comp = getattr(self, compname)
        if isinstance(comp, Component):
            vargraph = self._var_graph
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
        vargraph = self._var_graph
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
        
        for src,sink in _filter_internal_edges(to_remove):
            self.workflow.disconnect(src, sink)
            utup = src.split('.', 1)
            vtup = sink.split('.', 1)
            if len(vtup) > 1:
                getattr(self, vtup[0]).remove_source(vtup[1])
            if len(utup) > 1:
                getattr(self, utup[0]).unlink_output(utup[1])
        
        vargraph.remove_edges_from(to_remove)
        self._io_info = (self._io_info[0], None)  # io graph has changed

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
                compnames = set([n for n in self.list_containers() 
                                 if isinstance(getattr(self, n), Component)])
                vargraph = self.get_var_graph()
                comp_info = self._get_required_comp_info(vargraph, required_outputs)
                self.workflow.run(comp_info)
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
                    try:
                        #print '%s: Transfer--> %s to %s (%s)' % (self.name,src,out,self.get(src))
                        setattr(self, out, self.get(src))
                    except Exception, err:
                        self.raise_exception("cannot set '%s' with '%s': %s" %
                                             (out, src, str(err)), type(err))

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
        graphid, iograph = self.get_io_info()
        invalid_outs = set(self.list_outputs(valid=False))
        invalid_ins = set(self.list_inputs(valid=False))
        skip = set([out for inp,out in iograph.edges() if inp in invalid_ins])
        final = invalid_outs - skip
        for name in final:
            self.set_valid(name, True)

    def step(self):
        """ Execute one workflow 'step'. """
        nodes = None
        if self.workflow.is_active:
            try:
                self.workflow.step()
            except StopIteration:
                pass
        else:
            self._update_inputs_from_boundary(self.list_inputs(valid=False))
            try:
                self.workflow.step(dict([(name,None) for name in self.workflow.nodes()]))
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
        vargraph = self.get_var_graph()
        if show_passthrough:
            return _filter_internal_edges(vargraph.edges())
        else:
            return _filter_internal_edges([(outname,inname) for outname,inname in 
                                                vargraph.edges_iter() 
                                                if '.' in outname and '.' in inname])

    def add_trait(self, name, *trait):
        """Overrides base definition of add_trait in order to
        update the vargraph.
        """
        super(Assembly, self).add_trait(name, *trait)
        self._var_graph.add_node(name)

    def make_inputs_valid(self, compname, inputs):
        """ Make inputs for child component valid. To do this requires that
        any outputs those inputs depend on must be made valid (if necessary)
        and transferred to those inputs.
        """
        vargraph = self.get_var_graph()
        trace_inputs = [(compname, inputs)]
        need_valid = set()
        comp_infos = {}  # comp name dict with required outputs as data
        invalid_inputs = {}
        while trace_inputs:
            dest, dst_inputs = trace_inputs.pop()
            if dest:
                destpaths = ['.'.join((dest, name)) for name in dst_inputs]
            else:
                destpaths = dst_inputs
                
            req_inputs = {} # comp names dict with required inputs as data
            for destpath in destpaths:
                for srcpath in vargraph.predecessors(destpath):
                    srcparts = srcpath.split('.')
                    if len(srcparts) > 1:
                        srcname = srcparts[0]
                        comp_info = comp_infos.get(srcname)
                        if comp_info is None:
                            src = getattr(self, srcname)
                            inv_inputs = src.list_inputs(valid=False)
                            invalid_inputs[srcname] = inv_inputs
                            comp_infos[srcname] = set([srcparts[1]])
                        else:
                            inv_inputs = invalid_inputs[srcname]
                            comp_infos[srcname].add(srcparts[1])
                        
                        req = [name for name in inv_inputs 
                                if _path_exists(vargraph, '.'.join((srcname,name)), 
                                                destpath)]
                        if req:
                            req_inputs[srcname] = req
                    else:
                        need_valid.add(srcpath)
            for srcname, reqs in req_inputs.items():
                trace_inputs.append((srcname, reqs))

        # If any of our inputs are needed and not valid, call parent.
        if need_valid:
            self.parent.make_inputs_valid(self.name, need_valid)
            self._update_inputs_from_boundary(need_valid)

        # Evaluate selected components.
        self.workflow.run(comp_infos)

    def _update_inputs_from_boundary(self, inputs):
        """ Transfer boundary inputs to internal destinations. """
        vargraph = self.get_var_graph()
        for src in inputs:
            for src_path, dst_path in vargraph.edges(src):
                try:
                    dstcompname, dstvarname = dst_path.split('.', 1)
                except ValueError:
                    dstcomp = self
                    dstvarname = dst_path
                else:
                    dstcomp = getattr(self, dstcompname)
                if not dstcomp.get_valid(dstvarname):
                    if self.get_enabled(src):
                        if self.get_valid(src):
                            try:
                                #print '%s: -->Transfer %s to %s (%s)' % (self.name,src_path, dst_path,getattr(self,src))
                                self.set(dst_path, self.get_wrapped_attr(src), srcname=src_path)
                            except Exception, err:
                                self.raise_exception("cannot set '%s' from '%s' : %s" %
                                                     (dst_path, src, str(err)), type(err))
                        else:
                            self.raise_exception("input variable %s is not valid!" % src,
                                                 RuntimeError)
                    else: # disabled
                        dstcomp.set_enabled(dstvarname, False)

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
        vargraph = self.get_var_graph()
        for source in sources:
            if comp_name:
                edges = vargraph.edges('.'.join((comp_name, source)))
            else:
                edges = vargraph.edges(source)
            for src, dest in edges:
                destparts = dest.split('.', 1)
                if len(destparts) == 1:
                    if self.set_valid(dest, False):
                        invalidated.append(dest)
                else:
                    getattr(self, destparts[0]).set_valid(destparts[1], False)
        if invalidated:
            self._valid = False
            if self.parent:
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
        vargraph = self.get_var_graph()
        for source in outputs:
            for src, dest in vargraph.edges('.'.join((comp_name, source))):
                destparts = dest.split('.', 1)
                if len(destparts) == 1:
                    if self.set_enabled(dest, False):
                        disabled.append(dest)
                else:
                    getattr(self, destparts[0]).set_disabled(destparts[1])
        if disabled and self.parent:
            self.parent.disable_dependent_inputs(self.name, invalidated)
