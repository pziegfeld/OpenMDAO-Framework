"""
Workflow support.
"""

import atexit
import platform
import Queue
import threading
import traceback

import networkx as nx
from networkx.algorithms.traversal import is_directed_acyclic_graph, strongly_connected_components

from openmdao.main.exceptions import RunStopped, CircularDependencyError
from openmdao.main.workflow import Workflow
from openmdao.main.component import Component

__all__ = ('AsyncWorkflow',)

# 'fake' component nodes in the workflow used to represent data
# flow to/from the boundary of the Assembly
BOUNDARY_IN = '@in'
BOUNDARY_OUT = '@out'

class PicklableQueue(Queue.Queue):
    def __getstate__(self):
        state = self.__dict__.copy()
        to_remove = ['mutex', 'not_empty', 'not_full', 'all_tasks_done']
        for name in to_remove:
            del state[name]
        return state
    
    def __setstate__(self, state):
        state['mutex'] = threading.Lock()
        state['not_empty'] = threading.Condition(state['mutex'])
        state['not_full'] = threading.Condition(state['mutex'])
        state['all_tasks_done'] = threading.Condition(state['mutex'])
        self.__dict__ = state

class AsyncWorkflow(Workflow):
    """
    Evaluates components in lazy dataflow fashion.

    - If `sequential` is True, evaluate sequentially.
    - If `record_states` is True, then `dispatch_table` is filled.
    - `dispatch_table` is a list of the names of dispatched components.

    During evaluation, each component which is ready to run based on its
    input state is dispatched to a separate thread which runs the component,
    copies outputs to the inputs of any dependent components, and queues
    results. Subsequently, any now-ready dependent components are queued for
    later execution.
    """

    def __init__(self, scope=None, verbose=True):
        super(AsyncWorkflow, self).__init__(scope=scope)
        
        self._comp_graph = nx.DiGraph()
        self._comp_graph.add_nodes_from([BOUNDARY_IN, BOUNDARY_OUT])
        self._verbose = verbose

        self.sequential = False
        self.record_states = True
        self.dispatch_table = []
        
        self._comp_info = {}         # dict of names of components and their required outputs. (None==all)
        self._stop = False           # If True, stop evaluation.
        self._active = []            # List of components currently active.
        self._ready_q = []           # List of components ready to run.
        #self._done_q = Queue.Queue() # Queue of completion info tuples.
        self._done_q = PicklableQueue() # Queue of completion info tuples.
        self._pool = WorkerPool(self._service_loop) # Pool of worker threads.
        
        
    def add_node(self, node):
        """ Add a new node to the end of the flow. """
        if isinstance(getattr(self.scope, node), Component):
            self._comp_graph.add_node(node)
            self._comp_info[node] = None
        else:
            raise TypeError('%s is not a Component' % 
                            '.'.join((self.scope.get_pathname(),node)))
        
    def nodes(self):
        return [n for n in self._comp_graph.nodes() if not n.startswith('@')]
        
    def remove_node(self, node):
        """Remove a component from this Workflow and any of its children."""
        self._comp_graph.remove_node(node)
        if node in self._comp_info:
            del self._comp_info[node]

    def run(self, comp_info=None):
        """
        Execute the given components. `comp_info` is a dictionary
        mapping components to required outputs (or None for all).
        """
        if comp_info is None:
            comp_info = dict([(name,None) for name in self.nodes()])
        self._setup(comp_info)
        if not self._ready_q:
            return
        self.resume()
    
    def steppable(self):
        """ Return True if it makes sense to 'step' this component. """
        return len(self._comp_graph) > 1

    def _parse_pathnames(self, srcpath, destpath):
        srcparts = srcpath.split('.', 1)
        if len(srcparts) > 1:
            srccompname, srcvarname = srcpath.split('.', 1)
        else:
            srccompname = BOUNDARY_IN
            srcvarname = srcpath
        destparts = destpath.split('.', 1)
        if len(destparts) > 1:
            destcompname, destvarname = destpath.split('.', 1)
        else:
            destcompname = BOUNDARY_OUT
            destvarname = destpath
        return (srccompname, srcvarname, destcompname, destvarname)
        
    def connect(self, srcpath, destpath):
        """Add an edge to our Component graph from *srccompname* to *destcompname*.
        The *srcvarname* and *destvarname* args are for data reporting only.
        """
        srccompname, srcvarname, destcompname, destvarname = \
                                     self._parse_pathnames(srcpath, destpath)
        
        # if an edge already exists between the two components, 
        # just increment the ref count
        graph = self._comp_graph
        try:
            data = graph[srccompname][destcompname]
        except KeyError:
            io_connects = {srcvarname:[destvarname]}
            graph.add_edge(srccompname, destcompname, io_connects=io_connects)
            
            if not is_directed_acyclic_graph(graph):
                # do a little extra work here to give more info to the user in the error message
                strongly_connected = strongly_connected_components(graph)
                # put the graph back the way it was before the cycle was created
                io_connects[srcvarname].remove(destvarname)
                if len(io_connects[srcvarname]) == 0:
                    del io_connects[srcvarname]
                    if len(io_connects) == 0:
                        graph.remove_edge(srccompname, destcompname)
                raise CircularDependencyError(
                        'circular dependency (%s) would be created by connecting %s to %s' %
                        (str(strongly_connected[0]), 
                         '.'.join([srccompname,srcvarname]), 
                         '.'.join([destcompname,destvarname]))) 
        
    def disconnect(self, srcpath, destpath):
        """Decrement the ref count for the edge in the dependency graph 
        between the two components, or remove the edge if the ref count
        reaches 0.
        """
        srccompname, srcvarname, destcompname, destvarname = \
                                      self._parse_pathnames(srcpath, destpath)
        try:
            io_connects = self._comp_graph[srccompname][destcompname]['io_connects']
        except KeyError:
            return # ignore disconnection of things that aren't connected
        io_connects[srcvarname].remove(destvarname)
        if len(io_connects[srcvarname]) == 0:
            del io_connects[srcvarname]
            if len(io_connects) == 0:
                self._comp_graph.remove_edge(srccompname, destcompname)

    @property
    def is_active(self):
        """ Return True if the workflow is currently active. """
        return self._active or self._ready_q

    def resume(self):
        """ Resume execution. """
        if not self._ready_q:
            raise RuntimeError('Run already complete')

        # Process results until all activity is complete.
        self._start_ready_comps()
        self._stop = False
        while self.is_active:
            if self._stop:
                raise RunStopped()
            try:
                self.step()
            except StopIteration:
                break

        # At this point all runnable components have been run.
        # Check for failures.
        failures = []
        for compname in self._comp_info:
            comp = getattr(self.scope, compname)
            if not comp.ok():
                failures.append(compname)
        if failures:
            raise RuntimeError('the following components failed: %s' % failures)

    def step(self, comp_info=None):
        """
        If any workers are 'active', process the next one that finishes. If
        `comp_info` is specified, then start a new evaluation.  `comp_info` is
        an iterator of names of components to run (or None for all).
        """
        if comp_info:
            self._setup(comp_info)

        self._start_ready_comps()
        if self.is_active:
            worker, compname, err, trace_str, ready = self._done_q.get()
            self._pool.release(worker)
            try:
                self._active.remove(compname)
            except ValueError:
                pass
            if err is None:
                # Note that multiple workers may think that they are
                # the one that made a dependent component ready.
                for dependent in ready:
                    if dependent.name in self._comp_info and dependent not in self._ready_q:
                        dependent.set_ready()
                        self._ready_q.append(dependent)
            else:
                raise err
                #raise RuntimeError('Component failed: %s' % msg)
        else:
            raise StopIteration

    def stop(self):
        """ Stop the evaluation (eventually). """
        self._stop = True
        for name in self.nodes():
            getattr(self.scope, name).stop()

    def _setup(self, comp_info):
        """ Setup to begin evaluation. 
        Puts all ready components on the ready queue.
        """
        self._comp_info = comp_info
        self._active = []
        self.dispatch_table = []

        # Drain _done_q.
        while True:
            try:
                self._done_q.get_nowait()
            except Queue.Empty:
                break

        # Get components that are ready-to-run, as well as components that will
        # be ready to run as soon as they get some inputs from other components that
        # are already valid
        comps = [getattr(self.scope, cname) for cname in self._comp_info]
        ready_set = set()
        runinfos = [(comp, comp.get_run_info()) for comp in comps]
        for comp, stuff in runinfos:
            ready, outputs = stuff
            if ready:
                if not comp in ready_set:
                    ready_set.add(comp)
            else:
                for name, val in outputs.items():
                    srcpath = '.'.join((comp.name, name))
                    for src,dest in self.scope._var_graph.edges(srcpath):
                        tup = dest.split('.', 1)
                        if len(tup) == 2:
                            destcomp = getattr(self.scope, tup[0])
                            if destcomp.get_valid(tup[1]) is False:
                                try:
                                    #print '%s: Transfer %s to %s (%s)' % (self.scope.name,srcpath,dest,val)
                                    destcomp.set(tup[1], val, srcname=srcpath)
                                except Exception, err:
                                    self.scope.raise_exception("cannot set '%s' from '%s': %s" %
                                                               (dest, srcpath, str(err)),
                                                               type(err))
                            if destcomp.is_ready():
                                ready_set.add(destcomp)
        self._ready_q = list(ready_set)

        
    def _start_ready_comps(self):
        """ Start all runnable components (unless sequential). """
        launched = []
        while self._ready_q:
            # Grab a worker and use it to run the next component.
            worker_q = self._pool.get()
            comp = self._ready_q.pop(0)
            worker_q.put(comp)
            self._active.append(comp.name)
            launched.append(comp.name)
            if self.sequential:
                break
        if self.record_states and launched:
            self.dispatch_table.append(launched)

    def _service_loop(self, request_q):
        """
        Get component, run it, update dependent inputs, and queue results.
        """
        vargraph = self.scope.get_var_graph()
        while True:
            comp = request_q.get()
            if comp is None:
                request_q.task_done()
                return  # Shutdown.

            err = trace_str = None
            ready = []
            try:
                req_outs = self._comp_info[comp.name]
                comp.run(required_outputs=req_outs)
            except Exception, err:
                trace_str = traceback.format_exc()
            else:
                # Update dependent components.
                try:
                    outdata = comp.get_available_linked_outputs()
                    updated = set()
                    for name, val in outdata.items():
                        srcpath = '.'.join((comp.name, name))
                        for src,dest in vargraph.edges(srcpath):
                            tup = dest.split('.', 1)
                            if len(tup) == 2:
                                destcomp = getattr(self.scope, tup[0])
                                destvar = tup[1]
                            else: # boundary output
                                destcomp = self.scope
                                destvar = tup[0]
                                
                            if not destcomp.get_valid(destvar):
                                try:
                                    #print '%s: Transfer %s to %s (%s)' % (self.scope.name,srcpath,dest,val)
                                    destcomp.set(destvar, val, srcname=srcpath)
                                except Exception, err:
                                    self.scope.raise_exception("cannot set '%s' from '%s': %s" %
                                                               (dest, srcpath, str(err)),
                                                               type(err))
                            if destcomp is not self.scope:
                                updated.add(destcomp)
    
                    # Check if updated components are now ready.
                    ready = [ucomp for ucomp in updated if ucomp.is_ready()]
                except Exception, err:
                    trace_str = traceback.format_exc()

            request_q.task_done()
            self._done_q.put((request_q, comp.name, err, trace_str, ready))


class WorkerPool(object):
    """ Pool of worker threads, grows as necessary. """

    def __init__(self, target):
        self._target = target
        self._idle = []     # Queues of idle workers.
        self._workers = {}  # Maps queue to worker.
        atexit.register(self.cleanup)

    def cleanup(self):
        """ Cleanup resources (worker threads). """
        for queue in self._workers:
            queue.put(None)
            self._workers[queue].join(1)
            if float(platform.python_version()[:3]) < 2.6:
                alive = self._workers[queue].isAlive()
            else:
                alive = self._workers[queue].is_alive()
            if alive:
                print 'Worker join timed-out.'
            try:
                self._idle.remove(queue)
            except ValueError:
                pass  # Never released due to some other issue...
        self._workers.clear()

    def get(self):
        """ Get a worker queue from the pool. """
        try:
            return self._idle.pop()
        except IndexError:
            queue = Queue.Queue()
            worker = threading.Thread(target=self._target, args=(queue,))
            if float(platform.python_version()[:3]) < 2.6:
                worker.setDaemon(True)
            else:
                worker.daemon = True
            worker.start()
            self._workers[queue] = worker
            return queue
                
    def release(self, queue):
        """ Release a worker queue back to the pool. """
        self._idle.append(queue)

