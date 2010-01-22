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

verbose = True

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
        self._verbose = verbose

        self.sequential = False
        self.record_states = True
        self.dispatch_table = []
        
        self._compnames = {}         # iterator of names of components to run.
        self._stop = False           # If True, stop evaluation.
        self._active = []            # List of components currently active.
        self._ready_q = []           # List of components ready to run.
        self._done_q = Queue.Queue() # Queue of completion info tuples.
        self._pool = WorkerPool(self._service_loop) # Pool of worker threads.
        
        
    def add_node(self, node):
        """ Add a new node to the end of the flow. """
        if isinstance(getattr(self.scope, node), Component):
            self._comp_graph.add_node(node)
        else:
            raise TypeError('%s is not a Component' % 
                            '.'.join((self.scope.get_pathname(),node)))
        
    def nodes(self):
        return self._comp_graph.nodes()
        
    def remove_node(self, node):
        """Remove a component from this Workflow and any of its children."""
        self._comp_graph.remove_node(node)

    def run(self, compnames=None):
        """
        Execute the given components. `compnames` is a dictionary
        mapping components to required outputs (or None for all).
        """
        if compnames is None:
            compnames = self._comp_graph.nodes()
        self._setup(compnames)
        if not self._ready_q:
            return
        self.resume()
    
    def steppable(self):
        """ Return True if it makes sense to 'step' this component. """
        return len(self._comp_graph) > 1

    def connect(self, srcpath, destpath):
        """Add an edge to our Component graph from *srccompname* to *destcompname*.
        The *srcvarname* and *destvarname* args are for data reporting only.
        """
        srccompname, srcvarname = srcpath.split('.', 1)
        destcompname, destvarname = destpath.split('.', 1)
        
        # if an edge already exists between the two components, 
        # just increment the ref count
        graph = self._comp_graph
        try:
            graph[srccompname][destcompname]['refcount'] += 1
        except KeyError:
            graph.add_edge(srccompname, destcompname, refcount=1)
            
        if not is_directed_acyclic_graph(graph):
            # do a little extra work here to give more info to the user in the error message
            strongly_connected = strongly_connected_components(graph)
            refcount = graph[srccompname][destcompname]['refcount'] - 1
            if refcount == 0:
                graph.remove_edge(srccompname, destcompname)
            else:
                graph[srccompname][destcompname]['refcount'] = refcount
            for strcon in strongly_connected:
                if len(strcon) > 1:
                    raise CircularDependencyError(
                        'circular dependency (%s) would be created by connecting %s to %s' %
                                 (str(strcon), 
                                  '.'.join([srccompname,srcvarname]), 
                                  '.'.join([destcompname,destvarname]))) 
        
    def disconnect(self, srcpath, destpath):
        """Decrement the ref count for the edge in the dependency graph 
        between the two components, or remove the edge if the ref count
        reaches 0.
        """
        comp1name, var1name = srcpath.split('.', 1)
        comp2name, var2name = destpath.split('.', 1)
        refcount = self._comp_graph[comp1name][comp2name]['refcount'] - 1
        if refcount == 0:
            self._comp_graph.remove_edge(comp1name, comp2name)
        else:
            self._comp_graph[comp1name][comp2name]['refcount'] = refcount

#=================================================================================

    @property
    def is_active(self):
        """ Return True if the workflow is currently active. """
        return self._active or self._ready_q

    def resume(self):
        """ Resume execution. """
        if not self._ready_q:
            raise RuntimeError('Run already complete')

        # Process results until all activity is complete.
        self._start()
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
        for compname in self._compnames:
            comp = getattr(self.scope, compname)
            if not comp.is_valid() and comp.is_ready():
                failures.append(compname)
        if failures:
            raise RuntimeError('the following components failed: %s' % failures)

    def step(self, compnames=None):
        """
        If any workers are 'active', process the next one that finishes. If
        `compnames` is specified, then start a new evaluation.  `compnames` is
        an iterator of names of components to run (or None for all).
        """
        if compnames:
            self._setup(compnames)

        self._start()
        if self.is_active:
            worker, compname, msg, ready = self._done_q.get()
            self._pool.release(worker)
            self._active.remove(compname)
            if msg is None:
                # Note that multiple workers may think that they are
                # the one that made a dependent component ready.
                for dependent in ready:
                    if dependent.name in self._compnames:
                        if dependent not in self._ready_q:
                            dependent.set_ready()
                            self._ready_q.append(dependent)
            else:
                raise RuntimeError('Component failed: %s' % msg)
        else:
            raise StopIteration

    def stop(self):
        """ Stop the evaluation (eventually). """
        self._stop = True

    def _setup(self, compnames):
        """ Setup to begin evaluation. """
        self._compnames = compnames
        self._active = []
        self.dispatch_table = []

        # Drain _done_q.
        while True:
            try:
                self._done_q.get_nowait()
            except Queue.Empty:
                break

        # Get components that are ready-to-run.
        self._ready_q = [comp for comp in [getattr(self.scope, cname) for cname in self._compnames] 
                             if comp.is_ready()]
        
    def _start(self):
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
        while True:
            comp = request_q.get()
            if comp is None:
                request_q.task_done()
                return  # Shutdown.

            msg = None
            ready = []
            try:
                # FIXME: put partial execution stuff back later
                comp.run()
            except Exception:
                msg = traceback.format_exc()
            else:
                # Update dependent components.
                try:
                    outdata = comp.get_outgoing_data()
                    updated = set()
                    for name, val in outdata.items():
                        srcpath = '.'.join((comp.name, name))
                        for dest in self.scope._var_graph.succ[srcpath]:
                            tup = dest.split('.', 1)
                            if len(tup) == 2:
                                destcomp = getattr(self.scope, tup[0])
                                destcomp.set(tup[1], val, 
                                             srcname=srcpath)
                                updated.add(destcomp)
                            else: # boundary output
                                self.scope.set(dest, val, 
                                               srcname=srcpath)
                                if self._verbose: 
                                    print 'Transfer %s.%s to %s' % (comp.name, name, dest)
    
                    # Check if updated components are now ready.
                    ready = [ucomp for ucomp in updated if ucomp.is_ready()]
                except Exception:
                    msg = traceback.format_exc()

            request_q.task_done()
            self._done_q.put((request_q, comp.name, msg, ready))


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

