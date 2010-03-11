
from enthought.traits.api import Float

from openmdao.main.api import Component

class Doubler(Component):
    x = Float(0.0, iotype='in')
    y = Float(0.0, iotype='out')

    def execute(self, required_outputs=None):
        self.y = self.x * 2

