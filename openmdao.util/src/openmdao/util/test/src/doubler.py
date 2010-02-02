
from enthought.traits.api import Float

from openmdao.main.api import Component

class Doubler(Component):
    x = Float(0.0, io_direction='in')
    y = Float(0.0, io_direction='out')

    def execute(self, required_outputs=None):
        self.y = self.x * 2

