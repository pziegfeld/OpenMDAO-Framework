from enthought.traits.api import Float

from openmdao.main.api import Component

class Box(Component):
    # set up interface to the framework
    width = Float(1., io_direction='in', units='cm')
    height = Float(1., io_direction='in', units='cm')
    depth = Float(1., io_direction='in', units='cm')
    thickness = Float(0.05, io_direction='in', units='cm')
    density = Float(0.01, io_direction='in', units='g/cm**3')
    
    mass = Float(0., io_direction='out', units='g')
    volume = Float(0., io_direction='out', units='cm**3')
    surface_area = Float(0., io_direction='out', units='cm**2')
        
    def __init__(self, doc=None, directory=''):
        super(Box, self).__init__(doc, directory)
        
    def execute(self, required_outputs=None):
        self.surface_area = (self.width*(self.height+self.depth)+
                     self.depth*self.height)*2
        self.mass = self.surface_area*self.thickness*self.density        
        self.volume = self.width*self.height*self.depth

