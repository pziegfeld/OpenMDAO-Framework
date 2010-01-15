#public symbols
__all__ = ['StringRef', 'StringRefArray']


from enthought.traits.api import BaseStr, List, TraitError
from enthought.traits.trait_handlers import NoDefaultSpecified

from openmdao.main.api import ExprEvaluator

class DumbDefault(object):
    def __getattr__(self, name):
        raise TraitError('StringRef: string reference is undefined')
            
class StringRef(BaseStr):
    """A trait that references, via a pathname, another trait in the
    framework. If it has io_direction of 'out', then the string may only be the pathname of
    a single variable (with optional array indexing), but if io_direction is 'in',
    it may be any valid expression and may reference any number of other
    variables.
    """
    
    def __init__(self, default_value=NoDefaultSpecified, **metadata):
        if default_value is NoDefaultSpecified:
            default_value = DumbDefault()
        super(StringRef, self).__init__(default_value, **metadata)

    def validate(self, object, name, value):
        s = super(StringRef, self).validate(object, name, value) # normal string validation
        try:
            if self.io_direction == 'out':
                s = ExprEvaluator(s, object, single_name=True)
            else:
                s = ExprEvaluator(s, object)
            s._parse()
        except RuntimeError:
            raise TraitError("invalid %sput ref variable value '%s'"%(self.io_direction,
                                                                        str(value)))
        return s
    
    
class StringRefArray(List):
    """A List of StringRef traits."""
    
    def __init__(self, **metadata):
        self.io_direction = metadata.get('io_direction', 'in')
        super(StringRefArray, self).__init__(trait=StringRef(io_direction=self.io_direction), 
                                             **metadata)
    
