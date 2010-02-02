

def deep_setattr(obj, path, value):
    """A multi-level setattr, setting the value of an
    attribute specified by a dotted path. For example,
    deep_settattr(obj, 'a.b.c', value).
    """
    tup = path.split('.')
    for name in tup[:-1]:
        obj = getattr(obj, name)
    setattr(obj, tup[-1], value)

def deep_getattr(obj, path):
    """A multi-level getattr, returning the value of an
    attribute specified by a dotted path. For example,
    deep_gettattr(obj, 'a.b.c').
    """
    for name in path.split('.'):
        obj = getattr(obj, name)
    return obj
