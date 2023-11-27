import inspect
import marshal
import types

def update_function_state(func, state):
    func.__globals__.update(state['function_globals'])
    func.__defaults__ = state['defaults']
    func.__kwdefaults__ = state['kwdefaults']

def deserialize_function(name, qualname, func_code, closure):
    func_code = marshal.loads(func_code)
    global_scope = {'__builtins__': __builtins__}
    deserialized_func = types.FunctionType(func_code, global_scope, name, closure=closure)
    deserialized_func.__qualname__ = qualname
    return deserialized_func

def get_global_references(func_code, global_scope, function_globals):
    for constant in func_code.co_consts:
        if inspect.iscode(constant):
            closure = tuple(types.CellType(None) for _ in range(len(constant.co_freevars)))
            nested_func = types.FunctionType(constant, global_scope, 'nested_func', closure=closure)
            function_globals.update(inspect.getclosurevars(nested_func).globals)
            get_global_references(constant, global_scope, function_globals)

def serialize_function_data(func):
    closure_vars = inspect.getclosurevars(func)
    func_code = marshal.dumps(func.__code__)
    function_globals = dict(closure_vars.globals)
    function_definition = (func.__name__, func.__qualname__, func_code, func.__closure__)
    get_global_references(func.__code__, func.__globals__, function_globals)
    function_context = {
        'function_globals': function_globals,
        'defaults': func.__defaults__,
        'kwdefaults': func.__kwdefaults__
    }
    return deserialize_function, function_definition, function_context, update_function_state