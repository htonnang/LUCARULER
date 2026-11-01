"""Bounded object-aware composition after legacy exact search is exhausted."""
import time
from . import object_diff, recursive_world, set_object_reducer, imported_primitives, primitives
from .object_support import view, selected_objects, render, fits
from .diagnostics import event, record_exception
from .primitives import Law, GEOM_OPS

def pool(pairs):
    laws=[]
    for module,fn in [('objects',object_diff.fit_all_object_diff_traced),
                      ('recursive',recursive_world.fit_all_recursive_traced),
                      ('sets',set_object_reducer.fit_all_set_reducer_traced),
                      ('supplemental',imported_primitives.fit_all_imported_traced)]:
        event('fitter_called:'+module)
        try: laws.extend(law for law,_ in fn(pairs))
        except Exception: record_exception('enhanced_fitter:'+module)
    return laws

def operations():
    for conn in (4,8):
        for selector in ('largest','smallest','unique_shape','repeated_shape'):
            for crop in (True,False):
                def select(grid,conn=conn,selector=selector,crop=crop):
                    bg,objs=view(grid,(0,conn,True))
                    return render(grid,bg,selected_objects(objs,selector),crop=crop)
                yield f'select_{selector}_{conn}_crop{crop}',select
    yield 'compress_empty_lines',imported_primitives.compress_background
    yield 'minimal_tile',imported_primitives.minimal_tile
    for name in ('reflect_h','reflect_v','rot90','rot180','transpose'):
        yield name,GEOM_OPS[name]

def search(pairs,seconds=8.0):
    from .budget import check as _budget_check
    _budget_check()
    started=time.monotonic()
    found=pool(pairs)
    if found:return sorted(found,key=lambda law:10*law.op_count+2*law.constant_bits)
    seen={tuple(pairs)}
    for name,fn in operations():
        if time.monotonic()-started>seconds:
            event('enhanced_budget_exhausted')
            break
        try:intermediate=[(fn(x),y) for x,y in pairs]
        except (ValueError,IndexError,KeyError):
            event('composition_precondition_failed');continue
        key=tuple(intermediate)
        if key in seen:continue
        seen.add(key);event('composition_state_examined')
        # Shape-changing object selection can precede a normal geometric,
        # recolouring, scaling or panel operation without first solving the task.
        terminal=[law for law,_ in primitives.fit_all_single_traced(intermediate)]
        terminal.extend(pool(intermediate))
        for law in terminal:
            def predict(grid,fn=fn,law=law):return law.predict(fn(grid))
            if fits(predict,pairs):
                found.append(Law(name+'+'+law.name,predict,
                                 f'{name} -> {law.description}',law.constant_bits,law.op_count+1))
        if found:break
    return sorted(found,key=lambda law:10*law.op_count+2*law.constant_bits)
