"""Shared bounded perception and rendering for new object reasoning."""
from collections import Counter
from functools import lru_cache
from .grid_world import extract_objects, extract_holes, guess_backgrounds
from .primitives import Law
from .diagnostics import event, record_exception
CONFIGS = [(rank, conn, multi) for rank in (0, 1) for conn in (4, 8) for multi in (False, True)]

@lru_cache(maxsize=1024)
def view(grid, config):
    rank, conn, multi = config
    if isinstance(rank, str):
        from .backgrounds import resolve
        bg = resolve(grid, rank)
    else:
        bgs = guess_backgrounds(grid, k=2)
        if rank >= len(bgs): return bgs[0], ()
        bg = bgs[rank]
    objs = tuple(extract_objects(grid, bg, conn, multi))
    if len(objs) > 40:
        event('perception_object_limit')
        return bg, ()
    event('representation_examined', config)
    return bg, objs

def local_grid(obj, bg):
    r0, c0, r1, c1 = obj.bbox
    return tuple(tuple(obj.multicolor.get((r, c), bg) for c in range(c0, c1 + 1)) for r in range(r0, r1 + 1))

def features(obj, bg):
    r0, c0, r1, c1 = obj.bbox
    holes = extract_holes(local_grid(obj,bg), bg)
    return {'area': obj.area, 'height': r1-r0+1, 'width': c1-c0+1,
            'holes': len(holes), 'hole_area': sum(o.area for o in holes),
            'colors': len(set(obj.multicolor.values())), 'bbox_empty': (r1-r0+1)*(c1-c0+1)-obj.area}

def selected_objects(objs, selector):
    if selector == 'all': return list(objs)
    if not objs: return []
    if selector in ('largest', 'smallest'):
        value = (max if selector == 'largest' else min)(o.area for o in objs)
        return [o for o in objs if o.area == value]
    if selector in ('unique_shape', 'repeated_shape'):
        counts = Counter(o.shape() for o in objs)
        return [o for o in objs if (counts[o.shape()] == 1) == (selector == 'unique_shape')]
    if selector in ('rare_color', 'common_color'):
        counts = Counter(o.color for o in objs)
        value = (min if selector == 'rare_color' else max)(counts.values())
        return [o for o in objs if counts[o.color] == value]
    axis = 'height' if selector in ('tallest', 'shortest') else 'width'
    values = {o.id: (o.bbox[2]-o.bbox[0]+1 if axis == 'height' else o.bbox[3]-o.bbox[1]+1) for o in objs}
    value = (max if selector in ('tallest', 'widest') else min)(values.values())
    return [o for o in objs if values[o.id] == value]

SELECTORS = ('largest', 'smallest', 'unique_shape', 'repeated_shape', 'rare_color', 'common_color', 'tallest', 'shortest', 'widest', 'narrowest')

def render(grid, bg, objs, crop=False):
    if not objs: raise ValueError('selector chose no objects')
    cells = {cell: col for obj in objs for cell, col in obj.multicolor.items()}
    if crop:
        r0,c0 = min(r for r,c in cells), min(c for r,c in cells)
        r1,c1 = max(r for r,c in cells), max(c for r,c in cells)
    else: r0,c0,r1,c1 = 0,0,len(grid)-1,len(grid[0])-1
    return tuple(tuple(cells.get((r,c), bg) for c in range(c0,c1+1)) for r in range(r0,r1+1))

def fits(predict, pairs):
    from .budget import check as _budget_check
    _budget_check()
    event('candidate_tried')
    try:
        for x,y in pairs:
            pred = predict(x)
            if pred != y:
                event('rejected_dimensions' if len(pred)!=len(y) or len(pred[0])!=len(y[0]) else 'rejected_cells')
                return False
    except (ValueError, KeyError, IndexError) as exc:
        event('rejected_precondition:' + type(exc).__name__)
        return False
    except Exception:
        record_exception('new_candidate')
        return False
    event('exact_candidate')
    return True

def accepted(name, predict, pairs, description, bits=0, ops=1):
    return Law(name,predict,description,bits,ops) if fits(predict,pairs) else None

def clear_views(): view.cache_clear()
