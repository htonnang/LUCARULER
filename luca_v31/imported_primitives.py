"""New supplemental primitives, NOT recovered third-party code."""
from .object_support import accepted
from .grid_world import guess_backgrounds, grid_shape
MISSING=False

def compress_background(grid):
    bg=guess_backgrounds(grid)[0]
    rows=[r for r,row in enumerate(grid) if any(v!=bg for v in row)]
    cols=[c for c in range(len(grid[0])) if any(row[c]!=bg for row in grid)]
    if not rows or not cols:raise ValueError('no content')
    return tuple(tuple(grid[r][c] for c in cols) for r in rows)

def minimal_tile(grid):
    h,w=grid_shape(grid)
    ph=next(p for p in range(1,h+1) if h%p==0 and all(grid[r]==grid[r%p] for r in range(h)))
    pw=next(p for p in range(1,w+1) if w%p==0 and all(row[c]==row[c%p] for row in grid for c in range(w)))
    return tuple(row[:pw] for row in grid[:ph])

def fit_all_imported_traced(pairs):
    from .budget import check as _budget_check
    _budget_check()
    laws=[]
    for name,fn in [('compress_empty_lines',compress_background),('minimal_repeated_tile',minimal_tile)]:
        law=accepted(name,fn,pairs,name)
        if law:laws.append((law,fit_all_imported))
    x,y=pairs[0];h,w=grid_shape(x);oh,ow=grid_shape(y)
    if oh and ow and h%oh==0 and w%ow==0 and (h>oh or w>ow):
        kh,kw=h//oh,w//ow
        def downscale(g):
            h,w=grid_shape(g)
            if h%kh or w%kw:raise ValueError('incompatible scale')
            out=[]
            for r in range(0,h,kh):
                row=[]
                for c in range(0,w,kw):
                    block={g[rr][cc] for rr in range(r,r+kh) for cc in range(c,c+kw)}
                    if len(block)!=1:raise ValueError('nonuniform block')
                    row.append(block.pop())
                out.append(tuple(row))
            return tuple(out)
        law=accepted(f'downscale_{kh}_{kw}',downscale,pairs,f'uniform block reduction {kh}x{kw}',bits=2)
        if law:laws.append((law,fit_all_imported))
    return laws

def fit_all_imported(pairs):return [law for law,_ in fit_all_imported_traced(pairs)]
