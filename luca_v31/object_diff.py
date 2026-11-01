"""New replacement: foreground translation, duplication, alignment and anchor recolouring."""
from .object_support import CONFIGS, view, selected_objects, accepted
from .grid_world import guess_backgrounds, grid_shape
MISSING=False

def foreground(grid,bg):
    return {(r,c):v for r,row in enumerate(grid) for c,v in enumerate(row) if v!=bg}

def paint_shift(grid,cells,bg,dr,dc,duplicate=False):
    h,w=grid_shape(grid);out=[list(row) for row in grid]
    if not duplicate:
        for r,c in cells:out[r][c]=bg
    for (r,c),color in cells.items():
        rr,cc=r+dr,c+dc
        if not (0<=rr<h and 0<=cc<w):raise ValueError('outside canvas')
        if out[rr][cc] not in (bg,color):raise ValueError('incompatible overlap')
        out[rr][cc]=color
    return tuple(map(tuple,out))

def match_object_differences(x,y):
    result={'input_shape':grid_shape(x),'output_shape':grid_shape(y)}
    if grid_shape(x)!=grid_shape(y):result['effect']='shape_change';return result
    bg=guess_backgrounds(x)[0];a,b=foreground(x,bg),foreground(y,bg)
    result.update(added_cells=len(b.keys()-a.keys()),removed_cells=len(a.keys()-b.keys()),
                  recolored_cells=sum(a[k]!=b[k] for k in a.keys()&b.keys()))
    return result

def fit_all_object_diff_traced(pairs):
    from .budget import check as _budget_check
    _budget_check()
    if not pairs or any(grid_shape(x)!=grid_shape(y) for x,y in pairs):return []
    laws=[];x,y=pairs[0]
    for rank in (0,1):
        bgs=guess_backgrounds(x,k=2)
        if rank>=len(bgs):continue
        bg=bgs[rank];cells=foreground(x,bg)
        if not cells:continue
        (r,c),color=next(iter(cells.items()))
        offsets={(rr-r,cc-c) for rr,row in enumerate(y) for cc,v in enumerate(row) if v==color and (rr,cc)!=(r,c)}
        for dr,dc in sorted(offsets,key=lambda v:(abs(v[0])+abs(v[1]),v))[:60]:
            for duplicate in (False,True):
                def predict(g,rank=rank,dr=dr,dc=dc,duplicate=duplicate):
                    gs=guess_backgrounds(g,k=2)
                    if rank>=len(gs):raise ValueError('background unavailable')
                    bg=gs[rank]
                    return paint_shift(g,foreground(g,bg),bg,dr,dc,duplicate)
                law=accepted(('duplicate' if duplicate else 'translate')+f'_{dr}_{dc}_bg{rank}',predict,pairs,
                             f'foreground offset ({dr},{dc}), copy={duplicate}, background_rank={rank}',bits=2)
                if law:laws.append((law,fit_all_object_diff))
    for config in CONFIGS:
        for selector in ('largest','smallest'):
            for action in ('left','right','up','down','anchor_row','anchor_column','anchor_color'):
                def predict(g,config=config,selector=selector,action=action):
                    bg,objs=view(g,config);chosen=selected_objects(objs,selector)
                    if len(chosen)!=1:raise ValueError('ambiguous object')
                    obj=chosen[0];r0,c0,r1,c1=obj.bbox;dr=dc=0
                    if action=='left':dc=-c0
                    elif action=='right':dc=len(g[0])-1-c1
                    elif action=='up':dr=-r0
                    elif action=='down':dr=len(g)-1-r1
                    else:
                        anchors=[o for o in objs if o.id!=obj.id]
                        if len(anchors)!=1:raise ValueError('ambiguous anchor')
                        anchor=anchors[0]
                        if action=='anchor_color':
                            out=[list(row) for row in g]
                            for r,c in obj.cells:out[r][c]=anchor.color
                            return tuple(map(tuple,out))
                        if action=='anchor_row':dr=anchor.bbox[0]-r0
                        else:dc=anchor.bbox[1]-c0
                    return paint_shift(g,obj.multicolor,bg,dr,dc)
                law=accepted(f'object_{selector}_{action}_{config}',predict,pairs,
                             f'SELECT({selector}) -> {action}; view={config}',ops=2)
                if law:laws.append((law,fit_all_object_diff))
    return laws

def fit_all_object_diff(pairs):return [law for law,_ in fit_all_object_diff_traced(pairs)]
