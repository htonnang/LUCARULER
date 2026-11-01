"""V11: explicit backgrounds, enclosed regions, frames and patterned panels.

Additional hypotheses run after V10 fails to produce an executable exact-fit law.
No task identifiers or test-output data enter these functions.
"""
from collections import Counter
from contextvars import ContextVar
from dataclasses import asdict
import json
import time
from .backgrounds import resolve, policies
from .grid_world import extract_holes, extract_objects
from .panel_composition import Layout, select_panels
from .primitives import Law, GEOM_OPS
from .relational_programs import fit as fit_objects, immutable, shape
from .diagnostics import event

_mode=ContextVar('v11_extension_mode',default='full')

def configure(mode='full'):
    if mode not in ('full','background','off'):raise ValueError('invalid extension mode')
    return _mode.set(mode)

def reset(token):_mode.reset(token)


def region_paint(g,spec):
    bg=resolve(g,spec['background']);regions=extract_holes(g,bg)
    if not regions:return g
    selected=[]
    for obj in regions:
        r0,c0,r1,c1=obj.bbox;h,w=r1-r0+1,c1-c0+1
        rectangular=obj.area==h*w
        predicate=spec['predicate']
        yes={'all':True,'rectangle':rectangular,'square':rectangular and h==w,
             'non_square':not(rectangular and h==w),'largest':obj.area==max(o.area for o in regions),
             'smallest':obj.area==min(o.area for o in regions)}[predicate]
        if yes:selected.append(obj)
    # Empty selections are legitimate on some demonstrations (no square holes).
    out=[list(row) for row in g]
    for obj in selected:
        for r,c in obj.cells:out[r][c]=spec['paint']
    return immutable(out)


def frames(g,bg):
    found=[]
    for obj in extract_objects(g,bg,4,False):
        r0,c0,r1,c1=obj.bbox
        if r1-r0<2 or c1-c0<2:continue
        border={(r,c) for r in range(r0,r1+1) for c in range(c0,c1+1) if r in (r0,r1) or c in (c0,c1)}
        if all(g[r][c]==obj.color for r,c in border):found.append((obj.bbox,obj.color))
    return found


def frame_transform(g,spec):
    bg=resolve(g,spec['background']);boxes=frames(g,bg)
    if len(boxes)!=1:raise ValueError('unique rectangular frame required')
    (r0,c0,r1,c1),frame_color=boxes[0]
    points=[(r,c) for r in range(r0+1,r1) for c in range(c0+1,c1) if g[r][c] not in (bg,frame_color)]
    if not points:raise ValueError('empty frame payload')
    if spec['family']=='frame_count':
        colors={g[r][c] for r,c in points}
        if len(colors)!=1:raise ValueError('count requires one payload colour')
        h,w=spec['output_shape'];count=len(points)
        if not(1<=h<=30 and 1<=w<=30 and count<=h*w):raise ValueError('count exceeds output')
        color=next(iter(colors))
        return tuple(tuple(color if r*w+c<count else bg for c in range(w)) for r in range(h))
    pr0,pc0=min(r for r,c in points),min(c for r,c in points)
    pr1,pc1=max(r for r,c in points),max(c for r,c in points)
    payload=tuple(tuple(g[r][c] for c in range(pc0,pc1+1)) for r in range(pr0,pr1+1))
    payload=GEOM_OPS[spec.get('geometry','identity')](payload)
    ih,iw=r1-r0-1,c1-c0-1;ph,pw=shape(payload)
    if ih%ph or iw%pw:raise ValueError('payload does not scale integrally')
    sy,sx=ih//ph,iw//pw
    if min(sy,sx)<1:raise ValueError('payload larger than frame')
    out=[list(row) for row in g]
    for r in range(ih):
        for c in range(iw):out[r+r0+1][c+c0+1]=payload[r//sy][c//sx]
    if spec['layout']=='crop':out=[row[c0:c1+1] for row in out[r0:r1+1]]
    return immutable(out)


def panel_table(g,spec):
    bg=resolve(g,spec['background']);layout=Layout(**spec['layout'])
    panels=select_panels(g,layout);h,w=shape(panels[0].cells)
    table=spec['table'];out=[]
    for r in range(h):
        row=[]
        for c in range(w):
            key=''.join('1' if p.cells[r][c]!=bg else '0' for p in panels)
            if key not in table:raise ValueError('unobserved panel occupancy pattern')
            value=table[key];row.append(bg if value=='background' else value)
        out.append(row)
    return immutable(out)


def template_repeat(g,spec):
    bg=resolve(g,spec['background']);h,w=shape(g)
    if h*h>30 or w*w>30:raise ValueError('expanded grid exceeds ARC size')
    role=spec['selector']
    if role=='foreground':chosen={v for row in g for v in row if v!=bg}
    elif isinstance(role,int):chosen={role}
    else:
        counts=Counter(v for row in g for v in row if v!=bg)
        if not counts:raise ValueError('empty foreground')
        target=(max if role=='most_frequent' else min)(counts.values())
        chosen={v for v,n in counts.items() if n==target}
        if len(chosen)!=1:raise ValueError('ambiguous foreground frequency')
    return tuple(tuple(g[r%h][c%w] if (g[r//h][c//w] in chosen)!=spec['invert'] else bg for c in range(w*w)) for r in range(h*h))


def execute(g,spec):
    family=spec['family']
    if family=='region_paint':return region_paint(g,spec)
    if family in ('frame_rescale','frame_count'):return frame_transform(g,spec)
    if family=='panel_table':return panel_table(g,spec)
    if family=='template_repeat':return template_repeat(g,spec)
    raise ValueError('unknown V11 program family')


def as_law(spec):
    spec=json.loads(json.dumps(spec));family=spec['family']
    law=Law('v11_'+family,lambda g:execute(g,spec),json.dumps(spec,sort_keys=True),
            constant_bits=1 if family=='frame_count' else 0,op_count=3)
    law.program_spec=spec;law.origin='v11_representation'
    return law


def infer_table(panels_by_example,pairs,bgs):
    table={}
    for panels,(_,y),bg in zip(panels_by_example,pairs,bgs):
        for r,row in enumerate(y):
            for c,v in enumerate(row):
                key=''.join('1' if p.cells[r][c]!=bg else '0' for p in panels)
                value='background' if v==bg else v
                if key in table and table[key]!=value:return None
                table[key]=value
    return table


def fit(pairs,mode=None,seconds=3.0):
    from .budget import check as _budget_check
    _budget_check()
    mode=mode or _mode.get()
    if mode=='off' or len(pairs)<2:return []
    start=time.monotonic();found=[]
    configs=[(m,conn,multi) for m in policies(pairs,novel_only=True) for conn in (4,8) for multi in (True,False)]
    if configs:
        event('v11_background_configurations',len(configs))
        found=fit_objects(pairs,seconds=min(1.0,seconds),configs_override=configs)
        for law in found:
            law.name='v11_background+'+law.name
            law.origin='v11_background'
        if found:return found
    if mode=='background':return []
    bgs=list(policies(pairs));same=all(shape(x)==shape(y) for x,y in pairs)
    # A conditional action may be inactive in some demonstrations.
    # Its paint colour need only occur in an active example, not every output.
    paints=sorted({v for x,y in pairs for row in y for v in row})
    budget_noted=False
    def expired():
        nonlocal budget_noted
        elapsed=time.monotonic()-start>=seconds
        if elapsed and not budget_noted:
            event('v11_search_budget_exhausted');budget_noted=True
        return elapsed
    def add(spec):
        try:
            event('v11_candidate_tried',spec['family'])
            if all(execute(x,spec)==y for x,y in pairs):
                found.append(as_law(spec));event('v11_exact_candidate',spec['family'])
        except (ValueError,KeyError,IndexError):event('v11_precondition_rejected')
    if same:
        for bg in bgs:
            for predicate in ('square','rectangle','all','non_square','largest','smallest'):
                for paint in paints:add(dict(family='region_paint',background=bg,predicate=predicate,paint=paint))
                if expired():return found
        if found:return found
    for bg in bgs:
        if expired():break
        # Frame decomposition supports both cropped and in-place payload expansion.
        for layout in (('original',) if same else ('crop',)):
            for geom in GEOM_OPS:
                add(dict(family='frame_rescale',background=bg,layout=layout,geometry=geom))
        if len({shape(y) for x,y in pairs})==1 and not same:
            add(dict(family='frame_count',background=bg,output_shape=shape(pairs[0][1])))
    if found or expired():return found
    # Equal panels are a representation; infer a conditional occupancy table.
    layouts=[Layout(rows,cols,sr,sc) for rows,cols in ((1,2),(2,1),(1,3),(3,1),(1,4),(4,1),(2,2))
             for sr in ((0,1) if rows>1 else (0,)) for sc in ((0,1) if cols>1 else (0,))]
    for layout in layouts:
        if expired():break
        try:
            if any(layout.shape(x)!=shape(y) for x,y in pairs):continue
            extracted=[select_panels(x,layout) for x,y in pairs]
        except ValueError:continue
        for bg in bgs:
            values=[resolve(x,bg) for x,y in pairs]
            table=infer_table(extracted,pairs,values)
            if table is None:continue
            # LOO re-estimates the entire table, with structure fixed.
            valid=True
            if len(pairs)>=3:
                for i in range(len(pairs)):
                    other=infer_table(extracted[:i]+extracted[i+1:],pairs[:i]+pairs[i+1:],values[:i]+values[i+1:])
                    if other is None or any(other.get(k)!=v for k,v in table.items()):valid=False;break
            if valid:add(dict(family='panel_table',background=bg,layout=asdict(layout),table=table))
        if found:return found
    if all(shape(y)==(len(x)**2,len(x[0])**2) for x,y in pairs):
        common=sorted(set.intersection(*({v for row in x for v in row} for x,y in pairs)))
        for bg in bgs:
            for selector in ['foreground','most_frequent','rarest']+common:
                for invert in (False,True):
                    if expired():return found
                    add(dict(family='template_repeat',background=bg,selector=selector,invert=invert))
    expired()
    return found


def upgrade_result(pairs,test_input,result):
    from .budget import check as _budget_check
    _budget_check()
    if not result.failure_category or _mode.get()=='off':return result
    event('v11_extension_entered')
    candidates=fit(pairs)
    preds=[];laws=[]
    for law in candidates:
        try:pred=law.predict(test_input)
        except (ValueError,KeyError,IndexError):
            event('v11_test_precondition_rejected');continue
        if pred in preds:continue
        preds.append(pred);laws.append(law)
        if len(preds)==2:break
    if not preds:return result
    return type(result)(preds,laws,'V11 exact demonstration fit in an alternate representation.',failure_category=None)
