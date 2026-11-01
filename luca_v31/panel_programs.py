"""Bounded temporal and transactional multi-panel program synthesis.

All correspondence, roles and programs are recomputed from the current input.
No stored example grids, task identifiers or output dimensions at inference.
"""
from itertools import product
import time
from .adaptive_panels import extract, bg, cut
from .panel_boolean import need
from .panel_relations import GEOMETRIES, transformed
from .grid_world import extract_objects
from .diagnostics import event


def tile(q, background, rule):
    h,w=len(q),len(q[0]);out=[[background]*w for _ in range(h)]
    geometry=rule['geometry'];dr,dc=rule['shift'];boundary=rule['boundary']
    need(boundary in ('reject','clip'), 'unknown boundary policy')
    for r,row in enumerate(q):
        for c,v in enumerate(row):
            if v==background:continue
            a,b=next(iter(transformed({(r,c)},h,w,geometry)));a+=dr;b+=dc
            if not (0<=a<h and 0<=b<w):
                need(boundary=='clip','region foreground leaves canvas');continue
            out[a][b]=v
    return tuple(map(tuple,out))


def components(q,background,connectivity):
    out={}
    for o in extract_objects(q,background,connectivity,False):
        r,c,_,_=o.bbox
        key=(o.color,tuple(sorted((a-r,b-c) for a,b in o.cells)))
        # Repeated identical objects require assignment search, not arbitrary ordering.
        need(key not in out,'ambiguous repeated shape/color correspondence')
        out[key]=(r,c)
    need(out,'empty temporal panel')
    return out


def next_motion(panels,backgrounds,p):
    need(len(panels)>=2,'need two sequence observations')
    h,w=len(panels[-1]),len(panels[-1][0])
    need(all((len(q),len(q[0]))==(h,w) for q in panels),'unequal temporal panels')
    need(len(set(backgrounds))==1,'temporal background changes')
    tracks=[components(q,b,p['connectivity']) for q,b in zip(panels,backgrounds)]
    need(all(set(t)==set(tracks[0]) for t in tracks),'objects appear/disappear or change shape')
    output=[[backgrounds[-1]]*w for _ in range(h)];occupied=set()
    for key in sorted(tracks[0]):
        velocities=[(b[key][0]-a[key][0],b[key][1]-a[key][1]) for a,b in zip(tracks,tracks[1:])]
        need(len(set(velocities))==1,'nonconstant object velocity')
        dr,dc=velocities[-1];r,c=tracks[-1][key];color,shape=key
        cells={(r+a+dr,c+b+dc) for a,b in shape}
        inside={(a,b) for a,b in cells if 0<=a<h and 0<=b<w}
        need(p['boundary']=='clip' or cells==inside,'predicted object leaves canvas')
        need(not occupied&inside,'predicted objects overlap')
        occupied|=inside
        for a,b in inside:output[a][b]=color
    return tuple(map(tuple,output))


def role(i,n,policy):
    if policy=='all':return 'all'
    if policy=='first_rest':return 'first' if i==0 else 'rest'
    if policy=='last_rest':return 'last' if i==n-1 else 'rest'
    if policy=='parity':return 'even' if i%2==0 else 'odd'
    raise ValueError('region role policy')


def apply(g,p):
    panels,bounds,_=extract(g,p);backgrounds=[bg(q,p['background']) for q in panels]
    if p['kind']=='temporal':
        order=p['order'];need(order in (1,-1),'sequence order')
        panels=panels[::order];backgrounds=backgrounds[::order]
        if p['model']=='motion':return next_motion(panels,backgrounds,p)
        need(p['model']=='geometry','unknown temporal model')
        need(all(tile(a,b,p['rule'])==z for a,b,z in zip(panels,backgrounds,panels[1:])), 'sequence transition mismatch')
        return tile(panels[-1],backgrounds[-1],p['rule'])
    need(p['kind']=='regions','unknown program kind')
    offset=p['source_offset'];need(offset in (-1,0,1),'region source offset')
    out=[list(row) for row in g];writes=[]
    for i,box in enumerate(bounds):
        j=(i+offset)%len(panels);rule=p['rules'][role(i,len(panels),p['roles'])]
        result=tile(panels[j],backgrounds[j],rule);r,c,rr,cc=box
        need((len(result),len(result[0]))==(rr-r,cc-c),'source/destination region shape mismatch')
        writes.append((box,result))
    # All source reads precede writes; updates cannot cascade between regions.
    for (r,c,rr,cc),result in writes:
        for a,row in enumerate(result):out[r+a][c:cc]=row
    return tuple(map(tuple,out))


def fit_rules(pairs,backgrounds,stop):
    """Infer geometry then translation from bounding-box correspondence."""
    from .budget import check as _budget_check
    _budget_check()
    if not pairs:return []
    found=[]
    for geom in GEOMETRIES:
        if stop():break
        x,y=pairs[0];h,w=len(x),len(x[0]);b=backgrounds[0]
        points={(r,c) for r,row in enumerate(x) for c,v in enumerate(row) if v!=b}
        dest={(r,c) for r,row in enumerate(y) for c,v in enumerate(row) if v!=b}
        try:points=transformed(points,h,w,geom)
        except ValueError:continue
        shifts=[(0,0)]
        if points and dest:
            delta=(min(r for r,c in dest)-min(r for r,c in points),min(c for r,c in dest)-min(c for r,c in points))
            if delta not in shifts:shifts.append(delta)
        for shift,boundary in product(shifts,('reject','clip')):
            rule=dict(geometry=geom,shift=list(shift),boundary=boundary)
            try:ok=all(tile(x,b,rule)==y for (x,y),b in zip(pairs,backgrounds))
            except ValueError:ok=False
            if ok:found.append(rule)
    return found[:8]


def operators(values,targets,stop=None):
    event('v22_panel_program_entered');end=time.monotonic()+0.4;emitted=0
    def done():return emitted>=16 or time.monotonic()>=end or (stop is not None and stop())
    for parser,background in product(('separator1','separator0','frames'),('majority','border','corners')):
        if done():return
        base=dict(parser=parser,divider='unique',background=background)
        try:scenes=[extract(g,base) for g in values];backgrounds=[[bg(q,background) for q in s[0]] for s in scenes]
        except ValueError:continue
        event('v22_panel_program_extraction',parser)
        for order in (1,-1):
            panels=[s[0][::order] for s in scenes];bgs=[b[::order] for b in backgrounds]
            if not all(all((len(q),len(q[0]))==(len(y),len(y[0])) for q in qs) for qs,y in zip(panels,targets)):continue
            event('v22_temporal_shape')
            for connectivity,boundary in product((4,8),('reject','clip')):
                if done():return
                p=dict(base,kind='temporal',model='motion',order=order,connectivity=connectivity,boundary=boundary)
                try:ok=all(apply(g,p)==y for g,y in zip(values,targets))
                except ValueError:ok=False
                if ok:emitted+=1;event('v22_temporal_exact');yield 'panel_program',p
            transitions=[];fills=[]
            for qs,bs,y in zip(panels,bgs,targets):
                transitions.extend(zip(qs,qs[1:]+[y]));fills.extend(bs)
            for rule in fit_rules(transitions,fills,done):
                if done():return
                p=dict(base,kind='temporal',model='geometry',order=order,rule=rule)
                if all(apply(g,p)==y for g,y in zip(values,targets)):
                    emitted+=1;event('v22_temporal_exact');yield 'panel_program',p
        # Targets constrain local rules; panel bounds still come only from input.
        compatible=True;local=[]
        for g,y,(_,bounds,_) in zip(values,targets,scenes):
            if (len(g),len(g[0]))!=(len(y),len(y[0])):compatible=False;break
            region_cells={(r,c) for a,b,aa,bb in bounds for r in range(a,aa) for c in range(b,bb)}
            if any(g[r][c]!=y[r][c] for r in range(len(g)) for c in range(len(g[0])) if (r,c) not in region_cells):compatible=False;break
            local.append([cut(y,b) for b in bounds])
        if not compatible:continue
        event('v22_regions_shape')
        for offset,policy in product((0,1,-1),('all','first_rest','last_rest','parity')):
            if done():return
            groups={};fills={}
            for (qs,_,_),ys,bs in zip(scenes,local,backgrounds):
                for i,y in enumerate(ys):
                    key=role(i,len(qs),policy);j=(i+offset)%len(qs)
                    groups.setdefault(key,[]).append((qs[j],y));fills.setdefault(key,[]).append(bs[j])
            candidates={k:fit_rules(v,fills[k],done) for k,v in groups.items()}
            if not all(candidates.values()):continue
            # Bound combinatorics; preserve reject/clip alternatives.
            keys=sorted(candidates)
            for index,rules in enumerate(product(*(candidates[k][:2] for k in keys))):
                if done() or index>=4:break
                p=dict(base,kind='regions',roles=policy,source_offset=offset,rules=dict(zip(keys,rules)))
                try:ok=all(apply(g,p)==y for g,y in zip(values,targets))
                except ValueError:ok=False
                if ok:emitted+=1;event('v22_regions_exact');yield 'panel_program',p
