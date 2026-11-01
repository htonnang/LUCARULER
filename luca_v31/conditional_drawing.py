"""Bounded, demonstration-fitted drawing and marker correspondence programs.

Inputs alone determine execution. Output examples are used only by fit().
Every admitted program reproduces every demonstration exactly; existing exact
LUCA programs retain precedence. No task identifiers or stored answers.
"""
from collections import Counter
from itertools import product
import json
from .primitives import Law
from .diagnostics import event

SCHEMA='luca-conditional-drawing-1'
def need(ok,msg):
    if not ok:raise ValueError(msg)
def freeze(g):return tuple(tuple(r) for r in g)
def dims(g):return len(g),len(g[0])
def cells(g,color):return [(r,c) for r,row in enumerate(g) for c,v in enumerate(row) if v==color]
def background(g):
    counts=Counter(v for row in g for v in row);m=max(counts.values())
    colors=[c for c,n in counts.items() if n==m];need(len(colors)==1,'ambiguous background');return colors[0]
def oriented(g,axis):return freeze(zip(*g)) if axis=='horizontal' else freeze(g)

def sequence(g,p):
    bg=background(g);points=cells(g,p['source']);need(len(points)>=3,'three sequence points required')
    # Ordered source points establish a nonzero integer vector; orientation is
    # a fitted policy, not selected by consulting the prediction target.
    points=sorted(points,reverse=p['reverse']);dr=points[1][0]-points[0][0];dc=points[1][1]-points[0][1]
    need((dr,dc)!=(0,0),'zero step')
    need(all((b[0]-a[0],b[1]-a[1])==(dr,dc) for a,b in zip(points,points[1:])),'irregular spacing')
    need(p['source']!=bg and p['draw']!=bg,'background role')
    out=[list(row) for row in g];h,w=dims(g);r,c=points[-1];changed=0
    while True:
        r+=dr;c+=dc
        if not (0<=r<h and 0<=c<w):break
        if out[r][c]!=bg:
            need(p['stop']=='obstacle','occupied continuation');break
        out[r][c]=p['draw'];changed+=1
    need(changed>0,'no continuation');return freeze(out)

def count_bar(g,p):
    q=oriented(g,p['axis']);bg=background(q);h,w=dims(q)
    cols=[]
    for c in range(w):
        pts=[r for r in range(h) if q[r][c]!=bg]
        if not pts:continue
        need(pts==list(range(min(pts),h)),'bars must share bottom baseline')
        need(len({q[r][c] for r in pts})==1,'monochrome bar required');cols.append(c)
    need(len(cols)>=2,'multiple bars required')
    gap=cols[1]-cols[0];need(gap>0 and all(b-a==gap for a,b in zip(cols,cols[1:])),'irregular bar spacing')
    counts=Counter(v for row in q for v in row)
    length=counts[p['positive']]-counts[p['negative']] if p['negative'] is not None else counts[p['positive']]
    need(0<length<=h,'bar length outside canvas')
    c=cols[-1]+gap;need(c<w,'no room for summary bar');need(p['draw']!=bg,'draw background')
    out=[list(row) for row in q]
    for r in range(h-length,h):need(out[r][c]==bg,'occupied summary bar');out[r][c]=p['draw']
    return oriented(out,p['axis'])

def panel_mode(g,p):
    q=oriented(g,p['axis']);h,w=dims(q)
    separators=[i for i,row in enumerate(q) if all(v==p['separator'] for v in row)]
    need(len(separators)==1,'unique separator row required');s=separators[0]
    need(0<s<h-1 and w%2==1,'nonempty panels and unique center')
    src=q[:s] if p['source_side']=='before' else q[s+1:]
    target=q[s+1:] if p['source_side']=='before' else q[:s]
    palette={v for row in target for v in row};need(len(palette)==1,'target must be uniform')
    counts=Counter(v for row in src for v in row);m=max(counts.values());modes=[v for v,n in counts.items() if n==m]
    need(len(modes)==1,'ambiguous mode');color=modes[0];need(color not in palette,'no edit')
    r=h-1 if p['source_side']=='before' else 0
    out=[list(row) for row in q];out[r][w//2]=color;return oriented(out,p['axis'])

def project_markers(g,p):
    if p['axis']=='auto':
        outputs=set()
        for axis in ('vertical','horizontal'):
            try:outputs.add(project_markers(g,dict(p,axis=axis)))
            except ValueError:pass
        need(len(outputs)==1,'ambiguous or absent line orientation');return next(iter(outputs))
    q=oriented(g,p['axis']);bg=background(q);h,w=dims(q)
    lines={}
    for c in range(w):
        vals={q[r][c] for r in range(h)}
        if len(vals)==1 and q[0][c]!=bg:
            color=q[0][c];need(color not in lines,'ambiguous same-color reference lines');lines[color]=c
    need(lines,'reference lines absent')
    line_cols=set(lines.values());markers=[(r,c,v) for r,row in enumerate(q) for c,v in enumerate(row) if v!=bg and c not in line_cols]
    need(markers,'markers absent')
    # Restrict to isolated markers, never silently erase larger objects.
    points={(r,c) for r,c,v in markers}
    need(all(not any((r+dr,c+dc) in points for dr,dc in ((1,0),(-1,0),(0,1),(0,-1))) for r,c in points),'non-singleton marker')
    out=[list(row) for row in q]
    for r,c,v in markers:out[r][c]=bg
    destinations={}
    for r,c,color in markers:
        if color not in lines:
            need(p['unmatched']=='delete','unmatched color');continue
        ref=lines[color];dest=ref+(-1 if c<ref else 1)
        need(0<=dest<w and dest not in line_cols,'invalid projection target')
        key=(r,dest);need(key not in destinations or destinations[key]==color,'conflicting projections')
        destinations[key]=color
    for (r,c),color in destinations.items():need(out[r][c]==bg,'occupied destination');out[r][c]=color
    return oriented(out,p['axis'])

OPS={'sequence':sequence,'count_bar':count_bar,'panel_mode':panel_mode,'project_markers':project_markers}
def apply(g,p):
    need(p.get('kind') in OPS,'unknown drawing kind');return OPS[p['kind']](g,p)
def replay(g,program):
    need(program.get('schema')==SCHEMA,'unknown schema');return apply(g,program['params'])

def fit(pairs):
    from .budget import check as _budget_check
    _budget_check()
    if len(pairs)<2 or any(dims(x)!=dims(y) for x,y in pairs):return []
    changed_colors={y[r][c] for x,y in pairs for r in range(len(x)) for c in range(len(x[0])) if x[r][c]!=y[r][c]}
    if not changed_colors:return []
    common=set.intersection(*[{v for row in x for v in row} for x,y in pairs]);union=set.union(*[{v for row in x for v in row} for x,y in pairs])
    candidates=[]
    for unmatched in ('reject','delete'):candidates.append(dict(kind='project_markers',axis='auto',unmatched=unmatched))
    for axis in ('vertical','horizontal'):
        for unmatched in ('reject','delete'):candidates.append(dict(kind='project_markers',axis=axis,unmatched=unmatched))
        for sep,side in product(sorted(common),('before','after')):candidates.append(dict(kind='panel_mode',axis=axis,separator=sep,source_side=side))
        for pos,neg,color in product(sorted(common),[None]+sorted(union),sorted(changed_colors)):
            if pos!=neg:candidates.append(dict(kind='count_bar',axis=axis,positive=pos,negative=neg,draw=color))
    for src,color,rev,stop in product(sorted(common),sorted(changed_colors),(False,True),('reject','obstacle')):
        candidates.append(dict(kind='sequence',source=src,draw=color,reverse=rev,stop=stop))
    found=[]
    for p in candidates:
        event('v23_drawing_candidate')
        try:
            if not all(apply(x,p)==y for x,y in pairs):continue
        except (ValueError,IndexError,KeyError):continue
        spec=json.loads(json.dumps(p));program={'schema':SCHEMA,'params':spec}
        law=Law('v23_'+p['kind'],lambda g,p=spec:apply(g,p),json.dumps(program,sort_keys=True),constant_bits=len(p),op_count=3)
        law.program_spec=program;law.origin='v23_conditional_drawing';found.append(law);event('v23_drawing_exact_fit')
    return found

def choose(pairs,test_input,prior):
    if prior:return prior
    event('v23_drawing_entered');laws=fit(pairs);result=[];seen=set()
    for law in laws:
        try:pred=law.predict(test_input)
        except (ValueError,IndexError,KeyError):event('v23_drawing_test_precondition');continue
        if pred in seen:continue
        seen.add(pred);result.append(law)
    return result
