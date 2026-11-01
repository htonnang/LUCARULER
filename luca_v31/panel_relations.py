"""Infer explicit operand/reference roles, align panels, then combine masks.

All roles and geometry are serialized; no task IDs or test outputs. Geometry is
shared by non-reference operands. Marker alignment excludes role markers from
occupancy. Out-of-canvas foreground is either rejected or explicitly clipped.
"""
from itertools import combinations, product
import time
from .panel_boolean import split, background, OPS, need
from .diagnostics import event

GEOMETRIES=('identity','flip_h','flip_v','rot90','rot180','rot270','transpose','anti_transpose')

def cells(q,color):return {(r,c) for r,row in enumerate(q) for c,v in enumerate(row) if v==color}
def prepare(g,p):
 panels=split(g,p);bg=background(g,p,panels);h,w=len(panels[0]),len(panels[0][0])
 excluded={bg}|set(p.get('ignore',[]))
 masks=[{(r,c) for r,row in enumerate(q) for c,v in enumerate(row) if v not in excluded} for q in panels]
 selector=p['operands']
 if selector=='all':ids=list(range(len(panels)))
 elif selector=='nonempty':ids=[i for i,m in enumerate(masks) if m]
 elif isinstance(selector,list):
  need(len(set(selector))==len(selector) and all(type(i)==int and 0<=i<len(panels) for i in selector),'operand indices');ids=selector
 else:raise ValueError('operand selector')
 need(len(ids)>=2,'at least two operand panels')
 reference=p['reference']
 if reference=='first':ref=ids[0]
 elif isinstance(reference,dict) and 'tag' in reference:
  matches=[i for i,q in enumerate(panels) if cells(q,reference['tag'])]
  need(len(matches)==1,'reference tag must identify one panel');ref=matches[0]
 elif reference=='empty':
  matches=[i for i,m in enumerate(masks) if not m]
  need(len(matches)==1,'workspace role ambiguous');ref=matches[0]
 else:raise ValueError('reference role')
 return panels,bg,masks,ids,ref,h,w

def transformed(points,h,w,kind):
 need(kind in GEOMETRIES,'geometry')
 if kind=='identity':return set(points)
 if kind=='flip_h':return {(r,w-1-c) for r,c in points}
 if kind=='flip_v':return {(h-1-r,c) for r,c in points}
 if kind=='rot180':return {(h-1-r,w-1-c) for r,c in points}
 need(h==w,'quarter rotation/transpose requires square panel')
 if kind=='rot90':return {(c,h-1-r) for r,c in points}
 if kind=='rot270':return {(w-1-c,r) for r,c in points}
 if kind=='transpose':return {(c,r) for r,c in points}
 return {(h-1-c,w-1-r) for r,c in points}

def origin(points):
 need(points,'empty alignment support');return min(r for r,c in points),min(c for r,c in points)

def aligned(g,p):
 panels,bg,masks,ids,ref,h,w=prepare(g,p);mode=p['alignment'];geom=p['geometry']
 if mode=='marker':
  marker=p['marker'];anchors=[cells(q,marker) for q in panels]
  need(len(anchors[ref])==1,'reference anchor must be unique');dest=next(iter(anchors[ref]))
 elif mode=='bbox':dest=origin(masks[ref])
 elif mode!='none':raise ValueError('alignment mode')
 out=[]
 for i in ids:
  kind='identity' if i==ref else geom;m=transformed(masks[i],h,w,kind);dr=dc=0
  if mode=='marker':
   need(len(anchors[i])==1,'operand anchor must be unique');src=next(iter(transformed(anchors[i],h,w,kind)));dr,dc=dest[0]-src[0],dest[1]-src[1]
  elif mode=='bbox':
   src=origin(m);dr,dc=dest[0]-src[0],dest[1]-src[1]
  moved={(r+dr,c+dc) for r,c in m};inside={(r,c) for r,c in moved if 0<=r<h and 0<=c<w}
  need(p['boundary'] in ('reject','clip'),'boundary policy')
  need(p['boundary']=='clip' or moved==inside,'aligned foreground leaves panel')
  out.append(sum(1<<(r*w+c) for r,c in inside))
 return out,bg,h,w

def combine(masks,op,h,w):
 need(op in OPS,'Boolean operation');both=masks[0];either=odd=0
 for m in masks:both &= m;either |= m;odd ^= m
 allbits=(1<<(h*w))-1
 return {'AND':both,'OR':either,'XOR':odd,'NAND':allbits^both,'NOR':allbits^either,'XNOR':allbits^odd}[op]

def apply(g,p):
 masks,bg,h,w=aligned(g,p);bits=combine(masks,p['operation'],h,w)
 true=p['true_color'];false=bg if p['false_color']=='input_background' else p['false_color']
 need(type(true)==type(false)==int and 0<=true<=9 and 0<=false<=9 and true!=false,'output palette')
 return tuple(tuple(true if bits & (1<<(r*w+c)) else false for c in range(w)) for r in range(h))

def palette(bitsets,targets,bgs):
 yes=set();no=set();follows=True
 for bits,y,bg in zip(bitsets,targets,bgs):
  w=len(y[0])
  for r,row in enumerate(y):
   for c,v in enumerate(row):
    if bits & (1<<(r*w+c)):yes.add(v)
    else:no.add(v);follows &= v==bg
    if len(yes)>1 or (len(no)>1 and not follows):return None
 if len(yes)!=1 or not no:return None
 true=next(iter(yes))
 if follows and all(bg!=true for bg in bgs):return true,'input_background'
 if len(no)==1 and true not in no:return true,next(iter(no))
 return None

def operators(values,targets,stop=None,max_configurations=1800):
 """Bound proposal work, including rejected configurations, inside parent budget."""
 if any(len({v for row in y for v in row})>2 for y in targets):return
 local_end=time.monotonic()+0.8;tried=0
 def exhausted():return tried>=max_configurations or time.monotonic()>=local_end or (stop is not None and stop())
 common=sorted(set.intersection(*({v for row in g for v in row} for g in values)))
 for rows,cols in ((1,2),(2,1),(1,3),(3,1),(1,4),(4,1),(2,2)):
  for rs,cs in product((0,1,2) if rows>1 else (0,),(0,1,2) if cols>1 else (0,)):
   layout=dict(rows=rows,cols=cols,row_gap=rs,col_gap=cs)
   if any(len(g)!=rows*len(y)+(rows-1)*rs or len(g[0])!=cols*len(y[0])+(cols-1)*cs for g,y in zip(values,targets)):continue
   try:panels=[split(g,layout) for g in values]
   except ValueError:continue
   event('v22_panel_relation_layout',layout)
   # Keep role-based background ahead of fixed-color hypotheses.
   seen_bg=set()
   for mode in ('shared_panel_color','corners','border',0,1,*('color:'+str(c) for c in common)):
    p=dict(layout,bg=mode)
    try:bgs=tuple(background(g,p,qs) for g,qs in zip(values,panels))
    except ValueError:continue
    if bgs in seen_bg:continue
    seen_bg.add(bgs)
    markers=[c for c in common if c not in bgs and all(sum(len(cells(q,c))==1 for q in qs)>=2 for qs in panels)]
    tags=[c for c in common if c not in bgs and all(sum(bool(cells(q,c)) for q in qs)==1 for qs in panels)]
    selectors=['all','nonempty']+[list(z) for z in combinations(range(rows*cols),2) if rows*cols>2]
    refs=['first']+[{'tag':c} for c in tags]+['empty']
    # Alignment role candidates precede unrestricted pair/geometry alternatives.
    for alignment,marker in [('marker',c) for c in markers]+[('none',None),('bbox',None)]:
     for selector,ref in product(selectors,refs):
      ignore=sorted(set(([marker] if marker is not None else [])+([ref['tag']] if isinstance(ref,dict) else [])))
      base=dict(p,operands=selector,reference=ref,alignment=alignment,ignore=ignore,marker=marker)
      for geometry in GEOMETRIES:
       for boundary in (('reject','clip') if alignment!='none' else ('reject',)):
        if exhausted():return
        event('v22_panel_relation_configuration');tried+=1;params=dict(base,geometry=geometry,boundary=boundary)
        try:prepared=[aligned(g,params) for g in values]
        except ValueError:continue
        for op in OPS:
         bits=[combine(m,op,h,w) for m,bg,h,w in prepared];colors=palette(bits,targets,bgs)
         if colors is not None:
          event('v22_panel_relation_exact_proposal');yield 'panel_relation',dict(params,operation=op,true_color=colors[0],false_color=colors[1])
