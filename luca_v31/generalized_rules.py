"""Demonstration-fitted object properties and explicit selection conditions."""
from collections import Counter
from .grid_world import extract_objects,guess_backgrounds
from .primitives import Law
from .diagnostics import event

def need(ok,message):
 if not ok:raise ValueError(message)
def scene(g,p):
 b=guess_backgrounds(g)[0] if p['bg']=='auto' else p['bg'];return b,extract_objects(g,b,p.get('conn',8),False)
def feature(o,objects,key):
 r,c,rr,cc=o.bbox;h,w=rr-r+1,cc-c+1;shape=o.shape()
 if key=='area':return str(o.area)
 if key=='aspect':return str((h>w)-(h<w))
 if key=='diagonal':
  pts=list(shape);n=len(pts);cov=n*sum(a*b for a,b in pts)-sum(a for a,b in pts)*sum(b for a,b in pts)
  return str((cov>0)-(cov<0))
 if key=='symmetry':return str(tuple(shape==s for s in ({(a,w-1-b) for a,b in shape},{(h-1-a,b) for a,b in shape},{(b,a) for a,b in shape})))
 if key=='area_extrema':
  areas=[v.area for v in objects];return 'min' if o.area==min(areas) else 'max' if o.area==max(areas) else 'middle'
 if key=='shape_frequency':return str(sum(v.shape()==shape for v in objects))
 raise ValueError('feature')

def panel_cycle(g,p):
 h,w=len(g),len(g[0]);need(h%2==1 and w%2==1 and h>=5 and w>=5,'2x2 panel dimensions')
 a,c=h//2,w//2;divider=g[a][c]
 need(all(v==divider for v in g[a]) and all(row[c]==divider for row in g),'missing central dividers')
 panels=[tuple(tuple(row[j:j+c]) for row in g[i:i+a]) for i,j in ((0,0),(0,c+1),(a+1,0),(a+1,c+1))]
 missing=[i for i,q in enumerate(panels) if len({v for row in q for v in row})==1 and q[0][0]!=divider]
 need(len(missing)==1,'placeholder must be unique solid non-divider panel');target=missing[0]
 def turn(q):
  kind=p['geometry']
  if kind=='rot90':return tuple(zip(*q[::-1]))
  if kind=='rot270':return tuple(zip(*q))[::-1]
  if kind=='rot180':return tuple(row[::-1] for row in q[::-1])
  if kind=='flip_h':return tuple(row[::-1] for row in q)
  if kind=='flip_v':return q[::-1]
  if kind=='transpose':return tuple(zip(*q))
  raise ValueError('cycle geometry')
 start=next(i for i in range(4) if i!=target);tile=panels[start];predictions={start:tile}
 for step in range(1,4):
  tile=turn(tile);predictions[(start+step)%4]=tile
 need(all(predictions[i]==q for i,q in enumerate(panels) if i!=target),'observed panels violate cycle')
 return predictions[target]

def apply(g,p):
 if p['kind']=='panel_cycle':return panel_cycle(g,p)
 b,objs=scene(g,p);kind=p['kind'];out=[list(row) for row in g]
 if kind=='recolor_property':
  for o in objs:
   value=feature(o,objs,p['feature']);need(value in p['mapping'],'unseen object feature value')
   color=p['mapping'][value];color=b if color=='background' else color
   for r,c in o.cells:out[r][c]=color
 elif kind=='select_object':
  need(objs,'no objects');mode=p['selector']
  if mode=='largest':score=lambda o:o.area
  elif mode=='most_frequent_shape':score=lambda o:sum(z.shape()==o.shape() for z in objs)
  elif mode=='least_frequent_shape':score=lambda o:-sum(z.shape()==o.shape() for z in objs)
  else:raise ValueError('selector')
  m=max(map(score,objs));chosen=[o for o in objs if score(o)==m]
  crops=[]
  for o in chosen:
   r,c,rr,cc=o.bbox;crops.append(tuple(tuple(g[a][z] if (a,z) in o.cells else b for z in range(c,cc+1)) for a in range(r,rr+1)))
  need(all(x==crops[0] for x in crops),'tied selection gives different outputs');return crops[0]
 elif kind=='marker_recolor_delete':
  bodies=[o for o in objs if o.area>1];markers=[o for o in objs if o.area==1];need(bodies and markers,'body/marker missing')
  need(len({o.color for o in bodies})==1,'multiple body color roles');bodycolor=bodies[0].color
  markers=[o for o in markers if o.color!=bodycolor];need(markers,'no foreign-color markers')
  for m in markers:
   for r,c in m.cells:out[r][c]=b
  for o in bodies:
   adjacent=[m for m in markers if any(abs(r-a)+abs(c-z)==1 for r,c in o.cells for a,z in m.cells)]
   need(len(adjacent)<=1,'ambiguous adjacent marker')
   if adjacent:
    for r,c in o.cells:out[r][c]=adjacent[0].color
 elif kind=='keep_middle_column':
  need(len(g[0])%2==1,'middle column not unique');mid=len(g[0])//2
  out=[[v if c==mid else b for c,v in enumerate(row)] for row in g]
 else:raise ValueError('rule kind')
 return tuple(map(tuple,out))

def fitted(pairs):
 from .budget import check as _budget_check
 _budget_check()
 found=[]
 def accept(p):
  try:ok=all(apply(x,p)==y for x,y in pairs)
  except ValueError:ok=False
  if ok:
   import json
   spec=json.loads(json.dumps(p));law=Law('v22_general_'+p['kind']+'_'+p.get('feature',p.get('selector','')),lambda g,p=spec:apply(g,p),json.dumps(spec,sort_keys=True),constant_bits=len(p.get('mapping',{})),op_count=2)
   law.generalized_spec=spec;law.program_spec={'schema':'luca-generalized-1','params':spec};law.origin='v22_generalized_rules';found.append(law)
 for geometry in ('rot90','rot270','rot180','flip_h','flip_v','transpose'):accept(dict(kind='panel_cycle',geometry=geometry))
 for bg in ('auto',0):
  for conn in (4,8):
   base=dict(bg=bg,conn=conn)
   for selector in ('most_frequent_shape','least_frequent_shape','largest'):accept(dict(base,kind='select_object',selector=selector))
   accept(dict(base,kind='marker_recolor_delete'))
   for key in ('diagonal','aspect','area_extrema','symmetry','shape_frequency','area'):
    mapping={};valid=True
    for x,y in pairs:
     if (len(x),len(x[0]))!=(len(y),len(y[0])):valid=False;break
     b,objects=scene(x,base)
     if not objects:valid=False;break
     for o in objects:
      outputs={y[r][c] for r,c in o.cells}
      if len(outputs)!=1:valid=False;break
      color=next(iter(outputs));color='background' if color==b else color;value=feature(o,objects,key)
      if value in mapping and mapping[value]!=color:valid=False;break
      mapping[value]=color
     if not valid:break
    if valid and len(mapping)>=2:accept(dict(base,kind='recolor_property',feature=key,mapping=mapping))
  accept(dict(bg=bg,kind='keep_middle_column'))
 return found

def choose(pairs,test_input,prior):
 """Prefer relational alternatives to memorized shape rules, preserving other fits.
 A changed choice must fit every demonstration and apply to the novel input.
 """
 # Explicit structural rules retain precedence over newly inferred properties.
 if prior and not ('recolor_by_shape' in prior[0].name or prior[0].name in ('crop_largest_object','crop_smallest_object')):
  return prior
 laws=fitted(pairs)
 if not laws:return prior
 # These simple properties have an explicit domain; literal area tables remain
 # fallback alternatives, not a general replacement for all exact programs.
 preferred=[l for l in laws if l.generalized_spec['kind'] in ('marker_recolor_delete','keep_middle_column','panel_cycle') or (l.generalized_spec['kind']=='recolor_property' and l.generalized_spec['feature'] in ('diagonal','aspect','area_extrema','symmetry')) or (l.generalized_spec['kind']=='select_object' and l.generalized_spec['selector']=='most_frequent_shape')]
 # Prefer monochrome 4-connected denoising crops only in demonstrably noisy scenes.
 if any(l.name=='crop_largest_object' for l in prior):
  noisy=all((lambda obs:len(obs)>=3 and sum(o.area==1 for o in obs)>=len(obs)/2)(scene(x,dict(bg='auto',conn=4))[1]) for x,y in pairs)
  if noisy:preferred=[l for l in laws if l.generalized_spec['kind']=='select_object' and l.generalized_spec['selector']=='largest' and l.generalized_spec['conn']==4]+preferred
 candidates=preferred+prior;out=[]
 for law in candidates:
  try:law.predict(test_input)
  except ValueError:continue
  out.append(law)
 return out
