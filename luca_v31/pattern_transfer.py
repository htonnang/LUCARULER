"""Demonstration-fitted pattern transfer and marker-directed extension."""
from collections import Counter
from dataclasses import replace
from .primitives import Law
from .structural_rules import grid,components
from .region_reasoning import regions,bounds

def predict(g,m):
 name,bg,*args=m;g=grid(g);h,w=len(g),len(g[0]);out=[list(row) for row in g]
 if bg==-1:
  counts=Counter(v for row in g for v in row);mx=max(counts.values());cs=[v for v,n in counts.items() if n==mx]
  if len(cs)!=1:return None
  bg=cs[0]
 pts=[(r,c) for r in range(h) for c in range(w) if g[r][c]!=bg]
 if not pts:return None
 if name=='alternating_marker_stripes':
  if len(pts)!=2:return None
  (a,b),(x,y)=pts
  if g[a][b]==g[x][y]:return None
  axis=0 if h>w else 1 if w>h else None
  if axis is None:return None
  positions=sorted((r if axis==0 else c,g[r][c]) for r,c in pts);(p,v),(q,z)=positions;step=q-p
  if step<=0:return None
  for k,pos in enumerate(range(p,h if axis==0 else w,step)):
   col=v if k%2==0 else z
   for j in range(w if axis==0 else h):out[pos if axis==0 else j][j if axis==0 else pos]=col
 elif name=='rectangle_marker_rays':
  parts=components(g,bg);boxes=[];markers=[]
  for col,ps in parts:
   a,b,x,y=bounds(ps)
   if len(ps)==1:markers.append((col,ps[0]))
   elif len(ps)==(x-a+1)*(y-b+1) and x>a and y>b:boxes.append((col,a,b,x,y))
   else:return None
  if len(boxes)!=1 or not markers:return None
  _,a,b,x,y=boxes[0];changed=False
  for col,(r,c) in markers:
   path=[]
   if a<=r<=x and (c<b or c>y):path=[(r,j) for j in range(c+1,b)] if c<b else [(r,j) for j in range(y+1,c)]
   elif b<=c<=y and (r<a or r>x):path=[(i,c) for i in range(r+1,a)] if r<a else [(i,c) for i in range(x+1,r)]
   for i,j in path:
    if out[i][j] not in (bg,col):return None
    out[i][j]=col;changed=True
  if not changed:return None
 elif name=='opposite_corner_stamps':
  a,b,x,y=bounds(pts)
  if x-a!=1 or y-b!=1 or len(pts)!=4 or len({g[r][c] for r,c in pts})!=4:return None
  for dr,dc,v in [(-2,-2,g[x][y]),(-2,2,g[x][b]),(2,-2,g[a][y]),(2,2,g[a][b])]:
   for r in range(a+dr,a+dr+2):
    for c in range(b+dc,b+dc+2):
     if 0<=r<h and 0<=c<w:out[r][c]=v
 elif name=='panel_union_template':
  separator=args[0]
  if separator==-1:
   possible=[v for v in {k for row in g for k in row}-{bg} if any(set(row)=={v} for row in g) and any(all(g[r][c]==v for r in range(h)) for c in range(w))]
   if len(possible)!=1:return None
   separator=possible[0]
  rows=[r for r,row in enumerate(g) if set(row)=={separator}];cols=[c for c in range(w) if all(g[r][c]==separator for r in range(h))]
  if not rows or not cols:return None
  rb=[-1]+rows+[h];cb=[-1]+cols+[w];rects=[(rb[i]+1,cb[j]+1,rb[i+1],cb[j+1]) for i in range(len(rb)-1) for j in range(len(cb)-1)]
  sizes={(x-a,y-b) for a,b,x,y in rects}
  if len(sizes)!=1 or any(v<=0 for v in next(iter(sizes))):return None
  shape=set()
  for a,b,x,y in rects:
   for r in range(a,x):
    for c in range(b,y):
     if g[r][c] not in (bg,separator):shape.add((r-a,c-b))
  if not shape:return None
  for a,b,x,y in rects:
   for r,c in shape:
    if out[a+r][b+c]==bg:out[a+r][b+c]=separator
 elif name=='fragment_template_complete':
  groups=regions([[v!=bg for v in row] for row in g],True);templates=[];fragments=[]
  for ps in groups:
   if len({g[r][c] for r,c in ps})>=2:templates.append(ps)
   else:fragments.append(ps)
  if not templates or not fragments:return None
  changed=False
  for ps in fragments:
   options={}
   for ts in templates:
    if len(ts)<=len(ps):continue
    for ar,ac in ps[:1]:
     for br,bc in ts:
      if g[ar][ac]!=g[br][bc]:continue
      dr,dc=ar-br,ac-bc;mapping={(r+dr,c+dc):g[r][c] for r,c in ts}
      if not all(mapping.get((r,c))==g[r][c] for r,c in ps):continue
      mapping={(r,c):v for (r,c),v in mapping.items() if 0<=r<h and 0<=c<w}
      if any(g[r][c] not in (bg,v) for (r,c),v in mapping.items()):continue
      key=tuple(sorted((r,c,v) for (r,c),v in mapping.items()));options[key]=sum(g[r][c]==v for (r,c),v in mapping.items())
   if not options:return None
   best=max(options.values());options={k for k,v in options.items() if v==best}
   if len(options)!=1:return None
   for r,c,v in next(iter(options)):
    if out[r][c] not in (bg,v):return None
    changed|=out[r][c]!=v;out[r][c]=v
  if not changed:return None
 else:return None
 return grid(out)

def candidates(pairs):
 if any(len(x)!=len(y) or len(x[0])!=len(y[0]) for x,y in pairs):return
 common=set.intersection(*[{v for row in x for v in row} for x,y in pairs])
 for bg in [-1]+sorted(common):
  for name in ['alternating_marker_stripes','rectangle_marker_rays','opposite_corner_stamps','fragment_template_complete']:yield(name,bg)
  for sep in [-1]+sorted(common-{bg}):yield('panel_union_template',bg,sep)

def fits(pairs):
 if len(pairs)<2:return []
 return [m for m in candidates(pairs) if all(predict(x,m)==grid(y) for x,y in pairs)]

def induce(pairs):
 models=fits(pairs)
 if models and len(pairs)>=3:
  names={m[0] for m in models}
  for i,(x,y) in enumerate(pairs):
   other=fits(pairs[:i]+pairs[i+1:]);names={n for n in names if {predict(x,m) for m in other if m[0]==n}=={grid(y)}}
  models=[m for m in models if m[0] in names]
 return models

def upgrade_many(train_pairs,test_inputs,results):
    pairs=list(train_pairs);models=induce(pairs)
    if not models:return results
    out=[]
    for g,result in zip(test_inputs,results):
        predictions={predict(g,m) for m in models}
        if None in predictions or len(predictions)!=1:out.append(result);continue
        prediction=next(iter(predictions));model=models[0]
        if result.predictions and grid(result.predictions[0])==prediction:out.append(result);continue
        law=Law(name='pattern_transfer:'+model[0],predict=lambda x,m=model:predict(x,m),description='Demonstration-fitted object relations with omitted-pair checks and model agreement.',constant_bits=16,op_count=6)
        law.program_spec=dict(op='pattern_transfer',parameters=model);law.origin='pattern_transfer'
        guesses=[prediction];laws=[law]
        if result.predictions:
            guesses.append(result.predictions[0])
            if result.laws:laws.append(result.laws[0])
        diagnostics=dict(result.diagnostics or {});diagnostics['pattern_transfer']=dict(models=models,exact_demonstrations=len(pairs),omitted_pair_checks=len(pairs) if len(pairs)>=3 else 0)
        out.append(replace(result,predictions=guesses,laws=laws,failure_category=None,diagnostics=diagnostics,explanation=result.explanation+' | Pattern transfer supplied attempt 1.'))
    return out
