"""Demonstration-fitted relations between aligned markers.

One reusable grammar: select colour cells or cross centres, group by colour or
pool them, connect aligned points or extend through them, paint background cells.
Crossing policies are explicit; no task identifiers or answer tables.
"""
from itertools import product
from collections import defaultdict
from time import monotonic
import json
from .grid_world import guess_backgrounds
from .primitives import Law
from .budget import check
SCHEMA='luca-line-relations-1'

def _most_common(g):
 ct=defaultdict(int)
 for row in g:
  for v in row:ct[v]+=1
 return min(ct,key=lambda k:(-ct[k],k))

def execute(g,p):
 h,w=len(g),len(g[0]);bg=p['background'];groups=defaultdict(list)
 if bg=='auto':bg=_most_common(g)  # OPT188: background as a role (each grid's most frequent colour)
 for r,row in enumerate(g):
  for c,v in enumerate(row):
   if v==bg:continue
   if p['selector']=='centres':
    if not(0<r<h-1 and 0<c<w-1):continue
    arms=[g[r-1][c],g[r+1][c],g[r][c-1],g[r][c+1]]
    if len(set(arms))!=1 or arms[0] in (bg,v):continue
   source=p['source']
   if isinstance(source,int) and v!=source:continue
   groups['any' if source=='any' else v].append((r,c,v))
 if not groups:raise ValueError('no markers')
 edits={};priorities={};axes={'rows':[(0,1)],'columns':[(1,0)],'both':[(0,1),(1,0)],'diagonals':[(1,1),(1,-1)]}[p['axes']]
 for key,points in sorted(groups.items(),key=lambda x:str(x[0])):
  for dr,dc in axes:
   lines=defaultdict(list)
   for r,c,v in points:lines[r*dc-c*dr].append((r,c,v))
   for pts in lines.values():
    if p['extent']=='between' and len(pts)<2:continue
    color=p['paint'] if isinstance(p['paint'],int) else pts[0][2]
    if p['paint']=='source' and len({v for r,c,v in pts})!=1:raise ValueError('mixed source paint')
    r,c,_=min(pts);cells=[]
    if p['extent']=='full':
     while 0<=r-dr<h and 0<=c-dc<w:r-=dr;c-=dc
     while 0<=r<h and 0<=c<w:cells.append((r,c));r+=dr;c+=dc
    else:
     end=max(pts)
     while (r,c)!=(end[0],end[1]):
      if not(0<=r<h and 0<=c<w):raise ValueError('invalid line')
      cells.append((r,c));r+=dr;c+=dc
     cells.append((r,c))
    for cell in cells:
     if g[cell[0]][cell[1]]!=bg:continue
     if cell in edits and edits[cell]!=color:
      if p['crossing']=='reject':raise ValueError('conflicting lines')
      if p['crossing']=='first':continue
      if p['crossing']=='vertical' and priorities[cell]>=abs(dr):continue
      if p['crossing']=='horizontal' and priorities[cell]>=abs(dc):continue
     edits[cell]=color
     priorities[cell]=abs(dc) if p['crossing']=='horizontal' else abs(dr)
 out=[list(row) for row in g]
 for (r,c),v in edits.items():out[r][c]=v
 return tuple(map(tuple,out))

def law(p):
 spec=dict(schema=SCHEMA,params=p)
 l=Law('line_relations',lambda g:execute(g,p),json.dumps(spec,sort_keys=True),2,3)
 l.program_spec=spec;l.origin='v27_line_relations';return l

def fit(pairs,seconds=1.5,limit=1800):
 check()
 if len(pairs)<2:return []
 if any((len(x),len(x[0]))!=(len(y),len(y[0])) for x,y in pairs):return []
 backgrounds=set.intersection(*(set(guess_backgrounds(x,2)) for x,y in pairs))
 if len({_most_common(x) for x,y in pairs})>1:backgrounds=set(backgrounds)|{'auto'}
 colors=sorted(set.intersection(*({v for row in x for v in row} for x,y in pairs)))
 paint=sorted({b for x,y in pairs for ra,rb in zip(x,y) for a,b in zip(ra,rb) if a!=b})
 start=monotonic();found=[];attempts=0
 for bg in sorted(backgrounds,key=str):
  if any(a!=b and a!=(_most_common(x) if bg=='auto' else bg) for x,y in pairs for ra,rb in zip(x,y) for a,b in zip(ra,rb)):continue
  for selector,source,extent,axes,draw,crossing in product(('cells','centres'),['each','any']+[c for c in colors if c!=bg and bg!='auto'],('between','full'),('both','rows','columns','diagonals'),['source']+paint,('reject','vertical','horizontal','first','last')):
   check();attempts+=1
   if attempts>limit or monotonic()-start>seconds:return found
   p=dict(background=bg,selector=selector,source=source,extent=extent,axes=axes,paint=draw,crossing=crossing)
   try:
    if all(execute(x,p)==y for x,y in pairs):found.append(law(p))
   except ValueError:continue
   if len(found)>=24:return found
 return found

def upgrade_result(pairs,g,result):
 check()
 if not result.failure_category:return result
 preds=[];laws=[]
 for candidate in fit(pairs):
  try:out=candidate.predict(g)
  except ValueError:continue
  if out in preds:continue
  preds.append(out);laws.append(candidate)
  if len(preds)==2:break
 if not preds:return result
 return type(result)(preds,laws,'Aligned-marker relation fits every demonstration.',failure_category=None)
