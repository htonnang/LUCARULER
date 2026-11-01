"""Input-derived reference roles with color-independent relationships.
Each operator performs one scene operation; compositions are searched elsewhere.
"""
from .backgrounds import resolve
from .grid_world import extract_objects

def need(ok,why):
 if not ok:raise ValueError(why)
def perceive(g,p):
 bg=resolve(g,p['bg']);objects=extract_objects(g,bg,p.get('conn',4),True)
 refs=[o for o in objects if o.area==1];need(len(refs)==1,'unique singleton reference required')
 ref=refs[0];bodies=[o for o in objects if o is not ref];need(bodies,'no target objects')
 return bg,ref,bodies

def choose(bodies,ref,relation):
 mr,mc=next(iter(ref.cells))
 if relation=='only':need(len(bodies)==1,'unique body required');return bodies[0]
 if relation in ('nearest','farthest'):
  scores=[min(abs(r-mr)+abs(c-mc) for r,c in o.cells) for o in bodies]
 elif relation in ('largest','smallest'):scores=[o.area for o in bodies]
 elif relation in ('row_aligned','column_aligned'):
  matched=[o for o in bodies if (o.bbox[0]<=mr<=o.bbox[2] if relation=='row_aligned' else o.bbox[1]<=mc<=o.bbox[3])]
  need(len(matched)==1,'unique aligned target required');return matched[0]
 else:raise ValueError('unknown relation')
 value=(max if relation in ('farthest','largest') else min)(scores)
 need(scores.count(value)==1,'reference relation tied');return bodies[scores.index(value)]

def apply(g,op,p):
 bg,ref,bodies=perceive(g,p);mr,mc=next(iter(ref.cells));color=g[mr][mc]
 if op=='role_select':
  target=choose(bodies,ref,p['relation']);cells=set(target.cells)
  if p['keep_reference']:cells|=ref.cells
  return tuple(tuple(v if (r,c) in cells else bg for c,v in enumerate(row)) for r,row in enumerate(g))
 if op=='role_recolor':
  cells=set().union(*(o.cells for o in bodies))
  return tuple(tuple(color if (r,c) in cells else (bg if (r,c) in ref.cells and not p['keep_reference'] else v) for c,v in enumerate(row)) for r,row in enumerate(g))
 if op=='role_place':
  target=choose(bodies,ref,p['relation']);r0,c0,r1,c1=target.bbox
  anchor=p['anchor']
  if anchor=='top_left':dr,dc=mr-r0,mc-c0
  elif anchor=='bottom_right':dr,dc=mr-r1,mc-c1
  elif anchor=='center':
   need((r0+r1)%2==0 and (c0+c1)%2==0,'nonintegral center');dr,dc=mr-(r0+r1)//2,mc-(c0+c1)//2
  else:raise ValueError('anchor')
  out=[list(row) for row in g];out[mr][mc]=bg
  for r,c in target.cells:out[r][c]=bg
  for (r,c),v in target.multicolor.items():
   rr,cc=r+dr,c+dc
   need(0<=rr<len(g) and 0<=cc<len(g[0]),'placement outside canvas')
   need(out[rr][cc]==bg,'placement collision');out[rr][cc]=v
  return tuple(tuple(row) for row in out)
 raise ValueError('unknown role operator')


def operators(values,focused=False):
 for bg in ((0,) if focused else (0,'color:0',1)):
  for conn in ((4,) if focused else (4,8)):
   p=dict(bg=bg,conn=conn)
   try:
    scenes=[perceive(g,p) for g in values]
   except (ValueError,IndexError):continue
   for relation in ('nearest','farthest','largest','smallest','row_aligned','column_aligned'):
    try:
     for _,ref,bodies in scenes:choose(bodies,ref,relation)
    except ValueError:continue
    for keep in (True,False):yield 'role_select',dict(p,relation=relation,keep_reference=keep)
   for keep in (False,True):yield 'role_recolor',dict(p,keep_reference=keep)
   for relation in ('only','largest'):
    for anchor in ('top_left','bottom_right','center'):yield 'role_place',dict(p,relation=relation,anchor=anchor)
