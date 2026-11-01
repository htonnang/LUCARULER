"""Demonstration-verified marker constructions and variable output structure.
Rules are general templates; no ARC IDs or test targets are used.
"""
from .object_support import accepted,view
from .grid_world import guess_backgrounds
from .primitives import Law

def colors(g):return {v for row in g for v in row}
def cells(g,c):return [(r,k) for r,row in enumerate(g) for k,v in enumerate(row) if v==c]
def immutable(g):return tuple(map(tuple,g))

def scale(g,feature):
 bg=guess_backgrounds(g)[0]
 n=len(colors(g)-({0} if feature=='nonzero_colors' else {bg}))
 if not 1<=n<=10 or len(g)*n>30 or len(g[0])*n>30:raise ValueError('invalid scale')
 return tuple(tuple(v for v in row for _ in range(n)) for row in g for _ in range(n))

def recolor_marker(g,conn):
 bg,objs=view(g,(0,conn,False))
 if len(objs)!=2:raise ValueError('requires one object and one singleton')
 objs=sorted(objs,key=lambda o:o.area);marker,body=objs
 if marker.area!=1 or body.area<=1:raise ValueError('ambiguous marker')
 out=[[bg]*len(g[0]) for _ in g]
 for r,c in body.cells:out[r][c]=marker.color
 return immutable(out)

def stamp(g,bg,marker,conn,keep_source=True):
 _,objects=view(g,(0,conn,True))
 # Require chosen background to agree with this component representation.
 if guess_backgrounds(g)[0]!=bg:raise ValueError('background mismatch')
 templates=[o for o in objects if o.area>1 and sum(g[r][c]==marker for r,c in o.cells)==1]
 if len(templates)!=1:raise ValueError('ambiguous template')
 template=templates[0];anchors=cells(g,marker)
 if len(anchors)<2:raise ValueError('no destination markers')
 if any(g[r][c]!=marker and (r,c) not in template.cells for r,row in enumerate(g) for c,v in enumerate(row) if v!=bg):raise ValueError('unexplained foreground')
 ar,ac=next((r,c) for r,c in template.cells if g[r][c]==marker)
 pattern=[(r-ar,c-ac,g[r][c]) for r,c in template.cells if g[r][c]!=marker]
 out=[[bg]*len(g[0]) for _ in g]
 if keep_source:out[ar][ac]=marker
 for r,c in anchors:
  for dr,dc,col in pattern:
   rr,cc=r+dr,c+dc
   if not(0<=rr<len(g) and 0<=cc<len(g[0])):raise ValueError('stamp outside grid')
   if out[rr][cc] not in (bg,col):raise ValueError('stamp collision')
   out[rr][cc]=col
 return immutable(out)

def dilate(g,bg,src,dst,radius,metric):
 if colors(g)-{bg,src}:raise ValueError('unexpected colour')
 anchors=cells(g,src)
 if not anchors:raise ValueError('no markers')
 out=[[bg]*len(g[0]) for _ in g]
 for r,c in anchors:
  for dr in range(-radius,radius+1):
   for dc in range(-radius,radius+1):
    if metric=='cross' and abs(dr)+abs(dc)>radius:continue
    rr,cc=r+dr,c+dc
    if 0<=rr<len(g) and 0<=cc<len(g[0]):out[rr][cc]=dst
 return immutable(out)

def cross_guides(g,bg,paint,row_edge,col_edge):
 h,w=len(g),len(g[0]);ri=0 if row_edge=='top' else h-1;ci=0 if col_edge=='left' else w-1
 rows=[r for r in range(h) if r!=ri and g[r][ci]!=bg]
 cols=[c for c in range(w) if c!=ci and g[ri][c]!=bg]
 if not rows or not cols:raise ValueError('no crossing guides')
 out=[list(row) for row in g]
 for r in rows:
  for c in cols:
   if out[r][c]!=bg:raise ValueError('intersection occupied')
   out[r][c]=paint
 return immutable(out)

def panel_nor(g,bg,paint,axis,separator):
 h,w=len(g),len(g[0]);n=w if axis=='columns' else h
 if (n-separator)%2:raise ValueError('unequal panels')
 k=(n-separator)//2
 if separator:
  line=[g[r][k] for r in range(h)] if axis=='columns' else list(g[k])
  if len(set(line))!=1:raise ValueError('not a separator')
 if axis=='columns':a=tuple(row[:k] for row in g);b=tuple(row[k+separator:] for row in g)
 else:a=g[:k];b=g[k+separator:]
 return tuple(tuple(paint if x==bg and y==bg else bg for x,y in zip(ra,rb)) for ra,rb in zip(a,b))

def fit(pairs):
 from .budget import check as _budget_check
 _budget_check()
 if len(pairs)<2:return []
 found=[]
 def add(name,fn,desc,bits=0,ops=1):
  law=accepted(name,fn,pairs,desc,bits,ops)
  if law:found.append(law)
 shape_change=any((len(x),len(x[0]))!=(len(y),len(y[0])) for x,y in pairs)
 if shape_change:
  for feature in ('nonzero_colors','foreground_colors'):
   add('adaptive_scale_'+feature,lambda g,f=feature:scale(g,f),'nearest-neighbour scale from number of '+feature)
  # Explicit output-layout alternatives, rather than one guessed split.
  for bg in range(10):
   paints=set(v for _,y in pairs for row in y for v in row)-{bg}
   if len(paints)!=1:continue
   paint=next(iter(paints))
   for axis in ('columns','rows'):
    for sep in (0,1):
     add(f'panel_nor_{axis}_{sep}_bg{bg}',lambda g,bg=bg,p=paint,a=axis,s=sep:panel_nor(g,bg,p,a,s),'Split equal panels; paint where both cells are background',bits=2,ops=3)
  return found
 for conn in (4,8):
  add('marker_recolor_remove_'+str(conn),lambda g,c=conn:recolor_marker(g,c),'Recolour large object from singleton marker, then remove marker',ops=2)
 # Constants are inferred by enumeration and accepted only with complete fit.
 input_colors=set.intersection(*(colors(x) for x,_ in pairs))
 output_colors=set.intersection(*(colors(y) for _,y in pairs))
 for bg in sorted(input_colors & output_colors):
  for marker in sorted(input_colors-{bg}):
   if all(sum(v!=bg for row in x for v in row)>1 for x,_ in pairs):
    for conn in (4,8):
     add(f'marker_stamp_{bg}_{marker}_{conn}',lambda g,bg=bg,m=marker,c=conn:stamp(g,bg,m,c),'Copy a unique anchored template to matching marker positions, remove markers',bits=2,ops=3)
   if all(colors(x)<={bg,marker} for x,_ in pairs):
    for dst in sorted(output_colors-{bg}):
     for radius in (1,2):
      for metric in ('square','cross'):
       add(f'marker_dilate_{marker}_{dst}_{radius}_{metric}',lambda g,b=bg,s=marker,d=dst,r=radius,m=metric:dilate(g,b,s,d,r,m),'Expand marker neighbourhoods',bits=3,ops=2)
  changed={y[r][c] for x,y in pairs for r in range(len(x)) for c in range(len(x[0])) if x[r][c]!=y[r][c]}
  if len(changed)==1:
   paint=next(iter(changed))
   for re in ('top','bottom'):
    for ce in ('left','right'):
     add(f'border_cross_{re}_{ce}_bg{bg}',lambda g,b=bg,p=paint,re=re,ce=ce:cross_guides(g,b,p,re,ce),'Paint intersections projected from boundary guides',bits=2,ops=2)
 return found
