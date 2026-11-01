"""Input-derived variable panels, role colors and bounded panel programs.

No target dimensions are used to extract panels at prediction time. Layout,
background, alignment, reference and rendering policies are serialized.
"""
from collections import Counter
from itertools import product
import time
from .panel_boolean import need,OPS
from .panel_relations import transformed,combine
from .diagnostics import event

def runs(indices):
 out=[]
 for i in sorted(indices):
  if not out or i!=out[-1][-1]+1:out.append([i])
  else:out[-1].append(i)
 return [(v[0],v[-1]+1) for v in out]
def cut(g,b):
 r,c,rr,cc=b;return tuple(tuple(row[c:cc]) for row in g[r:rr])
def separator_candidates(g,axis):
 h,w=len(g),len(g[0]);lines=[i for i in range(h if axis==0 else w) if len({g[i][j] if axis==0 else g[j][i] for j in range(w if axis==0 else h)})==1]
 colors={g[i][0] if axis==0 else g[0][i] for i in lines};out=[]
 for color in sorted(colors):
  band=[i for i in lines if (g[i][0] if axis==0 else g[0][i])==color]
  # Separator color must occur only in full bands, not as panel content.
  if sum(v==color for row in g for v in row)!=len(band)*(w if axis==0 else h):continue
  spans=runs(set(range(h if axis==0 else w))-set(band))
  if not 2<=len(spans)<=8:continue
  bounds=[(a,0,b,w) if axis==0 else (0,a,h,b) for a,b in spans]
  out.append((color,bounds))
 return out

def frame_candidates(g):
 h,w=len(g),len(g[0]);out=[]
 for color in sorted({v for row in g for v in row}):
  boxes=set()
  for r in range(h-2):
   for c,cc in runs(i for i,v in enumerate(g[r]) if v==color):
    if cc-c<3:continue
    for rr in range(r+2,h):
     if any(g[k][c]!=color or g[k][cc-1]!=color for k in range(r+1,rr+1)):break
     if all(g[rr][j]==color for j in range(c,cc)) and any(g[i][j]!=color for i in range(r+1,rr) for j in range(c+1,cc-1)):
      boxes.add((r+1,c+1,rr,cc-1));break
  boxes=sorted(boxes)
  if not 2<=len(boxes)<=8:continue
  if any(max(a[0]-1,b[0]-1)<min(a[2]+1,b[2]+1) and max(a[1]-1,b[1]-1)<min(a[3]+1,b[3]+1) for i,a in enumerate(boxes) for b in boxes[i+1:]):continue
  out.append((color,boxes))
 return out

def extract(g,p):
 mode=p['parser'];choices=frame_candidates(g) if mode=='frames' else separator_candidates(g,int(mode[-1]))
 selector=p.get('divider','unique')
 if selector=='unique':need(len(choices)==1,'ambiguous/missing panel divider');color,bounds=choices[0]
 else:
  choices=[z for z in choices if z[0]==selector];need(len(choices)==1,'literal divider absent');color,bounds=choices[0]
 return [cut(g,b) for b in bounds],bounds,color

def bg(q,mode):
 if type(mode)==int:return mode
 h,w=len(q),len(q[0])
 if mode=='corners':
  v={q[r][c] for r,c in ((0,0),(h-1,0),(0,w-1),(h-1,w-1))};need(len(v)==1,'panel corners disagree');return next(iter(v))
 vals=[v for r,row in enumerate(q) for c,v in enumerate(row) if mode=='majority' or r in (0,h-1) or c in (0,w-1)]
 ranked=Counter(vals).most_common();need(len(ranked)==1 or ranked[0][1]>ranked[1][1],'background tie');return ranked[0][0]

def perceive(g,p,extracted=None):
 panels,bounds,divider=extracted or extract(g,p);bgs=[bg(q,p['background']) for q in panels]
 ignored=set(p.get('ignore',[]));masks=[{(r,c) for r,row in enumerate(q) for c,v in enumerate(row) if v!=b and v not in ignored} for q,b in zip(panels,bgs)]
 refs=p['reference']
 if refs=='first':ref=0
 elif refs=='last':ref=len(panels)-1
 elif refs in ('largest','smallest'):
  sizes=[len(q)*len(q[0]) for q in panels];size=(max if refs=='largest' else min)(sizes);found=[i for i,a in enumerate(sizes) if a==size];need(len(found)==1,'reference size tie');ref=found[0]
 elif refs=='empty':
  found=[i for i,m in enumerate(masks) if not m];need(len(found)==1,'empty reference ambiguous');ref=found[0]
 else:raise ValueError('reference policy')
 ids=list(range(len(panels))) if p['operands']=='all' else [i for i,m in enumerate(masks) if m]
 need(len(ids)>=2,'fewer than two operands')
 return panels,bounds,divider,bgs,masks,ref,ids

def aligned(g,p,scene=None):
 panels,bounds,divider,bgs,masks,ref,ids=scene or perceive(g,p);h,w=len(panels[ref]),len(panels[ref][0]);mode=p['alignment'];marker=p.get('marker');geometry=p['geometry']
 def anchor(q):
  a=[(r,c) for r,row in enumerate(q) for c,v in enumerate(row) if v==marker];need(len(a)==1,'missing/ambiguous marker');return a[0]
 def origin(m):need(m,'empty alignment support');return min(r for r,c in m),min(c for r,c in m)
 if mode=='marker':dest=anchor(panels[ref])
 elif mode=='bbox':dest=origin(masks[ref])
 elif mode!='none':raise ValueError('alignment')
 output=[];colored=[]
 for i in ids:
  q=panels[i];ph,pw=len(q),len(q[0]);kind='identity' if i==ref else geometry
  m=transformed(masks[i],ph,pw,kind);dr=dc=0
  if mode=='marker':src=next(iter(transformed({anchor(q)},ph,pw,kind)));dr,dc=dest[0]-src[0],dest[1]-src[1]
  elif mode=='bbox':src=origin(m);dr,dc=dest[0]-src[0],dest[1]-src[1]
  moved={(r+dr,c+dc) for r,c in m};inside={(r,c) for r,c in moved if 0<=r<h and 0<=c<w}
  need(p['boundary']=='clip' or moved==inside,'aligned foreground outside reference')
  output.append(sum(1<<(r*w+c) for r,c in inside));paint={}
  for r,c in masks[i]:
   a,b=next(iter(transformed({(r,c)},ph,pw,kind)));a+=dr;b+=dc
   if 0<=a<h and 0<=b<w:paint[a,b]=q[r][c]
  colored.append(paint)
 return output,colored,(panels,bounds,divider,bgs,masks,ref,ids),h,w

def color_value(role,scene):
 panels,bounds,divider,bgs,masks,ref,ids=scene
 if type(role)==int:return role
 if role=='reference_bg':return bgs[ref]
 if role=='divider':return divider
 if role=='reference_fg':colors={panels[ref][r][c] for r,c in masks[ref]}
 elif role=='shared_fg':colors=set.intersection(*({panels[i][r][c] for r,c in masks[i]} for i in ids))
 else:raise ValueError('color role')
 need(len(colors)==1,'foreground role ambiguous');return next(iter(colors))

def render(g,p,scene,tile):
 if p['render']=='extract':return tile
 panels,bounds,divider,bgs,masks,ref,ids=scene
 if p['render']=='extract_frame':
  need(p['parser']=='frames','not framed');r,c,rr,cc=bounds[ref];out=[list(row) for row in cut(g,(r-1,c-1,rr+1,cc+1))]
  for i,row in enumerate(tile):out[i+1][1:1+len(row)]=row
 else:
  need(p['render']=='replace_reference','render mode');out=[list(row) for row in g];r,c,rr,cc=bounds[ref]
  for i,row in enumerate(tile):out[r+i][c:cc]=row
 return tuple(map(tuple,out))

def apply(g,p):
 bitmasks,colored,scene,h,w=aligned(g,p)
 if p['operation']=='overlay':
  fill=color_value(p['false_color'],scene);tile=[[fill]*w for _ in range(h)]
  for paint in colored[::p.get('priority',1)]:
   for (r,c),v in paint.items():tile[r][c]=v
 else:
  bits=combine(bitmasks,p['operation'],h,w);true=color_value(p['true_color'],scene);false=color_value(p['false_color'],scene);need(true!=false,'equal output colors')
  tile=[[true if bits&(1<<(r*w+c)) else false for c in range(w)] for r in range(h)]
 return render(g,p,scene,tuple(map(tuple,tile)))

def operators(values,targets,stop=None):
 event('v22_adaptive_entered');end=time.monotonic()+0.9;tried=0
 def exhausted():return tried>=1200 or time.monotonic()>=end or (stop is not None and stop())
 common=sorted(set.intersection(*({v for row in g for v in row} for g in values)))
 for parser in ('separator1','separator0','frames'):
  for divider in ['unique']+common:
   base=dict(parser=parser,divider=divider)
   if exhausted():event('v22_adaptive_budget');return
   try:extracted=[extract(g,base) for g in values]
   except ValueError:continue
   event('v22_adaptive_extraction',parser)
   # Preserve distinct functional policies even if they agree on demonstrations.
   for background in ['majority','border','corners']+common:
    emitted=0;signatures=set()
    for reference,operands in product(('largest','smallest','empty','first','last'),('all','nonempty')):
     if emitted>=4:break
     p=dict(base,background=background,reference=reference,operands=operands,ignore=[])
     try:scenes=[perceive(g,p,e) for g,e in zip(values,extracted)]
     except ValueError:continue
     event('v22_adaptive_roles')
     modes=[]
     for mode in ('extract','replace_reference','extract_frame'):
      if mode=='extract_frame' and parser!='frames':continue
      compatible=True
      for g,y,scene in zip(values,targets,scenes):
       panels,bounds,sep,bgs,masks,ref,ids=scene;h,w=len(panels[ref]),len(panels[ref][0]);shape=(len(g),len(g[0])) if mode=='replace_reference' else (h+2,w+2) if mode=='extract_frame' else (h,w)
       if shape!=(len(y),len(y[0])):compatible=False;break
       if mode=='replace_reference':
        r,c,rr,cc=bounds[ref]
        if any(g[i][j]!=y[i][j] for i in range(len(g)) for j in range(len(g[0])) if not(r<=i<rr and c<=j<cc)):compatible=False;break
      if compatible:modes.append(mode)
     if not modes:continue
     event('v22_adaptive_target_frame')
     markers=[c for c in common if all(sum(v==c for row in q for v in row)==1 for scene in scenes for q in scene[0])]
     for alignment,marker in [('none',None),('bbox',None)]+[('marker',c) for c in markers]:
      if emitted>=4:break
      pp=dict(p,alignment=alignment,marker=marker,ignore=[] if marker is None else [marker])
      try:ss=[perceive(g,pp,e) for g,e in zip(values,extracted)]
      except ValueError:continue
      for geometry,boundary in product(('identity','flip_h','flip_v','rot180','rot90','rot270','transpose','anti_transpose'),('reject','clip')):
       if emitted>=4:break
       if exhausted():event('v22_adaptive_budget');return
       tried+=1;event('v22_adaptive_configuration');params=dict(pp,geometry=geometry,boundary=boundary)
       try:aa=[aligned(g,params,scene) for g,scene in zip(values,ss)]
       except ValueError:continue
       event('v22_adaptive_alignment')
       for mode in modes:
        parameters=dict(params,render=mode);tiles=[]
        for y,a in zip(targets,aa):
         scene=a[2];r,c,rr,cc=scene[1][scene[5]]
         tiles.append(cut(y,(r,c,rr,cc)) if mode=='replace_reference' else cut(y,(1,1,len(y)-1,len(y[0])-1)) if mode=='extract_frame' else y)
        roles=['reference_bg','reference_fg','shared_fg','divider']+sorted(set(v for y in tiles for row in y for v in row))
        resolved={}
        for role in roles:
         try:resolved[role]=[color_value(role,a[2]) for a in aa]
         except ValueError:pass
        for op in OPS:
         desired=[];valid=True
         for a,y in zip(aa,tiles):
          bits=combine(a[0],op,a[3],a[4]);yes=set();no=set()
          for r,row in enumerate(y):
           for c,v in enumerate(row):(yes if bits&(1<<(r*a[4]+c)) else no).add(v)
          if len(yes)!=1 or len(no)!=1 or yes==no:valid=False;break
          desired.append((next(iter(yes)),next(iter(no))))
         if not valid:continue
         tr=[role for role,vals in resolved.items() if vals==[v[0] for v in desired]];fr=[role for role,vals in resolved.items() if vals==[v[1] for v in desired]]
         for true,false in product(tr[:2],fr[:2]):
          fitted=dict(parameters,operation=op,true_color=true,false_color=false)
          if all(apply(g,fitted)==y for g,y in zip(values,targets)):
           signature=repr(fitted)
           if emitted<4 and signature not in signatures:
            signatures.add(signature);emitted+=1;event('v22_adaptive_exact');yield 'adaptive_panel',fitted
        for fill in ['reference_bg']:
         for priority in (1,-1):
          fitted=dict(parameters,operation='overlay',false_color=fill,priority=priority)
          try:ok=all(apply(g,fitted)==y for g,y in zip(values,targets))
          except ValueError:ok=False
          if ok and emitted<4 and repr(fitted) not in signatures:
           signatures.add(repr(fitted));emitted+=1;event('v22_adaptive_exact');yield 'adaptive_panel',fitted
