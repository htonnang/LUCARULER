"""Explicit six-operation Boolean family on aligned equal-sized panels.

XOR is odd parity across all panels; XNOR is even parity. NAND and NOR
negate n-ary AND and OR once (they are not sequential binary folds).
"""
from collections import Counter
from itertools import product
from .backgrounds import resolve
OPS=('AND','OR','XOR','NAND','NOR','XNOR')
def need(x,msg):
 if not x:raise ValueError(msg)
def truth(bits,op):
 need(op in OPS,'unknown Boolean operation');need(len(bits)>=2,'at least two inputs')
 if op=='AND':return all(bits)
 if op=='OR':return any(bits)
 if op=='XOR':return sum(bits)%2==1
 if op=='NAND':return not all(bits)
 if op=='NOR':return not any(bits)
 return sum(bits)%2==0

def split(g,p):
 rows,cols=p['rows'],p['cols'];rs,cs=p['row_gap'],p['col_gap'];h,w=len(g),len(g[0])
 need(rows*cols in (2,3,4) and rows>=1 and cols>=1,'unsupported panel count')
 need(rs in (0,1,2) and cs in (0,1,2),'separator thickness')
 hh=h-(rows-1)*rs;ww=w-(cols-1)*cs
 need(hh>0 and ww>0 and hh%rows==0 and ww%cols==0,'unequal panels');ph,pw=hh//rows,ww//cols
 for k in range(1,rows):
  for r in range(k*ph+(k-1)*rs,k*(ph+rs)):need(len(set(g[r]))==1,'nonuniform row separator')
 for k in range(1,cols):
  for c in range(k*pw+(k-1)*cs,k*(pw+cs)):need(len({row[c] for row in g})==1,'nonuniform column separator')
 return tuple(tuple(tuple(g[r*(ph+rs)+i][c*(pw+cs):c*(pw+cs)+pw]) for i in range(ph)) for r,c in product(range(rows),range(cols)))

def background(g,p,panels=None):
 if p['bg']!='shared_panel_color':return resolve(g,p['bg'])
 panels=split(g,p) if panels is None else panels
 common=set.intersection(*({v for row in q for v in row} for q in panels))
 need(len(common)==1,'shared panel background ambiguous')
 return next(iter(common))

def mask(g,p):
 panels=split(g,p);bg=background(g,p,panels);h,w=len(panels[0]),len(panels[0][0])
 return tuple(tuple(truth([q[r][c]!=bg for q in panels],p['operation']) for c in range(w)) for r in range(h))

def apply(g,p):
 bits=mask(g,p);bg=background(g,p);false=bg if p['false_color']=='input_background' else p['false_color'];true=p['true_color']
 need(type(false)==int and 0<=false<=9 and type(true)==int and 0<=true<=9 and false!=true,'invalid output palette')
 return tuple(tuple(true if v else false for v in row) for row in bits)

def operators(values,targets):
 # Infer panel layout from demonstration dimensions, not task identifiers.
 common=set.intersection(*({v for row in g for v in row} for g in values));modes=[0,1,'corners','border']+['color:'+str(c) for c in sorted(common)]
 bgs=[];seen=set()
 for mode in modes:
  try:key=tuple(resolve(g,mode) for g in values)
  except ValueError:continue
  if key not in seen:seen.add(key);bgs.append((mode,key))
 for rows,cols in ((1,2),(2,1),(1,3),(3,1),(1,4),(4,1),(2,2)):
  for rs,cs in product((0,1,2) if rows>1 else (0,),(0,1,2) if cols>1 else (0,)):
   p=dict(rows=rows,cols=cols,row_gap=rs,col_gap=cs)
   if any(len(g)!=rows*len(y)+(rows-1)*rs or len(g[0])!=cols*len(y[0])+(cols-1)*cs for g,y in zip(values,targets)):continue
   try:
    for g in values:split(g,p)
   except ValueError:continue
   layout_bgs=list(bgs)
   try:
    shared=tuple(background(g,dict(p,bg='shared_panel_color')) for g in values)
    layout_bgs.insert(0,('shared_panel_color',shared))
   except ValueError:pass
   for mode,bgvalues in layout_bgs:
    for op in OPS:
     params=dict(p,bg=mode,operation=op);false_values=set();true_values=set();false_is_bg=True
     for g,y,bg in zip(values,targets,bgvalues):
      bits=mask(g,params)
      for r,row in enumerate(y):
       for c,v in enumerate(row):
        if bits[r][c]:true_values.add(v)
        else:false_values.add(v);false_is_bg &= v==bg
     # Do not guess an unobserved output label.
     if len(true_values)!=1 or not false_values:continue
     true=next(iter(true_values))
     if false_is_bg and all(true!=bg for bg in bgvalues):false='input_background'
     elif len(false_values)==1 and true not in false_values:false=next(iter(false_values))
     else:continue
     yield 'panel_boolean',dict(params,true_color=true,false_color=false)
