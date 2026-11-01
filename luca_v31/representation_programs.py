"""Typed ObjectSet -> Grid -> Grid search, exact on all demonstrations.

Selection predicates are relational, not task identifiers. Each representation
is a competing interpretation. No beam drops a representation for poor pixel
accuracy. Full traces distinguish domain rejection, residual errors and budgets.
"""
from collections import Counter
from contextvars import ContextVar
from fractions import Fraction
from functools import lru_cache
from itertools import product
import json,time
from .grid_world import extract_objects,extract_holes,guess_backgrounds,Entity
from .primitives import Law,GEOM_OPS
from .diagnostics import event
SCHEMA='luca-representation-program-1'
MODE=ContextVar('v25_mode',default='full')
LAST=ContextVar('v25_trace',default=None)
LIMIT=3000
SECONDS=2.0

def imm(g):return tuple(map(tuple,g))
def need(ok,message):
 if not ok:raise ValueError(message)
def objects(g,p):return scene(g,p['background_rank'],p['view'],p.get('connectivity',4))
@lru_cache(maxsize=128)
def scene(g,rank,kind,conn):
 bgs=guess_backgrounds(g,2);need(rank<len(bgs),'background rank absent');bg=bgs[rank]
 if kind=='holes':obs=extract_holes(g,bg)
 elif kind=='color_groups':
  obs=[]
  for c in sorted({v for row in g for v in row}-{bg}):
   cells={(r,j) for r,row in enumerate(g) for j,v in enumerate(row) if v==c}
   obs.append(Entity(len(obs),cells,c,{q:c for q in cells}))
 else:
  need(kind in ('mono','multi'),'unknown representation')
  obs=extract_objects(g,bg,conn,kind=='multi')
 need(0<len(obs)<=40,'object count outside domain');return bg,obs

def select(obs,kind):
 signatures=Counter(o.shape() for o in obs)
 if kind=='unique_shape':chosen=[o for o in obs if signatures[o.shape()]==1]
 elif kind=='repeated_shape':chosen=[o for o in obs if signatures[o.shape()]>1]
 else:
  feature={'largest':lambda o:o.area,'smallest':lambda o:-o.area,
           'most_colors':lambda o:len(set(o.multicolor.values())),
           'fewest_colors':lambda o:-len(set(o.multicolor.values())),
           'densest':lambda o:Fraction(o.area,(o.bbox[2]-o.bbox[0]+1)*(o.bbox[3]-o.bbox[1]+1)),
           'sparsest':lambda o:-Fraction(o.area,(o.bbox[2]-o.bbox[0]+1)*(o.bbox[3]-o.bbox[1]+1))}
  need(kind in feature,'unknown selector');scores=[feature[kind](o) for o in obs];chosen=[o for o,s in zip(obs,scores) if s==max(scores)]
 need(chosen,'empty selection')
 # Repeated shapes can select multiple copies only when cropped color grids
 # agree. This is checked after rendering, not by arbitrary scan order.
 return chosen

def extract(g,p):
 bg,obs=objects(g,p);chosen=select(obs,p['selector']);outs=[]
 for obj in chosen:
  r,c,rr,cc=obj.bbox
  if p['render']=='context':out=tuple(tuple(g[i][j] for j in range(c,cc+1)) for i in range(r,rr+1))
  elif p['render']=='isolated':out=tuple(tuple(g[i][j] if (i,j) in obj.cells else bg for j in range(c,cc+1)) for i in range(r,rr+1))
  else:raise ValueError('unknown rendering')
  outs.append(out)
 need(all(v==outs[0] for v in outs),'selection ambiguous');return outs[0]

def transform(g,p):
 kind=p['kind']
 if kind=='geometry':return GEOM_OPS[p['geometry']](g)
 if kind=='recolor':
  need(all(str(v) in p['mapping'] for row in g for v in row),'unobserved color');return tuple(tuple(p['mapping'][str(v)] for v in row) for row in g)
 if kind in ('scale','tile'):
  a,b=p['rows'],p['cols'];need(a>=1 and b>=1 and len(g)*a<=30 and len(g[0])*b<=30,'invalid expansion')
  return tuple(tuple(g[r//a][c//b] if kind=='scale' else g[r%len(g)][c%len(g[0])] for c in range(len(g[0])*b)) for r in range(len(g)*a))
 if kind=='compress_runs':
  rows=[row for i,row in enumerate(g) if i==0 or row!=g[i-1]]
  cols=[c for c in range(len(rows[0])) if c==0 or any(row[c]!=row[c-1] for row in rows)]
  return tuple(tuple(row[c] for c in cols) for row in rows)
 raise ValueError('unknown grid transform')

def execute(g,spec):
 need(spec.get('schema')==SCHEMA,'unknown program schema');steps=spec['steps'];need(1<=len(steps)<=2,'program depth')
 need(steps[0]['op']=='extract_object_grid','first operator type')
 out=extract(g,steps[0]['params'])
 if len(steps)==2:
  need(steps[1]['op']=='grid_transform','second operator type');out=transform(out,steps[1]['params'])
 return out

def law(spec):
 l=Law('v25_representation_program',lambda g:execute(g,spec),json.dumps(spec,sort_keys=True),op_count=len(spec['steps']))
 l.program_spec=spec;l.origin='v25_representation_search';return l

def second_ops(values,targets):
 for name in GEOM_OPS:
  if name!='identity':yield dict(kind='geometry',geometry=name)
 yield dict(kind='compress_runs')
 ratios=[]
 for x,y in zip(values,targets):
  if len(y)%len(x) or len(y[0])%len(x[0]):break
  ratios.append((len(y)//len(x),len(y[0])//len(x[0])))
 if len(ratios)==len(values) and len(set(ratios))==1 and ratios[0]!=(1,1):
  a,b=ratios[0]
  for kind in ('scale','tile'):yield dict(kind=kind,rows=a,cols=b)
 if all((len(x),len(x[0]))==(len(y),len(y[0])) for x,y in zip(values,targets)):
  mapping={};ok=True
  for x,y in zip(values,targets):
   for row,out in zip(x,y):
    for a,b in zip(row,out):
     if str(a) in mapping and mapping[str(a)]!=b:ok=False
     mapping[str(a)]=b
  if ok and any(int(a)!=b for a,b in mapping.items()):yield dict(kind='recolor',mapping=mapping)

def residual(values,targets):
 # All demonstrations contribute; no truncation at the first mismatch.
 wrong=0;total=0;shape_errors=0
 for x,y in zip(values,targets):
  total+=sum(map(len,y))
  if (len(x),len(x[0]))!=(len(y),len(y[0])):shape_errors+=1;wrong+=sum(map(len,y));continue
  wrong+=sum(a!=b for row,out in zip(x,y) for a,b in zip(row,out))
 return dict(wrong_cells=wrong,target_cells=total,shape_errors=shape_errors)

def fit(pairs,mode=None,seconds=SECONDS,limit=LIMIT,trace=None):
 from .budget import check as _budget_check
 _budget_check()
 scene.cache_clear();mode=mode or MODE.get();need(mode in ('full','atomic','mono'),'unknown search mode');start=time.monotonic();stats=Counter();reasons=Counter();found=[];best=None;best_spec=None;seen=set()
 inputs=tuple(x for x,y in pairs);targets=tuple(y for x,y in pairs)
 def exhausted():
  if stats['candidate_slots']>=limit:stats['candidate_limit']=1;return True
  if time.monotonic()-start>=seconds:stats['time_limit']=1;return True
  return False
 def check(values,spec):
  nonlocal best,best_spec
  errors=residual(values,targets)
  if best is None or (errors['shape_errors'],errors['wrong_cells'])<(best['shape_errors'],best['wrong_cells']):best,best_spec=errors,spec
  if values==targets:
   found.append(law(spec));stats['exact_fits']+=1
  else:stats['residual_mismatch']+=1
 if pairs:
  for rank,view,conn,selector,render in product(range(2),('multi','mono','color_groups','holes'),(4,8),('unique_shape','most_colors','largest','smallest','repeated_shape','densest','sparsest','fewest_colors'),('isolated','context')):
   if exhausted() or len(found)>=32:break
   if view in ('color_groups','holes') and conn==8:continue
   stats['candidate_slots']+=1
   if mode=='mono' and view!='mono':stats['representation_disabled']+=1;continue
   p=dict(background_rank=rank,view=view,connectivity=conn,selector=selector,render=render)
   spec=dict(schema=SCHEMA,steps=[dict(op='extract_object_grid',params=p)])
   try:values=tuple(extract(x,p) for x in inputs)
   except ValueError as e:reasons[str(e)]+=1;stats['precondition_rejected']+=1;continue
   stats['representation_executed']+=1;check(values,spec)
   # Keep semantically distinct extraction policies: same demonstration state
   # can mean different test behavior. No state-equivalence pruning here.
   for op in second_ops(values,targets):
    if exhausted() or len(found)>=32:break
    stats['candidate_slots']+=1
    if mode=='atomic':stats['composition_disabled']+=1;continue
    try:outputs=tuple(transform(v,op) for v in values)
    except ValueError as e:reasons[str(e)]+=1;stats['precondition_rejected']+=1;continue
    stats['composition_executed']+=1
    check(outputs,dict(schema=SCHEMA,steps=spec['steps']+[dict(op='grid_transform',params=op)]))
 result=dict(mode=mode,stats=dict(stats),preconditions=dict(reasons),best_residual=best,best_program=best_spec,elapsed_seconds=round(time.monotonic()-start,6),limits=dict(seconds=seconds,candidate_slots=limit),residual_scope='all demonstrations; shape mismatch counts every target cell wrong')
 if trace is not None:trace.update(result)
 LAST.set(result)
 event('v25_search_summary')
 for k,v in stats.items():
  if v:event('v25_'+k,str(v))
 return sorted(found,key=lambda l:(len(l.program_spec['steps']),len(l.description)))

def upgrade_result(pairs,test_input,result):
 from .budget import check as _budget_check
 _budget_check()
 LAST.set(None)
 if not result.failure_category:return result
 event('v25_extension_entered');laws=fit(pairs);preds=[];used=[]
 for candidate in laws:
  try:prediction=candidate.predict(test_input)
  except ValueError:event('v25_test_precondition_rejected');continue
  if prediction in preds:continue
  preds.append(prediction);used.append(candidate)
  if len(preds)==2:break
 if not preds:return result
 return type(result)(preds,used,'V25 representation program fits every demonstration.',failure_category=None)
