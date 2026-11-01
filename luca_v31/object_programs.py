"""Bounded object-role programs with transactional erase/place composition.

All scene bindings are resolved on the original input. Edits are then committed
together, so erasing an object cannot accidentally erase a later destination.
Only demonstration-exact programs are admitted. No task IDs or answer storage.
"""
from collections import Counter
from contextvars import ContextVar
from functools import lru_cache
from itertools import product
import json,time
from .grid_world import extract_objects,guess_backgrounds,Entity
from .primitives import Law
from .diagnostics import event

SCHEMA='luca-object-edit-1'
MODE=ContextVar('v24_object_mode',default='complete')
LIMIT=2500
SECONDS=1.0
def need(ok,msg):
    if not ok:raise ValueError(msg)
def imm(g):return tuple(map(tuple,g))
@lru_cache(maxsize=256)
def scene(g,rank,conn):
    bgs=guess_backgrounds(g,2);need(rank<len(bgs),'background policy')
    bg=bgs[rank];objects=extract_objects(g,bg,conn,False)
    need(0<len(objects)<=40,'object count outside domain')
    return bg,tuple(objects)
def select(objects,spec,g):
    if not objects:return []
    kind=spec['kind'];counts=Counter(o.color for o in objects)
    if kind=='color':return [o for o in objects if o.color==spec['value']]
    if kind=='singletons':return [o for o in objects if o.area==1]
    if kind=='repeated_color':return [o for o in objects if counts[o.color]>1]
    if kind=='unique_color':return [o for o in objects if counts[o.color]==1]
    if kind=='largest':return [o for o in objects if o.area==max(v.area for v in objects)]
    if kind=='smallest':return [o for o in objects if o.area==min(v.area for v in objects)]
    if kind=='border_line':
        h,w=len(g),len(g[0])
        return [o for o in objects if (o.area==h and o.bbox[1]==o.bbox[3] and o.bbox[1] in (0,w-1)) or (o.area==w and o.bbox[0]==o.bbox[2] and o.bbox[0] in (0,h-1))]
    if kind=='all':return list(objects)
    raise ValueError('unknown selector')
def distance(a,b):
    r,c,rr,cc=a.bbox;s,t,ss,tt=b.bbox
    return max(s-rr,r-ss,0)+max(t-cc,c-tt,0)
def bind(source,refs,policy):
    candidates=[o for o in refs if o.id!=source.id]
    if policy=='aligned_nearest':
        r,c,rr,cc=source.bbox
        candidates=[o for o in candidates if not (o.bbox[3]<c or o.bbox[1]>cc) or not (o.bbox[2]<r or o.bbox[0]>rr)]
    need(candidates,'reference absent')
    if policy=='unique':need(len(candidates)==1,'reference ambiguous');return candidates[0]
    d=min(distance(source,o) for o in candidates);best=[o for o in candidates if distance(source,o)==d]
    need(len(best)==1,'reference distance tie');return best[0]
def assignment(sources,refs):
    """Unique minimum-cost injective alignment, rejecting global ties."""
    need(len(sources)<=12 and len(refs)<=12,'assignment object limit')
    options=[]
    for obj in sources:
        r,c,rr,cc=obj.bbox
        row=[(i,distance(obj,o)) for i,o in enumerate(refs) if not(o.bbox[3]<c or o.bbox[1]>cc) or not(o.bbox[2]<r or o.bbox[0]>rr)]
        need(row,'assignment reference absent');options.append(row)
    @lru_cache(None)
    def solve(index,used):
        if index==len(sources):return 0,1,()
        best=float('inf');count=0;path=()
        for i,d in options[index]:
            if used&(1<<i):continue
            cost,n,rest=solve(index+1,used|(1<<i));cost+=d
            if cost<best:best,count,path=cost,n,(i,)+rest
            elif cost==best:count=min(2,count+n)
        return best,count,path
    cost,count,path=solve(0,0);need(count==1,'assignment absent or ambiguous')
    return {o.id:refs[i] for o,i in zip(sources,path)}
def destination(obj,ref,p):
    r,c,rr,cc=obj.bbox;a,b,aa,bb=ref.bbox;h,w=rr-r+1,cc-c+1;kind=p['kind']
    if kind=='contact':
        # Move the source box against the reference along each separated axis.
        dr=(a-h-r if rr<a else aa+1-r if r>aa else 0)
        dc=(b-w-c if cc<b else bb+1-c if c>bb else 0)
        need(dr or dc,'already touching or overlapping');return dr,dc
    if kind=='rotate':
        need(ref.area==1,'rotation pivot must be singleton')
        dr,dc=r-a,c-b
        for _ in range(p['turns']):dr,dc=-dc,dr
        return a+dr-r,b+dc-c
    if kind=='corner':
        # Assign the source to a quadrant around a reference bounding box.
        sr=(r+rr)-(a+aa);sc=(c+cc)-(b+bb);need(sr and sc,'source on quadrant boundary')
        if p['opposite']:sr=-sr;sc=-sc
        margin=p['margin'];tr=a-margin if sr<0 else aa+margin-h+1;tc=b-margin if sc<0 else bb+margin-w+1
        return tr-r,tc-c
    if kind=='edge':
        side=p['side'];return {'top':(a-r,0),'bottom':(aa-rr,0),'left':(0,b-c),'right':(0,bb-cc)}[side]
    if kind=='anchor':return a-r,b-c
    if kind=='stay':return 0,0
    raise ValueError('destination unavailable')

def apply(g,program):
    need(program.get('schema')==SCHEMA,'program schema')
    p=program['params'];bg,objects=scene(g,p['background_rank'],p['connectivity'])
    sources=select(objects,p['source'],g);need(sources,'empty source selection')
    sourceids={o.id for o in sources}
    refs=select([o for o in objects if o.id not in sourceids],p['reference'],g)
    need(refs,'empty reference selection')
    if p.get('source_parts','objects')=='cells':
        need(sum(o.area for o in sources)<=40,'source partition limit')
        sources=[Entity(-i-1,{pt},color,{pt:color}) for i,(pt,color) in enumerate((pair for obj in sources for pair in sorted(obj.multicolor.items())))]
    assigned=assignment(sources,refs) if p['binding']=='aligned_assignment' else None
    out=[list(row) for row in g];h,w=len(g),len(g[0]);writes={};erasures=set();steps=program['steps']
    need(steps in (['erase','place'],['place'],['erase'],['recolor'],['erase','trail','place']),'unknown edit sequence')
    for obj in sources:
        ref=assigned[obj.id] if assigned is not None else bind(obj,refs,p['binding']);dr,dc=destination(obj,ref,p['destination'])
        color=p['color'];paint=ref.color if color=='reference' else obj.color if color=='preserve' else color
        if 'erase' in steps:erasures.update(obj.cells)
        if steps==['recolor']:
            for pt in obj.cells:writes[pt]=paint
            continue
        if 'place' not in steps:continue
        for r,c in obj.cells:
            pt=(r+dr,c+dc)
            if not (0<=pt[0]<h and 0<=pt[1]<w):need(p['boundary']=='clip','outside canvas');continue
            need(pt not in writes or writes[pt]==paint,'conflicting placements');writes[pt]=paint
        if 'trail' in steps:
            need(dr==0 or dc==0 or abs(dr)==abs(dc),'nonlinear trail')
            n=max(abs(dr),abs(dc));sr=0 if dr==0 else (1 if dr>0 else -1);sc=0 if dc==0 else (1 if dc>0 else -1)
            for k in range(1,n):
                for r,c in obj.cells:
                    pt=(r+k*sr,c+k*sc)
                    if 0<=pt[0]<h and 0<=pt[1]<w and g[pt[0]][pt[1]]==bg:writes.setdefault(pt,paint)
    for r,c in erasures:out[r][c]=bg
    for (r,c),value in writes.items():
        if out[r][c]!=bg and (r,c) not in erasures and steps!=['recolor']:
            need(p['overlap']=='reference' and any((r,c) in o.cells for o in refs),'occupied destination')
        out[r][c]=value
    return imm(out)

def _candidates(pairs):
    # Output examples constrain candidate color bindings, never execution.
    common=set.intersection(*[{v for row in x for v in row} for x,y in pairs])
    removed=set.intersection(*[{x[r][c] for r in range(len(x)) for c in range(len(x[0])) if x[r][c]!=y[r][c]} for x,y in pairs])
    added=set.intersection(*[{y[r][c] for r in range(len(x)) for c in range(len(x[0])) if x[r][c]!=y[r][c]} for x,y in pairs])
    sources=[{'kind':k} for k in ('singletons','repeated_color','smallest')]+[dict(kind='color',value=c) for c in sorted(removed&common)]
    references=[{'kind':k} for k in ('largest','unique_color','all','border_line')]
    colors=['preserve','reference']+sorted(added)
    geometries=[dict(kind='contact'),*[dict(kind='rotate',turns=n) for n in (1,2,3)],*[dict(kind='corner',margin=m,opposite=o) for m in (0,1) for o in (False,True)],*[dict(kind='edge',side=s) for s in ('top','bottom','left','right')],dict(kind='anchor'),dict(kind='stay')]
    # Breadth-first stages: prioritize relational bindings with preserved color,
    # then inferred recoloring, then alternate background/connectivity views.
    for rank,conn in ((0,4),(0,8),(1,4)):
        for color in colors:
            for geo in geometries:
                for src,ref in product(sources,references):
                    bindings=('aligned_nearest','aligned_assignment') if geo['kind']=='rotate' else ('aligned_nearest','nearest') if geo['kind']=='contact' else ('nearest',)
                    for binding in bindings:
                        base=dict(background_rank=rank,connectivity=conn,source=src,reference=ref,binding=binding,destination=geo,color=color,boundary='clip' if geo['kind']=='rotate' else 'reject',overlap='reference' if geo['kind'] in ('corner','edge','anchor') else 'reject')
                        steps=['recolor'] if geo['kind']=='stay' else ['erase','place']
                        yield dict(schema=SCHEMA,params=base,steps=steps)
                        # Same geometry, independent edit alternatives: these are
                        # real atomic candidates and support the ablation arm.
                        if color=='preserve' and geo['kind'] in ('contact','anchor','edge'):
                            for actions in (['place'],['erase'],['erase','trail','place']):yield dict(schema=SCHEMA,params=base,steps=actions)

def candidates(pairs):
    for program in _candidates(pairs):
        yield program
        p=program['params']
        if p['destination']['kind']!='corner':continue
        try:
            _,objects=scene(pairs[0][0],p['background_rank'],p['connectivity'])
            selected=select(objects,p['source'],pairs[0][0])
            split=selected and any(o.area>1 for o in selected) and all(o.area<=4 for o in selected)
        except ValueError:split=False
        if split:yield dict(program,params=dict(p,source_parts='cells'))

def residual(pred,target,original):
    changed=right=stable=damage=0
    for a,b,c in zip((v for row in original for v in row),(v for row in target for v in row),(v for row in pred for v in row)):
        if a!=b:changed+=1;right+=c==b
        else:stable+=1;damage+=c!=b
    return dict(required=changed,correct_changes=right,unchanged=stable,damage=damage)

def fit(pairs,mode=None,limit=LIMIT,seconds=SECONDS,trace=None):
    from .budget import check as _budget_check
    _budget_check()
    mode=mode or MODE.get();need(mode in ('complete','atomic'),'unknown search arm')
    if len(pairs)<2 or any((len(x),len(x[0]))!=(len(y),len(y[0])) for x,y in pairs):event('v24_representation_outside_domain');return []
    start=time.monotonic();stats=Counter();best=None;found=[]
    for i,program in enumerate(candidates(pairs)):
        if i>=limit or time.monotonic()-start>=seconds:
            stats['candidate_limit' if i>=limit else 'time_limit']+=1;break
        stats['candidate_slots']+=1
        if mode=='atomic' and len(program['steps'])>1:stats['composition_disabled']+=1;continue
        stats['attempted']+=1;valid=True
        for x,y in pairs:
            try:pred=apply(x,program)
            except (ValueError,KeyError,IndexError) as exc:
                stats['precondition_rejected']+=1;stats['reason:'+str(exc)]+=1;valid=False;break
            if pred!=y:
                stats['residual_mismatch']+=1;score=residual(pred,y,x)
                rank=(score['correct_changes']/max(1,score['required']),-score['damage'])
                if best is None or rank>best[0]:best=(rank,score,program)
                valid=False;break
        if valid:
            spec=json.loads(json.dumps(program));law=Law('v24_object_edit_'+program['params']['destination']['kind'],lambda g,p=spec:apply(g,p),json.dumps(spec,sort_keys=True),constant_bits=3,op_count=len(spec['steps'])+3)
            law.program_spec=spec;law.origin='v24_object_programs';found.append(law);stats['exact_fit']+=1
            if len(found)>=32:break
    for key,value in stats.items():
        for _ in range(value):event('v24_'+key)
    event('v24_search_summary',dict(stats))
    if best:event('v24_best_partial',dict(scope='first mismatching demonstration per candidate',residual=best[1],program=best[2]))
    if trace is not None:trace.update(dict(stats=stats,best_partial=None if best is None else dict(scope='first mismatching demonstration per candidate',residual=best[1],program=best[2]),seconds=time.monotonic()-start,mode=mode))
    return found
def choose(pairs,test_input,prior):
    if prior:return prior
    event('v24_extension_entered');out=[];seen=set()
    for law in fit(pairs):
        try:pred=law.predict(test_input)
        except (ValueError,KeyError,IndexError):event('v24_test_precondition_rejected');continue
        if pred in seen:continue
        seen.add(pred);out.append(law)
    return out
