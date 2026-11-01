"""Bounded Grid-typed expression synthesis with reusable object operators.

Search composes intermediate states even when they do not solve a task.
Scores use demonstration changed/foreground cells and shape; outputs are
accepted only on exact fit. Object matching supplies translation/geometry
proposals. No task IDs, diagnostic oracle import, or test-output access.
"""
from collections import Counter,defaultdict
from itertools import product
from math import isqrt
import json
import time
from .grid_world import extract_objects,guess_backgrounds
from .backgrounds import resolve
from .primitives import Law,GEOM_OPS
from .diagnostics import event
from .correspondence import movement_hints

SCHEMA='luca-composition-10'
MAX_DEPTH=3
CANDIDATE_LIMIT=4500
BEAM_WIDTH=16
SECONDS=3.0

def imm(g):return tuple(tuple(row) for row in g)
def shape(g):return len(g),len(g[0])
def need(ok,msg):
    if not ok:raise ValueError(msg)
def box(cells):return min(r for r,c in cells),min(c for r,c in cells),max(r for r,c in cells),max(c for r,c in cells)
def crop(g,bg,pad=0):
    cells=[(r,c) for r,row in enumerate(g) for c,v in enumerate(row) if v!=bg];need(cells,'empty crop')
    r,c,rr,cc=box(cells);r-=pad;c-=pad;rr+=pad;cc+=pad
    need(rr-r<30 and cc-c<30,'crop too large')
    return tuple(tuple(g[i][j] if 0<=i<len(g) and 0<=j<len(g[0]) else bg for j in range(c,cc+1)) for i in range(r,rr+1))
def object_view(g,p):return extract_objects(g,resolve(g,p['bg']),p.get('conn',4),p.get('multi',True))
def selected(objs,selector):
    need(objs,'no objects')
    if selector=='all':return objs
    if selector in ('largest','smallest'):
        n=(max if selector=='largest' else min)(o.area for o in objs);return [o for o in objs if o.area==n]
    if isinstance(selector,int):return [o for o in objs if selector in o.multicolor.values()]
    raise ValueError('selector')






def chain(g,p):
    bg=resolve(g,p['bg']);marker=p['marker'];objs=sorted(extract_objects(g,bg,4,True),key=lambda o:o.bbox[1])
    need(2<=len(objs)<=12,'chain needs multiple parts');canvas={};last=None
    for i,o in enumerate(objs):
        markers=[q for q in o.cells if g[q[0]][q[1]]==marker];body=set(o.multicolor.values())-{marker}
        need(markers and len(body)==1,'invalid marked fragment')
        left=min(markers,key=lambda q:(q[1],q[0]));right=max(markers,key=lambda q:(q[1],q[0]))
        delta=(0,0) if last is None else (last[0]-left[0],last[1]+p['gap']-left[1])
        for r,c in o.cells:
            q=(r+delta[0],c+delta[1]);need(0<=q[0]<len(g) and 0<=q[1]<30,'chain out of bounds')
            color=next(iter(body)) if p['erase_marker'] else g[r][c]
            need(q not in canvas or (p['gap']==0 and canvas[q]==color),'chain collision');canvas[q]=color
        last=(right[0]+delta[0],right[1]+delta[1])
    width=max(c for r,c in canvas)+1
    return tuple(tuple(canvas.get((r,c),bg) for c in range(width)) for r in range(len(g)))




def apply(g,op,p):
    if op=='line_relations':
        from .line_relations import execute as draw
        if p.get('require_paint_absent') and any(p['paint'] in row for row in g):raise ValueError('temporary paint colour already present')
        return draw(g,p)
    if op=='representation_extract':
        from .representation_programs import extract
        return extract(g,p)
    if op=='panel_program':
        from .panel_programs import apply as panel_apply
        return panel_apply(g,p)
    if op=='adaptive_panel':
        from .adaptive_panels import apply as adaptive_apply
        return adaptive_apply(g,p)
    if op=='panel_relation':
        from .panel_relations import apply as relation_apply
        return relation_apply(g,p)
    if op=='panel_boolean':
        from .panel_boolean import apply as boolean_apply
        return boolean_apply(g,p)
    if op in ('role_select','role_recolor','role_place'):
        from .reference_relations import apply as role_apply
        return role_apply(g,op,p)
    if op=='geometry':return GEOM_OPS[p['kind']](g)
    if op=='recolor':return tuple(tuple(p['mapping'].get(str(v),v) for v in row) for row in g)
    if op=='crop':return crop(g,resolve(g,p['bg']),p.get('pad',0))
    if op in ('select','boxes','shift'):
        bg=resolve(g,p['bg']);objs=selected(object_view(g,p),p['selector']);need(objs,'empty selection')
        out=[[bg]*len(g[0]) for _ in g] if op=='select' else [list(row) for row in g]
        if op=='shift':
            for o in objs:
                for r,c in o.cells:out[r][c]=bg
        for o in objs:
            r,c,rr,cc=o.bbox
            if op=='select':
                for q,v in o.multicolor.items():out[q[0]][q[1]]=v
            elif op=='boxes':
                colors=Counter(o.multicolor.values());top=colors.most_common();need(len(top)==1 or top[0][1]>top[1][1],'body color tied');color=top[0][0]
                for i in range(r,rr+1):
                    for j in range(c,cc+1):
                        if p['mode']=='fill' or i in (r,rr) or j in (c,cc):out[i][j]=color
            else:
                for (i,j),v in o.multicolor.items():
                    a,b=i+p['dr'],j+p['dc'];need(0<=a<len(g) and 0<=b<len(g[0]),'shift bounds')
                    need(out[a][b] in (bg,v),'shift collision');out[a][b]=v
        return imm(out)
    if op=='chain':return chain(g,p)
    if op=='squares':
        bg=resolve(g,p['bg']);counts=Counter(v for row in g for v in row if v!=bg);need(counts,'empty counts')
        need(all(isqrt(n)**2==n for n in counts.values()),'non-square counts');need(len(set(counts.values()))==len(counts),'order tied')
        blocks=sorted((isqrt(n),c) for c,n in counts.items());blocks=blocks if p['ascending'] else blocks[::-1]
        h=max(s for s,c in blocks);w=sum(s for s,c in blocks)+p['gap']*(len(blocks)-1);need(w<=30 and h<=30,'packing bounds')
        out=[[bg]*w for _ in range(h)];offset=0
        for side,col in blocks:
            start=h-side if p['align']=='bottom' else 0
            for r in range(start,start+side):
                for c in range(offset,offset+side):out[r][c]=col
            offset+=side+p['gap']
        return imm(out)
    raise ValueError('unknown unary operator')


def merge(a,b,op,p):
    if op=='hstack':need(len(a)==len(b),'row mismatch');return tuple(x+y for x,y in zip(a,b))
    if op=='vstack':need(len(a[0])==len(b[0]),'column mismatch');return a+b
    need(shape(a)==shape(b),'overlay shapes');bg=resolve(a,p['bg'])
    if op=='overlay':return tuple(tuple(b[r][c] if b[r][c]!=bg else v for c,v in enumerate(row)) for r,row in enumerate(a))
    if op=='intersection':return tuple(tuple(v if v==b[r][c] else bg for c,v in enumerate(row)) for r,row in enumerate(a))
    raise ValueError('unknown binary operator')

UNARY={'line_relations','representation_extract','panel_program','adaptive_panel','panel_relation','panel_boolean','role_select','role_recolor','role_place','geometry','recolor','crop','select','boxes','shift','chain','squares'}
BINARY={'hstack','vstack','overlay','intersection'}

def execute(g,program):
    need(program.get('schema')==SCHEMA,'schema');nodes=program['nodes'];need(0<len(nodes)<=15,'node count')
    values={};depth={}
    for node in nodes:
        name,op=node['id'],node['op'];args=node.get('args',[]);p=node.get('params',{})
        need(name not in values,'duplicate node');need(all(a in values for a in args),'forward reference')
        if op=='input':need(not args,'input operands');out=imm(g);d=0
        elif op in UNARY:
            need(len(args)==1,'unary signature');out=apply(values[args[0]],op,p);d=depth[args[0]]+1
        elif op in BINARY:
            need(len(args)==2,'binary signature');out=merge(values[args[0]],values[args[1]],op,p);d=max(depth[a] for a in args)+1
        else:raise ValueError('operator')
        need(0<len(out)<=30 and 0<len(out[0])<=30 and all(len(row)==len(out[0]) and all(type(v)==int and 0<=v<=9 for v in row) for row in out),'invalid grid')
        values[name]=out;depth[name]=d
    need(program['return'] in values,'return');return values[program['return']]


def source():return dict(schema=SCHEMA,nodes=[dict(id='n0',op='input',args=[],params={})],**{'return':'n0'})
def compose(a,op,p,b=None):
    nodes=[];maps=[]
    for prog in (a,) if b is None else (a,b):
        mapping={}
        for node in prog['nodes']:
            # Sharing identical nodes preserves true DAG branching.
            args=[mapping[k] for k in node['args']]
            identical=next((n for n in nodes if n['op']==node['op'] and n['args']==args and n['params']==node['params']),None)
            if identical:mapping[node['id']]=identical['id'];continue
            name='n'+str(len(nodes));mapping[node['id']]=name;nodes.append(dict(id=name,op=node['op'],args=args,params=node['params']))
        maps.append(mapping[prog['return']])
    name='n'+str(len(nodes));nodes.append(dict(id=name,op=op,args=maps,params=p))
    return dict(schema=SCHEMA,nodes=nodes,**{'return':name})
def law(program):
    spec=json.loads(json.dumps(program));ops=[n['op'] for n in spec['nodes'][1:]]
    result=Law('v22_'+'+'.join(ops),lambda g:execute(g,spec),json.dumps(spec,sort_keys=True),op_count=len(ops))
    result.program_spec=spec;result.origin='v22_compositional_search';return result


def operators(values,targets,hints,focused=False,stop=None):
    if not focused:
        from .panel_programs import operators as program_operators
        yield from program_operators(values,targets,stop=stop)
        from .adaptive_panels import operators as adaptive_operators
        yield from adaptive_operators(values,targets,stop=stop)
        from .panel_boolean import operators as boolean_operators
        yield from boolean_operators(values,targets)
        from .panel_relations import operators as relation_operators
        yield from relation_operators(values,targets,stop=stop)
    from .reference_relations import operators as role_operators
    yield from role_operators(values,focused=focused)
    common=sorted(set.intersection(*({v for row in g for v in row} for g in values)))
    support=Counter(v for y in targets for row in y for v in row)
    explicit=sorted(set(common)|set(support),key=lambda c:(-support[c],c))[:3]
    bgs=[0,1]+['color:'+str(c) for c in explicit]
    # Per-state palette inference is useful after geometry/cropping, even when
    # recoloring the original alone could not have explained the examples.
    if all(shape(x)==shape(y) for x,y in zip(values,targets)):
        mapping={};consistent=True;changes=Counter()
        for x,y in zip(values,targets):
            for row,out in zip(x,y):
                for a,b in zip(row,out):
                    if a!=b:changes[a,b]+=1
                    if str(a) in mapping and mapping[str(a)]!=b:consistent=False
                    mapping[str(a)]=b
        if consistent and any(int(k)!=v for k,v in mapping.items()):yield 'recolor',dict(mapping=mapping)
        for (a,b),n in changes.most_common(8):yield 'recolor',dict(mapping={str(a):b})
    for kind in sorted(GEOM_OPS,key=lambda k:(k not in hints['geometries'],list(GEOM_OPS).index(k))):
        if kind!='identity':yield 'geometry',dict(kind=kind)
    if focused:
        for pad in (0,1):yield 'crop',dict(bg=0,pad=pad)
        return
    for bg in bgs:
        try:bvals=[resolve(g,bg) for g in values]
        except (ValueError,IndexError):continue
        for pad in (0,1):yield 'crop',dict(bg=bg,pad=pad)
        counts=[Counter(v for row in g for v in row if v!=b) for g,b in zip(values,bvals)]
        if all(c and len(c)<=8 and all(isqrt(n)**2==n for n in c.values()) for c in counts):
            for ascending,gap,align in product((True,False),(0,1),('bottom','top')):
                yield 'squares',dict(bg=bg,ascending=ascending,gap=gap,align=align)
        for conn in (4,8):
            multi_objects=[extract_objects(g,b,conn,True) for g,b in zip(values,bvals)]
            if all(2<=len(objs)<=8 for objs in multi_objects):
                for marker in common:
                    if not all(all(marker in o.multicolor.values() and len(set(o.multicolor.values())-{marker})==1 for o in objs) for objs in multi_objects):continue
                    bodies=set.intersection(*({next(iter(set(o.multicolor.values())-{marker})) for o in objs} for objs in multi_objects))
                    if conn==4:
                        for gap,erase in product((0,1),(False,True)):yield 'chain',dict(bg=bg,marker=marker,gap=gap,erase_marker=erase)
            for multi in (True,False):
                objects=multi_objects if multi else [extract_objects(g,b,conn,False) for g,b in zip(values,bvals)]
                if not all(1<=len(o)<=40 for o in objects):continue
                # Selection provides intermediate grids for later crop/geometry,
                # not only standalone solutions.
                for selector in ['largest','smallest']+common:
                    yield 'select',dict(bg=bg,conn=conn,multi=multi,selector=selector)
                for mode in ('outline','fill'):
                    yield 'boxes',dict(bg=bg,conn=conn,multi=multi,selector='all',mode=mode)
                if conn==4 and multi:
                    for dr,dc in [q for q in hints['offsets'] if q!=(0,0)][:4]:
                        for selector in ('largest','smallest','all'):
                            yield 'shift',dict(bg=bg,conn=conn,multi=multi,selector=selector,dr=dr,dc=dc)


def score(values,targets,original):
    """Demonstration-only foreground/changed-region priority, never acceptance."""
    total=0.0
    for g,y,x in zip(values,targets,original):
        if shape(g)!=shape(y):
            h,w=shape(g);hh,ww=shape(y)
            structural=0.0
            # A selected object may already have the target structure while
            # still occupying a large original canvas. Retain it for a later
            # crop/geometry step instead of rewarding canvas size alone.
            bg=guess_backgrounds(x,1)[0]
            try:
                a=crop(g,bg);b=crop(y,bg)
                if any(fn(a)==b for fn in GEOM_OPS.values()):structural=0.65
                else:
                    aa=tuple(tuple(int(v!=bg) for v in row) for row in a)
                    bb=tuple(tuple(int(v!=bg) for v in row) for row in b)
                    if any(fn(aa)==bb for fn in GEOM_OPS.values()):structural=0.45
            except ValueError:pass
            total+=0.1*(min(h,hh)/max(h,hh)+min(w,ww)/max(w,ww))+structural
            continue
        bg=guess_backgrounds(y,1)[0];weighted=weight=0
        for r,row in enumerate(y):
            for c,v in enumerate(row):
                n=1+(3 if v!=bg else 0)+(4 if shape(x)==shape(y) and x[r][c]!=v else 0)
                weight+=n;weighted+=n*(g[r][c]==v)
        total+=0.3+0.7*weighted/weight
    return total/len(values)


def _fit(pairs,seconds=SECONDS,limit=CANDIDATE_LIMIT,max_depth=MAX_DEPTH,beam_width=BEAM_WIDTH,trace=None,focused=False,usage=None):
    from .budget import check as _budget_check
    _budget_check()
    if len(pairs)<2:return []
    started=time.monotonic();original=tuple(imm(x) for x,y in pairs);targets=tuple(imm(y) for x,y in pairs)
    hints=movement_hints(pairs);event('v22_correspondence_offsets',hints['offsets'])
    root=dict(program=source(),values=original,score=score(original,targets,original),last='input')
    beam=[root];archive=[root];found=[];seen={original:1};tried=0;budget=False
    def exhausted():
        nonlocal budget
        if tried>=limit or time.monotonic()-started>=seconds:
            if not budget:event('v22_search_budget_exhausted');budget=True
            return True
        return False
    def accept(values,program,last,generated):
        # Keep up to two demonstration-equivalent expressions so one may
        # generalize when another has different test preconditions.
        # Preserve boundary alternatives even when all demonstrations are interior.
        if last in ('adaptive_panel','panel_program') and values==targets:
            found.append(law(program));event('v22_exact_candidate');return
        if seen.get(values,0)>=2:return
        seen[values]=seen.get(values,0)+1
        if values==targets:
            found.append(law(program));event('v22_exact_candidate');return
        generated.append(dict(program=program,values=values,score=score(values,targets,original),last=last))
    for depth in range(1,max_depth+1):
        event('v22_depth_entered',depth);generated=[]
        # Binary composition works on retained intermediate branches, with
        # output shape used to avoid impossible final merges.
        if depth>=2 and not exhausted():
            pool=archive[:12]+beam[:8]
            binary_tried=0
            for a,b in product(pool,pool):
                if a is b:continue
                for op in ('hstack','vstack','overlay','intersection'):
                    if exhausted() or binary_tried>=256:break
                    if op=='hstack':legal=all(len(x)==len(z)==len(y) and len(x[0])+len(z[0])==len(y[0]) for x,z,y in zip(a['values'],b['values'],targets))
                    elif op=='vstack':legal=all(len(x[0])==len(z[0])==len(y[0]) and len(x)+len(z)==len(y) for x,z,y in zip(a['values'],b['values'],targets))
                    else:legal=all(shape(x)==shape(z)==shape(y) for x,z,y in zip(a['values'],b['values'],targets))
                    if not legal:continue
                    tried+=1;binary_tried+=1;event('v22_candidate_tried');p=dict(bg=0)
                    try:
                        values=tuple(merge(x,z,op,p) for x,z in zip(a['values'],b['values']))
                        program=compose(a['program'],op,p,b['program'])
                        if len(program['nodes'])<=15:accept(values,program,op,generated)
                    except (ValueError,KeyError,IndexError):event('v22_precondition_rejected')
                if exhausted() or binary_tried>=256:break
        if found or exhausted():break
        for state in beam:
            for op,p in operators(state['values'],targets,hints,focused=focused,stop=exhausted):
                if exhausted():break
                tried+=1;event('v22_candidate_tried')
                try:
                    values=tuple(apply(g,op,p) for g in state['values'])
                    if values==state['values'] or any(len(g)>30 or len(g[0])>30 for g in values):continue
                    program=compose(state['program'],op,p)
                    if len(program['nodes'])<=15:accept(values,program,op,generated)
                except (ValueError,KeyError,IndexError):event('v22_precondition_rejected')
            if exhausted():break
        if found or exhausted():break
        generated.sort(key=lambda s:(-s['score'],len(s['program']['nodes'])))
        # Preserve some operator diversity rather than all near-identical
        # high-background-accuracy states occupying the frontier.
        beam=[];families=Counter()
        for state in generated:
            if len(beam)<beam_width//2 or families[state['last']]<2:
                beam.append(state);families[state['last']]+=1
            if len(beam)>=beam_width:break
        if trace is not None:trace.append([(state['score'],state['program']) for state in beam])
        archive=sorted(archive+beam,key=lambda s:-s['score'])[:16]
        if not beam:break
    if usage is not None:usage['candidates']=tried
    return found[:32]


def fit(pairs,seconds=SECONDS,limit=CANDIDATE_LIMIT,max_depth=MAX_DEPTH,beam_width=BEAM_WIDTH,trace=None):
    # Share one budget across focused and general search. This is a static
    # scene-based scheduler, not extra time or learned rule scheduling.
    from .budget import check as _budget_check
    _budget_check()
    if len(pairs)<2 or seconds<=0 or limit<=0:return []
    start=time.monotonic();eligible=True
    from .reference_relations import perceive
    try:
        for x,y in pairs:perceive(x,dict(bg=0,conn=4))
    except (ValueError,IndexError):eligible=False
    used={}
    if eligible:
        event('v22_role_focused_entered')
        found=_fit(pairs,seconds=min(0.75,seconds),limit=min(1500,limit),max_depth=max_depth,beam_width=min(8,beam_width),trace=trace,focused=True,usage=used)
        if found:return found
    remaining=seconds-(time.monotonic()-start);remaining_candidates=limit-used.get('candidates',0)
    if remaining<=0 or remaining_candidates<=0:return []
    return _fit(pairs,seconds=remaining,limit=remaining_candidates,max_depth=max_depth,beam_width=beam_width,trace=trace)


def upgrade_result(pairs,test_input,result):
    from .budget import check as _budget_check
    _budget_check()
    if not result.failure_category:return result
    event('v22_extension_entered');laws=fit(pairs);preds=[];used=[]
    for candidate in laws:
        try:prediction=candidate.predict(test_input)
        except (ValueError,KeyError,IndexError):event('v22_test_precondition_rejected');continue
        if prediction in preds:continue
        preds.append(prediction);used.append(candidate)
        if len(preds)==2:break
    if not preds:return result
    return type(result)(preds,used,'V22 composed program fits all demonstrations.',failure_category=None)
