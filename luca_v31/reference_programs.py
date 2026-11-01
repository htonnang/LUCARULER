"""Bounded synthesis of named-role, branching grid programs.

Programs are JSON DAGs with checked operand types and no task identifiers.
References are resolved anew in each input. Selection ambiguity rejects a
candidate; fitting sees demonstration pairs, never held-out answers.
"""
from collections import Counter
from itertools import product
import json
from .backgrounds import resolve
from .grid_world import extract_objects
from .primitives import GEOM_OPS, Law
from .diagnostics import event

SCHEMA = 'luca-role-dag-1'

def imm(g): return tuple(tuple(row) for row in g)
def shape(g): return len(g), len(g[0])
def need(condition, message):
    if not condition: raise ValueError(message)
def unique(values):
    need(len(values) == 1, 'reference is absent or ambiguous')
    return values[0]

def predicate(o, name):
    r0,c0,r1,c1=o.bbox
    if name=='singleton': return o.area==1
    if name=='legend': return r1-r0==c1-c0==1 and o.area==4 and len(set(o.multicolor.values()))==4
    if name=='frame':
        return r1-r0>=3 and c1-c0>=3 and o.cells=={(r,c) for r in range(r0,r1+1) for c in range(c0,c1+1) if r in (r0,r1) or c in (c0,c1)}
    raise ValueError('unknown object predicate')

# Signatures also make malformed graph references and type errors explicit.
SIGNATURES = {
 'input': ((), 'Grid'), 'background': (('Grid',), 'Color'),
 'literal': ((), 'Color'), 'objects': (('Grid','Color'), 'Objects'),
 'filter': (('Objects',), 'Objects'), 'unique': (('Objects',), 'Entity'),
 'remaining': (('Objects','Entity'), 'Objects'), 'cells': (('Entity',), 'Mask'),
 'object_color': (('Entity',), 'Color'), 'geometry': (('Grid',), 'Grid'),
 'equal': (('Grid','Grid'), 'Mask'), 'foreground': (('Grid','Color'), 'Mask'),
 'nonfixed': (('Grid',), 'Mask'), 'intersect': (('Mask','Mask'), 'Mask'),
 'border': (('Grid',), 'Mask'), 'blank': (('Grid','Color'), 'Grid'),
 'paint': (('Grid','Mask','Color'), 'Grid'), 'paint_empty': (('Grid','Mask','Color','Color'), 'Grid'), 'hstack': (('Grid','Grid'), 'Grid'),
 'vstack': (('Grid','Grid'), 'Grid'), 'legend_map': (('Grid','Entity'), 'Palette'),
 'recolor': (('Grid','Palette','Mask'), 'Grid'),
 'rows': (('Grid',), 'Rows'), 'full_rows': (('Rows','Color'), 'Rows'),
 'unique_row': (('Rows',), 'Row'), 'prefix_complete': (('Grid','Row','Color'), 'Grid'),
 'interior': (('Entity',), 'Mask'),
 'quadrant_field': (('Grid','Entity','Objects','Color'), 'Grid'),
 'where': (('Grid','Mask','Grid'), 'Grid'),
 'extensions': (('Objects',), 'Mask'),
 'color_cells': (('Grid','Color'), 'Mask'), 'rail_vector': (('Mask','Mask'), 'Vector'),
 'translate': (('Grid','Mask','Vector','Color'), 'Grid'),
 'template_mask': (('Grid','Color'), 'Mask'),
 'repeat_at': (('Grid','Mask','Color'), 'Grid'),
}

def validate(spec):
    need(spec.get('schema')==SCHEMA, 'unknown program schema')
    need(0<len(spec['nodes'])<=32, 'node bound exceeded')
    types={}
    for n in spec['nodes']:
        need(n['id'] not in types, 'duplicate role')
        need(n['op'] in SIGNATURES, 'unknown operator')
        args=n.get('args',[]); signature,out=SIGNATURES[n['op']]
        need(len(args)==len(signature), 'operand count mismatch')
        need(all(types.get(a)==t for a,t in zip(args,signature)), 'forward reference or operand type mismatch')
        types[n['id']]=out
    need(types.get(spec['return'])=='Grid', 'program must return a grid')
    return spec


def operate(op,a,p,g):
    if op=='input': return g
    if op=='background': return resolve(a[0],p['policy'])
    if op=='literal':
        need(isinstance(p['value'],int) and 0<=p['value']<=9, 'invalid color');return p['value']
    if op=='objects': return extract_objects(a[0],a[1],p.get('connectivity',4),p.get('multicolor',True))
    if op=='filter': return [o for o in a[0] if predicate(o,p['predicate'])]
    if op=='unique':
        obj=unique(a[0])
        if 'predicate' in p:need(predicate(obj,p['predicate']),'reference predicate failed')
        return obj
    if op=='remaining': return [o for o in a[0] if o is not a[1]]
    if op=='cells': return set(a[0].cells)
    if op=='object_color':
        colors=set(a[0].multicolor.values()) if a[0].multicolor else {a[0].color}
        return unique(list(colors))
    if op=='geometry': return GEOM_OPS[p['geometry']](a[0])
    if op=='equal':
        x,y=a;need(shape(x)==shape(y),'comparison shape mismatch')
        return {(r,c) for r,row in enumerate(x) for c,v in enumerate(row) if v==y[r][c]}
    if op=='foreground': return {(r,c) for r,row in enumerate(a[0]) for c,v in enumerate(row) if v!=a[1]}
    if op=='nonfixed':
        h,w=shape(a[0]);coords=tuple(tuple((r,c) for c in range(w)) for r in range(h))
        moved=GEOM_OPS[p['geometry']](coords);need(shape(moved)==(h,w),'geometry changes shape')
        return {(r,c) for r in range(h) for c in range(w) if moved[r][c]!=(r,c)}
    if op=='intersect': return a[0]&a[1]
    if op=='border':
        h,w=shape(a[0]);return {(r,c) for r in range(h) for c in range(w) if r in (0,h-1) or c in (0,w-1)}
    if op=='blank': return tuple(tuple(a[1] for _ in row) for row in a[0])
    if op in ('paint','paint_empty'):
        x,mask,color=a[:3];h,w=shape(x);need(all(0<=r<h and 0<=c<w for r,c in mask),'paint out of bounds')
        if op=='paint_empty':need(all(x[r][c] in (a[3],color) for r,c in mask),'paint collision')
        return tuple(tuple(color if (r,c) in mask else v for c,v in enumerate(row)) for r,row in enumerate(x))
    if op=='hstack':
        need(len(a[0])==len(a[1]),'row mismatch');return tuple(x+y for x,y in zip(*a))
    if op=='vstack':
        need(len(a[0][0])==len(a[1][0]),'column mismatch');return a[0]+a[1]
    if op=='legend_map':
        x,o=a;r,c,rr,cc=o.bbox;need(predicate(o,'legend'),'invalid legend')
        values=((x[r][c],x[r][cc]),(x[rr][c],x[rr][cc]))
        if p['axis']=='vertical':values=tuple(zip(*values))
        return {u:v for x,y in values for u,v in ((x,y),(y,x))}
    if op=='recolor':
        x,mapping,protected=a
        return tuple(tuple(v if (r,c) in protected else mapping.get(v,v) for c,v in enumerate(row)) for r,row in enumerate(x))
    if op=='rows': return list(enumerate(a[0]))
    if op=='full_rows': return [(r,row) for r,row in a[0] if all(v!=a[1] for v in row)]
    if op=='unique_row': return unique(a[0])
    if op=='prefix_complete':
        x,(guide_index,guide),bg=a;out=[list(row) for row in x]
        for r,row in enumerate(x):
            if r==guide_index or all(v==bg for v in row):continue
            n=next((c for c,v in enumerate(row) if v==bg),len(row))
            need(n>0 and all(v==bg for v in row[n:]),'target is not a foreground prefix')
            mapping={}
            for source,target in zip(guide[:n],row[:n]):
                need(source not in mapping or mapping[source]==target,'inconsistent local mapping');mapping[source]=target
            need(all(v in mapping for v in guide),'reference palette not covered by prefix')
            out[r]=[mapping[v] for v in guide]
        return imm(out)
    if op=='interior':
        r,c,rr,cc=a[0].bbox;return {(i,j) for i in range(r+1,rr) for j in range(c+1,cc)}
    if op=='quadrant_field':
        x,frame,markers,bg=a;r0,c0,r1,c1=frame.bbox
        need((r1-r0-1)%2==0 and (c1-c0-1)%2==0,'odd frame interior')
        need(len(markers)==2 and all(o.area==1 for o in markers),'two singleton markers required')
        colors={}
        for o in markers:
            r,c=next(iter(o.cells));need(not(r0<=r<=r1 and c0<=c<=c1),'marker must be external')
            key=(int(2*r>r0+r1)+int(2*c>c0+c1))%2
            need(key not in colors,'marker roles overlap');colors[key]=o.color
        return tuple(tuple(colors[(int(2*r>r0+r1)+int(2*c>c0+c1))%2] for c in range(len(x[0]))) for r in range(len(x)))
    if op=='where':
        x,mask,y=a;need(shape(x)==shape(y),'branch shape mismatch')
        return tuple(tuple(y[r][c] if (r,c) in mask else v for c,v in enumerate(row)) for r,row in enumerate(x))
    if op=='extensions':
        need(bool(a[0]),'no target objects');mask=set()
        for o in a[0]:
            r,c,rr,cc=o.bbox
            n=len(set(o.multicolor.values())) if p['feature']=='color_count' else (rr-r+1 if p['feature']=='height' else cc-c+1)
            mask|={(i,j) for i in range(rr+1,rr+n+1) for j in range(c,cc+1)}
        return mask
    if op=='color_cells': return {(r,c) for r,row in enumerate(a[0]) for c,v in enumerate(row) if v==a[1]}
    if op=='rail_vector':
        markers,body=a;need(len(markers)>=2 and bool(body),'rail and body required')
        vertical=len({c for r,c in markers})==1;horizontal=len({r for r,c in markers})==1
        need(vertical!=horizontal,'unique rail axis required');axis=0 if vertical else 1
        coords=sorted(q[axis] for q in markers);gaps={y-x for x,y in zip(coords,coords[1:])}
        step=unique(list(gaps));direction=p['direction']
        if direction=='longer_side':
            center=(min(q[axis] for q in body)+max(q[axis] for q in body))/2
            delta=sum(q>center for q in coords)-sum(q<center for q in coords)
            need(delta!=0,'rail direction tied');direction=1 if delta>0 else -1
        step*=direction;return (step,0) if vertical else (0,step)
    if op=='translate':
        x,cells,(dr,dc),bg=a;h,w=shape(x);out=[list(row) for row in x]
        for r,c in cells:out[r][c]=bg
        for r,c in cells:
            rr,cc=r+dr,c+dc;need(0<=rr<h and 0<=cc<w,'translation out of bounds')
            need(out[rr][cc]==bg,'translation collision');out[rr][cc]=x[r][c]
        return imm(out)
    if op=='template_mask':
        x,bg=a;selector=p['selector'];counts=Counter(v for row in x for v in row)
        if selector in ('most_frequent','rarest'):
            score=max(counts.values()) if selector=='most_frequent' else min(counts.values())
            selector=unique([v for v,n in counts.items() if n==score])
        return {(r,c) for r,row in enumerate(x) for c,v in enumerate(row) if ((v!=bg if selector=='foreground' else v==selector)!=p.get('invert',False))}
    if op=='repeat_at':
        x,mask,bg=a;h,w=shape(x);need(h*h<=30 and w*w<=30,'expanded canvas too large')
        return tuple(tuple(x[r%h][c%w] if (r//h,c//w) in mask else bg for c in range(w*w)) for r in range(h*h))
    raise ValueError('unknown operation')


def execute(g,spec):
    validate(spec);g=imm(g);values={}
    for node in spec['nodes']:
        values[node['id']]=operate(node['op'],[values[k] for k in node.get('args',[])],node.get('params',{}),g)
    result=values[spec['return']]
    need(0<len(result)<=30 and 0<len(result[0])<=30,'invalid output extent')
    need(all(len(row)==len(result[0]) and all(isinstance(v,int) and 0<=v<=9 for v in row) for row in result),'invalid output grid')
    return result

class Builder:
    def __init__(self,family,bg=None):
        self.spec=dict(schema=SCHEMA,family=family,nodes=[],**{'return':'canvas'})
        self.add('canvas','input')
        if bg is not None:self.add('background','background','canvas',policy=bg)
    def add(self,role,op,*args,**params):
        self.spec['nodes'].append(dict(id=role,op=op,args=list(args),params=params));self.spec['return']=role
        return role
    def objects(self,multicolor=True): self.add('objects','objects','canvas','background',multicolor=multicolor)
    def reference(self,pred):
        self.add('references','filter','objects',predicate=pred);self.add('reference','unique','references')
    def law(self):
        spec=json.loads(json.dumps(validate(self.spec)))
        law=Law('v12_'+spec['family'],lambda g:execute(g,spec),json.dumps(spec,sort_keys=True),op_count=len(spec['nodes']))
        law.program_spec=spec;law.origin='v12_reference_graph';return law


def candidates(pairs):
    """Bounded schema expansion. Every branch and parameter is demo-derived.

    Stable family/parameter order is the tie-breaker. Equivalent demo behavior
    is NOT grounds to discard programs: their test predictions may differ.
    """
    same=all(shape(x)==shape(y) for x,y in pairs)
    colors=sorted({v for x,y in pairs for grid in (x,y) for row in grid for v in row})
    paints=sorted({v for x,y in pairs for row in y for v in row})
    common=sorted(set.intersection(*({v for row in x for v in row} for x,y in pairs)))
    # Literals supported by outputs remain legal even when absent from an input.
    bgs=['color:'+str(c) for c in colors]+['corners','border',0,1]
    if same:
        for bg in bgs:
            for orient in ('identity','transpose'):
                b=Builder('guide_prefix',bg);b.add('working','geometry','canvas',geometry=orient)
                b.add('rows','rows','working');b.add('guides','full_rows','rows','background');b.add('guide','unique_row','guides')
                b.add('completed','prefix_complete','working','guide','background');b.add('result','geometry','completed',geometry=orient);yield b
            for axis in ('horizontal','vertical'):
                b=Builder('legend_substitution',bg);b.objects();b.reference('legend')
                b.add('palette','legend_map','canvas','reference',axis=axis);b.add('protected','cells','reference')
                b.add('result','recolor','canvas','palette','protected');yield b
            b=Builder('reference_border',bg);b.objects();b.reference('singleton')
            # Exactly one object, not one singleton amid other foreground objects.
            b.spec['nodes'][-1]['args']=['objects'];b.spec['nodes'][-1]['params']={'predicate':'singleton'}
            b.add('color','object_color','reference');b.add('mask','border','canvas');b.add('empty','blank','canvas','background')
            b.add('result','paint','empty','mask','color');yield b
            b=Builder('frame_quadrants',bg);b.objects(False);b.reference('frame')
            b.add('markers','remaining','objects','reference');b.add('interior','interior','reference')
            b.add('field','quadrant_field','canvas','reference','markers','background')
            b.add('result','where','canvas','interior','field');yield b
            for geom in list(GEOM_OPS)[1:]:
                for paint in paints:
                    b=Builder('correspondence_paint',bg);b.add('partner','geometry','canvas',geometry=geom)
                    b.add('matches','equal','canvas','partner');b.add('foreground','foreground','canvas','background')
                    b.add('distinct','nonfixed','canvas',geometry=geom);b.add('selected','intersect','matches','foreground')
                    b.add('mask','intersect','selected','distinct');b.add('color','literal',value=paint)
                    b.add('result','paint','canvas','mask','color');yield b
            for feature in ('color_count','height','width'):
                for paint in paints:
                    b=Builder('feature_extension',bg);b.objects();b.add('mask','extensions','objects',feature=feature)
                    b.add('color','literal',value=paint);b.add('result','paint_empty','canvas','mask','color','background');yield b
            for body in common:
                for marker in common:
                    if body==marker or bg=='color:'+str(body) or bg=='color:'+str(marker):continue
                    for direction in (1,-1,'longer_side'):
                        b=Builder('rail_translation',bg);b.add('body_color','literal',value=body);b.add('marker_color','literal',value=marker)
                        b.add('body','color_cells','canvas','body_color');b.add('markers','color_cells','canvas','marker_color')
                        b.add('vector','rail_vector','markers','body',direction=direction)
                        b.add('result','translate','canvas','body','vector','background');yield b
    # Independent D4 branches inferred from each demonstration quadrant.
    if all(shape(y)==(2*len(x),2*len(x[0])) for x,y in pairs):
        choices=[]
        for qr,qc in ((0,0),(0,1),(1,0),(1,1)):
            legal=[]
            for geom,fn in GEOM_OPS.items():
                if all(fn(x)==tuple(row[qc*len(x[0]):(qc+1)*len(x[0])] for row in y[qr*len(x):(qr+1)*len(x)]) for x,y in pairs):legal.append(geom)
            choices.append(legal)
        for i,geoms in enumerate(product(*choices)):
            if i>=128:break
            b=Builder('independent_view_assembly')
            for k,geom in enumerate(geoms):b.add('view'+str(k),'geometry','canvas',geometry=geom)
            b.add('top','hstack','view0','view1');b.add('bottom','hstack','view2','view3');b.add('result','vstack','top','bottom');yield b
    if all(shape(y)==(len(x)**2,len(x[0])**2) for x,y in pairs):
        for bg in bgs:
            for selector in ['foreground','most_frequent','rarest']+common:
                for invert in (False,True):
                    b=Builder('reference_template',bg);b.add('locations','template_mask','canvas','background',selector=selector,invert=invert)
                    b.add('result','repeat_at','canvas','locations','background');yield b


def fit(pairs,limit=12000):
    from .budget import check as _budget_check
    _budget_check()
    if len(pairs)<2:return []
    found=[];seen=set()
    for i,builder in enumerate(candidates(pairs)):
        if i>=limit:event('v12_candidate_budget_exhausted');break
        event('v12_candidate_tried',builder.spec['family'])
        try:
            law=builder.law()
            if all(law.predict(x)==y for x,y in pairs):
                key=json.dumps(law.program_spec,sort_keys=True)
                if key not in seen:
                    seen.add(key);found.append(law);event('v12_exact_candidate',builder.spec['family'])
                if len(found)>=128:event('v12_candidate_pool_full');break
        except (ValueError,KeyError,IndexError):event('v12_precondition_rejected')
    return found


def upgrade_result(pairs,test_input,result):
    from .budget import check as _budget_check
    _budget_check()
    if not result.failure_category:return result
    event('v12_extension_entered');laws=fit(pairs);preds=[];used=[]
    for law in laws:
        try:prediction=law.predict(test_input)
        except (ValueError,KeyError,IndexError):event('v12_test_precondition_rejected');continue
        if prediction in preds:continue
        preds.append(prediction);used.append(law)
        if len(preds)==2:break
    if not preds:return result
    return type(result)(preds,used,'V12 named-reference graph fits every demonstration.',failure_category=None)
