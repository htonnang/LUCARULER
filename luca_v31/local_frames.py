"""Bounded local-coordinate programs. Parameters are fitted on demonstrations only.

No task IDs, demonstration tables or test labels are consulted. Boundary behavior
is explicit. This finite grammar is not a general physical simulator.
"""
import json
from itertools import product
from .primitives import Law
from .diagnostics import event

def need(ok):
    if not ok:raise ValueError('local frame precondition')

def imm(g):return tuple(tuple(row) for row in g)

def execute(g,spec):
    bg=spec['background'];h,w=len(g),len(g[0]);out=[list(row) for row in g]
    if spec['operation']=='align_rows':
        active=[r for r,row in enumerate(g) if any(v!=bg for v in row)]
        need(len(active)>=2)
        reference=active[0] if spec['reference']=='first' else active[-1]
        cols=[c for c,v in enumerate(g[reference]) if v!=bg]
        need(len(cols)>=2)
        anchor=cols[0] if spec['side']=='left' else cols[-1]
        color=g[reference][anchor]
        for r in active:
            hits=[c for c,v in enumerate(g[r]) if v==color];need(len(hits)==1)
            delta=anchor-hits[0];out[r]=[bg]*w
            for c,v in enumerate(g[r]):
                if v!=bg:
                    need(0<=c+delta<w);out[r][c+delta]=v
    elif spec['operation']=='dominoes':
        direction=spec['direction'];bars=[]
        for c in range(w):
            rows=[r for r in range(h) if g[r][c]!=bg]
            if not rows:continue
            need(rows==list(range(min(rows),max(rows)+1)))
            need(len({g[r][c] for r in rows})==1)
            bars.append((c,g[rows[0]][c],rows))
        need(len(bars)>=2 and len({max(rs) for c,color,rs in bars})==1)
        need(any(color==spec['trigger'] for c,color,rs in bars))
        base=max(bars[0][2]);reach=None
        for c,color,rows in sorted(bars,reverse=direction<0):
            if color==spec['trigger'] or (reach is not None and direction*c<=reach):
                end=c+direction*(len(rows)-1)
                if spec['boundary']=='reject':need(0<=end<w)
                for r in rows:out[r][c]=bg
                for j in range(len(rows)):
                    cc=c+direction*j
                    if 0<=cc<w:out[base][cc]=color
                reach=direction*c+len(rows)
            else:reach=None
    else:raise ValueError('unknown local operation')
    return imm(out)

def as_law(spec):
    encoded=json.dumps(spec,sort_keys=True)
    law=Law('local_frame_'+spec['operation'],lambda g:execute(g,spec),encoded,4,1)
    law.program_spec=spec;law.origin='v26_local_frames';return law

def fit(pairs):
    from .budget import check as _budget_check
    _budget_check()
    if len(pairs)<2 or any((len(x),len(x[0]))!=(len(y),len(y[0])) for x,y in pairs):return []
    colors=sorted(set.intersection(*({v for row in x for v in row} for x,y in pairs)))
    found=[]
    for bg in colors:
        specs=[dict(schema='luca-local-frame-1',operation='align_rows',background=bg,reference=ref,side=side)
               for ref,side in product(('first','last'),('left','right'))]
        specs += [dict(schema='luca-local-frame-1',operation='dominoes',background=bg,trigger=trigger,direction=direction,boundary=boundary)
                  for trigger,direction,boundary in product(colors,(1,-1),('reject','clip')) if trigger!=bg]
        for spec in specs:
            try:
                if all(execute(x,spec)==y for x,y in pairs):found.append(as_law(spec))
            except ValueError:continue
    event('v26_local_exact_fits',len(found));return found

def upgrade_result(pairs,test_input,result):
    from .budget import check as _budget_check
    _budget_check()
    if not result.failure_category:return result
    predictions=[];laws=[]
    for law in fit(pairs):
        try:prediction=law.predict(test_input)
        except ValueError:continue
        if prediction in predictions:continue
        predictions.append(prediction);laws.append(law)
        if len(predictions)==2:break
    if not predictions:return result
    return type(result)(predictions,laws,'Local-frame program fits all demonstrations.',failure_category=None)
