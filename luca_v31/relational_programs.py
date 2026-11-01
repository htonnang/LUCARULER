"""Bounded, serializable object programs. Fitting never receives test answers.

Grammar: perceive -> select by property/relation -> edit -> layout -> D4
          -> optional fitted palette map.
Additional relational action: match object masks -> transfer template pixels.
This is an enumerated typed grammar, not arbitrary program synthesis.
"""
from collections import Counter
from functools import lru_cache
import json
import time
from .grid_world import guess_backgrounds, extract_objects, extract_holes
from .object_support import view, local_grid, selected_objects
from .primitives import Law, GEOM_OPS
from .diagnostics import event

class PreconditionError(ValueError):
    pass


def immutable(g):
    return tuple(tuple(row) for row in g)


def shape(g):
    return len(g), len(g[0])


def crop(g, bg):
    cells=[(r,c) for r,row in enumerate(g) for c,v in enumerate(row) if v!=bg]
    if not cells: raise PreconditionError('empty crop')
    r0,c0=min(r for r,c in cells),min(c for r,c in cells)
    r1,c1=max(r for r,c in cells),max(c for r,c in cells)
    return tuple(row[c0:c1+1] for row in g[r0:r1+1])


def mask(obj):
    r0,c0,r1,c1=obj.bbox
    return tuple(tuple(int((r,c) in obj.cells) for c in range(c0,c1+1)) for r in range(r0,r1+1))


def canonical_mask(obj):
    m=mask(obj)
    return min(fn(m) for fn in GEOM_OPS.values())


def unique_major(obj):
    counts=Counter(obj.multicolor.values()); most=counts.most_common()
    if len(most)>1 and most[0][1]==most[1][1]: raise PreconditionError('tied major colours')
    return most[0][0]


def select(g, bg, objs, selector):
    name=selector[0]
    if name=='all': return list(objs)
    if name in ('largest','smallest','unique_shape','repeated_shape','tallest','widest'):
        return selected_objects(objs,name)
    if name in ('unique_d4','repeated_d4'):
        counts=Counter(canonical_mask(o) for o in objs)
        return [o for o in objs if (counts[canonical_mask(o)]==1)==(name=='unique_d4')]
    if name in ('near_marker','aligned_marker','inside_marker_bbox'):
        markers=[o for o in objs if set(o.multicolor.values())=={selector[1]}]
        if len(markers)!=1: raise PreconditionError('marker must be unique')
        marker=markers[0];mr0,mc0,mr1,mc1=marker.bbox
        if name=='near_marker':
            return [o for o in objs if o.id!=marker.id and any((r+dr,c+dc) in marker.cells for r,c in o.cells for dr,dc in ((-1,0),(1,0),(0,-1),(0,1))) ]
        if name=='aligned_marker':
            return [o for o in objs if o.id!=marker.id and (o.bbox[0]<=mr1 and mr0<=o.bbox[2] or o.bbox[1]<=mc1 and mc0<=o.bbox[3])]
        return [o for o in objs if o.id!=marker.id and mr0<=o.bbox[0]<=o.bbox[2]<=mr1 and mc0<=o.bbox[1]<=o.bbox[3]<=mc1]
    chosen=[]
    for o in objs:
        r0,c0,r1,c1=o.bbox;h,w=r1-r0+1,c1-c0+1
        colors=set(o.multicolor.values())
        if name=='singleton': ok=o.area==1
        elif name=='non_singleton': ok=o.area>1
        elif name=='multicolor': ok=len(colors)>1
        elif name=='monochrome': ok=len(colors)==1
        elif name=='square': ok=h==w
        elif name=='rectangle': ok=o.area==h*w
        elif name=='border': ok=r0==0 or c0==0 or r1==len(g)-1 or c1==len(g[0])-1
        elif name=='interior': ok=r0>0 and c0>0 and r1<len(g)-1 and c1<len(g[0])-1
        elif name=='contains_color': ok=selector[1] in colors
        elif name=='symmetric_h': ok=mask(o)==GEOM_OPS['reflect_h'](mask(o))
        elif name=='symmetric_v': ok=mask(o)==GEOM_OPS['reflect_v'](mask(o))
        elif name=='has_hole': ok=bool(extract_holes(local_grid(o,bg),bg))
        else: raise PreconditionError('unknown selector')
        if ok: chosen.append(o)
    return chosen


def paint_color(obj, paint, bg, objs):
    if paint=='self': return unique_major(obj)
    if paint=='background': return bg
    if paint=='singleton':
        cs={o.color for o in objs if o.area==1 and o.id!=obj.id}
        if len(cs)!=1: raise PreconditionError('ambiguous singleton colour')
        return next(iter(cs))
    if isinstance(paint,int) and 0<=paint<=9: return paint
    raise PreconditionError('unknown paint')


@lru_cache(maxsize=4096)
def edit(g, config, selector, action, paint, layout):
    bg,objs=view(g,config)
    if not objs: raise PreconditionError('no objects')
    chosen=select(g,bg,objs,selector)
    if not chosen: raise PreconditionError('empty selection')
    out=[[bg]*len(g[0]) for _ in g] if action=='keep' or layout=='crop' else [list(row) for row in g]
    if layout=='crop':
        for obj in chosen:
            for (r,c),v in obj.multicolor.items(): out[r][c]=v
    for obj in chosen:
        r0,c0,r1,c1=obj.bbox
        if action=='keep':
            for (r,c),v in obj.multicolor.items(): out[r][c]=v
        elif action=='delete':
            for r,c in obj.cells: out[r][c]=bg
        elif action in ('recolor','fill_bbox','fill_rows','fill_cols','fill_holes'):
            col=paint_color(obj,paint,bg,objs)
            if action=='recolor': cells=obj.cells
            elif action=='fill_bbox': cells={(r,c) for r in range(r0,r1+1) for c in range(c0,c1+1) if g[r][c]==bg}
            elif action=='fill_rows':
                cells=set()
                for r in range(r0,r1+1):
                    cs=[c for rr,c in obj.cells if rr==r]
                    if cs: cells.update((r,c) for c in range(min(cs),max(cs)+1) if g[r][c]==bg)
            elif action=='fill_cols':
                cells=set()
                for c in range(c0,c1+1):
                    rs=[r for r,cc in obj.cells if cc==c]
                    if rs: cells.update((r,c) for r in range(min(rs),max(rs)+1) if g[r][c]==bg)
            else:
                cells={(r+r0,c+c0) for hole in extract_holes(local_grid(obj,bg),bg) for r,c in hole.cells if g[r+r0][c+c0]==bg}
            for r,c in cells: out[r][c]=col
        elif action in ('normalize_minor','remove_minor'):
            major=unique_major(obj)
            if len(set(obj.multicolor.values()))<2: raise PreconditionError('no minor colour')
            for (r,c),v in obj.multicolor.items():
                if v!=major: out[r][c]=major if action=='normalize_minor' else bg
        else: raise PreconditionError('unknown action')
    out=immutable(out)
    return crop(out,bg) if layout=='crop' else out


def copy_corresponding(g,spec):
    bg,objs=view(g,tuple(spec['config']))
    sources=[o for o in objs if len(set(o.multicolor.values()))>1]
    if len(sources)!=1: raise PreconditionError('one multicolour template required')
    source=sources[0]; transform=GEOM_OPS[spec['geometry']]
    template=transform(local_grid(source,bg)); footprint=transform(mask(source))
    targets=[o for o in objs if o.id!=source.id and len(set(o.multicolor.values()))==1 and mask(o)==footprint]
    if not targets: raise PreconditionError('no matching target masks')
    out=[list(row) for row in g]
    for target in targets:
        r0,c0,_,_=target.bbox
        for r,row in enumerate(footprint):
            for c,on in enumerate(row):
                if on: out[r+r0][c+c0]=template[r][c]
    return immutable(out)


def move_toward(g,spec):
    bg,objs=view(g,tuple(spec['config']))
    sources=[o for o in objs if set(o.multicolor.values())=={spec['source']}]
    targets=[o for o in objs if set(o.multicolor.values())=={spec['target']}]
    if len(sources)!=1 or len(targets)!=1: raise PreconditionError('unique source/target required')
    src,dst=sources[0],targets[0]
    sr0,sc0,sr1,sc1=src.bbox;tr0,tc0,tr1,tc1=dst.bbox
    sign=lambda x:(x>0)-(x<0)
    dr,dc=sign(tr0+tr1-sr0-sr1),sign(tc0+tc1-sc0-sc1)
    if spec['axis']=='row': dc=0
    if spec['axis']=='column': dr=0
    if dr==dc==0: raise PreconditionError('no direction')
    moved={(r+dr,c+dc):col for (r,c),col in src.multicolor.items()}
    if any(not(0<=r<len(g) and 0<=c<len(g[0])) or (g[r][c]!=bg and (r,c) not in src.cells) for r,c in moved):
        raise PreconditionError('collision or boundary')
    out=[list(row) for row in g]
    for r,c in src.cells: out[r][c]=bg
    for (r,c),col in moved.items(): out[r][c]=col
    return immutable(out)


def execute(g,spec):
    if spec['family']=='copy_correspondence': return copy_corresponding(g,spec)
    if spec['family']=='move_toward': return move_toward(g,spec)
    if spec['family']!='object_program': raise PreconditionError('unknown program family')
    out=edit(g,tuple(spec['config']),tuple(spec['selector']),spec['action'],spec['paint'],spec['layout'])
    out=GEOM_OPS[spec['geometry']](out)
    palette=spec.get('palette',{})
    if palette: out=tuple(tuple(palette.get(str(v),v) for v in row) for row in out)
    return out


def complexity(spec):
    if spec['family']=='copy_correspondence': return 4
    if spec['family']=='move_toward': return 4
    return 2+(spec['selector'][0]!='all')+(spec['layout']=='crop')+(spec['geometry']!='identity')+bool(spec.get('palette'))


def as_law(spec, origin='search'):
    # JSON is a safe, portable program description; no eval/pickle or task answers.
    spec=json.loads(json.dumps(spec))
    name='v10_'+spec['family']+'+'+spec.get('action','match')+'+'+spec.get('geometry',spec.get('axis',''))
    law=Law(name,lambda g:execute(g,spec),json.dumps(spec,sort_keys=True),0,complexity(spec))
    law.program_spec=spec
    law.origin=origin
    return law


def infer_palette(outputs, targets):
    mapping={}
    for a,b in zip(outputs,targets):
        if shape(a)!=shape(b): return None
        for ra,rb in zip(a,b):
            for u,v in zip(ra,rb):
                if u in mapping and mapping[u]!=v: return None
                mapping[u]=v
    return {str(k):v for k,v in mapping.items() if k!=v}


def fit_spec(spec,pairs):
    """Revalidate a stored *structure*, refitting its palette on current demos."""
    from .budget import check as _budget_check
    _budget_check()
    base=dict(spec); base.pop('palette',None)
    try: outputs=[execute(x,base) for x,y in pairs]
    except (PreconditionError, ValueError, IndexError, KeyError): return None
    targets=[y for x,y in pairs]
    if outputs==targets: return base
    if not spec.get('palette'): return None
    palette=infer_palette(outputs,targets)
    if palette is None: return None
    fitted=dict(base,palette=palette)
    return fitted if palette_loo(outputs,targets) else None


def palette_loo(outputs,targets):
    # Refit colour parameters while holding out each demonstration.
    # The structural program is fixed: this is NOT full search-level LOO.
    if len(outputs)<3: return True
    for i in range(len(outputs)):
        mp=infer_palette(outputs[:i]+outputs[i+1:],targets[:i]+targets[i+1:])
        if mp is None: return False
        pred=tuple(tuple(mp.get(str(v),v) for v in row) for row in outputs[i])
        if pred!=targets[i]: return False
    return True


def selectors(pairs):
    names=['all','non_singleton','singleton','multicolor','monochrome','square','rectangle',
           'symmetric_h','symmetric_v','has_hole','largest','smallest','unique_shape',
           'repeated_shape','unique_d4','repeated_d4','tallest','widest','border','interior']
    yield from ((n,) for n in names)
    common=set.intersection(*({v for row in x for v in row} for x,y in pairs))
    for color in sorted(common):
        for relation in ('contains_color','near_marker','aligned_marker','inside_marker_bbox'):
            yield (relation,color)


def _fit_first(pairs,seconds=3.0,limit=30000,configs_override=None,collect_pool=False):
    from .budget import check as _budget_check
    _budget_check()
    if len(pairs)<2: return []
    start=time.monotonic(); attempts=0;found=[];seen=set()
    targets=[y for x,y in pairs];same=all(shape(x)==shape(y) for x,y in pairs)
    configs=[(0,4,True),(0,8,True),(0,4,False),(0,8,False),(1,4,True),(1,8,True),(1,4,False),(1,8,False)]
    if configs_override is not None: configs=list(configs_override)
    if not configs: return []
    def expired(): return attempts>=limit or len(found)>=32 or time.monotonic()-start>=seconds
    def accept(spec):
        nonlocal attempts
        attempts+=1
        try:
            if all(execute(x,spec)==y for x,y in pairs):
                found.append(as_law(spec));event('v10_exact_program',spec['family']);return True
        except (ValueError,IndexError,KeyError): event('v10_precondition_rejected')
        return False
    # Genuine input-object correspondence, with colour-preserving template transfer.
    if same:
        for cfg in (configs[:2] if configs_override is None else [c for c in configs if c[2]]):
            for geom in GEOM_OPS:
                accept(dict(family='copy_correspondence',config=cfg,geometry=geom))
        common=sorted(set.intersection(*({v for row in x for v in row} for x,y in pairs)))
        for source in (common if configs_override is None else []):
            for target in common:
                if source==target: continue
                for axis in ('both','row','column'):
                    accept(dict(family='move_toward',config=(0,4,False),source=source,target=target,axis=axis))
        if found and not collect_pool: return sorted(found,key=lambda law:law.op_count)[:(32 if collect_pool else 8)]
    paints=['self','singleton']+sorted({v for y in targets for row in y for v in row})
    actions=[('keep',None),('delete',None),('normalize_minor',None),('remove_minor',None)]
    if same:
        actions += [(action,paint) for action in ('recolor','fill_bbox','fill_rows','fill_cols','fill_holes') for paint in paints]
    for cfg in configs:
        selections_seen=set()
        for selector in selectors(pairs):
            if expired(): break
            try:
                selection_signature=tuple(tuple(o.id for o in select(x,*view(x,cfg),selector)) for x,y in pairs)
                if any(not v for v in selection_signature):continue
            except (ValueError,IndexError,KeyError):continue
            # Equivalent demonstration states still can generalize differently;
            # keep first selector by declared structural ordering.
            if selection_signature in selections_seen and not collect_pool: continue
            selections_seen.add(selection_signature)
            for action,paint in actions:
                if expired(): break
                for layout in (('original',) if same else ('crop',)):
                    base=dict(family='object_program',config=cfg,selector=selector,action=action,paint=paint,layout=layout,geometry='identity')
                    attempts+=1
                    try: outputs=[execute(x,base) for x,y in pairs]
                    except (ValueError,IndexError,KeyError):continue
                    signature=tuple(outputs)
                    if signature in seen and not collect_pool:continue
                    seen.add(signature)
                    for geom,fn in GEOM_OPS.items():
                        converted=[fn(o) for o in outputs]
                        if any(shape(a)!=shape(b) for a,b in zip(converted,targets)):continue
                        spec=dict(base,geometry=geom)
                        if converted==targets:
                            found.append(as_law(spec));event('v10_exact_program','object_program')
                        elif not same:
                            palette=infer_palette(converted,targets)
                            if palette and palette_loo(converted,targets):
                                spec['palette']=palette;found.append(as_law(spec));event('v10_exact_program','palette_composition')
                    if found and not collect_pool: break
                if found and not collect_pool:break
            if found and not collect_pool:break
        if (found and not collect_pool) or expired(): break
    event('v10_search_attempts',attempts)
    if expired():event('v10_search_budget_exhausted')
    return sorted(found,key=lambda law:law.op_count)[:(32 if collect_pool else 8)]


def clear_cache():
    edit.cache_clear()


def fit(pairs,seconds=3.0,limit=30000,configs_override=None):
    """Retain the first search result, then revalidate its representation variants.

    This bounded completion does not enumerate unrelated action structures.
    Demonstration-equivalent configurations must not be merged before prediction.
    """
    from .budget import check as _budget_check
    _budget_check()
    first=_fit_first(pairs,seconds,limit,configs_override)
    if not first:return first
    configs=configs_override if configs_override is not None else [
        (rank,conn,multi) for rank in (0,1) for conn in (4,8) for multi in (True,False)]
    out=list(first);seen={json.dumps(l.program_spec,sort_keys=True) for l in out}
    for law in first[:8]:
        spec=getattr(law,'program_spec',{})
        if 'config' not in spec:continue
        for cfg in configs:
            alt=dict(spec,config=cfg);key=json.dumps(alt,sort_keys=True)
            if key in seen:continue
            seen.add(key)
            try:
                if all(execute(x,alt)==y for x,y in pairs):
                    out.append(as_law(alt));event('v26_representation_alternative')
            except (ValueError,IndexError,KeyError):continue
            if len(out)>=32:return out
    return out
