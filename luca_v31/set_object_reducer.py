"""New replacement: group objects by shape/color and filter or extract sets."""
from .object_support import CONFIGS, SELECTORS, view, selected_objects, render, accepted
MISSING=False

def fit_all_set_reducer_traced(pairs):
    """OPT173 (DSL port 3): retired. Object selection now lives in the LUCA DSL (select_objects),
    where it composes with other operations; the original fitter is in docs/retired/."""
    return []


def _retired_fit_all_set_reducer_traced(pairs):
    from .budget import check as _budget_check
    _budget_check()
    laws=[]
    for config in CONFIGS:
        if any(not view(x,config)[1] for x,_ in pairs):continue
        for selector in SELECTORS:
            for mode in ('keep','delete','crop'):
                def predict(grid,config=config,selector=selector,mode=mode):
                    bg,objs=view(grid,config);selected=selected_objects(objs,selector)
                    if mode=='delete':
                        ids={o.id for o in selected};selected=[o for o in objs if o.id not in ids]
                    return render(grid,bg,selected,crop=mode=='crop')
                law=accepted('set_'+mode+'_'+selector+'_'+str(config),predict,pairs,
                             f'SELECT({selector}) -> {mode}; view={config}',ops=2)
                if law:laws.append((law,fit_all_set_reducer))
    return laws

def fit_all_set_reducer(pairs):return [law for law,_ in fit_all_set_reducer_traced(pairs)]
