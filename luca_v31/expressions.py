"""expressions.py -- RECONSTRUCTED (original not uploaded).

synthesize_scalar_expression(contexts, targets) returns the simplest
expression over integer context features that reproduces every target, or
None.  MDL order: constant -> feature -> feature+c -> c*feature -> c-feature.
Returned object exposes .name, .is_constant, .eval(ctx).
"""
from __future__ import annotations
from .diagnostics import record_exception
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional


@dataclass
class Expr:
    name: str
    fn: Callable[[Dict], Optional[int]]
    is_constant: bool = False

    def eval(self, ctx: Dict) -> Optional[int]:
        try:
            return self.fn(ctx)
        except Exception:
            record_exception('expressions.py:22')
            return None


def _original_synthesize_scalar_expression(contexts: List[Dict], targets: List) -> Optional[Expr]:
    if not contexts or len(contexts) != len(targets):
        return None
    if len(set(targets)) == 1:
        v = targets[0]
        return Expr(f"const({v})", lambda ctx, v=v: v, is_constant=True)
    keys = [k for k in contexts[0] if all(isinstance(c.get(k), int) for c in contexts)]
    for k in keys:
        if all(c[k] == t for c, t in zip(contexts, targets)):
            return Expr(k, lambda ctx, k=k: ctx.get(k))
    for k in keys:
        diffs = {t - c[k] for c, t in zip(contexts, targets)}
        if len(diffs) == 1:
            d = diffs.pop()
            return Expr(f"{k}{d:+d}", lambda ctx, k=k, d=d: ctx.get(k) + d)
        if all(c[k] for c in contexts):
            ratios = {t / c[k] for c, t in zip(contexts, targets)}
            if len(ratios) == 1:
                m = ratios.pop()
                if m == int(m):
                    m = int(m)
                    return Expr(f"{m}*{k}", lambda ctx, k=k, m=m: m * ctx.get(k))
        sums = {t + c[k] for c, t in zip(contexts, targets)}
        if len(sums) == 1:
            s = sums.pop()
            return Expr(f"{s}-{k}", lambda ctx, k=k, s=s: s - ctx.get(k))
    return None


def synthesize_scalar_expression(contexts, targets):
    original=_original_synthesize_scalar_expression(contexts,targets)
    if original is not None or not contexts or len(contexts)!=len(targets):
        return original
    keys=sorted(k for k in contexts[0] if all(isinstance(c.get(k),int) for c in contexts))
    for i,a in enumerate(keys):
        for b in keys[i+1:]:
            for symbol,op in [('+',lambda x,y:x+y),('-',lambda x,y:x-y),
                              ('reverse_subtract',lambda x,y:y-x),('*',lambda x,y:x*y),
                              ('abs_difference',lambda x,y:abs(x-y)),
                              ('min',min),('max',max)]:
                if all(op(c[a],c[b])==t for c,t in zip(contexts,targets)):
                    return Expr(f'{symbol}({a},{b})',lambda c,a=a,b=b,op=op:op(c[a],c[b]))
    return None
