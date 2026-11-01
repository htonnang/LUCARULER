"""Evidence rule for learned table entries (shared by every fitter that learns a table).

policy 'demos' (default until OPT184): an entry must be seen in >= 2 demonstrations.
policy 'occ'   (OPT185 candidate): an entry must be seen >= 2 times in total, in any demonstrations
               (two objects of the same kind in one demonstration also count), so tasks with only
               2 demonstrations are not blocked when one of them shows the case twice.
Chosen by the LUCA_EVIDENCE environment variable so both can be measured on the same code.
"""
import os

POLICY = os.environ.get("LUCA_EVIDENCE", "demos")
# OPT187: inside the leave-one-demonstration-out refit the evidence rule is relaxed to 'seen once'.
# The refit only has to PREDICT the held-out demonstration; an entry it never saw makes that prediction
# fail on its own, so the cross-check stays strict while 3-demo tasks are no longer blocked.
REFIT = [False]


def enough(demos, occurrences):
    """demos: set (or count) of demonstration indices showing the entry; occurrences: total count."""
    n_demos = demos if isinstance(demos, int) else len(demos)
    if REFIT[0]:
        return occurrences >= 1
    if POLICY == "occ":
        return occurrences >= 2
    return n_demos >= 2
