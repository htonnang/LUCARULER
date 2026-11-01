"""Split an ARC 'challenges' JSON (and optionally its 'solutions' JSON) into one file per task,
the format LUCA and the hybrid read.

Usage: python3 hybrid/split_tasks.py CHALLENGES.json OUT_DIR [SOLUTIONS.json]
"""
import json, os, sys


def main():
    ch = json.load(open(sys.argv[1]))
    out = sys.argv[2]
    sol = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else {}
    os.makedirs(out, exist_ok=True)
    for tid, t in ch.items():
        for i, tt in enumerate(t["test"]):
            if tid in sol:
                tt["output"] = sol[tid][i]
        json.dump(t, open(os.path.join(out, tid + ".json"), "w"))
    print(len(ch), "tasks written to", out)


if __name__ == "__main__":
    main()
