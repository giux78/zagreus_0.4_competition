"""Per-domain net effect of reasoning (slow vs fast prompt, robust scoring)."""
import glob
import json
import re
import sys
from collections import defaultdict


def extract_answer_fast(output):
    L = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    mi = min([output.find(c) for c in L if c in output], default=-1)
    return "" if mi == -1 else output[mi]


def extract_answer(output):
    def _find(pattern, text, ic=True):
        flags = re.DOTALL | (re.IGNORECASE if ic else 0)
        m = re.search(pattern, text, flags)
        if m:
            a = re.sub(r"[è:)(?+\-,;.]", "", m.group(1)).strip()
            a = re.sub(r"^(?:sarà\s+la\s+|la\s+)?", "", a).strip()
            return extract_answer_fast(a) if a else ""
        return ""
    dp = r"Risposta:\s*(.*?)\s*(?=\n[A-Z]\)|\Z)"
    fb = [
        r"quindi, la risposta è\s*(.*?)\s*(?=\n[A-Z]\)|\Z)",
        r"risposta\s*(?:corretta|giusta|appropriata|esatta|migliore|ottimale|finale|definitiva)?\s*[:è]*\s*(.*?)\s*(?=\n[A-Z]\)|\Z)",
        r"risposta\s*più\s*(?:corretta|appropriata)\s*[:è]*\s*(.*?)\s*(?=\n[A-Z]\)|\Z)",
        r"(?:soluzione|opzione|scelta|alternativa)\s*(?:corretta)?\s*[:è]*\s*(.*?)\s*(?=\n[A-Z]\)|\Z)",
        r"(?:quindi|in\s*conclusione,?)?\s*(?:la\s*)?risposta\s*è\s*(.*?)\s*(?=\n[A-Z]\)|\Z)",
        r"(?:la\s*)?(?:risposta|opzione|scelta)\s*(?:corretta|giusta|esatta)\s*è\s*(?:la\s*)?(?:lettera\s*)?([A-Z])",
    ]
    a = _find(dp, output, ic=False)
    if a:
        return a
    if "nessuna delle opzioni" in output.lower():
        return ""
    for p in fb:
        a = _find(p, output)
        if a:
            return a
    return ""


def robust(t):
    return extract_answer(t) or extract_answer_fast(t)


italic = [json.loads(l) for l in open("italic.jsonl")]
macro = {i: r["macro_category"] for i, r in enumerate(italic)}


def files(name):
    fs = [f for f in glob.glob(f"results/*{name}*.json") if "checkpoint" not in f]
    return [f for f in fs if "_fast_" in f][0], [f for f in fs if "_fast_" not in f][0]


for name in sys.argv[1:] or ["opd250", "nesso3b"]:
    ff, sf = files(name)
    fast = {r["index"]: r for r in json.load(open(ff))["results"]}
    slow = {r["index"]: r for r in json.load(open(sf))["results"]}
    agg = defaultdict(lambda: [0, 0, 0])  # fixed, broke, n
    for i in set(fast) & set(slow):
        fr = robust(fast[i]["output"]) == fast[i]["answer"]
        sr = robust(slow[i]["output"]) == slow[i]["answer"]
        m = macro[i]
        agg[m][2] += 1
        if not fr and sr:
            agg[m][0] += 1
        if fr and not sr:
            agg[m][1] += 1
    print(f"=== {name}: reasoning net effect by domain (helps=+, hurts=-)")
    for m, (fx, bk, n) in sorted(agg.items(), key=lambda x: (x[1][0] - x[1][1]) / x[1][2]):
        print(f"   {m:<26} {(fx - bk) / n * 100:+6.1f}%  (fixed {fx}, broke {bk}, n={n})")
    print()
