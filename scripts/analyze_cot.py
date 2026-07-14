"""Why does CoT (slow mode) underperform on ITALIC? Decompose the fast/slow gap
using the already-saved run_eval.py result JSONs (each record has the raw `output`).

For each model we recompute predictions on BOTH the fast-prompt and cot-prompt files
with the SAME robust extractor, so the only variable left is the prompt, not the parser:
  robust(text) = extract_answer(text) or extract_answer_fast(text)

We report:
  - official score (parser as the harness used it: fast->extract_answer_fast, slow->extract_answer)
  - robust score (both files parsed identically) -> isolates prompt effect from parser effect
  - blank rate (model produced nothing parseable) -> the format/compliance tax
  - fast x slow crosstab under robust scoring -> does reasoning flip right->wrong?
"""
import glob
import json
import re
import sys
from collections import Counter

RES = sys.argv[1] if len(sys.argv) > 1 else "results"


def extract_answer_fast(output: str) -> str:
    L = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    mi = min([output.find(c) for c in L if c in output], default=-1)
    return "" if mi == -1 else output[mi]


def extract_answer(output: str) -> str:
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


def load(path):
    return {r["index"]: r for r in json.load(open(path))["results"]}


def tag_of(fname):
    b = fname.split("custom_openai_")[1]
    name = b.split("_italic_")[0]
    mode = "fast" if "_fast_" in b else "slow"
    return name, mode


files = [f for f in glob.glob(f"{RES}/*.json") if "checkpoint" not in f]
by_model = {}
for f in files:
    name, mode = tag_of(f)
    by_model.setdefault(name, {})[mode] = f

for name, modes in by_model.items():
    print("=" * 72)
    print(name)
    per = {}
    for mode, f in modes.items():
        recs = load(f)
        n = len(recs)
        off_parser = extract_answer_fast if mode == "fast" else extract_answer
        off = sum(off_parser(r["output"]) == r["answer"] for r in recs.values()) / n
        rob = sum(robust(r["output"]) == r["answer"] for r in recs.values()) / n
        blank = sum(robust(r["output"]) == "" for r in recs.values()) / n
        avglen = sum(len(r["output"]) for r in recs.values()) / n
        per[mode] = recs
        print(f"  {mode:4}  official={off:.4f}  robust={rob:.4f}  blank={blank:.4f}  avg_len={avglen:5.0f}")
    if "fast" in per and "slow" in per:
        idx = set(per["fast"]) & set(per["slow"])
        cc = cw = wc = ww = 0
        for i in idx:
            fr = robust(per["fast"][i]["output"]) == per["fast"][i]["answer"]
            sr = robust(per["slow"][i]["output"]) == per["slow"][i]["answer"]
            cc += fr and sr; cw += fr and not sr
            wc += (not fr) and sr; ww += (not fr) and not sr
        print(f"  crosstab (robust scoring, n={len(idx)}):")
        print(f"    fast✓slow✓={cc}  fast✓slow✗={cw}  (reasoning BROKE a right answer)")
        print(f"    fast✗slow✓={wc}  (reasoning FIXED a wrong answer)  fast✗slow✗={ww}")
        print(f"    net effect of reasoning: {wc - cw:+d} questions")


# qualitative: opd250 questions where fast(robust) right but slow reasoning wrong
print("=" * 72)
print("SAMPLE: opd250 fast-right but slow-wrong (reasoning talked itself out of it)")
opd = by_model.get("opd250")
if opd and "fast" in opd and "slow" in opd:
    fast, slow = load(opd["fast"]), load(opd["slow"])
    shown = 0
    for i in sorted(set(fast) & set(slow)):
        if shown >= 3:
            break
        fr = robust(fast[i]["output"]) == fast[i]["answer"]
        sp = robust(slow[i]["output"])
        if fr and sp != slow[i]["answer"]:
            q = [m for m in slow[i]["messages"] if m["role"] == "user"][-1]["content"]
            print(f"\n[gold={slow[i]['answer']} slow_pred={sp}] {q.splitlines()[-6] if q.splitlines() else ''}")
            print("  reasoning:", repr(slow[i]["output"][:400]))
            shown += 1
