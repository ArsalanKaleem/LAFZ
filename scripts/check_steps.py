import json, numpy as np
from harf.tokenizer.common import load_tokenizer
from harf.analysis.extra import rep4

base = load_tokenizer("Qwen/Qwen2.5-1.5B")
bins = [0, 30, 40, 50, 60, 75, 1000]
for lang, code in (("ur", "urd_Arab"), ("sd", "snd_Arab")):
    ext = load_tokenizer(f"runs/Qwen2.5-1.5B/{lang}/D/tokenizer")
    print(f"\n{lang}: loop rate by number of tokens needed to write the reference (n in brackets)")
    print("steps".ljust(16) + "".join(f"{bins[i]}-{bins[i+1]}".rjust(14) for i in range(len(bins) - 1)))
    for cond, tok in (("baseline", base), ("baseline_lora", base), ("D", ext), ("D_mean", ext)):
        try:
            rows = [json.loads(l) for l in open(f"runs/Qwen2.5-1.5B/{lang}/eval/flores_{cond}_eng_Latn-{code}.jsonl", encoding="utf-8")]
        except FileNotFoundError:
            continue
        steps = np.array([len(tok.encode(r["ref"], add_special_tokens=False)) for r in rows])
        loop = np.array([rep4(r["hyp"]) >= 0.5 for r in rows])
        cells = []
        for i in range(len(bins) - 1):
            m = (steps >= bins[i]) & (steps < bins[i + 1])
            cells.append(f"{100 * loop[m].mean():.0f}% ({m.sum()})" if m.sum() >= 15 else "-")
        print(cond.ljust(16) + "".join(c.rjust(14) for c in cells))