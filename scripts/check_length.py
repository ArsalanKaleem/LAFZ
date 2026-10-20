import json, numpy as np
from scipy import stats
from harf.tokenizer.common import load_tokenizer
from harf.analysis.extra import rep4

tok = load_tokenizer("Qwen/Qwen2.5-1.5B")
for lang, code in (("ur", "urd_Arab"), ("sd", "snd_Arab")):
    for cond in ["baseline", "baseline_lora", "D", "D_mean"]:
        try:
            rows = [json.loads(l) for l in open(f"runs/Qwen2.5-1.5B/{lang}/eval/flores_{cond}_eng_Latn-{code}.jsonl", encoding="utf-8")]
        except FileNotFoundError:
            continue
        L = np.array([len(tok.encode(r["ref"], add_special_tokens=False)) for r in rows])
        R = np.array([rep4(r["hyp"]) for r in rows])
        rho, p = stats.spearmanr(L, R)
        q = np.digitize(L, np.quantile(L, [0.25, 0.5, 0.75]))
        rates = [f"{100 * (R[q == k] >= 0.5).mean():.0f}%" for k in range(4)]
        print(f"{lang} {cond:14s} rho(length, rep4)={rho:+.3f} p={p:.2g}  loop rate by length quartile: {rates}")