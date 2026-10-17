"""Download public text for HARF languages into data/raw/<lang>/ (streams; keeps only what is needed).

Usage:
    python scripts/download_data.py --langs ps am hi ta --max-docs 200000
    python scripts/download_data.py --langs ur sd            # the original two languages

Sources (check each dataset card for current license and config names before publishing):
    FineWeb-2 (HuggingFaceFW/fineweb-2, ODC-By) and Wikipedia (wikimedia/wikipedia, CC BY-SA).
If a config name fails, open the dataset page on Hugging Face, find the language under "Subsets",
and pass it with --fineweb-config / --wiki-config.
"""
import argparse
import json
import os

LANGS = {  # lang -> (FineWeb-2 config, Wikipedia config)
    "ur": ("urd_Arab", "20231101.ur"),
    "sd": ("snd_Arab", "20231101.sd"),
    "ps": ("pbt_Arab", "20231101.ps"),
    "am": ("amh_Ethi", "20231101.am"),
    "hi": ("hin_Deva", "20231101.hi"),
    "ta": ("tam_Taml", "20231101.ta"),
}
LICENSES = {"HuggingFaceFW/fineweb-2": "ODC-By-1.0", "wikimedia/wikipedia": "CC-BY-SA-4.0"}


def dump(name: str, config: str, limit: int, out: str) -> int:
    from datasets import load_dataset
    ds = load_dataset(name, config, split="train", streaming=True)
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for row in ds:
            if n >= limit:
                break
            text = row.get("text")
            if not text:
                continue
            f.write(json.dumps({"id": n, "text": text, "source": f"{name}/{config}",
                                "license": LICENSES[name]}, ensure_ascii=False) + "\n")
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", required=True, choices=sorted(LANGS))
    ap.add_argument("--max-docs", type=int, default=200_000, help="per source")
    ap.add_argument("--wiki-docs", type=int, default=100_000)
    ap.add_argument("--fineweb-config", default=None, help="override (only with a single --langs)")
    ap.add_argument("--wiki-config", default=None, help="override (only with a single --langs)")
    a = ap.parse_args()
    for lang in a.langs:
        fw, wk = LANGS[lang]
        if len(a.langs) == 1:
            fw, wk = a.fineweb_config or fw, a.wiki_config or wk
        os.makedirs(f"data/raw/{lang}", exist_ok=True)
        for name, config, limit in (("HuggingFaceFW/fineweb-2", fw, a.max_docs),
                                    ("wikimedia/wikipedia", wk, a.wiki_docs)):
            out = f"data/raw/{lang}/{name.split('/')[-1]}_{config}.jsonl"
            if os.path.exists(out) and os.path.getsize(out) > 0:
                print(f"skip {out} (exists)")
                continue
            try:
                n = dump(name, config, limit, out)
                print(f"{lang}: {out}  {n:,} docs")
            except Exception as e:  # keep going; report clearly
                print(f"{lang}: FAILED {name}/{config}: {e}")
                if os.path.exists(out):
                    os.remove(out)


if __name__ == "__main__":
    main()
