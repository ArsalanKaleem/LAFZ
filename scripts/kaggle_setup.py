"""Kaggle session setup in one command (run from the repo root after `pip install`).

    python scripts/kaggle_setup.py --langs ta
    python scripts/kaggle_setup.py --langs ur --restore baseline_lora D D_mean

1. Copies data/processed/<lang> and runs/<model>/<lang> from an attached dataset named harf-<lang>.
2. --restore: copies trained conditions (those with lora/adapter) and eval files from ANY attached
   notebook output, so earlier results can be compared and re-evaluated.
3. Downloads FLORES-200 into data/flores200_dataset if missing.
"""
import argparse
import glob
import os
import shutil
import tarfile
import urllib.request

FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"


def find_dataset(root: str, lang: str):
    hits = [p for p in glob.glob(f"{root}/**/harf-{lang}", recursive=True) if os.path.isdir(p)]
    return min(hits, key=len) if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", required=True)
    ap.add_argument("--restore", nargs="*", default=[], help="condition names to restore from notebook outputs")
    ap.add_argument("--model", default="Qwen2.5-1.5B")
    ap.add_argument("--input-root", default="/kaggle/input")
    ap.add_argument("--no-flores", action="store_true")
    a = ap.parse_args()
    ok = True
    for lang in a.langs:
        R = f"runs/{a.model}/{lang}"
        ds = find_dataset(a.input_root, lang)
        if ds is None:
            print(f"[{lang}] MISSING dataset harf-{lang}: attach it via Add Input → Your Datasets")
            ok = False
            continue
        src_data, src_runs = f"{ds}/data/processed/{lang}", f"{ds}/runs/{a.model}/{lang}"
        for s, d in ((src_data, f"data/processed/{lang}"), (src_runs, R)):
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
                print(f"[{lang}] copied {s} -> {d}")
            else:
                print(f"[{lang}] MISSING {s}")
                ok = False
        for cond in a.restore:
            hits = [p for p in glob.glob(f"{a.input_root}/**/runs/{a.model}/{lang}/{cond}", recursive=True)
                    if os.path.isdir(f"{p}/lora/adapter")]
            if hits:
                shutil.copytree(hits[0], f"{R}/{cond}", dirs_exist_ok=True)
                print(f"[{lang}] restored {cond} from {hits[0]}")
            else:
                print(f"[{lang}] could not restore {cond}: attach the notebook output that trained it")
                ok = False
        if a.restore:
            for ev in sorted(glob.glob(f"{a.input_root}/**/runs/{a.model}/{lang}/eval", recursive=True)):
                shutil.copytree(ev, f"{R}/eval", dirs_exist_ok=True)
            print(f"[{lang}] merged earlier eval files")
    if not a.no_flores and not os.path.isdir("data/flores200_dataset/devtest"):
        os.makedirs("data", exist_ok=True)
        print("downloading FLORES-200 ...")
        urllib.request.urlretrieve(FLORES_URL, "data/flores.tgz")
        with tarfile.open("data/flores.tgz") as t:
            t.extractall("data")
        os.remove("data/flores.tgz")
    if not a.no_flores:
        print("FLORES:", "OK" if os.path.isdir("data/flores200_dataset/devtest") else "MISSING")
    print("SETUP OK" if ok else "SETUP INCOMPLETE — fix the MISSING lines above before running experiments")


if __name__ == "__main__":
    main()
