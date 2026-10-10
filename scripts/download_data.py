# Public Urdu/Sindhi text. Check each dataset card for the current license and config names.
import json, os
from datasets import load_dataset

SOURCES = {
    "ur": [("HuggingFaceFW/fineweb-2", "urd_Arab", 200_000),
           ("wikimedia/wikipedia", "20231101.ur", 100_000)],
    "sd": [("HuggingFaceFW/fineweb-2", "snd_Arab", 200_000),
           ("wikimedia/wikipedia", "20231101.sd", 100_000)],
}
for lang, sources in SOURCES.items():
    os.makedirs(f"data/raw/{lang}", exist_ok=True)
    for name, config, limit in sources:
        out = f"data/raw/{lang}/{name.split('/')[-1]}_{config}.jsonl"
        ds = load_dataset(name, config, split="train", streaming=True)
        with open(out, "w", encoding="utf-8") as f:
            for i, row in enumerate(ds):
                if i >= limit:
                    break
                f.write(json.dumps({"id": i, "text": row["text"], "source": f"{name}/{config}",
                                    "license": "see dataset card"}, ensure_ascii=False) + "\n")
        print(lang, out, i)