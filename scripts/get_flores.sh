#!/usr/bin/env bash
# FLORES-200 (CC BY-SA 4.0). If this URL has moved, use FLORES+ (openlanguagedata/flores_plus on Hugging Face)
# and export each language/split as data/flores200_dataset/{dev,devtest}/<code>.{dev,devtest}, one sentence per line.
set -euo pipefail
mkdir -p data && cd data
curl -L -o flores200.tar.gz https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz
tar xzf flores200.tar.gz
ls flores200_dataset/devtest/{eng_Latn,urd_Arab,snd_Arab}.devtest
