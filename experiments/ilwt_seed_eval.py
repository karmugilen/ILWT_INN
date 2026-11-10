#!/usr/bin/env python
"""Score a stego set with a given ILWT-INN checkpoint (same metrics and summary as cross_dataset_eval.py)."""
import os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
model = sys.argv[sys.argv.index("--model") + 1]
del sys.argv[sys.argv.index("--model"):sys.argv.index("--model") + 2]
import eval_common as E
E.load_model.__defaults__ = (Path(model),)
import cross_dataset_eval
cross_dataset_eval.main()
