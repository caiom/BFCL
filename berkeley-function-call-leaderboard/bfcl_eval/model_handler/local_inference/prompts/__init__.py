import os

dir_path = os.path.dirname(os.path.realpath(__file__))

prompts = {}

# Prompt templates used by Phi-4 prompt ablations.
fnames = [
    "bfcl_simple_ll.md",
    "bfcl_simple_lll.md",
    "bfcl_simple_llll.md",
    "bfcl_simple_parallel.md",
]
for fname in fnames:
    with open(f"{dir_path}/{fname}", encoding="utf-8") as f:
        prompts[fname.split(".")[0]] = f.read()
