# YourMT3+ local setup

`third_party/YourMT3/` is a vendored clone of the upstream HuggingFace Space
(its own git repo, ~2 GB with venv + checkpoint), so it is gitignored in this
repo. The two project-authored files that live inside it are versioned here;
`migs-tab doctor` validates the install.

## Recreate from scratch

```bash
mkdir -p third_party && cd third_party
git clone https://huggingface.co/spaces/mimbres/YourMT3   # needs git-lfs for the checkpoint under amt/logs/
cd YourMT3
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-mac.txt
cp ../../scripts/yourmt3/migs_driver.py .
cp ../../scripts/yourmt3/requirements-mac.txt .   # if not already present
```

Then run `uv run migs-tab doctor` from the project root — it checks the
driver, the venv, and that the checkpoint was actually pulled via LFS
(>10 MB guard against LFS pointer files).

## Files

- `migs_driver.py` — project-authored subprocess driver: runs YourMT3+ on one
  audio file, writes MIDI to a caller-specified path, prefers MPS over CUDA/CPU.
  Canonical copy lives HERE; keep `third_party/YourMT3/migs_driver.py` in sync
  when editing (doctor does not yet hash-compare them).
- `requirements-mac.txt` — macOS/MPS-specific pins layered on upstream
  requirements.txt.
