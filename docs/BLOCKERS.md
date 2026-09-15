# Blockers

None currently open.

## Resolved

### SAM 3D Body checkpoint download — gated repo (resolved 2026-09-15)

Was blocked on Nam accepting the HF gated-repo terms (`hf download facebook/sam-3d-body-dinov3` failed
with "Access denied. This repository requires approval."). Access was granted later in the same session;
`model.ckpt` (2.0GB), `assets/mhr_model.pt` (664MB), and `model_config.yaml` are now downloaded into
`data/models/sam3db/`.
