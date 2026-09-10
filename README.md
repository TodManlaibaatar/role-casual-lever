# Role Readout or Causal Lever?

MATS application research project: test whether a prompt-injection steering effect survives removal of its component along a validated User/Tool probe.

## Layout

- `prompt-injection-as-role-confusion/`: independent upstream clone. Its revision is recorded in `configs/provenance.json`; this outer repository ignores the clone.
- `scripts/`: our executable experiments and checks.
- `configs/`: experiment settings and provenance.
- `results/`: local copies of run reports, transcripts, and figures; ignored by Git except the directory placeholder. Back these up separately.
- `writeup/`: research report and executive summary drafts.
- `timeline.md`: active project time, maintained by the researcher.

## Working from VS Code

Open this outer project folder. The existing Colab CLI environment remains inside the upstream clone:

```bash
source prompt-injection-as-role-confusion/.venv-colab/bin/activate
```

The currently allocated Colab session is `mats-role`. Its kernel already contains the loaded model and tokenizer. The smoke test below depends on that state; it is not a standalone model loader.

```bash
colab exec -s mats-role --timeout 900 -f scripts/01_agent_smoke_test.py
colab download -s mats-role /content/mats-role/benign_smoke.json ./results/benign_smoke.json
```

This smoke test simulates tool execution. It does not run model-generated shell commands. Its transcript needs manual review, and it is not a reproduction of CAMBRIA's steering result.

Colab runtime storage is temporary. Download reports after runs. Keep source scripts in this local repository and record experimental settings before the main evaluation.

## Reproducing the source checkout

```bash
git clone https://github.com/role-confusion/prompt-injection-as-role-confusion.git
git -C prompt-injection-as-role-confusion checkout --detach ec333c40fd43fe991e1ebf66765051b6d7e35784
```

See `configs/provenance.json` for source/model revisions and the environment reported during setup. These are observations, not yet a complete dependency lockfile.
