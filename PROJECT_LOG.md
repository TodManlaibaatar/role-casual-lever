# Project Research Log

Approximate timeline of ~20 hours of active work on this project. GPU jobs sometimes ran in parallel with other work; the times below refer to active project time rather than wall-clock runtime.

## Hours 0–1 — Project selection and initial critique
- Brainstormed and stress-tested several possible project ideas with GPT and Claude.
- Settled on testing whether the causal effect of prompt-injection steering is actually mediated by a linearly readable User-vs-Tool representation.
- Focused the project around a falsifiable decomposition experiment rather than a broad exploratory study.

## Hours 1–2 — Scope, setup, and experimental design
- Refined the research question and experimental scope.
- Set up the upstream prompt-injection codebase and GPT-OSS-20B environment.
- Validated the safe simulated tool environment and intervention machinery.
- Developed the independent User-vs-Tool probe setup and decomposition design.

## Hours 2–3 — Primary causal decomposition
- Implemented and reviewed `scripts/09_primary_component_eval.py`.
- Set up the four primary conditions: baseline, full steering, probe-aligned steering, and probe-orthogonal residual steering.
- Checked paired-seed evaluation, masking, scoring, and downstream probe readouts.

## Hours 3–4.5 — Primary evaluation
- Ran the primary causal-component experiment.
- Debugged an evaluator issue involving combined safe `curl -sL` flags.
- Fixed the parser without outcome-dependent resampling and resumed only affected trajectories.

## Hours 4.5–5 — Primary-result analysis
- Inspected raw and aggregate outputs and sanity-checked the analysis.
- Identified the main behavior/readout dissociation: aligned and residual steering had similar observed behavioral effects despite very different downstream L14 role-probe shifts.
- Refined the interpretation to avoid claiming that the orthogonal residual was “non-role.”

## Hours 5–7 — Specificity and benign-utility controls
- Implemented and ran `scripts/10_random_direction_null.py`.
- Implemented and ran `scripts/11_benign_utility_control.py`.
- Checked whether the effect could be explained by generic activation perturbation, indiscriminate tool use, or model degradation.
- Inspected results and challenged the AI-assisted interpretation against the raw outputs.

## Hours 7–9 — Fresh-prompt replication
- Implemented and ran `scripts/12_fresh_heldout_replication.py`.
- After seeing that the component estimates were imprecise, added a second disjoint fresh batch with `scripts/13_fresh_heldout_extension.py`.
- Distinguished the strongly replicated full-vector effect from the weaker component-vs-baseline evidence.
- Identified that the behavior/readout dissociation replicated more cleanly than the individual component effects.

## Hours 9–12 — Equal-norm control and debugging
- Developed `scripts/14_norm_matched_component_control.py` to address the confound that the raw residual component had larger norm than the aligned component.
- Ran the experiment for roughly 2.5 hours before discovering that excessive backup logging had created major runtime overhead.
- Recovered completed trajectories from the console output.
- Diagnosed a checkpoint-directory mismatch during resume and restored the recovered trajectories to the expected path.

## Hours 12–13 — Complete equal-norm evaluation
- Resumed from 30 valid recovered trajectories and completed the remaining runs.
- Analyzed the equal-norm results.
- Updated the interpretation: aligned steering is more potent per unit norm, but the residual effect does not collapse when norm-matched.

## Hours 13–14 — Figures and final analyses
- Built the main causal-decomposition, control, and per-prompt readout figures.
- Added a within-condition association analysis to test whether L14 role margins predict attack success within intervention conditions.
- Checked figure numbers against the saved aggregate results.

## Hours 14–20 — Write-up and application
- Wrote the executive summary and technical report.
- Audited the claim hierarchy against the fresh-prompt results and statistical evidence.
- Added explicit limitations, including the substantial alignment between the L11 decomposition probe and L14 downstream probe.
- Documented that L18 was originally intended as the downstream readout but was demoted after failing cross-domain validation, before component outcomes were evaluated.
- Revised claims where the evidence was weaker than the initial interpretation.
- Completed the MATS application questions and finalized the public repository.

## AI use

I used GPT and Claude extensively for brainstorming, code generation/review, statistical-analysis suggestions, debugging, and adversarial critique of interpretations. I inspected experiment outputs myself, checked proposed conclusions against the saved results, and made the final decisions about experimental design, follow-up controls, and claims. Several important changes came from explicitly challenging AI-assisted interpretations rather than accepting them at face value.