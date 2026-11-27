# Research journal (sanitized snapshot)

This is a sanitized research-journal snapshot dated **2026-11-27 07:46 Europe/Berlin**. It records scientific status, publication status and named git heads for [nca_research](https://github.com/cap-jmk-real/nca_research). Operational hostnames, machine paths and process identifiers are omitted on purpose. This file is not permission to merge a pull request or to submit or upload a paper.

Canonical operational handover notes live outside this repository.

## Goal

NCA paper (*Micro-Routing with Graph Neural Cellular Automata*, paper-db id `nca`): keep the manuscript honest (tree GNCA is competitive and cheap versus tuned baselines after equal-budget HPO, **not** significantly more accurate — claim as of 2026-11-10), hold paper-review / klaut.pro ≥8.5 with findings fixed, and only then show Julian M. Kleber a Zenodo draft (his name, ORCID 0000-0001-5518-0932) before any Zenodo publish. aiXiv id **aixiv.261101.000002** (v1.5 submitted 2026-11-02 by Julian via web UI). Outreach drafts only after aiXiv acceptance, and only sent with Julian's OK.

## Publication status

- **Stage:** drafting / post-TMLR. TMLR rejected NCA (Julian, 2026-11-24). Path is Zenodo after paper-review and klaut.pro ≥8.5 with findings fixed; show Julian a draft first. Further aiXiv versions and outreach need his yes. The merge of PR #19 is **not** a Zenodo publish OK.
- **paper-db:** id `nca`. Best dated paper-review on the #19 lineage: **8.55 / 10**, 0 blockers, 2026-11-24 17:34 Europe/Berlin, “PR #19 head b51ca4a” (TMLR anonymous build). A 9.0 appears on 2026-11-10 (PR #11 / #9) — do not mix scores across heads. Julian approved resubmitting on aiXiv at paper-review 9.0.
- **Preprint / aiXiv:** **aixiv.261101.000002** (v1.5 submitted 2026-11-02 by Julian via web UI; v1.0 / v1.5 / v7.7 recorded in paper-db). No Zenodo DOI.
- **Venue target:** diamond open access, no APC, Scopus/WoS or similar (Julian, 2026-11-24 17:29 Europe/Berlin).
- **Open manuscript PRs (2026-11-27):** draft [#21](https://github.com/cap-jmk-real/nca_research/pull/21) `31d51e8601c74745f57399c8f14d9198e5d1a955` (klaut panel 171); also open drafts #15, #14 and #13 (both marked superseded by #16), #12, #11, #9 (conflicting), #8 (conflicting), #7. #16, #19 and #20 are merged. Default `main` is `0f8355c0ae843e77a1b933efe5ea5371cd7df47f`.
- **Ready to submit / Zenodo upload?** No. Need Julian's yes on a named head; a trusted klaut panel is not complete; third paired BFCL repeats are still queued behind other GPU work.

## Current step

No NCA training job is running. Latest merged paper work is on `main` (PR #19 and #20). Draft PR #21 is the newest open paper/code PR. Remaining compute: paired BFCL gate repeats, still held until the GPU is free and no BFCL process is running. Export work `nca_bfcl_exports` already finished on 2026-11-25 10:44 Europe/Berlin (0.017 h). Do not start or touch a live BFCL process.

## Decision log

- **2026-11-27 07:39–07:40 Europe/Berlin — Julian via Grok Bot:** keep this journal in the four-paper set (with Publication status) and open a sanitized journal PR on this repo. Do not merge the PR until his yes.
- **2026-11-25 — Julian, via Grok Bot:** third paired BFCL repeat approved so every entry has three paired runs; klaut reviews wait for a stronger local model; no further Azure spend beyond the existing cap.
- **2026-11-24 20:27:45 Europe/Berlin — GitHub:** PR #19 merged by `cap-jmk-real` (merge commit `57091b0c491878b3b05721680ab78ca07c71dae6`, head `5653c6c0a3ed94f992e4fa17a7627bd9f7169a43`). Do not treat that merge as permission to publish on Zenodo.
- **2026-11-24 — Julian:** TMLR rejection; Zenodo after ≥8.5; show drafts first.
- **2026-11-02 — Julian:** submitted NCA aixiv.261101.000002 → v1.5 via web UI. Further submits hold until his Go.
- **2026-11-10 — equal-budget HPO claim:** tree GNCA is competitive and cheap versus tuned baselines, not significantly more accurate.

## Results (selected)

- paper-review 8.55 / 10, 0 blockers, 2026-11-24 17:34 Europe/Berlin, “PR #19 head b51ca4a”. Earlier same day: 8.49 on `4f24294`, then 8.45, 8.42, 8.2. A 9.0 appears on 2026-11-10 (PR #11 / #9). Use the dated row; do not mix scores across heads.
- Live default `main` is `0f8355c0ae843e77a1b933efe5ea5371cd7df47f` (pushed 2026-11-25 10:26:19 Europe/Berlin), after MIT/CC BY PR #20 (merged 2026-11-25 07:36:50 Europe/Berlin, head `4078567dfb867cd6d73994c172215887eaae5ecc`).
- A MiniMax-M2.5 NCA klaut run was stopped on 2026-11-25 07:38:02 Europe/Berlin (switch to another local model). Not a finished trusted panel.

## Blockers

- Zenodo / any further aiXiv version / outreach: need Julian's yes on a named head.
- Trusted klaut panel not complete.
- Hosted-LLM router baseline spend still waiting for Julian (do not spend).
- BFCL repeats wait on GPU idle and on no live BFCL process.

## Next steps

1. After the GPU is free, let the already-guarded queue run the BFCL repeats; do not touch a live BFCL process.
2. Re-run a trusted klaut panel only on a model Julian has accepted.
3. Prepare a Zenodo draft for Julian; do not upload.
4. Outreach only after aiXiv acceptance, and only with Julian's OK.

## Last updated

2026-11-27 07:46 Europe/Berlin.
