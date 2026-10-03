# First run: Hermes traces into a text NCA

**Date:** 2026-10-03

Hermes Agent (Nous Research) is the first environment. The local install already has the traces: `~/.hermes/state.db` holds 35 sessions and 3,297 messages (104 user, 1,421 assistant, 1,772 tool). The tool mass is small enough to be a label set: `terminal` (644), `browser_exec` (517), `read_file` (125), `write_file` (116), `patch` (103), then `execute_code`, `web_search`, `search_files`, and a long tail. `effect_disposition` is empty on almost every row, so outcome quality is not a free label.

## Rating

The direction is right: an NCA that is useful inside Hermes has to imitate a trace, and text is the right first modality.

The two-model sandwich is the part to postpone. Distilling a small teacher from stronger models, then training the automaton on that teacher, puts two unmeasured approximations in front of the only question this run can answer: can a local grid predict Hermes's next action better than a flat model on the same text? Generate new traces only after that comparison, and only to fill tools the 35 sessions barely contain (`memory`, `delegate_task`, `process`).

Direct training on traces is the first run. A teacher is a second run, and its job is to clean targets, not to become another policy the NCA copies.

## What a training row is

One row is one decision point inside a session, split so no session is in both train and test.

- **Goal.** The latest user message, written onto the instruction channels.
- **Recent trace.** The last few steps, each as tool name plus a truncated result, laid out as rows on the grid. This uses the same spatial write the later vision channels will use. The cells hold text features, not pixels.
- **Hidden channels.** Carried from the previous decision in the same session. Resetting them is the ablation for memory.
- **Target.** The next `tool_name`, or a single `respond` class when the assistant message has no tool call. Full reply text is not the target. Hermes's value here is the tool loop, and a closed label set is the only loss this corpus can support.

Thirty-five sessions can show whether the rule overfits and whether it beats a baseline. They cannot support a claim about new tasks.

## Order of work

1. **Export, don't generate.** Read `messages` joined to `sessions`. Keep `role`, `tool_name`, `tool_calls`, `content` truncated to a fixed cap, `timestamp`. Drop compacted summaries, or mark them so they are not targets. Write a session-level split.
2. **Train the automaton on those rows.** Instruction band, trace rows, K local updates, readout over the tool vocabulary plus `respond`. Loss is cross-entropy on the next label.
3. **Beat a flat baseline on the same rows.** Same text, no spatial update. That is hypothesis 4 in [research-design.md](research-design.md). If the grid loses, stop and change the write pattern before adding a teacher.
4. **Bring in a teacher only for bad targets.** `qwen3.5:9b` is already on engine. Use it to shorten tool results into a fixed schema, and to mark loops that repeat the same failing call. Train the NCA on those cleaned labels. Do not distill the teacher's weights or hidden states in this run.
5. **Generate traces for the holes.** After step 3 has a number, run Hermes on a short task list that forces the rare tools. Append those sessions with the same schema.
6. **Add a modality without changing the action head.** Screenshots and files already cached under `~/.hermes` land in the vision channels. The tool label set stays. That is the multimodal step, and it waits until the text rule is measurable.

## What this run refuses

- Next-token generation of Hermes replies.
- A distilled student model sitting between the traces and the NCA.
- Training on all 3,297 messages as if they were independent. The split is by session.
- Treating the current logs as a general agent dataset. They are a pilot distribution dominated by terminal and browser calls.
