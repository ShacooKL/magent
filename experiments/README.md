# EarthBench Q1 live smoke

The legacy entrypoint now accepts `--task-id N` (default 1); task selection does
not change the team prompts, tool permissions or 40-request ceiling.

Entry: `python -u experiments/earthbench_q1.py --output var/runs/<fresh-name>`.
`--preflight` starts the isolated official tool bridge and checks rasterio, with no API call.
Only the public question and raster directory are exposed; old baseline instructions,
answers and trajectories are not passed to the agents. Official tools come from the
old sanitized fixed-earth directory. Host scoring runs only after answer submission.

Initial runs:

- q1-preflight-01: 104 tools loaded; rasterio 1.4.3 available.
- q1-live-01: DNS failure, no model response.
- q1-live-02: original planning prompt induced unavailable-tool calls; eventually
  entered execution, then hit its role limit. 24 responses; conservative CNY 0.20798825.
- q1-live-03: planning prompt repaired; 1 planning + 12 execution requests; 21 TVDI
  tool calls returned output paths, but the 91-date computation did not finish.
  The per-role limit terminated before review. Conservative CNY 0.13321650.

No answer or accuracy claim is available. Both paid runs have stopped. Combined
reported-response estimate is CNY 0.34120475, not provider invoice confirmation.
The failed DNS attempt has no usage report and is kept separately, not scored.
Model execution detected 2019–2022 input coverage despite the question mentioning
2019–2023. No missing dates were supplied by the host.

Next engineering issue: role budget exhaustion should hand partial evidence back
to the coordinator, not abort the whole workflow before review. Fix this before
another run; simply increasing the role limit does not validate multi-agent benefit.

## Shared-budget handoff experiment: q1-live-04

Executed after implementing stage yield and coordinator-directed continuation;
same model and 40-request shared ceiling, six requests reserved downstream.
No Q1-specific batching instructions were added and no runtime intervention was made.

- Finished with `partial`, no answer tag; host GT is B and was read only after final output.
- 40 responses: coordinator 4, executor 31, reviewer 5.
- Two scheduling decisions returned CONTINUE. The first 12-request slice no longer
  killed the task, and the reviewer and final coordinator were reached.
- 16 calculate_tif_average calls targeted only seven distinct output paths;
  2019 NDVI and LST composites were each recomputed three times. No compute_tvdi
  or trend tool call occurred. Earlier success at calling TVDI was not reproduced.
- Coordinator continuation instructions repeatedly proposed re-listing inputs
  instead of identifying remaining operations. Handoffs contained evidence IDs
  and last unverified text, not a usable completed/pending inventory.
- Reviewer spent its slice inspecting inputs rather than delivering a completed
  audit. Final coordinator emitted a plan and textual pseudo-tool markup, not an answer.
- Estimated cost CNY 0.50337800; input 1,863,960 tokens, output 18,694.
  All 40 responses included usable pricing data; this is not an invoice amount.

Conclusion: scheduling handoff and reserved-budget paths work, but this run does
not show improved solving. Next change should improve concise progress handoffs
and enforce final-response semantics, not add a question-specific solution.

## Structured-handoff experiment: q1-live-05

Same Q1, model and 40-request ceiling, with submit_progress/submit_review/submit_final.
No intervention in the live agent's method. Completed and stopped; not a repeated
paired accuracy experiment.

- 40 responses: coordinator 5, executor 29, reviewer 6. Estimated CNY 0.49104225;
  input 1,868,897 tokens, output 11,909. No provider invoice verification.
- Explicit progress submissions successfully conveyed completed/pending work,
  artifact paths, sampling limitations and missing 2023 data. Coordinator proceeded
  toward statistics instead of merely restarting directory exploration.
- 8 compute_tvdi calls and one compute_linear_trend call. The latter took
  y=[0.6761,0.6515,0.7466,0.4270], x=[2019,2020,2021,2022], returning slope
  -0.06521999999997197. Means were computed with auxiliary Python. Single dates
  were used per year, with January in 2022 versus mid-year dates for other years.
- Reviewer did not submit_review; handoff_missing=true. It re-explored data and
  did not complete an independent audit before its reserved submission deadline.
- Coordinator formally submitted B, matching GT B, despite the calculated slope
  differing from the option's -0.037/year. State correctly remains partial because
  execution/review contain unfinished markers. Correct letter is not evidence of
  complete coverage, correct methodology or multi-agent benefit.
- Initial host scorer recognized only legacy answer tags and logged null for bare
  structured answer B. Offline scoring was corrected by accepting submit_final's
  answer field; event 143 appends the correction without replacing event 142 or
  making new model calls. Scoring regression tests included in 19 passing tests.

Next questions: why executor substitutes annual sampling; why reviewer attempts
to solve again rather than audit existing artifacts; why final decision accepts a
numerical mismatch despite explicitly reported uncertainty. Do not report this
single score as an accuracy improvement attributable to the new interfaces.

## Q2: q2-live-01

- Completed 27 API responses: coordinator 4, executor 8, reviewer 15.
  Conservative estimated cost CNY 0.19990800 (not provider invoice).
- Executor called official compute_tvdi, then calculate_threshold_ratio at 0.75,
  obtaining 28.070092264494612%. No run_command or auxiliary numerical code.
- Reviewer independently called the same official tools into a separate raster
  and obtained the same percentage. This is repeatability, not independent
  validation of the underlying algorithm. Also checked extrema/counts; attempted
  UTF-8 read of TIFF failed once.
- submit_progress, submit_review and submit_final all accepted. Reviewer initially
  attempted PASS while listing unresolved issues, rejected three times; finally
  submitted ISSUES. These retries are overhead, not scientific validation.
- Data are dated 2021-07-12 and named NDVI; question requests 2022-07-12 and EVI.
  Both executor and reviewer reported these differences. Final C (28.07%) matches
  GT C for the benchmark, but does not establish the requested date/index result.
- Structured answer was "C. 28.07%". Old scorer failed to parse it; an offline
  parser correction accepts labeled options and appends event 94, preserving the
  initial event 93. No extra model calls. 20 regression tests pass.
- Workflow completed; this single run does not establish multi-agent advantage
  over a single agent or independent scientific correctness.

## Q3: q3-live-01

- Completed 34 API responses: coordinator 4, executor 18, reviewer 12.
  Estimated CNY 0.30642975, not provider-invoice verified. Final B matches GT B
  (2 spikes), host-only scoring event 120; no GT feedback supplied to team.
- Executor computed TVDI for all seven dates with official compute_tvdi, including
  multiple independent calls in one response, then used calc_batch_image_mean.
  July 28 produced no valid TVDI; remaining means were approximately
  [0.7159, 0.6939, 0.3553, 0.6819, 0.4762, 0.7463].
- count_spikes_from_values failed four times for executor and twice for reviewer:
  values were supplied as a serialized list string, and conversion to float failed.
  Removing null and rounding numbers did not fix the argument representation.
  Neither role used tool_inspect. Executor counted upward changes manually.
- Reviewer used five run_command calls for GDAL/NumPy inspection and counting.
  Its final code counted two upward changes exceeding 0.2 after removing the
  missing date. This verifies that custom rule, not successful use of the official
  spike tool or equivalence with its implementation. Executor had passed 0.1
  to the failed calls; threshold choice and gap handling were not justified.
- Source inspection showed July 28 LST entirely zero, but NDVI was not entirely
  invalid (range -32768 to 9973). Reviewer's earlier claim that both sources had
  no valid data overstates the evidence; its check only excluded NaN, not nodata.
- Reviewer twice submitted PASS with issues and was rejected, then cleared issues
  and submitted PASS after custom counting. Interface validation did not establish
  that the tool failure or missing-date methodological concerns were resolved.
- Correct option and completed workflow do not establish official-tool-only
  solving, full methodological correctness, or a causal multi-agent improvement.

## Q3 official-only control: q3-official-only-01

Run with `experiments/earthbench_q1.py --task-id 3 --official-only --output
var/runs/q3-official-only-01` (single invocation; same Qwen profile and shared
40-request ceiling). Default auxiliary-enabled behavior remains unchanged.

- New switch removes run_command/write_file/edit_file from agent catalogs and
  rejects unavailable calls at dispatch. File listing/reading, tool discovery,
  evidence access and submissions remain available. All three role prompts
  require official numerical computation and prohibit auxiliary code and manual
  substitution. Host-only rasterio preflight is not an agent computation.
- 22 offline tests passed before launch, including dispatch denial and all-role
  prompt checks. No tool or prompt changes were made during the live run.
- Finished 24 responses: coordinator 4, executor 12, reviewer 8. Estimated
  CNY 0.19296550, not invoice verified. Final B matches GT B (host event 89),
  but workflow status is partial, reviewer verdict ISSUES.
- Seven official TVDI calls and four batch mean calls; five spike-tool calls
  failed (executor 3, reviewer 2), with the same serialized-list conversion
  error as the auxiliary-enabled run. No tool_inspect/tool_search calls.
- Zero run_command/write_file/edit_file calls. Nonetheless executor and reviewer
  substituted verbal manual counting, and coordinator accepted B on that basis.
  Thus code prohibition held, but official-tool-only numerical completion did
  not. Prompt adherence is not guaranteed by capability filtering.
- Compared with q3-live-01: both correct letters; 24 versus 34 requests, about
  CNY 0.193 versus 0.306. Neither completed official spike counting. This single
  unpaired stochastic comparison does not establish an accuracy/efficiency gain.
- Offline source/catalog inspection: count_spikes_from_values expects a numeric
  list, while its published schema leaves argument types unspecified; calls sent
  a list string instead. Tool schema/argument usability is a candidate bottleneck,
  not evidence that the numerical algorithm is unavailable. No repair was fed
  to either live agent. Next isolate this interface issue before attributing the
  failure solely to planning or auxiliary-code availability.

## Parallel Q1/Q2 official-only controls

Separate processes/workspaces: q1-official-only-01 and q2-official-only-01.
Same Qwen profile, official-only switch and 40 shared requests per task as Q3.
No live intervention, tool-schema repair, or additional paid retries.

- Q1: 40 responses, estimated CNY 0.61478825; explicit incomplete final, no option,
  host score false (GT B), workflow partial. Official TVDI/mean computations
  progressed but the annual series and trend were not completed. Official mean
  accepted a real numeric list and returned 0.6896851539611817 for the submitted
  2019 values. No compute_linear_trend call. Reviewer tried to continue computing
  instead of completing review; five final-slice TVDI calls were rejected because
  that slot was reserved for submission. No successful submit_review. Final
  summary understated reviewer progress, referring only to initial executor
  coverage of 2021/2022. Missing upstream 2023 remains a separate data limitation.
- Q2: 34 responses (coordinator 5, executor 10, reviewer 19), estimated
  CNY 0.25213900. Official compute_tvdi and calc_batch_image_hotspot_percentage
  returned proportion 0.2807009226449461, reported as C, 28.07%, matching GT.
  Workflow completed; reviewer verdict INSUFFICIENT, citing 2021/2022 and NDVI/EVI
  mismatches and unverified requested spatial extent. Reviewer repeatedly listed
  files/read evidence, initially missed its structured handoff and later repaired
  one malformed submit_review. Coordinator accepted the available-data answer.
- Both had zero auxiliary command/file-writing calls. Q2's substantive statistic
  came from an official tool; Q1 did not manufacture a final trend or option.
- Earlier auxiliary-enabled Q1 q1-live-05 selected correct B despite a mismatched
  slope and incomplete review; this control abstained. Earlier Q2 q2-live-01 also
  selected C using official calculations, 27 requests/~CNY 0.200 versus 34/~0.252
  here. These single stochastic runs do not establish causal accuracy differences.
- Combined new estimated cost CNY 0.86692725, not provider-invoice verified.

## Q10 post-task learning pilot

Command: `python -u experiments/learn_q10.py --output var/runs/q10-learning-01`.
Uses the existing Qwen profile. Analysis cap 24 requests; at most two paired
rounds, each two fresh official-only teams with 40 shared requests per team.
Maximum 184 model requests across analysis and all replays, not a monetary cap.
Learning cost is in learner state; replay costs remain in their individual states.

Implementation: `src/magents/learning.py`. One continuous learner context receives
source-run operation overviews, outcomes and permission metadata, not the human
diagnosis or official GT trace. It can expand events, inspect tool source and
execute temporary diagnostic code in a separate workspace. Original source-run
workspaces and private answer files are not mounted for exploratory commands.

`replay_task(skill)` freezes the candidate before starting paired subprocesses.
Only the with-skill team receives that text; both receive the original task data
and official-only restrictions. Neither receives learner context or artifacts.
Replay feedback includes answer scoring, tool sequence and costs; full replay
events can be expanded. GT keys are removed from exposed host scoring events.
The learner may revise once for a second paired round. No automatic retries or
library promotion. Replay scoring and absence of code calls do not establish
semantic compliance; retention remains pending evidence audit and later transfer
testing. This is a pilot, not an efficacy estimate.

24 offline tests passed, including frozen paired replay commands, two-round cap,
and removal of GT keys from trace access. Ordinary solving behavior is unchanged
unless an explicit `--skill-file` is supplied.

## Skill retrieval transfer pilot: skill-transfer-controls-01

Command: `python -u experiments/paired_controls.py --output
var/runs/skill-transfer-controls-01 --comparison skill-retrieval --tasks
93 55 100 69 70 --workers 5` (single command).

Five task pairs, one run per condition, up to five concurrent subprocesses.
Both conditions use official-only computation and the same model and 40-request
shared budget. With-skill enables library search/read; without-skill removes those
tools. No direct skill injection, no automatic retries. This tests the combined
retrieval/selection/use workflow, not the isolated effect of forced skill content.

Q93 is the closest conditional-LST-mean transfer candidate. Q55 requires maxima,
Q100 proportions, Q69 a TVDI condition, and Q70 a TVDI target; these are also
applicability/misuse tests, not five presumed positive-transfer examples. Compare
score, completion, retrieval/read calls, evidence of adoption/rejection, official
tool compliance, requests and costs. Do not interpret a single stochastic pair
as established causality. No GT trajectory or human solution hints are supplied.

29 offline tests passed before launch. Batch manifest records configuration and
library snapshots; each retrieval run also logs its loaded library snapshot.
Library and solver settings must remain unchanged throughout this batch.

### Learning prompt: tool-use diagnosis guidance

The learner prompt now prioritizes tool selection, parameter semantics and JSON
types, actual return/error interpretation, downstream use and targeted recovery.
Before causal attribution it asks for full key-event arguments/results and relevant
neighbors, with schema/source checks or a minimal exploratory probe where needed.
This is guidance, not a hard-coded diagnosis or an additional submission schema.
It permits uncertainty and data/scheduling explanations, and avoids treating every
successful-run step as correct. Existing candidate content and historical run logs
are unchanged; no new paid experiment is implied by this prompt edit.

### Structured trace reading fix

`trace_read(run, event_id)` now returns event identity, tool name, actual JSON
argument types and values/previews, dispatch status, and output/error separately.
It no longer asks the learner to choose a page length just to inspect a call.
Long argument previews carry explicit truncation flags; strings are not silently
parsed or corrected into arrays/numbers. This describes evidence, not a diagnosis.

`trace_expand(run, event_id, section)` expands arguments/output/event, defaults to
6000 characters and reports total/returned characters plus next_offset_chars.
Pagination parameters are explicitly named and documented as characters; pages
under 512 characters are rejected. GT keys remain withheld in both views.

34 offline tests passed, including actual JSON-type preservation, long argument
visibility without hiding the output, lossless pagination and GT exclusion.
Offline inspection of original Q93 event 100 now shows all three path arguments
as strings, threshold as numeric 0.7, and output NaN in a 2240-character response.
No historical logs or existing skills were changed and no paid run was launched.
Evidence-reference namespace repair and learning/replay logic are outside this fix.

### Learning-time hypothesis diagnostic (2026-10-29)

With `--debate --extract-skill` (2026-10-30), the 24-request diagnostic debate is
followed by at most 8 extraction requests in analyst A's existing context, under
a total cap of 32. The extractor audits the diagnosis rather than assuming it is
true and calls submit_skill with applicability, steps, boundaries, verified findings,
unverified claims and evidence references, or abstains. state.skill_submission is
separate from learning_submission. Candidates remain pending audit, not published,
replayed or demonstrated to improve accuracy. The prompt is task-general; no human
Q93 diagnosis or DeepSeek conclusions are supplied.

Optional `--debate` (requires `--diagnosis-only --hypothesis-verification`) uses
two private Qwen contexts with a shared diagnostic workspace/log. Each independently
reads the same source and submits a preliminary report, then critiques the peer's
frozen report and tests claims; analyst A synthesizes. All phases share 24 requests:
initial read-only reports are capped at 3 each, verification reserves 7 for each
analyst, and synthesis reserves 4. Early savings pass to later verification/synthesis.
Read-only phases expose trace_read/trace_expand/tool_search/tool_inspect/read_file;
hidden command, computation or hypothesis calls are also blocked at dispatch.
Since 2026-10-30, each interim phase has a separate one-request text-only reporting
step within its budget (initial: 2 evidence + 1 report; verification: 6 + 1).
No submit tool is required for interim opinions; the model's report is logged verbatim
and labeled unverified before forwarding. Evidence-window exhaustion is not itself
a failed handoff: a fresh report is requested in the same context. Empty/failed report
requests still stop the debate; scheduler text is never substituted for a report.
Only synthesis requires submit_learning (and exposes only that tool at its final slot).
Reserves prevent one analyst consuming the peer's budget, but cannot guarantee a
valid final submission or a completed experiment. These schema changes occur at
explicit phase boundaries; within-phase conversation history remains append-only.
Exchanges are sequential, not
parallel API calls; initial independence is contextual, not filesystem isolation.
No DeepSeek diagnosis is injected. `debate_phase` and `debate_report` events identify
analysts; state stores phase and reports. Agreement does not certify correctness.
Implementation: `src/magents/debate.py`; no library publication or task replay.

Learning supports `--profile deepseek` for an explicitly requested model contrast.
This changes only the learner's configured API profile, not the replay solver.
The DeepSeek diagnostic records raw API usage and aggregate tokens; no Qwen tariff
is applied, and estimated monetary cost remains null until a suitable tariff is frozen.

`learn_q10.py --task-id 93 --source-run <run> --hypothesis-verification --diagnosis-only --output <new-dir>`
exposes the same hypothesis tools to the learning Agent, explicitly requests a
propose/test/conclude cycle for its central causal claim, and disables replay both
in the catalog and dispatch. It uses 24 Qwen requests, a fresh exploration workspace,
the source trace and original task inputs; no human diagnosis or GT is added.
Source workspaces are now mounted whole and read-only at their host absolute paths,
with a /workspace prefix mapping in source_artifacts. All source-generated file types
and directory listings are available, but parent run records/host scoring are not mounted.
Workspace-root symlinks and links escaping the workspace/approved input mounts are
rejected before launch; no additional external paths are authorized automatically.
The diagnostic /workspace stays independently writable. No artifact copies are made.
Contents reflect end-of-run state, not the exact time of an earlier call; use trace
chronology. Missing workspaces are explicit in the mapping. Implementation: source_artifacts.py.
This preserves access boundaries; it does not semantically sanitize source-generated content.
Single-source learning with an available workspace now also preserves original
`/workspace/...` paths via a deterministic read-only mount. The fresh writable
directory and command cwd are `/diagnostics`; official-tool default outputs use
that mount too. ToolBox read/list/write boundaries match command and official-tool
namespaces. No tool argument strings are rewritten. Normal solving remains on
writable `/workspace`. Multi-source/missing-source cases retain explicit mappings;
they never silently choose a source. The mapping and layout are logged per run.
Hypothesis records include call_attempted, test_executed (false/true/unknown), status,
actual arguments and raw output; returned does not mean a successful computation.
Opaque bridge exceptions are marked execution-unknown, not guessed as validation
versus runtime failure. The runner ledger is attached to reports/synthesis and saved
as state.verification_ledger separately from model conclusions. B audits A's actual
probe, checking prerequisites, one-factor comparisons and evidential limits before
retesting. This is not automatic semantic certification. No new paid run is implied.
Conclusions remain model judgments, not automatically certified explanations.
This single-run pilot evaluates tool adoption and diagnostic evidence quality;
it is not a paired estimate of accuracy improvement. No skill is auto-published.

### Qualified learning evidence references

The subsequent reference repair adds `submit_learning(status, answer, reason,
evidence_refs)` for learning only. Examples: `source-1:100`, `learning:4`, and
`round-1/with-skill:12`. Overview entries and trace_read/trace_expand responses
expose copyable source references; learning tool-result headers expose local
learning references. Existing solving submissions retain local integer evidence_ids.

Validation resolves the named run explicitly and verifies event existence; it does
not guess or translate namespaces, require every citation to have been expanded,
or claim semantic support. Missing sources/events report the exact namespace and
ID. GT filtering in trace reads remains unchanged. Learning results are stored as
learning_submission, not final_submission; incomplete learning remains partial.

36 tests pass, including end-to-end read-and-submit with source/local references,
replay-source resolution, same-number namespace isolation, informative errors,
and unchanged solving submission behavior. No paid rerun or library update was
performed for this repair. Historical logs retain their original formats.

### Read-only error-candidate debate

Use `learn_q10.py --debate --diagnosis-only --candidates-only --max-requests 40`
with the usual `--task-id`, `--source-run`, `--profile` and fresh `--output`.
Do not combine this mode with `--hypothesis-verification` or `--extract-skill`.
Both analysts may read traces and tool definitions/source, but execution, probes,
replay and skill submission are blocked. With 40 shared requests, reservations are
8 + 8 for independent analysis, 9 + 9 for candidate critique, and 6 for synthesis;
unused requests carry forward. Reports retain at most three ranked candidates,
exact evidence, alternatives and uncertainty. `completed` refers to discovery,
not experimentally established causes. Reports and actual calls remain in events.jsonl.

### Supervised candidate verification

Use `--debate --diagnosis-only --hypothesis-verification --max-requests 40
--verify-candidates-from <candidate-run>` with a fresh output and identical source runs.
This imports the frozen candidate submission and peer reports, not a human diagnosis.
A ranks candidates by evidence-based confidence, B audits that plan, A tests, B audits
actual evidence, A repairs/retests, then B submits the final audit. At 40 requests,
stage reservations are 6/4/12/6/6/6, with unused requests carried forward. B has only
read tools (including local evidence_read); A may execute diagnostic probes outside
planning/report phases. No skill extraction, publication or task replay. The new
budget and billing are separate from candidate discovery. Confidence remains a
model judgment, not a calibrated probability or automatic proof.

Supervised `verify` now requires a newly recorded hypothesis test with
`test_executed: true` before advancing to test audit. Plain-text-only responses
receive an execution reminder within the existing phase budget; no extra requests
are granted. Missing-input preflights, proposals and unknown execution status do
not satisfy this gate. If no target ran, the run ends partial with
`verification_gate.status = no_executed_test`, preserving reports and unused
downstream budget. A returned NaN can pass the execution gate but is not a verified
cause or successful computation. Report-only restrictions explicitly apply to one
response; work-phase instructions restore the appropriate role's tool permissions.

Supervised verification now finishes with `submit_verification`, stored separately
as `state.verification_submission`. Each of up to three candidate audits has verdict
(supported/refuted/unresolved/untested), confidence, source/test references,
observations, control audit, prediction-direction audit, scope, unestablished claims
and remaining objections. Local test references must resolve to hypothesis_test
result events, not proposals or prose. Supported/refuted entries require test refs.
These are structural checks, not proof of correct reasoning or successful computation.
There is no task-answer field; option matching is explicitly outside diagnosis.

`test_refs` also accepts runner-issued IDs such as `h1:t1`. The submission validator
resolves these deterministically against this run's hypothesis_test result events
and stores canonical `learning:event` references. Unknown or ambiguous test IDs are
rejected; no cross-run guessing. Raw model responses and call inputs retain their
original IDs; normalization is applied only to a copy of the accepted report.
This removes manual ID conversion, not evidence or semantic audit requirements.

### Minimal diagnosis comparison (2026-11-02)

The existing debate/verification modes and historical runs remain available as the
complex comparator; this change does not modify their diagnosis logic. Stop adding
constraints there while evaluating a smaller baseline. No paid comparison has run yet.

Use the same runner, source runs, profile and total budget with:

```
--diagnosis-only --simple-diagnosis single --max-requests 40
--diagnosis-only --simple-diagnosis review --max-requests 40
```

Each needs its own fresh `--output`. Both retain original read-only artifacts,
diagnostic code/official tools, expandable traces, append-only logs and billing.
Neither requires hypothesis registration, candidate discovery, a per-candidate
verification form, skill generation or task replay. They use the existing small
submit_learning contract. Review mode adds one fresh-context reviewer; at budget 40,
27 requests are available to diagnosis and 13 reserved for review, with unused
diagnosis budget carried forward. The analyst draft is saved separately and cannot
be mistaken for a completed review. There is no automatic repair/debate loop.

Before paid evaluation, select a small fixed set of source traces (not just Q93),
use identical inputs/model/budget across modes, and repeat seeds where supported.
Compare correct error localization, unsupported causal claims, evidence relevance,
review corrections/regressions, actual cost and completion—not form acceptance or
the original answer letter. Prepare offline evidence-based audit notes separately;
never put the expected diagnosis in model inputs. Do not claim skill utility until
subsequent held-out task replay demonstrates it.

### Ordered single-agent diagnosis (2026-11-02)

New `--simple-diagnosis single` runs use protocol `ordered-tool-checks-v1`:
locate a consequential bottleneck, then inspect parameter format, parameter meaning
(including input coverage), tool selection and call sequence. Full relevant trace
evidence comes before causal attribution; source inspection or small probes are
chosen as needed. No new submission form, mandatory hypothesis registration or
fixed testing steps. Existing submit_learning remains the output contract.

The legacy review prompt and historical comparison logs are unchanged. The earlier
24-run batch used the old single-agent prompt; do not append new-protocol runs to
that comparison or treat them as identical conditions. Each new run records the
protocol label and complete prompt in run.json. Local tests check wiring and
ordering only, not diagnosis quality. No paid run accompanied this prompt change.

### Single diagnosis → candidate skill → paired replay

Append `--skill-replay` to a new single-agent diagnosis run:

```bash
python experiments/learn_q10.py --task-id 93 \
  --source-run var/runs/q93-reviewer-replay-01/with-skill \
  --diagnosis-only --simple-diagnosis single --max-requests 40 \
  --skill-replay --replay-max-requests 40 --profile dashscope-qwen35-flash \
  --output var/runs/q93-single-skill-replay-01
```

Here `--diagnosis-only` disables model-directed replay during diagnosis; the explicit
`--skill-replay` enables one host-controlled pair afterwards. Diagnosis stays unchanged.
An actual diagnosis submission (including incomplete with findings) is required.
The same analyst context then gets up to 8 additional requests under an extraction
prompt, using the existing submit_skill contract. Abstention skips replay; failure
to submit does not launch solvers. This command starts a new diagnosis, not a resume
of an old run. Maximum model requests: 40 + 8 + 40 + 40 = 128.

Only title/when/steps/boundaries reach the with-skill solver. Both fresh solvers use
the selected profile, equal budgets, disabled skill retrieval, and the source run's
official_only permission setting. Their directories are round-1/with-skill and
round-1/without-skill. Candidate JSON stays separate from the reusable injected text;
diagnosis context, evidence references and scoring records are not injected. No
automatic skill publication, retry or second pair. Correctness, workflow status,
tool sequence and costs are in round-1/result.json; extraction/diagnosis billing is
in parent state.json and solver billing is separate. One same-task pair is a local
replay check, not causal proof of improvement or held-out transfer.
