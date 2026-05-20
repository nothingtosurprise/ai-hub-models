# Nightly Failure Analysis

Analyze nightly CI failures and post a structured comment on the GitHub issue.

**Budget: You have ~35 tool calls. Be efficient. Batch queries. Do NOT read files one-by-one.**

## Context Variables

Injected via the workflow prompt:
- `RUN_ID` — the failed workflow run ID
- `REPO` — the repository (owner/name)
- `RUN_URL` — direct link to the workflow run

## Critical `gh` CLI Rules

- ALWAYS use `--json` and `--jq` flags — never parse human-readable text
- NEVER use `!=` in `--jq` expressions — bash mangles `!`. Use `select(.conclusion == "failure")` instead
- Check run status before using `--log-failed`

## Step 1: Get Failed Jobs and Test Results (~3 tool calls)

1. Get failed jobs in ONE call:
   ```
   gh run view $RUN_ID --json jobs --jq '.jobs[] | select(.conclusion == "failure") | {name, conclusion}'
   ```

2. Try downloading test results artifact:
   ```
   gh run download $RUN_ID -n nightly-test-results -D /tmp/nightly-results
   ```

3. If download succeeds, read `/tmp/nightly-results/summary.md` — it has pre-built failure tables with test names, errors, and stack traces. This is your primary data source. If download fails, work from job-level pass/fail only.

## Step 2: Summarize + Categorize Failures (~3 tool calls)

From summary.md or job data, group failures:

| Category | Detection |
|----------|-----------|
| Unit Test | `*-unit-tests-junit.xml` or "QAIHM Tests" |
| Model Test | `*-model-tests-junit.xml` or "Model Tests" |
| Workbench Job | `*-verify-workbench-jobs-junit.xml` |
| Workflow/Infra | Jobs that failed without XML |
| Cross-Version | Same test fails across 3.10/3.11/3.12/3.13 |

**Dedup rule:** Same test failing across all Python versions = report once as "cross-version".

For each category: count, unique error signatures, first stack trace (max 10 lines).

## Step 3: Find Breaking Commits (~5 tool calls)

1. Find last successful nightly + current SHA in ONE call each:
   ```
   gh run list --workflow=nightly.yml --status=success --limit=1 --json headSha,createdAt,databaseId
   gh run view $RUN_ID --json headSha --jq '.headSha'
   ```

2. Get commit range with file changes in ONE call:
   ```
   gh api repos/$REPO/compare/{last_sha}...{current_sha} \
     --jq '.commits[] | {sha: .sha[0:8], date: .commit.author.date[0:16], author: .commit.author.name, message: .commit.message | split("\n")[0], files: [.files[].filename]}'
   ```
   **If this doesn't return files**, get the stat summary instead:
   ```
   gh api repos/$REPO/compare/{last_sha}...{current_sha} --jq '{total_commits: .total_commits, files: [.files[].filename]}'
   ```

3. Cross-reference changed files with failing tests. Do NOT check each commit individually.
   For each suspect commit, note which specific failure(s) it relates to (use the failure # from the summary table).
   **PR references:** Always use fully-qualified format `qcom-ai-hub/ai-hub-models-internal#N` — comments are posted on tetracode issues, so bare `#N` resolves to the wrong repo.
   General rules:
   - `models/<id>/*` changed → `models/<id>/test*` failures
   - `utils/`, `configs/`, `test/` changed → unit test failures
   - `global_requirements.txt`, `pyproject.toml` changed → cross-version failures
   - `scorecard/` changed → scorecard failures

4. If 0 commits between last success and current failure, note: "No new commits. Likely external dependency, flaky test, or infrastructure issue."

## Step 4: Root Cause Analysis + Triage (~5 tool calls)

**First, check `.claude/triage/historical-patterns.md` for known recurring patterns.**
Many nightly failures match historical signatures (transient host outages, service timeouts,
dependency breakages). If the failure matches a known pattern, classify it immediately
without deeper investigation. Only dig into commits/code if the pattern is novel.

**CRITICAL — Find the root cause BEFORE proposing any fix.**

Do NOT propose workarounds that mask the real issue. A fix that tolerates bad data is a
bandaid — the right fix addresses WHY the data is bad in the first place.

**Root cause checklist:**
1. **Where does the unexpected data/state originate?** Trace the error upstream:
   - If our code crashes on unexpected input, ask: is the input wrong, or is our code wrong?
   - If a dependency produces unexpected output (e.g. renamed tensors, wrong formats),
     the fix belongs in that dependency — file a ticket to the responsible team.
   - If our code is too strict/too loose for valid behavior, the fix is in our code.
2. **Search for existing issues BEFORE proposing a fix:**
   ```
   gh issue list --repo qcom-ai-hub/tetracode --search "<model_name> OR <error_keyword>" --state open --limit 5 --json number,title,labels,assignees
   ```
   If an existing issue tracks this failure, reference it instead of proposing a new fix.
3. **Is this a bug in an external dependency?** (AIMET, QNN compiler, Hub API, etc.)
   - If yes: recommend filing a ticket to the owning team with the job ID and repro steps.
   - Only recommend a workaround in our code if clearly labeled as temporary.
   - **AIMET signals:** `QcQuantizeOp_` prefix, `_q` suffix, `w8a8`/`w8a16` precision failures
     → Route to `Quantization` team. Do NOT propose fuzzy-match fixes in our code.
4. **Is this a regression from a recent PR?** Cross-reference with suspect commits from Step 3.

**In your output, always state:**
- **Root cause:** one sentence on what's actually wrong and where
- **Owner:** which team/component owns the fix
- **Recommended action:** file ticket / fix in our code / both (temporary workaround + ticket)

Use `.claude/triage/` files for routing. Key decision process:

**Check error origin FIRST:**

1. **Stack trace in `qai_hub_models/`** → likely `ai-hub-models`, BUT check what produced the bad state:
   - `QcQuantizeOp_` prefix in tensor names → `Quantization` (AIMET bug, not ours)
   - Compiler renamed outputs (no `QcQuantizeOp_`) → `Compiler/ONNX2EP`
   - Our logic error on valid data → `ai-hub-models`

2. **External system error** → route per `.claude/triage/error-patterns.md`:
   - Compile failures ("Cannot capture", shape errors) → `Compiler/ONNX2EP`
   - Context binary exit codes (malformed binary) → `Compiler/ONNX2EP`
   - QNN runtime crash ("NPU crashed", "graph execute error") → `Tungsten`
   - TFLite delegate issues → `Compiler/ONNX2EP`
   - OOM / timeout / HTTP 5xx from Hub → `Cloud services`

3. **Dependency / transient** → see `.claude/triage/historical-patterns.md`

**Error severity:**
- ImportError/ModuleNotFoundError → **Blocking** (dependency change)
- TypeError/AttributeError → **Blocking** (API change)
- TimeoutError/connection errors → **Non-blocking** (transient, re-run)
- OOM/exit 137 → **Blocking** (`Cloud services`)

**NEVER assign to a specific person — the czar rotates weekly.**

## Step 5: Post Comment on GitHub Issue (~3 tool calls)

1. Find today's nightly failure issue(s):

   **Preferred:** Use the `ISSUE_URLS` context variable (passed from the workflow). It contains the
   exact issue URL(s) created by the notify_failure job for THIS run. Extract the issue number from
   the URL (e.g., `https://github.com/qcom-ai-hub/tetracode/issues/19062` → `19062`) and post there.

   **Fallback (only if ISSUE_URLS is empty or unavailable):**
   ```
   gh issue list --repo qcom-ai-hub/tetracode \
     --search "[QAIHM Nightly]" \
     --label "p3" --label "ai-hub-models" \
     --state open --limit 5 --json number,title,createdAt
   ```
   There may be two issues: "[QAIHM Nightly] Test Failures" and "[QAIHM Nightly] Workbench Job Failures".
   Post your analysis on the most relevant issue (test issue for test failures, workbench issue for job failures).
   If both exist and your analysis covers both, post on the test failures issue.
   If not found, retry once after 30 seconds.

2. Post the comment using the format below. Keep it under 65,000 characters.

## Output Format

```markdown
## Breeze AI Nightly Analysis

**Run:** [View Workflow]($RUN_URL) | **Date:** YYYY-MM-DD | **Failures:** X across Y suites

---

### Failure Summary

| Category | Count | Key Error | Python Versions |
|----------|------:|-----------|-----------------|
| ... | ... | ... | ... |

<details>
<summary>Category Name (N failures)</summary>

| Test | Error | File |
|------|-------|------|
| ... | ... | ... |

</details>

---

### Suspect Commits

**Last passing:** YYYY-MM-DD HH:MM — `sha` | **Current failing:** YYYY-MM-DD HH:MM — `sha` | **Commits in range:** N

| Commit | Date (UTC) | Author | Message | Related Failure | Suspect? |
|--------|------------|--------|---------|-----------------|----------|
| ... | MM-DD HH:MM | ... | ... | #1 (model_name) | reason |

---

### Root Cause Analysis & Triage

| # | Failure | Root Cause | Owner | Recommended Action | Severity |
|---|---------|------------|-------|-------------------|----------|
| ... | ... | ... | ... | ... | ... |

> Soft recommendations for the nightly czar. Do not assign to individuals.
> If the root cause is in an external dependency, file a ticket to the owning team.

---

<details>
<summary>Agent Reasoning Trace</summary>

**Patterns matched:**
- List which error-patterns.md entries matched each failure
- Note confidence level (HIGH/MEDIUM/LOW)

**Team routing decisions:**
- For each triage recommendation, explain: why this team? what was ruled out?
- If stack trace was in qai_hub_models/ → note "our code, not external"

**Commit bisection:**
- Last passing SHA → current failing SHA
- Which files changed → which tests broke (correlation)
- Commits ruled out and why

**Uncertainties:**
- Any failures where confidence < HIGH
- Patterns not found in KB
- Ambiguous stack traces

**Job logs:** [Full agent output]($RUN_URL) (see "AI Nightly Failure Analysis" job)

</details>

---
*Generated by Breeze AI nightly analyst*
```

## Rules

- Batch `gh` calls — never loop over commits one-by-one
- If JUnit XML parsing fails, fall back to summary.md; if that fails, use job-level data
- Be concise — the czar needs actionable info, not prose
- Include UTC timestamps
- Keep comment under 65,000 characters
- Use fully-qualified cross-repo references (`qcom-ai-hub/ai-hub-models-internal#N`) for all PR/issue numbers — comments are posted on `qcom-ai-hub/tetracode`, so bare `#N` links to the wrong repo
