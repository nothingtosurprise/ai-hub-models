# Scorecard Regression Analysis

Analyze scorecard performance and numerics regressions, classify by trend,
cluster by root cause, and post a structured triage comment on the GitHub issue.

**Budget: You have ~35 tool calls. Be efficient. Batch queries. Do NOT read files one-by-one.**

## Context Variables

Injected via the workflow prompt:
- `ISSUE_URL` — the regression issue URL in qcom-ai-hub/tetracode
- `RUN_ID` — GitHub Actions run ID
- `RUN_URL` — direct link to the scorecard workflow run
- `DEPLOYMENT` — "prod" or "dev"/"staging"
- `ARTIFACT_NAME` — name of the artifact bundle to download
- `TREND_SUMMARY` — JSON with trend counts (passed inline, no download needed):
  `{"lookback": N, "threshold": N, "total_current": N, "sustained_count": N, "new_count": N, "flaky_count": N, "recovered_count": N}`
  Use these counts directly in the Summary table. If "unavailable", classify all trends as unknown.

## Critical Rules

- ALWAYS use `--json` and `--jq` flags for `gh` API calls — never parse human-readable text
- NEVER use `!=` in `--jq` expressions — bash mangles `!`. Use `select(.x == "y")` instead
- NEVER assign to a specific person — route to teams only (czar rotates weekly)
- Use fully-qualified cross-repo references: `qcom-ai-hub/ai-hub-models-internal#N`
- Keep the final comment under 65,000 characters
- If a file is missing, note it and proceed with available data

## Step 1: Load Regression Data (~4 tool calls)

1. Download the scorecard artifacts:
   ```
   gh run download $RUN_ID -n "$ARTIFACT_NAME" -D /tmp/scorecard-artifacts
   ```
   The agent runner has `actions: read`, so this should succeed. If it fails
   (e.g., the artifact upload step itself failed), fall back to Step 1b.

2. If download succeeded, find and read the perf regressions JSON:
   ```
   find /tmp/scorecard-artifacts -name "perf-regressions-2x-*.json"
   cat <found_path>
   ```
   Each entry has: Model ID, Precision, Component, Device, Runtime, Prev Inference time,
   New Inference time, Kx slower, Job ID, Previous Job ID (prod/staging).

3. If download succeeded, find and read the numerics regressions JSON:
   ```
   find /tmp/scorecard-artifacts -name "numerics-regressions-*.json"
   cat <found_path>
   ```
   Each entry has: Model ID, Dataset Name, Metric Name, Device, Precision, Runtime,
   FP Accuracy, Device Accuracy, Previous FP Accuracy, Previous Device Accuracy.

4. If download succeeded, find and read the full trend report for per-regression classification:
   ```
   find /tmp/scorecard-artifacts -name "trend-report.json"
   cat <found_path>
   ```
   Contains: `sustained`, `new`, `flaky`, `recovered` lists + `summary` stats.

**Step 1b (fallback): If artifact download failed:**
- Read regression data from the issue body via `gh api repos/qcom-ai-hub/tetracode/issues/NUMBER`
- Use the `TREND_SUMMARY` variable (passed inline in your prompt) for aggregate counts
- Note in Reasoning Trace that per-regression trend classification is unavailable

**After loading, note:**
- Total perf regressions count
- Total numerics regressions count
- Trend summary (N new, N sustained, N flaky, N recovered) — from TREND_SUMMARY or trend-report.json

## Step 2: Analyze Trend Classifications (~2 tool calls)

From the trend report, classify perf regressions by priority:

**Priority order: NEW > SUSTAINED > FLAKY**

- **NEW** (first time seen) — highest priority, investigate root cause
- **SUSTAINED** (appeared in 2+ of last 4 runs) — known ongoing issues, link existing tickets
- **FLAKY** (intermittent, 1 of last 4 runs) — likely noise, note but don't escalate
- **RECOVERED** (previously sustained, now gone) — good news, mention briefly

For numerics regressions (no trend data in v1):
- Compare "Previous FP Accuracy" vs "FP Accuracy" — if FP accuracy changed, model weights were updated
- Compare "Previous Device Accuracy" vs "Device Accuracy" — if only device accuracy changed, runtime issue
- Flag regressions where Device Accuracy dropped significantly (>5% absolute) as HIGH priority

## Step 3: Root-Cause Clustering (~8 tool calls)

Group regressions by shared attributes to identify systemic patterns.
Read `.claude/triage/scorecard-patterns.md` for scorecard-specific signals.

### Performance Clustering

1. **By Device** — If 3+ models regress on same device:
   - All runtimes affected → firmware/driver issue (Tungsten)
   - Only one runtime → runtime-specific (Compiler or Tungsten)

2. **By Runtime** — If 3+ models regress on same runtime:
   - QNN only → QNN runtime update (Tungsten)
   - TFLite only → delegate issue (Compiler/ONNX2EP)
   - ONNX only → ORT update (Tungsten)

3. **By Model Family** — If multiple components of same model regress:
   - Across all devices → model-specific issue (AI Hub Models)
   - Single device → device interaction issue

4. **By Precision** — If w8a8/w8a16 cluster together:
   - Quantization tool update (Quantization team)

5. **By Severity** — Flag any >5x factor as CRITICAL regardless of cluster

### Numerics Clustering

1. **Same metric drops across many models** → calibration or AIMET update (Quantization)
2. **Same device, all metrics** → device accuracy regression (Tungsten)
3. **FP accuracy changed** → model weights updated (AI Hub Models)
4. **Single model, all devices** → model code change (AI Hub Models)

### Cross-Reference

Read `.claude/triage/runtime-guide.md` to confirm runtime → team mapping.
Check if any cluster pattern matches `.claude/triage/error-patterns.md` entries.

## Step 4: Soft Triage & Ownership Routing (~5 tool calls)

Read `.claude/triage/teams.md` for team routing rules.

**Map each cluster to a team:**

| Cluster Type | Primary Team | Escalation Condition |
|-------------|-------------|---------------------|
| Device-wide perf regression | Tungsten | >5x or sustained 3+ runs |
| QNN runtime regression | Tungsten | Any new regression |
| TFLite/ONNX compile path | Compiler/ONNX2EP | Any new regression |
| Quantization drift (w8a8/w8a16) | Quantization | >5% accuracy drop |
| Model-specific (all devices) | AI Hub Models | Check recent model PRs |
| Infrastructure (timeouts, OOM) | Cloud Services | Blocking multiple models |

**Severity classification:**

| Level | Criteria |
|-------|----------|
| CRITICAL | >5x factor AND new this run |
| HIGH | >5x factor OR (2-5x AND new AND 3+ models affected) |
| MEDIUM | 2-5x sustained (known ongoing) |
| LOW | Flaky (intermittent, single occurrence) |

**Search for existing tracking issues** (if time permits, 1 call):
```
gh issue list --repo qcom-ai-hub/tetracode --search "scorecard regression" --state open --limit 5 --json number,title,labels
```

## Step 5: Post Structured Comment (~3 tool calls)

1. Extract issue number from ISSUE_URL (e.g., `https://github.com/qcom-ai-hub/tetracode/issues/19062` → `19062`).

2. Build the comment body using the output format below.

3. Post the comment:
   ```
   gh api repos/qcom-ai-hub/tetracode/issues/NUMBER/comments \
     -X POST -f body="<comment_body>"
   ```

**If the comment exceeds 65K characters:**
- Truncate the detailed tables to top 10 entries per cluster
- Add "(N more — see full data in scorecard artifacts)" at the bottom

## Output Format

```markdown
## Breeze AI Scorecard Analysis

**Run:** [View Scorecard]($RUN_URL) | **Date:** YYYY-MM-DD | **Deployment:** $DEPLOYMENT

---

### Summary

| Category | Count | New | Sustained | Flaky |
|----------|------:|----:|----------:|------:|
| Perf Regressions (2x+) | N | X | Y | Z |
| Numerics Regressions | N | — | — | — |

---

### Critical Regressions (>5x slowdown)

| Model | Device | Runtime | Precision | Factor | Trend | Job |
|-------|--------|---------|-----------|--------|-------|-----|
| ... | ... | ... | ... | ... | NEW | [link] |

---

### Performance Clusters

#### Cluster 1: [Description] (N models affected)
**Signal:** [what pattern was detected]
**Team:** @team-name
**Trend:** [NEW/SUSTAINED/FLAKY]

| Model | Device | Runtime | Factor | Trend |
|-------|--------|---------|--------|-------|
| ... | ... | ... | ... | ... |

#### Cluster 2: ...

---

### Numerics Regressions

| Model | Dataset | Metric | Device | Precision | Runtime | Prev Accuracy | New Accuracy |
|-------|---------|--------|--------|-----------|---------|---------------|--------------|
| ... | ... | ... | ... | ... | ... | ... | ... |

**Analysis:**
- [Observations about numerics patterns — FP accuracy changes, device-only drops, etc.]

---

### Trend Context

- **N regressions are NEW** this run → investigate first
- **N are SUSTAINED** since run XXXXX → known ongoing issues
- **N are FLAKY** (appeared 1/4 recent runs) → likely noise
- **N RECOVERED** → previously sustained, now resolved

---

### Triage Recommendations

| # | Cluster/Issue | Severity | Owner | Recommended Action |
|---|--------------|----------|-------|-------------------|
| 1 | ... | CRITICAL | Tungsten | Investigate device firmware on [device] |
| 2 | ... | HIGH | Quantization | Check AIMET version bump |
| 3 | ... | MEDIUM | AI Hub Models | Known issue — link to existing ticket |

> Soft recommendations for the scorecard czar. Do not assign to individuals.
> When the symptom matches an entry in `teams.md` "External Destinations" (QAIRT/AISW JIRA, QDC JIRA, upstream ORT-QNN), add a `Likely tracked in: <link>` line under the affected row. Owner is still the tetracode team — destination is where the fix lands.

---

<details>
<summary>Agent Reasoning Trace</summary>

**Data loaded:**
- Perf regressions: N entries from perf-regressions-2x-TIMESTAMP.json
- Numerics regressions: N entries from numerics-regressions-TIMESTAMP.json
- Trend report: [available/unavailable]

**Clustering decisions:**
- [Which patterns from scorecard-patterns.md matched]
- [Why each cluster was grouped together]
- [Confidence level for each cluster: HIGH/MEDIUM/LOW]

**Team routing:**
- [For each recommendation: why this team? what was ruled out?]

**Uncertainties:**
- [Any regressions where confidence < HIGH]
- [Patterns not found in KB]
- [Ambiguous cases]

</details>

---
*Generated by Breeze AI scorecard analyst*
```

## Rules

- Batch operations — never loop over regressions one-by-one
- If trend-report.json is missing, proceed without trend data (classify all as "unknown trend")
- Be concise — the czar needs actionable info, not prose
- Include UTC timestamps and deployment info
- For job IDs, construct Hub links:
  - Prod: `https://workbench.aihub.qualcomm.com/jobs/JOB_ID`
  - Dev: `https://dev.aihub.qualcomm.com/jobs/JOB_ID`
- Always note which deployment this scorecard ran against
- If zero perf regressions but numerics regressions exist, still post analysis
- If zero regressions of both types, post a brief "no regressions detected" note
