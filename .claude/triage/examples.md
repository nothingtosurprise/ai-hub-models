# Triage Examples

Real triage decisions from the issue history, with reasoning.
Use these as reference for how to classify similar issues.

## Example 1: Clear Compiler/ONNX2EP Bug
**Issue:** #18492 "Cannot capture the entire model: Resize"
- **Labels assigned:** `bug, Compiler/ONNX2EP`
- **Assigned to:** Compiler/ONNX2EP team
- **Reasoning:** "Cannot capture the entire model" is a signature ONNX EP compilation error where QNN EP fails to capture an ONNX operator. The op name (Resize) confirms it's a missing/broken op in the EP.
- **Confidence:** HIGH
- **Similar:** #18490 (Identity), #18376 (BatchNormalization), #18353 (GEMM)

## Example 2: Context Binary Runtime Failure
**Issue:** #18300 "context binary exit code 14: Wrong number of Parameters 6"
- **Labels assigned:** `bug, Compiler/ONNX2EP`
- **Assigned to:** Compiler/ONNX2EP team
- **Reasoning:** "context binary exit code" means a compiled QNN context binary failed during on-device execution. Exit code 14 indicates a parameter count mismatch. This is a compiler bug producing invalid binaries.
- **Confidence:** HIGH
- **Similar:** #18299 (exit code 15: GatherPatches)

## Example 3: ORT Version Regression (ambiguous)
**Issue:** #18531 "Possible Regression in ORT 1.24.3 vs 1.24.1 on X2 Elite CRD only"
- **Labels assigned:** `P0` (no team label!)
- **Assigned to:** Compiler/ONNX2EP team
- **Reasoning:** ORT regression on a specific device (X2 Elite CRD). Could be `ONNXRuntime` or `Compiler/ONNX2EP` or `Devices` (device-specific). Assigned to Compiler/ONNX2EP because they own ORT perf investigations. The "CRD only" qualifier suggests it may also need device team input.
- **Confidence:** MEDIUM — ambiguous between runtime and device teams
- **Better labeling would be:** `P0, ONNXRuntime` or `P0, Compiler/ONNX2EP, Devices`

## Example 4: ONNX2TF Cross-Format Regression
**Issue:** #18530 "ONNX2TF Regression: 5+ Matmul created that is falling back to CPU"
- **Labels assigned:** `P0` (no team label!)
- **Assigned to:** Compiler Service team
- **Reasoning:** ONNX2TF is a conversion pathway in the Compiler Service. Matmul ops falling back to CPU means the converter isn't properly mapping ops to accelerated implementations. This is `Compiler Service` territory, not `Compiler/ONNX2EP`.
- **Confidence:** HIGH for Compiler Service
- **Better labeling would be:** `P0, Compiler Service`

## Example 5: Nightly Failure (auto-filed, needs triage)
**Issue:** #18544 "QAI Hub Models Nightly Tests Failed - 2026-04-08"
- **Labels assigned:** `P0, ai-hub-models`
- **Assigned to:** (unassigned — czar picks up)
- **Reasoning:** Auto-filed by CI. The czar needs to read the CI logs to determine root cause: could be model code change, dependency update, compiler regression, or infrastructure issue. Initial label is correct; downstream triage happens after log analysis.
- **Confidence:** HIGH for initial label, further triage needed

## Example 6: GPU Nightly (specific sub-team)
**Issue:** #18447 "GPU nightly Tests Failed - 2026-04-03"
- **Labels assigned:** `P0, ai-hub-models, gen-ai`
- **Assigned to:** gen-ai team
- **Reasoning:** GPU nightly tests are a specific subset of model tests focused on GenAI models with GPU inference. The `gen-ai` label distinguishes from regular model nightly failures.
- **Confidence:** HIGH

## Example 7: Deployment Failure (auto-filed)
**Issue:** #18461 "Deployment Build&Test Failure April 3, 2026"
- **Labels assigned:** `P0, Deployment`
- **Assigned to:** Deployment team
- **Reasoning:** Post-deployment test failure. Title pattern is consistent and auto-filed. Deployment team rotation handles these.
- **Confidence:** HIGH

## Example 8: Infrastructure OOM
**Issue:** #18040 "Nightly scorecard OOMKilled (exit code 137) in torch_3_of_4"
- **Labels assigned:** `P0, bug`
- **Assigned to:** Cloud services team
- **Reasoning:** Exit code 137 = OOMKilled by Linux kernel. "torch_3_of_4" identifies the CI split. This is an infrastructure issue (job memory limits too low), not a model bug.
- **Confidence:** HIGH for Cloud services
- **Better labeling would be:** `P0, bug, Cloud services`

## Example 9: Accuracy Gap (model team)
**Issue:** #18303 "Large accuracy gap between published benchmarks and scorecard eval for 16 models"
- **Labels assigned:** `P2, ai-hub-models`
- **Assigned to:** ai-hub-models team
- **Reasoning:** Discrepancy between published benchmarks and actual scorecard evaluation results. This is a model validation issue owned by the ai-hub-models team.
- **Confidence:** HIGH

## Example 10: LiteRT / Tungsten
**Issue:** #18547 "litert build path is showing up in logs"
- **Labels assigned:** `Tungsten, P2`
- **Assigned to:** Tungsten team
- **Reasoning:** LiteRT build issue. "litert" keyword maps directly to Tungsten team. Build path leaking into logs is a packaging/build issue.
- **Confidence:** HIGH

## Example 11: Batch Norm Regression (ambiguous)
**Issue:** #18511 "Regression: Possible miss in Batch Norm folding in ORT"
- **Labels assigned:** `P0` (no team label!)
- **Assigned to:** ONNXRuntime team
- **Reasoning:** Batch normalization folding is an ONNX Runtime optimization pass. "Possible miss" means the optimization isn't being applied, causing a perf regression. Could be `ONNXRuntime` or `Compiler/ONNX2EP`.
- **Confidence:** MEDIUM
- **Better labeling would be:** `P0, ONNXRuntime` or `P0, Compiler/ONNX2EP`

## Example 12: Device Enablement
**Issue:** #18352 "[Nord Robotics] Enable in AI Hub Models"
- **Labels assigned:** `P1, Blocked, Devices`
- **Assigned to:** ai-hub-models team
- **Reasoning:** New device (Nord Robotics / QAM8797P) needs to be added to AI Hub Models. The `Blocked` label indicates a dependency on other enablement steps. Part of a larger epic (#18347).
- **Confidence:** HIGH

## Example 13: AIMET Quantization Bug — NOT Our Code (CRITICAL)
**Nightly failure:** `ddrnet23_slim::test_export[w8a8-qnn_dlc-cs_8_elite]`
- **Error:** `ValueError: Output 'mask' from get_output_spec() not found in compiled model metadata. Available outputs: ['QcQuantizeOp_mask_q']`
- **Mentions:** QNN, quantization, DLC, compiled model metadata
- **Key signal:** `QcQuantizeOp_` prefix + `_q` suffix = **AIMET quantization artifacts**
- **WRONG triage:** `ai-hub-models` — "our code is too strict, fix merge_output_metadata()"
- **WRONG triage:** `Compiler/ONNX2EP` — "compiler renamed the outputs"
- **CORRECT triage:** `Quantization` — AIMET is leaking internal quantization op names into the compiled model's output names. AIMET should produce outputs with the original names (`mask`), not wrapped names (`QcQuantizeOp_mask_q`).
- **Confidence:** HIGH for Quantization
- **Real-world resolution:** Filed as tetracode#19066 + AIMET-4534. Assigned to Quantization team.
- **Lesson:** `QcQuantizeOp_` prefix is the signature of an AIMET bug. Do NOT propose "fuzzy match" workarounds in our code — that would mask the real issue. The fix belongs in AIMET.
- **How to verify:** Search tetracode: `gh issue list --repo qcom-ai-hub/tetracode --search "<model_name> AIMET" --label Quantization --state open`

## Example 14: Dependency Wheel Failure
**Nightly failure:** `pipertts_de/en/it::environment_setup` on py3.12 + py3.13
- **Error:** `Install QAIHM[dev,pipertts_*] (wheel) failed — piper-phonemize==1.1.0 has no matching wheel`
- **CORRECT triage:** `ai-hub-models` (model owner from the PR that added pipertts)
- **Reasoning:** Missing wheels for certain Python versions/platforms is a model dependency issue. The fix is either adding supported wheels, pinning compatible versions, or marking the model as unsupported on those Python versions.
- **Confidence:** HIGH

## Example 15: Link Job Failure — Compiler/ONNX2EP
**Issue:** #19030 "CVT w8a16_mixed_fp16 AOT fails, but JIT works"
- **Labels assigned:** `Compiler/ONNX2EP`
- **Reasoning:** Link job (ctx-bin-gen) fails with dtype mismatch for Depthwise Conv2d (Fp16 activation + Int8 weights). The DLC compiles fine but the AOT link step fails. Link job failures route to `Compiler/ONNX2EP`.
- **Confidence:** HIGH
- **Lesson:** Link job failures route to `Compiler/ONNX2EP`.

## Example 16: Link Job Failure — ai-hub-models (disabled model)
**Nightly failure:** `pipertts_*` link jobs failed on cs_8_elite and cs_x_elite
- **Error:** 6 pipertts encoder link jobs failed
- **Context:** Model was made unpublished in same nightly window (commit `47c58c40`)
- **CORRECT triage:** `ai-hub-models` — test infrastructure should skip link jobs for disabled/unpublished models
- **WRONG triage:** `Compiler/ONNX2EP` — this is not a compiler bug
- **Confidence:** HIGH
- **Lesson:** When a model is disabled/unpublished and its jobs start failing, the fix is in our test infrastructure (skip the jobs), not in the compiler.

## Example 17: TFLite Regression Coinciding with QAIRT Bump — Compiler/ONNX2EP
**Issue:** #18933 "SINet TFLite regression"
- **Labels assigned:** `P1, Compiler/ONNX2EP`
- **Reasoning:** TFLite regression appeared at same time as QAIRT 2.45 upgrade. TFLite ships its on-device compilation with QAIRT — all paths must compile to context binary on device.
- **WRONG triage:** `Tungsten` — this is not a device runtime issue
- **Confidence:** HIGH
- **Lesson:** TFLite regressions that coincide with QAIRT version bumps → `Compiler/ONNX2EP`.

## Example 18: Float Accuracy Drop on Specific Device — Compiler/ONNX2EP
**Issue:** #18939 "Swin accuracy regression on Samsung Galaxy S25"
- **Labels assigned:** `Compiler/ONNX2EP`
- **Context:** Float precision accuracy drops on S25 with ONNX runtime, other devices fine
- **WRONG triage:** `ai-hub-models` (accuracy issue)
- **CORRECT triage:** `Compiler/ONNX2EP` — device-specific float accuracy drop = EP execution issue
- **Confidence:** HIGH
- **Lesson:** Accuracy regressions that are device-specific AND float precision on ONNX runtime → `Compiler/ONNX2EP`, not `ai-hub-models`.

## Anti-Pattern: What NOT to Do

### Don't assume stack trace location determines ownership
A stack trace in `qai_hub_models/` does NOT always mean it's our bug. For example, output names being incorrect results in a raise in our code, but the root cause is in the compiler stack (the compiler renamed outputs incorrectly).

Check: "What produced the bad state?" — not "Where did the code crash?"
- Bad data from compiler (e.g., renamed outputs) → compiler team, even if our code raises
- Our logic error processing valid compiler output → ai-hub-models

### Don't confuse ONNX EP compilation with ONNX Runtime
- "Cannot capture" / "subgraph" / "exit code" → `Compiler/ONNX2EP` (compilation phase)
- "ORT version regression" / "segments" → `ONNXRuntime` (runtime phase)

### Link job failures → Compiler/ONNX2EP
Link job failures (DLC → context binary) should be routed to `Compiler/ONNX2EP`. This is consistent with the error patterns routing.

Exception: If a link failure is caused by our code sending the wrong input (e.g., "Input model must be a QNN DLC model"), route to `ai-hub-models`.

### Don't assign nightly failures to a specific person
The czar rotates weekly. Assign to the `ai-hub-models` label and let the current czar pick up.
