# Error Pattern Recognition

This file maps common error signatures and title patterns to labels and teams.
Only patterns relevant to AIHM nightly failures are included.
Patterns are ordered by confidence level: HIGH patterns are near-certain, MEDIUM need verification.

## HIGH Confidence Patterns

### Nightly/Weekly Test Failures (auto-filed)
| Title Pattern | Label | Notes |
|--------------|-------|-------|
| "QAI Hub Models Nightly Tests Failed" | `P0, ai-hub-models` | Czar triages further. |
| "GPU nightly Tests Failed" | `P0, ai-hub-models, gen-ai` | GPU model failures. |
| "GPU weekly Tests Failed" | `P0, ai-hub-models, gen-ai` | GPU model failures. |

### QAIHM Internal Code Failures
These errors originate in our own codebase (`qai_hub_models/`), NOT in the compiler or runtime.
Even if the error message mentions runtime keywords (QNN, DLC, ONNX), always check the stack trace first.
If the traceback points to our code, route to `ai-hub-models`.

| Error Signature | Source File | Label | Notes |
|----------------|-------------|-------|-------|
| "Output '<name>' not found in compiled model metadata" + `QcQuantizeOp_` in available outputs | `configs/model_metadata.py` (`merge_output_metadata`) | `Quantization` | **AIMET bug.** `QcQuantizeOp_` prefix = AIMET quantization artifacts leaking into output names. |
| "Output '<name>' not found in compiled model metadata" (NO `QcQuantizeOp_` prefix) | `configs/model_metadata.py` (`merge_output_metadata`) | `Compiler/ONNX2EP` | **Compiler bug.** Compiler is renaming outputs — our metadata merge expects stable output names. |
| "Input '<name>' not found in compiled model metadata" + `QcQuantizeOp_` in available inputs | `configs/model_metadata.py` (`merge_input_metadata`) | `Quantization` | **AIMET bug.** Same as output pattern — AIMET renaming inputs. |
| "Input '<name>' not found in compiled model metadata" (NO `QcQuantizeOp_` prefix) | `configs/model_metadata.py` (`merge_input_metadata`) | `Compiler/ONNX2EP` | **Compiler bug.** Compiler renaming inputs incorrectly. |
| ValueError/KeyError in `export.py` | `models/*/export.py` | `ai-hub-models` | Export script logic error. Check which PR last modified the export template or the model's export.py. |
| Errors in `testing_export_eval.py` | `utils/testing_export_eval.py` | `ai-hub-models` | Test infrastructure bug — compile_via_export, link_via_export, etc. |
| Errors in `scorecard/` | `scorecard/*.py` | `ai-hub-models` | Scorecard infrastructure — execution_helpers, collect_results, etc. |
| "Install QAIHM[dev,<model>] (wheel) failed" | Model's `requirements.txt` | `ai-hub-models` | Dependency doesn't have wheels for the CI Python version or platform. Route to model owner. |
| Codegen/template errors | `scripts/templates/*.j2` | `ai-hub-models` | Jinja template rendering failure in codegen. |
| "Check for code-gen changes" pre-commit failure | `scripts/run_codegen.py` | `ai-hub-models` | Committed code-gen output is stale. Re-run `run_codegen.py` for the affected model. |

**Key principle:** If the stack trace is in `qai_hub_models/`, it's almost always `ai-hub-models` regardless of which runtime or compiler is mentioned in the error message.

### ONNX EP Compiler Bugs
| Error Signature | Label | Notes |
|----------------|-------|-------|
| "Cannot capture the entire model: <OpName>" | `bug, Compiler/ONNX2EP` | Always this team. |
| "N subgraphs" or "incomplete capture" | `bug, Compiler/ONNX2EP` | Partial model capture. |
| "context binary exit code <N>:" | `bug, Compiler/ONNX2EP` | Device-specific compilation failure. |
| "OpType:<Op> requested by QnnExecutionProvider but not selected" | `bug, Compiler/ONNX2EP` | Missing QNN EP op support. |
| "[ShapeInferenceError] Incompatible dimensions" | `bug, Compiler/ONNX2EP` | Shape mismatch during EP capture. |
| "Type parameter (T) of Optype (<Op>): <type1> vs. <type2>" | `bug, Compiler/ONNX2EP` | Data type mismatch. |

### Device/Runtime Crashes
| Error Signature | Label | Notes |
|----------------|-------|-------|
| "NPU crashed. SSR detected." | `bug, Tungsten` | Hardware crash during on-device execution. |
| "QNN graph execute error. Error code:" | `bug, Tungsten` | Runtime execution failure. |

### Link Job Failures
**Important:** Link job failures route to `Compiler/ONNX2EP`.

| Error Signature | Label | Notes |
|----------------|-------|-------|
| "Weight Sharing is not supported on v<XX> target" | `Compiler/ONNX2EP` | Older chipsets lack weight-sharing support. Device limitation. |
| "Graph name is a duplicate. Aborting creation of graph" | `ai-hub-models` | Likely our serialization config issue. |
| AOT link fails but JIT compile succeeds (same model/device) | `Compiler/ONNX2EP` | QAIRT converter dtype mismatch in AOT path. |
| Link job fails for disabled/unpublished model | `ai-hub-models` | Model should be excluded from link test suite. Fix: update `code-gen.yaml` or test infrastructure. |
| "Input model must be a QNN DLC model" | `ai-hub-models` | Wrong compile target — model was compiled to context binary instead of DLC before linking. |

### Scorecard Auto-Filed Issues
| Title Pattern | Label | Notes |
|--------------|-------|-------|
| "[Scorecard] 2x+ Regressions Detected" | `P1, ai-hub-models` | Auto-filed by regression detection. Triaged by splitting into per-model sub-issues. |

### Transient / Sporadic Failures (No code fix needed)

**Important:** Before classifying a GitHub 5xx as transient, check whether the URL recently changed (repo transferred, branch renamed). Persistent 502s on a single URL with the rest of the test suite passing = stale URL, NOT a transient outage. See Example 19.

| Error Signature | Label | Notes |
|----------------|-------|-------|
| `images.cocodataset.org` connection timeout/reset | Sporadic | External dataset host. Resolves within hours. Re-run. |
| Imagenet download SSL error or timeout | Sporadic | External dataset. SSL cert rotation or network blip. |
| HuggingFace `HTTPError 429` or connection timeout | Sporadic | Rate limited or HF outage. Re-run after 30 min. |
| `git clone` timeout / GitHub API 5xx (multiple unrelated URLs) | Sporadic | Transient GitHub connectivity. Re-run. |
| GitHub 502 on ONE specific URL (license, model source) + other tests pass | `ai-hub-models` | **NOT transient.** Repo was likely transferred/renamed. Fix the URL. See Example 19. |
| `An action could not be found at the URI` + `codeload.github.com` | Sporadic | GitHub Actions CDN outage. All affected jobs share same error at "Set up job" step. Re-run. |
| Qualcomm internal network / proxy SSL failure | Sporadic | Internal network blip. Re-run. |
| `KeyError: 'Unable to synchronously open object'` | Sporadic | Corrupted HDF5 on CI machine. Re-run or clear cache. |
| "Internal compiler error" + "status code 500" + "CompleteMultipartUpload" | `Cloud services` | Transient S3 multipart upload failure. Re-run. |
| Job fails immediately after submission with no compile logs | `bug, Cloud services` | May be WorkerLostError with failed retry. |

### Workbench Service Issues (Not a compiler/code bug)
| Error Signature | Label | Notes |
|----------------|-------|-------|
| 3+ unrelated models timeout simultaneously (>60 min) | `Cloud services` | Service overload. Distinct from compiler bug (which hits 1-3 specific models). |
| `HTTP 503` / `504` from Hub API during job submission | `Cloud services` | Workbench service issue. Re-run after service recovers. |
| All quantize jobs timeout in same window | `Cloud services` | Quantize service capacity issue. |
| Dev environment SSL errors (internal, not external host) | `Cloud services` | Deployment issue. |

### Dependency Breakage (External change, fix in our code)
| Error Signature | Label | Notes |
|----------------|-------|-------|
| `ImportError: cannot import name ... from 'pkg_resources'` | `ai-hub-models` | setuptools removed pkg_resources. Pin or migrate. |
| `numpy` build/wheel failure on Python 3.12/3.13 | `ai-hub-models` | numpy wheels not yet available. Pin numpy version. |
| `ModuleNotFoundError: No module named 'tflite_runtime'` | `ai-hub-models` | Package renamed to `litert`. Update requirements. |
| QAIRT version not found / `hub.get_devices()` error after version removal | `ai-hub-models` | Old QAIRT version dropped from workbench. Upgrade pin. |
| Third-party model library API change (ultralytics, etc.) | `ai-hub-models` | Library updated, model code needs adapting. |
| Workbench client private API change (config class reordered) | `ai-hub-models` | Private API changed. Update client usage. |
| GitHub repo transferred → old URL returns persistent 502 | `ai-hub-models` | Update URL in info.yaml or `_info_yaml_enums.py`. NOT a transient failure. |

## MEDIUM Confidence Patterns

### Runtime/Compiler Regressions
| Error Signature | Likely Label | Disambiguation |
|----------------|-------------|----------------|
| "Regression" + "QNN" | `Compiler/ONNX2EP` or `perf` | Check if QAIRT version changed between runs. |
| "Regression" + "TFLite" or "LiteRT" | `Compiler/ONNX2EP` | Likely QAIRT version bump — all paths compile to context binary. |
| "Regression" + "ORT" or "ONNX Runtime" | `Tungsten` or `perf` | If perf regression → `perf`. If functional → `Tungsten`. |
| Float accuracy regression + device-specific + ONNX runtime | `Compiler/ONNX2EP` | Device-specific float accuracy drops on ORT = EP execution issue. |

### Model-Level Issues
| Error Signature | Likely Label | Disambiguation |
|----------------|-------------|----------------|
| "accuracy" + model name | `ai-hub-models` | Scorecard accuracy threshold violation. |
| "fails on all runtimes" | `ai-hub-models` | Check if model was recently updated. |
| Model test assertion failure (expected vs actual output mismatch) | `ai-hub-models` | Check if S3 test assets or model weights were updated without updating expected values. |

### Quantization / AIMET Issues
| Error Signature | Likely Label | Disambiguation |
|----------------|-------------|----------------|
| `QcQuantizeOp_` prefix or `_q` suffix on tensor names | `Quantization` | **AIMET artifact.** AIMET is leaking quantization op names into model outputs/inputs. This is an AIMET bug. Do NOT propose fuzzy-match workarounds in ai-hub-models. |
| "PSNR" or "calibration" | `Quantization` | Quantization quality metrics. |
| "QAIRT Converter" + quantization keyword | `Compiler/ONNX2EP` | Compiler-level quantization support. |

## Cross-Label Patterns

Some issues legitimately span multiple teams:

| Pattern | Primary Label | Secondary Label | Route to |
|---------|--------------|----------------|----------|
| Perf regression on ONNX EP | `perf` | `Compiler/ONNX2EP` | Primary: Compiler/ONNX2EP |
| Scorecard accuracy + compiler issue | `ai-hub-models` | `Compiler/ONNX2EP` | Primary: ai-hub-models, escalate if compiler root cause |
| LiteRT + device enablement | `Tungsten` | `Compiler/ONNX2EP` | Primary: Tungsten |
