# Scorecard-Specific Patterns

Patterns specific to scorecard performance and numerics regressions.
Use alongside `error-patterns.md` and `runtime-guide.md` for triage.

---

## Performance Regression Signatures

### Device Firmware / Driver Updates

| Signal | Confidence | Team |
|--------|-----------|------|
| Multiple unrelated models regress on same device, same run | HIGH | Tungsten |
| Regression appears only on one chipset generation | MEDIUM | Tungsten |
| All runtimes affected on same device | HIGH | Tungsten (firmware) |
| Only QNN runtime affected on a device, TFLite fine | MEDIUM | Compiler/ONNX2EP |

### Runtime Version Bumps

| Signal | Confidence | Team |
|--------|-----------|------|
| All ONNX Runtime models regress, QNN fine | HIGH | Tungsten (ORT) |
| All TFLite models regress | HIGH | Compiler/ONNX2EP (delegate) |
| Only context binary path affected | HIGH | Compiler/ONNX2EP |
| QNN models regress across all devices | HIGH | Tungsten (QNN runtime update) |

### Noise vs Real (Threshold Guidance)

| Factor | Duration | Classification |
|--------|----------|---------------|
| 2-3x | Single run, then recovers | FLAKY — likely device load/thermal |
| 2-3x | 3+ consecutive runs | SUSTAINED — real regression |
| >5x | Any | CRITICAL — always investigate |
| 2x | Absolute time diff < 1ms | NOISE — within measurement error |
| 2-3x | Only on automotive devices (SA8775P) | Check if device was under load — automotive benchmarks are noisier |

### Infrastructure / Cloud Issues

| Signal | Confidence | Team |
|--------|-----------|------|
| Regression coincides with known Hub maintenance window | HIGH | Cloud Services |
| Same model shows wildly different times across retries | MEDIUM | Cloud Services (device pool contention) |
| All models on one device show ~same factor increase | HIGH | Cloud Services (throttled device) |

---

## Numerics Regression Signatures

### Quantization Drift

| Signal | Confidence | Team |
|--------|-----------|------|
| w8a8/w8a16 precision only, float16 fine | HIGH | Quantization (AIMET) |
| PSNR/mAP drops across many models same run | HIGH | Quantization (calibration data or AIMET update) |
| Single model, single metric | MEDIUM | AI Hub Models (model code change) |
| `QcQuantizeOp_` in any related error | HIGH | Quantization |

### Reference Model Updates

| Signal | Confidence | Team |
|--------|-----------|------|
| FP Accuracy changed vs previous (compare "Previous FP Accuracy") | HIGH | AI Hub Models (torch model weights updated) |
| Device accuracy changed but FP accuracy stable | HIGH | Compiler/Tungsten (runtime or compile change) |
| New metric appears with no previous data | LOW | Not a regression — new coverage added |

### Device Accuracy Issues

| Signal | Confidence | Team |
|--------|-----------|------|
| Same model fails on all devices | MEDIUM | AI Hub Models or Quantization |
| Same model fails on one device only | HIGH | Tungsten (device-specific runtime bug) |
| Accuracy within 1% of threshold | LOW | Borderline — may be measurement variance |

---

## Deployment-Specific Patterns

| Pattern | Interpretation | Action |
|---------|---------------|--------|
| Regression in prod but NOT in dev | Prod-specific issue or already fixed in dev | Check dev scorecard results |
| Regression in dev but NOT in prod | Compiler/runtime team testing unreleased changes | Flag but don't escalate |
| Regression in BOTH prod and dev | Systemic issue (shared infrastructure or model change) | Escalate — affects all environments |
| New regression only in staging | Staging environment instability | Low priority unless sustained |

---

## Known Flaky Model/Device Combos

<!-- This section is populated over time by the kb-weekly-update agent.
     Add entries as patterns are confirmed via multiple scorecard runs. -->

| Model | Device | Runtime | Notes |
|-------|--------|---------|-------|
| (initially empty) | | | |

---

## Cross-Referencing with Trend Data

When the trend report (`trend-report.json`) is available:

- **NEW regressions** → investigate first (highest priority)
- **SUSTAINED regressions** → known issues, link to existing tracking tickets if available
- **FLAKY regressions** → likely noise, mention but don't escalate
- **RECOVERED regressions** → good news, mention briefly for awareness
