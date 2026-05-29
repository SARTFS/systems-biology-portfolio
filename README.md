# ACRV: Tat-Dependent Conditionally Replicating Virus
## Mathematical Modeling of a Triple-Safety HIV Gene Therapy Platform

**Author**: Lovinia  
**Background**: Computer Science undergraduate, self-studying Biochemistry & Molecular Biology  

---

## Project Overview

This project presents a deterministic ODE model of an engineered viral vector designed for HIV latency clearance. The system integrates three independent safety mechanisms, mathematically proven through parameter sensitivity analysis and design space optimization.

**Biological Design**:
- **Conditional replication**: Strictly dependent on HIV Tat protein (LTR-driven)
- **Gene silencing**: shRNA expression targeting HIV conserved regions
- **Immune recruitment**: Engineered Env protein (molecular design pending)

**Mathematical Proof**: Complete parameter scan (625 combinations) identifies a robust "Goldilocks window" where therapeutic efficacy, off-target safety, and self-clearance are simultaneously satisfied.

---

## Key Results

### Triple-Safety Mechanisms

| Layer | Mechanism | Mathematical Evidence |
|-------|-----------|----------------------|
| **1. Tat Threshold Gating** | LTR promoter requires Tat transactivation | Sharp activation threshold at Tat₀ > 0.05; complete silence below |
| **2. shRNA Self-Limiting** | Negative feedback loop: ACRV → shRNA → Tat silencing | Pulse-like kinetics with deterministic clearance |
| **3. Provirus Dependency** | No HIV DNA → no Tat source → no replication | Selectivity index: **169.2×** (infected/uninfected) |

### Critical Finding: False Optimum vs True Optimum

Initial analysis identified a parameter set with extreme selectivity (907×) but failed clearance validation:

| Parameter Set | Selectivity | ACRV Peak | t=300 Clearance | Status |
|--------------|-------------|-----------|-----------------|--------|
| k_sh=0.05, K_sil=5.0 | **907×** | 4.97 | **4.9 (FAIL)** |  False optimum |
| **k_sh=1.675, K_sil=1.325** | **169×** | 3.05 | **0.0998 (PASS)** | True optimum |

**Lesson**: Extreme selectivity without complete self-destruction creates chronic low-level replication. Balanced parameters achieve pulse amplification + full clearance.

### Design Space Analysis

**Scan grid**: 25×25 = 625 combinations (k_shrna: 0.05–2.0, K_silence: 0.1–5.0)

**True Goldilocks window**: 52 parameter combinations satisfying:
- Selectivity index > 100×
- ACRV peak > 2.0
- ACRV at t=300 < 0.1

**Optimal balance point**:

---

## Model Evolution

| Version | File | Variables | Milestone |
|---------|------|-----------|-----------|
| v0.1 | `Tat-LTR.py` | 2 (Tat, ACRV) | Baseline positive feedback |
| v0.2 | `shRNA.py` | 3 (+shRNA) | Self-limiting negative feedback |
| v0.3 | `ACRV-Provirus.py` | 4 (+Provirus) | Off-target safety proof |
| v1.0 | `Parameter_Sweep.py` | 4 + analysis | Threshold sensitivity |
| v1.1 | `ACRV-thermodynamic_1.py` | 4 + t=300 | Clearance validation |
| **v1.2** | **`ACRV-thermodynamic_2.py`** | **4 + 2D scan** | **True Goldilocks window** |

---

## Files

### Core Models

| File | Description | Key Output |
|------|-------------|------------|
| `Tat-LTR.py` | Baseline positive feedback | Tat-ACRV co-amplification curve |
| `Parameter_Sweep.py` | Initial Tat threshold scan | Threshold ~0.05 for ACRV activation |
| `shRNA.py` | Three-variable with negative feedback | Pulse-like ACRV kinetics |
| `ACRV-Provirus.py` | Four-variable with HIV provirus | Selectivity index calculation |
| `ACRV-thermodynamic_1.py` | Extended t=300 simulation | False optimum identification |
| `ACRV-thermodynamic_2.py` | **2D parameter sensitivity heatmap** | **52 Goldilocks combinations** |

### Figures

All figures in `project-01-tat-ltr/figures/`:

| Figure | Description |
|--------|-------------|
| `tat_ltr_baseline.png` | Baseline Tat-ACRV dynamics (lag/exponential/saturation phases) |
| `tat_ltr_with_shrna_feedback.png` | Self-limiting pulse: Tat, ACRV, shRNA, efficiency |
| `acrv_design_space_heatmap.png` | Initial 2D scan (3 metrics) |
| `acrv_extended_validation_t300.png` | t=300 clearance: optimal vs strong shRNA comparison |
| `acrv_true_design_space.png` | **True Goldilocks window (4-panel)** |

---
## Dependencies

- Python 3.10+
- NumPy
- SciPy
- Matplotlib

`Complete README with project results and design space analysis`
