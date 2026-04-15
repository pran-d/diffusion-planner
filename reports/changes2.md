# Thesis Discrepancy Report
**Thesis:** Imitation Learning from Sparse Demonstrations for Kinematic Motion Planning in Robot Loco-Manipulation
**Author:** Pranav Debbad

---

## 1. Structural & Outline Errors

### Chapter numbering mismatch in outline (§1.4) — *Error*
The thesis outline in §1.4 maps "Chapter 7 → limitations/practical implications" and "Chapter 8 → conclusion," but the actual thesis has Chapter 7 = RL Tracking, Chapter 8 = Discussion, and Chapter 9 = Conclusion. The outline is off by two chapters and omits Chapter 7 entirely.

### Table of contents lists 8 chapters; body has 9 — *Error*
The ToC lists chapters 1–9 correctly, but §1.4 (Thesis Outline) describes only 8 chapters: "Chapter 7 discusses limitations… Chapter 8 concludes." Chapter 7 (RL Tracking) is completely unmentioned in the written outline, suggesting it was inserted late without updating §1.4.

---

## 2. Internal Cross-Reference Errors

### §8.2 cites "Chapter 5 evaluation design" imprecisely — *Error*
The Discussion (§8.2) states "consistent with the Chapter 5 evaluation design, where endpoint error… are complementary rather than interchangeable." This is an over-broad chapter-level reference that should point to specific sections or tables rather than the whole chapter.

### §6.1 cites Equation 5.3 for an unrelated claim — *Inconsistency*
Section 6.1 states that reconstruction behavior is "consistent with… Equation 5.3 (used in the pipeline)." Equation 5.3 is the 6D rotation encoding formula — it is unrelated to the reconstruction quality claim being made. The intended citation is likely the diffusion reconstruction objective or the training setup.

### §8.1 omits Table 6.2 from its supporting citations — *Inconsistency*
The Discussion says "ablations in Table 6.1 and Table 6.3 further show that this representational benefit depends on conditioning design," while ignoring the auxiliary-loss ablation in Table 6.2, which is discussed in the same paragraph and is equally relevant to the same claim.

### §7.5 implies an unestablished causal link via Figure 6.5 — *Inconsistency*
Section 7.5 states: "This trend is consistent with the planner-side displacement evidence in Figure 6.5." The phrasing "this trend" implies a causal relationship between kinematic goal-following and evolutionary coverage expansion that is not empirically established in the thesis.

---

## 3. Content & Claim Inconsistencies

### Smoothness loss defined two different ways — *Error*
In §4.6, the smoothness loss is defined as matching **first-order temporal differences** against the reference (a velocity-matching loss). In §5.3.3 (Equation 5.8), it is implemented as a **second-difference penalty** (∥x_{t+1} − 2x_t + x_{t−1}∥²), which is an acceleration penalty. These are mathematically distinct formulations. The thesis never acknowledges the discrepancy or states that one supersedes the other.

### Grasp loss defined two different ways — *Error*
In §4.6, `L_grasp` uses a soft proximity gate γ_t based on object distance, making supervision strong when the object is close. In §5.3.3 (Equation 5.9), the grasp loss is re-defined using a binary contact mask `m^grasp_t` and a fixed reference offset `d_0`. The two formulations are not equivalent, and neither is stated to supersede the other.

### Abstract description of dynamics is potentially misleading — *Misleading*
The abstract correctly states that trajectories "are not dynamically solved during diffusion sampling." However, §7.2 describes the RL tracker as also acting as a "motion cleaner" correcting "small physical inconsistencies" in the diffusion reference. The abstract should more explicitly acknowledge this two-stage nature to avoid implying the diffusion output is used without dynamic correction.

### History-size ablation: best config unsupported by table — *Gap*
The caption for Figure 6.4 states "A history size of 2 works best for the current planning horizon," but Table 6.1 only includes history sizes 1 and 3 alongside the base configuration. History size 2 is never isolated in the table, leaving the claim without direct tabular support.

### Combined auxiliary loss underperforms single losses without explanation — *Gap*
Table 6.2 shows "grasp only" achieves the best mean error (0.4447 m) and Succ@20cm (0.5833) of any ablation, yet the combined "smoothness + grasp" configuration performs substantially worse (0.9531 m error). The text acknowledges a "non-trivial weighting trade-off" but never explains the mechanism behind this regression or what it implies for the final configuration choice.

### L=8 transformer blocks unjustified — *Gap*
The architecture section (§4.5) fixes L = 8 transformer blocks and 4 attention heads as design choices. However, §4.3.4 (ablation-informed design choices) does not include any depth or head-count ablation. This architectural decision is presented without empirical justification.

---

## 4. Bibliography Errors

### References [7] and [8] are incomplete — *Formatting Error*
References [7] (OmniRetarget) and [8] (DynaRetarget) are missing all author, venue, and page information. They appear as title-only strings in quotation marks, which is likely a placeholder that was never completed:

> [7] "OmniRetarget: Interaction-preserving data generation for humanoid whole-body loco-manipulation and scene interaction," 2025.
> [8] "DynaRetarget: A complete pipeline for retargeting human motions to humanoid control policies," 2026.

Both are cited prominently in §2.3 and §3.4 and should be completed with full author and venue information.

---

## 5. Minor / Notation Issues

### Research questions not revisited in conclusion — *Minor*
Two explicit research questions are posed in §1.2. Neither is directly answered in §9 (Conclusion), which instead lists general findings. A thesis should close the loop on its stated research questions.

### ε used for two different quantities — *Minor*
In §2.1, ε appears as an endpoint-error tolerance threshold (∥ψ(o_H, o_0) − g∥₂ ≤ ε_g). In §4.3.1 and §4.6, ε is used as a small numerical stability constant in division operations. The same symbol is used for fundamentally different quantities without disambiguation.

### H overloaded as both history length and prediction horizon — *Minor*
H is used for both the history window length (§2.1: "h_t = (s_{t−Hh+1}, …, s_t)") and the future prediction horizon ("outputs τ_{t+1:t+H}"). In several equations both roles appear simultaneously without subscript distinction (e.g., §4.2), creating ambiguity.