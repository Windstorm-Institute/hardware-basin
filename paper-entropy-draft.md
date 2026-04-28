# Entropy (MDPI) Submission Draft

# The Hardware Basin: Why the Quantization Cliff Is About Level Allocation, Not Bit Count

**Grant Lavell Whitmer III**

The Windstorm Institute, Fort Ann, NY 12827, USA; grantwhitmer3@gmail.com

---

## Abstract

Paper 7 reported a universal quantization cliff at INT4 to INT3. We investigated whether this cliff is a software artifact or a property of low-precision arithmetic. The cliff location depends on the quantization method. Symmetric uniform quantization produces catastrophic BPT approximately 17 at INT4 (versus 4.3 at FP16). bitsandbytes NF4, which places quantization levels at Gaussian quantiles, produces operational BPT approximately 3.9 to 4.7 at INT4. Same bit count, opposite outcomes. A Lloyd-Max (minimum-MSE) quantizer achieves per-matrix cosine 0.990 at INT4 and 0.965 at INT3. The result is universal across four architectures and consistent across all 24 layers of Pythia-410M. Version 2.1 added: a five-seed replication at Pythia-1.4B yielding Welch t = 633.74, p = 2.84e-15, Cohen's d = 400.81; a Lloyd-Max INT3 end-to-end test showing that per-matrix cosine overstates propagated quality (BPT = 11.74); and a three-model, three-seed robust replication confirming the cliff scales with model size. Version 2.2 (April 2026) adds a methodological-journey narrative (§1.4) tracing the seven rounds of follow-up — including the Round-5 thesis pivot from "cliff at INT4" to "cliff is about level allocation" — and an adversarial-review-defense table (§4.6); Zenodo 10.5281/zenodo.19672922 (concept DOI 10.5281/zenodo.19672921). The minimum viable inference specification is not "N-bit integer" but "N-bit with distribution-aware level allocation, validated end-to-end."

**Keywords:** quantization cliff; level allocation; NF4; symmetric quantization; Lloyd-Max; inference hardware; INT4; weight distribution; structural bonus

---

## 1. Introduction

Paper 7 of the Windstorm Series documented a universal INT4 to INT3 quantization cliff. This paper investigates whether the cliff is a software artifact or a fundamental property of low-precision arithmetic and shows it is neither. The cliff location depends on level allocation.

The full methods, results, statistical tests, the v2.1 verification round, and the v2.2 methodological-journey + adversarial-review-defense framework are in paper.pdf (current version v2.2) in this repository.

---

## References

Whitmer III, G.L. (2026a-g). Papers 1-7, Windstorm Institute.

*This is the MDPI Entropy submission draft. See also: paper-aap-draft.md, paper-arxiv.tex, paper-rsif-draft.md, paper.pdf.*
