# American Academic Publisher Draft

# The Hardware Basin: Why the Quantization Cliff Is About Level Allocation, Not Bit Count

**Grant Lavell Whitmer III**

The Windstorm Institute, Fort Ann, New York 12827, United States of America

Email: grantwhitmer3@gmail.com (Corresponding Author)

---

## Abstract

Paper 7 reported a universal quantization cliff at INT4 to INT3 using bitsandbytes software quantization. We investigated whether this cliff is a software artifact or a mathematical property of low-precision arithmetic. The cliff location depends on the quantization method. Under symmetric uniform quantization, the cliff is at INT8 to INT4: Pythia-410M and Pythia-1.4B both produce catastrophic BPT approximately 17 at INT4 (versus 4.3 at FP16). Under bitsandbytes NF4 (normal-float-4, which places quantization levels at the quantiles of a normal distribution), INT4 is operational: BPT approximately 3.9 to 4.7. Same bit count, opposite outcomes. The difference is level allocation. A Lloyd-Max (minimum-MSE) quantizer achieves cosine similarity 0.990 at INT4 and 0.965 at INT3, outperforming NF4. This result is universal across four model architectures and consistent across all 24 layers of Pythia-410M. Version 2.1 added three publication-grade verifications: GS3 (five-seed replication at Pythia-1.4B, Welch t = 633.74, p = 2.84e-15, Cohen's d = 400.81), Lloyd-Max INT3 end-to-end test (fails at BPT = 11.74), and R4 robust multi-model replication (three models, three seeds each). Version 2.2 (April 2026) adds the publication-readiness presentation: a methodological-journey narrative (§1.4) tracing the seven rounds of follow-up — including the Round-5 thesis pivot where the original "cliff at INT4" hypothesis was reframed to "cliff is about level allocation" — and an adversarial-review-defense table (§4.6) mapping eleven likely peer-review objections to the specific round and result that addresses each. Published v2.2 on Zenodo at 10.5281/zenodo.19672922 (concept DOI 10.5281/zenodo.19672921). The minimum viable inference specification is not "four-bit integer" but "four-bit with distribution-aware level allocation, validated end-to-end." This is the ninth paper in the Windstorm Series.

**Keywords:** quantization cliff, level allocation, NF4, symmetric quantization, Lloyd-Max, inference hardware, INT4, Pythia, Mamba, weight distribution, kurtosis, structural bonus, Cohen's d

---

## 1. Introduction

Paper 7 documented a quantization cliff at INT4 to INT3 across eight language models using bitsandbytes round-to-nearest weight quantization. A critic has a natural objection: bitsandbytes RTN is the crudest quantization method available, and the cliff may be an artifact of the algorithm rather than a property of the arithmetic.

This paper tests that objection directly. What we found was more nuanced: the cliff is real, but its location depends on how the available quantization levels are allocated across the weight distribution.

The full methods, results, statistical tests, the v2.1 verification round, and the v2.2 methodological-journey + adversarial-review-defense framework are in the accompanying paper.pdf (current version v2.2) and Grand Slam Supplementary Materials PDF.

---

## References

See paper.pdf for full reference list. All code and data: github.com/Windstorm-Institute/hardware-basin.

*This is the American Academic Publisher submission draft. See also: paper-arxiv.tex, paper-entropy-draft.md, paper-rsif-draft.md, paper.pdf.*
