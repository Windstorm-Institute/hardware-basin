# The Hardware Basin: Why the Quantization Cliff Is About Level Allocation, Not Bit Count

Grant Lavell Whitmer III

The Windstorm Institute, Fort Ann, NY 12827, USA

Email: grantwhitmer3@gmail.com

ORCID: 0009-0007-3224-755X

---

## Abstract

Paper 7 reported a universal quantization cliff at INT4 to INT3. We investigated whether this cliff is a software artifact or a property of low-precision arithmetic. The cliff location depends on the quantization method. Symmetric uniform quantization produces catastrophic BPT approximately 17 at INT4 (versus 4.3 at FP16). bitsandbytes NF4 produces operational BPT approximately 3.9--4.7 at INT4. Same bit count, opposite outcomes. A Lloyd-Max quantizer achieves per-matrix cosine 0.990 at INT4 and 0.965 at INT3. Universal across four architectures and consistent across all 24 layers. Version 2.1 adds three verifications: five-seed replication at Pythia-1.4B (Welch $t = 633.74$, $p = 2.84 \times 10^{-15}$, Cohen's $d = 400.81$); Lloyd-Max INT3 end-to-end failure (BPT = 11.74); and three-model, three-seed robust replication. The minimum viable inference specification is not a bit count but a level-allocation strategy.

**Keywords:** quantization cliff; level allocation; NF4; Lloyd-Max; inference hardware; structural bonus

---

## 1. Introduction

Paper 7 of the Windstorm Series documented a universal INT4--INT3 quantization cliff. This paper shows the cliff location depends on how quantization levels are allocated. The full text is in paper.pdf.

*This is the Royal Society Interface submission draft. See also: paper-aap-draft.md, paper-arxiv.tex, paper-entropy-draft.md, paper.pdf.*
