# Example Target Report (Synthetic)

**Context**: Demonstration oncotarget-lite run using cached synthetic GTEx/TCGA/DepMap-style summaries. Interpret as workflow illustration only.

## Top 5 scorecard hits

| Rank | Gene   | Score |
|------|--------|-------|
| 1    | MSLN   | 5.94  |
| 2    | CEACAM5 | 5.76  |
| 3    | CD274  | 5.71  |
| 4    | CD276  | 5.60  |
| 5    | EPCAM  | 5.55  |

## Spotlight: MSLN (Mesothelin)
- **Tumor bias**: High positive log2 fold-change across BRCA/LUAD/COAD (mean +1.40).
- **Normal safety margin**: Minimum normal TPM 0.52 (skin), with most tissues <1 TPM.
- **Dependency**: Mean DepMap score −0.01 (non-essential; good for ADC sparing).
- **Surface biology**: Cell-surface, signal peptide present; length 622 aa (within ADC-friendly 200–800 aa window).
- **Network degree**: Moderate PPI degree (32) reduces systemic pleiotropy risk.

## Caveats & next actions
- Synthetic summaries lack patient-level heterogeneity and immune infiltration context.
- Essentiality data does not cover primary cells; safety requires wet-lab validation.
- Prioritize orthogonal evidence: proteomics, IHC, safety knockout models.

_Run notebooks/02 and regenerate after modifying `data/raw/` with cohort-specific inputs._
