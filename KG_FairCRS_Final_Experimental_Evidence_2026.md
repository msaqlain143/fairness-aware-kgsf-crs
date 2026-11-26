# KG-FairCRS: Final Experimental Evidence and Research-Freeze Record

**Document status:** Consolidated evidence record; literature comparison is a targeted initial review, not an exhaustive systematic review.  
**Evidence archive:** `KG_FairCRS_FINAL_EVIDENCE_INPUTS.zip`  
**Primary research branch:** `dev/kg-rag-calibration`  
**Final reported backbone:** KGSF on ReDial  
**Experimental policy:** No new training, retrieval, parameter search, or experiment is proposed by this document.

---

## 1. Executive summary

KG-FairCRS is a post-ranking fairness framework evaluated on a frozen KGSF candidate population. Its implemented design combines:

1. category-conditioned accounting of item opportunity and accumulated exposure;
2. a relevance-eligibility gate using an operational score derived from frozen KGSF scores and KG-rank confidence; and
3. adaptive exposure-deficit calibration based on accumulated exposure within supported item-category/popularity groups.

The strongest reported final KGSF result is the 22.35J independent-test replay, packaged by 22.35L. Against the KGSF baseline, the primary `R0_filtered` KG-FairCRS configuration reports a 70.36% lower cumulative category-conditioned fairness objective, with HR@10 unchanged to the reported precision and small positive changes in NDCG@10 and MRR@10. These fairness results are descriptive: the experiment does not claim a test-set fairness p-value or confidence interval.

The 22.35N controlled validation ablation reports a 44.48% lower common category fairness loss for the full system than KGSF on the 624-event held-out validation split. Its utility differences are small and the paired bootstrap intervals include zero for the full system's HR@10, NDCG@10 and MRR@10 contrasts. The component ablations show that category conditioning and accumulated history affect the observed result, but they do not establish clean, independent additive causal contributions for every component.

**Research conclusion:** The evidence supports a promising, implementation-specific reduction in the reported cumulative category-conditioned fairness objective while largely preserving measured recommendation utility. It does not, by itself, establish universal fairness improvement, semantic relevance validity, cross-backbone generality, or CCF-A publication readiness.

## 2. Research scope and architecture

### 2.1 Problem addressed

The work studies item-side exposure disparity in knowledge-grounded conversational recommendation. Items are grouped by accepted category labels and training-derived popularity groups. The framework compares accumulated recommendation exposure against an operational opportunity estimate and adjusts ranking priority for eligible underexposed candidates.

The intended scope is not user-group fairness, provider fairness, or a general guarantee of equal item exposure. The implemented estimand is category-conditioned disparity across supported popularity groups.

### 2.2 Implemented pipeline

For each event:

1. Reuse the frozen KGSF Top-500 candidate list and original scores.
2. Construct an operational candidate relevance/opportunity distribution from KGSF softmax scores and KG-rank confidence.
3. Apply the fixed eligibility threshold.
4. Accumulate opportunity by accepted item category and popularity group.
5. Estimate group exposure rates from accumulated discounted Top-10 exposure.
6. Compute category-conditioned underexposure deficits.
7. Adjust eligible candidate scores and perform eligibility-prioritized Top-10 selection, with original-score fallback.
8. Update the exposure ledger.

The method is post-processing over frozen candidate scores; the final independent-test manifest states that training, retrieval, checkpoint modification and parameter retuning were not performed.

### 2.3 Frozen parameters

The 22.35J manifest records:

- `q_threshold = 0.0005`
- `R_min = 0.5`
- `lambda = 1.0`
- `KG_TAU = 50` in the operational KG-confidence construction documented in the mathematical specification.

The `R0_filtered` variant is primary; `R1_raw` is a sensitivity variant.

## 3. Experimental chronology and evidence separation

| Milestone | Backbone / role | Evidence in archive | Interpretation |
|---|---|---|---|
| 22.25 | KBRD historical component ablation | Aggregate and event-level CSVs, manifest, validation checks | Historical development evidence; do not pool its metrics with final KGSF results |
| 22.26 | KBRD paired inference | Point estimates, paired bootstrap inference, manifest, checks | Historical inferential evidence; Holm-adjusted comparisons are reported in Section 4 |
| 22.35F | KGSF validation/fairness evaluation | System summaries, event-level comparisons, fairness cells, parameter decisions | Main validation and parameter-decision lineage |
| 22.35G | Paired statistical comparison | Paired contrasts and bootstrap evidence | Utility inference on validation comparisons |
| 22.35H | Inferential robustness and sensitivity | BCa, sign-flip, tuning sensitivity, R0/R1 sensitivity, category decomposition | Robustness evidence; not an independent-test fairness significance analysis |
| 22.35J | KGSF independent-test replay | Event results, category fairness, exposure ledger, utility bootstrap, manifest and checks | Main final independent-test evidence; previously examined test split disclosed |
| 22.35L | Final evaluation package | Consolidated tables, figures, discussion, manifest and checks | Release package derived from 22.35J and 22.35F–H evidence |
| 22.35M | Mathematical specification | Formal definitions, conditional propositions, proof obligations and limitations | Implementation-aligned theory audit; no global monotonicity theorem |
| 22.35N | Controlled KGSF validation ablation | Six configurations, event-level outputs, fairness cells, paired bootstrap and checks | Final mechanism-attribution evidence on validation only |

**Do not conflate:** 22.25–22.26 KBRD metrics, 22.35 validation metrics, and 22.35J cumulative validation-plus-test fairness metrics. They have different backbones, estimands, splits and/or evaluation contexts.

## 4. Historical KBRD evidence: 22.25–22.26

### 4.1 22.25 aggregate results

The archived 22.25 aggregate reports 4,003 events and 1,283 conversations.

| Method | HR@10 | NDCG@10 | Category-group log-rate MAD | Fairness reduction vs A0 |
|---|---:|---:|---:|---:|
| A0 KBRD baseline | 0.170372 | 0.092041 | 13.246715 | 0.00% |
| A1 no category context | 0.169873 | 0.091713 | 9.643535 | 27.20% |
| A2 no KG evidence | 0.170122 | 0.092002 | 8.664587 | 34.59% |
| A3 no adaptive exposure deficit | 0.164627 | 0.087471 | 11.620301 | 12.28% |
| A4 full KG-FairCRS | 0.169873 | 0.091668 | 9.376413 | 29.22% |

These are historical KBRD results under the 22.25 metric and setup. They are not numerically comparable to the final KGSF proposal fairness objective.

### 4.2 22.26 paired inference

The 22.26 paired inference uses 10,000 conversation-cluster bootstrap replicates and Holm correction across the reported comparison family.

The archived table indicates that the full method versus the no-adaptive-exposure-deficit configuration has statistically significant differences after Holm correction for:
- HR@10: adjusted p = 0.036996;
- NDCG@10: adjusted p = 0.002400;
- fairness_MAD: adjusted p = 0.007699.

Comparisons of full KG-FairCRS against the no-category and no-KG configurations do not pass the reported Holm-adjusted 0.05 threshold. The full-method versus KBRD-baseline fairness_MAD comparison also does not pass that threshold (adjusted p = 1.0 in the archived table).

**Interpretation boundary:** These results provide historical evidence that the adaptive exposure component mattered in that KBRD experiment. They do not establish that each component independently contributes significantly in the final KGSF architecture.

## 5. Final KGSF independent-test results: 22.35J–22.35L

### 5.1 Main system comparison

The 22.35J output contains 4,003 test events and 1,283 conversations. The primary comparison is `R0_filtered`.

| System | HR@10 | NDCG@10 | MRR@10 | Cumulative fairness loss | Active categories |
|---|---:|---:|---:|---:|---:|
| A0 KGSF | 0.169623 | 0.091053 | 0.067311 | 1.769098 | 19 |
| A1 22.16 | 0.169373 | 0.090983 | 0.067293 | 1.124664 | 19 |
| A2 22.18 | 0.169873 | 0.091510 | 0.067816 | 1.671941 | 19 |
| A3 KG-FairCRS R0 | 0.169623 | 0.091112 | 0.067379 | **0.524308** | 19 |
| A3 KG-FairCRS R1 | 0.169623 | 0.091112 | 0.067379 | 0.524274 | 19 |

The 22.35L release table reports a 70.362985% reduction in cumulative fairness loss for A3 R0 relative to A0.

Utility differences are small:
- HR@10 delta: 0.000000 at the reported precision;
- NDCG@10 delta: +0.000059;
- MRR@10 delta: +0.000068.

The paired conversation-cluster bootstrap utility intervals in 22.35J include zero for the reported A3-vs-A0 utility contrasts. The bootstrap is a utility analysis, not a fairness-loss inferential test.

### 5.2 Fairness and coverage caveats

The 22.35J coverage summary reports:
- 2,001,500 candidate instances;
- 100,909 category-supported candidate instances;
- candidate-category coverage = 0.0504167 (about 5.04%);
- 1,283 test conversations and 4,003 events.

The final category comparison reports 16 improved categories and 3 worsened categories. The worsened categories are musical, action and horror. The fairness objective is therefore heterogeneous across categories; aggregate reduction must not be described as universal category improvement.

The 22.35J manifest also discloses:
- the independent-test split had been examined in historical experiments 22.22–22.34;
- this is a frozen-configuration re-evaluation, not a never-before-seen test evaluation;
- test events are replayed in ascending event ID, not claimed to be timestamp chronology;
- cumulative fairness is reported descriptively, without a test-set fairness p-value or confidence interval;
- the reported fairness estimand is cumulative validation-history plus independent-test exposure.

### 5.3 Interpretation

The independent-test result is encouraging: measured utility is essentially preserved while the chosen cumulative category-conditioned fairness objective is substantially lower. However, the fairness result is descriptive and conditional on the operational opportunity proxy, category mapping, support thresholds, replay order and previously disclosed test-history exposure.

## 6. Controlled mechanism attribution: 22.35N

### 6.1 Design and protocol

The 22.35N manifest and validation checks report:
- 3,010 validation events and 883 conversations;
- 569 tuning events and 624 held-out events;
- 176 held-out conversations;
- six configurations;
- identical held-out events and conversations across configurations;
- frozen Top-500 candidates per event;
- 2,000 paired bootstrap replicates clustered by conversation;
- independent test not loaded;
- no training, retrieval rerun, checkpoint modification or parameter retuning;
- validation checks passed.

### 6.2 Held-out results

| Configuration | HR@10 | NDCG@10 | MRR@10 | Common category fairness loss |
|---|---:|---:|---:|---:|
| N0 KGSF | 0.201923 | 0.107399 | 0.079019 | 1.786137 |
| Full KG-FairCRS | 0.200321 | 0.106936 | 0.078859 | **0.991597** |
| N1 eligibility only | 0.201923 | 0.107399 | 0.079019 | 1.786137 |
| N2 no category | 0.195513 | 0.104546 | 0.077120 | 2.082469 |
| N3 no KG | 0.200321 | 0.106936 | 0.078859 | 1.392882 |
| N4 no history | 0.201923 | 0.107399 | 0.079019 | 1.786137 |

Full KG-FairCRS has a 44.48% lower common category fairness loss than N0 on this held-out split. This is a point-estimate comparison, not a paired inferential claim about fairness.

### 6.3 Paired utility bootstrap

The 2,000-replicate conversation-cluster bootstrap reports the following full-method contrasts versus N0:

| Metric | Delta | 95% interval |
|---|---:|---:|
| HR@10 | -0.001603 | [-0.005000, 0.000000] |
| NDCG@10 | -0.000463 | [-0.001438, 0.000000] |
| MRR@10 | -0.000160 | [-0.000497, 0.000000] |

The intervals include zero at their upper boundary. Report the exact interval convention and bootstrap implementation in the manuscript; do not call these utility changes statistically significant.

### 6.4 Mechanism interpretation and caveats

- **Category conditioning:** Removing category conditioning increases common category fairness loss from 1.786137 to 2.082469 and worsens utility on the held-out split. This supports the relevance of category-conditioned accounting in this implementation.
- **Accumulated history:** N4 (no prior history) matches N1 (eligibility-only) in the reported held-out outputs. This is consistent with the current-event-only ledger producing no adaptive historical exposure correction.
- **Eligibility alone:** N1 matches baseline. Eligibility-first selection alone does not produce the full fairness improvement in this controlled setup.
- **KG evidence:** N3 has higher fairness loss than the full method (1.392882 vs 0.991597), while still improving over baseline. This supports a contribution from KG evidence within the combined system, but does not establish that KG evidence is individually necessary for any fairness improvement.
- **Interactions:** These are ablations of a coupled ranking and accounting pipeline. Differences should not be interpreted as independent additive causal effects without stronger factorial or interaction analysis. The current evidence is best described as controlled mechanism sensitivity/attribution under the frozen validation protocol.

## 7. Mathematical specification and theoretical boundaries: 22.35M

### 7.1 Operational opportunity

The inspected mathematical specification defines:
- KGSF candidate probability as a softmax over frozen Top-500 scores;
- KG confidence as `exp(-rank / 50)` for positive KG ranks and zero otherwise;
- normalized operational opportunity `q` proportional to `p_KGSF * (1 + KG_confidence)`;
- eligibility as `q >= 0.0005`.

The specification explicitly says `q` is an operational model-score/KG-rank proxy, not independently validated semantic or conversational relevance.

### 7.2 Fairness objective

For each category and popularity group, opportunity is accumulated from eligible candidates and exposure from displayed Top-10 items using discounted rank weights `1/log2(rank+1)`. Supported groups satisfy the minimum opportunity threshold `R_min=0.5` and positive epsilon support. The category objective is the opportunity-weighted variance of exposure-to-opportunity rates across supported popularity groups; the aggregate averages over active categories.

### 7.3 Verified conditional properties

The 22.35M document establishes:
- non-negativity of the weighted variance for a fixed active set;
- the standard pairwise identity for weighted variance;
- an exact one-group exposure-increment condition under fixed opportunity, weights and support;
- a conditional Top-K membership preservation result under an explicit score-margin bound;
- a fixed-support sensitivity bound for the exposure/opportunity rate.

### 7.4 Explicitly unproven properties

The document does **not** prove:
- global monotonic reduction of the full A3 fairness objective;
- universal Top-K utility preservation;
- global continuity across hard eligibility/support thresholds;
- that `q` is human-perceived conversational relevance;
- universal optimality or CCF-A-level novelty.

A particularly important implementation detail is that `lambda=0` need not recover baseline KGSF because eligibility-first selection can still change slate membership. The manuscript must accurately describe the complete selection rule, not simplify it to an additive score bonus alone.

## 8. Provenance and reproducibility audit

### 8.1 22.35J source-hash reconciliation

The 22.35J manifest records the following source hashes:

| Source | Manifest SHA-256 (CRLF snapshot) | Uploaded archive SHA-256 (LF bytes) | Result |
|---|---|---|---|
| `22_35F_validated_fairness_evaluation.py` | `fc3658bc607d13ec6c3be631fb707dd34a449c4b5fbd280e03bfcb0fb015457f` | `70e75346bfdb9886dc2970f3a5e1054575519f5f56ebae63da54635a8a90a398` | CRLF-normalized archive bytes match manifest |
| `22_35J_integrated_kg_faircrs_independent_test.py` | `42ba8ee6de15e100614d4d61b1635fee80701df59631fadae2efe3f9a48dcb3c` | `085eda027a431fccd251603c1e3ad24e011f93a8799c485ce22f0c99b8b444c3` | CRLF-normalized archive bytes match manifest |
| `22_16_category_context_utility_calibration.py` | `6555144efe624658eb32e8de48ee9e95ed94baded9486a8f6280d848ecbef751` | `a7cd6564fe3ad023a862150bee103be1b0f9cf71e5112d8dcc7cc5910e6d52b6` | CRLF-normalized archive bytes match manifest |

For all three files, converting the archive's LF line endings to CRLF reproduces the manifest SHA-256 exactly. This resolves the previously observed hash mismatch as a line-ending representation difference for these archived source files; it is not evidence of a code-content discrepancy.

The `22_35J_source_bundle/experiment_manifest.json` in the archive is a **22.35H** manifest, not a 22.35J source-bundle manifest. The independent-test execution manifest is separately located at `results_final_test_m2/experiment_22_35J/experiment_manifest.json`.

### 8.2 22.35L and 22.35N provenance

The 22.35L manifest contains hashes for its principal input result tables and validation artifacts. It is a consolidation package, not an independent rerun.

The 22.35N manifest records execution identity, split sizes and output inventory, but does not include a complete input-source hash map. The uploaded 22.35N source is available as `22_35N_controlled_mechanism_ablation.py`; its archive SHA-256 is:

`38f43bf7ed663ecac54e052588e6678efbacf73ae72c2c718b07113a38434d0a`

This hash is the uploaded archive file's current byte hash. It should be recorded alongside the Git commit identity for the source actually used in the run if the repository history confirms that identity.

### 8.3 Git source identity for 22.35N

The user-provided repository verification reports that `22_35N_controlled_mechanism_ablation.py` is tracked on `dev/kg-rag-calibration`, with commit `9a1225d` (`research: add controlled KG-FairCRS mechanism ablation`). The reported local HEAD and `origin/dev/kg-rag-calibration` point to the same commit, and `git status -sb` reports a clean branch. This establishes the reported Git identity for the 22.35N script. The experiment manifest itself does not record that commit hash, so the connection between the execution and this commit is supported by the supplied repository history, not by an embedded run-time Git field.

### 8.4 Test-history disclosure

The independent-test split had been inspected in earlier experiments 22.22–22.34. It must be described as a previously examined test split used for frozen-configuration re-evaluation, not as a pristine untouched holdout.

### 8.5 Reproduction scope

The archive contains final outputs, key scripts, selected manifests, theory and statistical tables. It does not contain every raw upstream dataset/export referenced by all manifests. Therefore, it is an evidence archive sufficient for this consolidated review, but not necessarily a complete standalone reproduction bundle from raw data to final result.

## 9. Closest-work literature comparison: initial targeted review

This is a focused comparison against selected primary publications. It is not an exhaustive systematic review, and the novelty claim must remain provisional until database searches and full-text screening are completed.

| Work | Main fairness/recommendation focus | Relationship to KG-FairCRS | Distinction to substantiate |
|---|---|---|---|
| FairCRS (RecSys 2024) | User-oriented fairness in conversational recommendation; model-agnostic framework | Directly establishes fairness-aware CRS as prior art | KG-FairCRS targets item-side category-conditioned exposure disparity rather than user-group recommendation-quality disparity |
| HyFairCRS (Findings of ACL 2025) | Multi-interest diversity fairness in conversational recommendation using hypergraph contrastive learning | Very close CRS fairness work; must be treated as a central comparator | Different fairness target and mechanism: multi-interest diversity learning versus accumulated category/popularity exposure calibration |
| FairMatch (ACM TOIS 2022) | Graph-based post-processing for multi-sided item/provider exposure bias | Establishes post-processing exposure redistribution as prior art | KG-FairCRS uses category-conditioned opportunity/exposure ledgers and KG-supported eligibility in a CRS setting |
| Recommendations with Minimum Exposure Guarantees (ESWA 2024) | Post-processing ILP with minimum exposure constraints | Establishes constrained exposure-aware reranking | KG-FairCRS uses adaptive accumulated deficits and category-conditioned opportunity rather than fixed minimum exposure guarantees |
| Make Fairness More Fair (KDD 2022) | Item utility estimation and exposure redistribution | Establishes relevance/utility-aware item fairness | KG-FairCRS uses an operational KGSF+KG opportunity proxy and category/popularity-conditioned cumulative objective |
| Fairness of Exposure in Dynamic Recommendation (2023 preprint) | Long-run exposure fairness in dynamic recommendation | Establishes dynamic/cumulative exposure as prior art | KG-FairCRS's claimed distinction must be the particular category-conditioned opportunity accounting, relevance gate and KG-grounded CRS integration, not simply accumulation over time |
| Relevance-aware Individual Item Fairness Measures (ACM TORS 2026) | Theoretical analysis of relevance-aware item fairness measures | Important for metric justification and limitations | KG-FairCRS's objective is a category/popularity-group opportunity-weighted variance; explain how it differs from individual-item measures |

### 9.1 What can be claimed cautiously

The evidence supports describing KG-FairCRS as an integrated post-processing framework that combines:
- category-conditioned opportunity and exposure accounting across popularity groups;
- an eligibility gate based on frozen recommender scores and KG-rank evidence; and
- adaptive accumulated-exposure deficit calibration in a conversational recommendation pipeline.

### 9.2 What cannot yet be claimed

Do not claim:
- the first fairness-aware conversational recommender;
- the first dynamic exposure-fairness method;
- the first relevance-aware item fairness objective;
- that no prior work combines category-aware fairness with KG evidence;
- that the method is model-agnostic based solely on one KGSF final evaluation;
- that the operational `q` is semantic relevance;
- that all categories improve;
- that the method guarantees fairness improvement.

The closest-work comparison needs a fuller search across ACM DL, IEEE Xplore, ACL Anthology, DBLP, Scopus/Web of Science and Google Scholar, with screening of papers on category-aware item fairness, popularity-aware exposure allocation, fairness-aware CRS, knowledge-graph recommendation and sequential/dynamic exposure fairness.

## 10. Supported claims and claims to avoid

| Proposed manuscript claim | Evidence status | Recommended wording |
|---|---|---|
| The method reduces the reported cumulative category-conditioned fairness objective on the evaluated KGSF ReDial test replay | Supported descriptively | “A3 reports a 70.36% lower cumulative objective than KGSF under the specified replay and metric.” |
| Utility is preserved | Supported within measured metrics and uncertainty | “HR@10 is unchanged to reported precision; small NDCG/MRR differences have bootstrap intervals including zero.” |
| Category conditioning matters | Supported by 22.35N point estimates | “Removing category conditioning worsens the common fairness objective in the controlled validation ablation.” |
| Accumulated history matters | Supported by N1/N4 equality and full-vs-no-history comparison | “The observed adaptive effect depends on historical exposure under the implemented ledger.” |
| KG evidence helps the full system | Partially supported | “Removing KG evidence increases the full system's fairness loss, although the no-KG variant remains better than baseline.” |
| All mechanisms independently improve performance | Not supported | Do not claim; mechanisms interact and N1/N4 equal baseline. |
| The system guarantees monotonic fairness improvement | Not supported | Do not claim. |
| The relevance gate measures true conversational relevance | Not supported | Call it an operational relevance/opportunity proxy. |
| The test is untouched | False | Disclose historical examination of the split. |
| CCF-A publication readiness is established | Not established | Present as a promising empirical contribution requiring full closest-work validation and reviewer-level scrutiny. |

## 11. Limitations to state in the manuscript

1. **Operational relevance proxy:** `q` is derived from KGSF scores and KG rank confidence, not direct human relevance judgments or calibrated conversational satisfaction.
2. **Limited category support:** only about 5.04% of test candidate instances are category-supported under the reported coverage definition.
3. **Category regressions:** three categories worsen in the independent-test decomposition.
4. **Threshold sensitivity and discontinuities:** hard eligibility and minimum-support thresholds can change active groups/categories discontinuously.
5. **No global monotonicity guarantee:** the one-step variance result is conditional on fixed support and a single-group exposure change.
6. **Single final backbone:** final KGSF evidence does not establish cross-backbone generality.
7. **Previously examined test split:** independent-test results are frozen re-evaluation, not untouched holdout evidence.
8. **Replay ordering:** ascending event ID is deterministic but is not established as actual temporal order.
9. **Fairness inference:** test fairness reduction is descriptive; no test-set fairness confidence interval or p-value is claimed.
10. **Multi-label attribution:** one item can contribute exposure to multiple accepted categories, so category accounting is not a conservation partition of total exposure.
11. **Source reproducibility:** the archive has selected source and result files but not every raw upstream input needed for a full end-to-end rerun.

## 12. Research freeze decision

### Frozen evidence

The following should be treated as the current reported evidence snapshots:
- historical KBRD 22.25–22.26 outputs;
- KGSF validation 22.35F–22.35H outputs;
- independent-test replay 22.35J outputs and its manifest;
- consolidation package 22.35L;
- mathematical specification 22.35M;
- controlled validation ablation 22.35N outputs and its manifest.

### Freeze conditions

Before declaring the repository's research record fully frozen:
1. Record the exact Git commit for each final source file, especially 22.35N.
2. Preserve the uploaded archive and its SHA-256 as an immutable evidence snapshot.
3. Keep generated result artifacts out of source-code commits unless the repository intentionally versions selected compact evidence tables.
4. Add the master evidence document and closest-work bibliography to version control.
5. Record the limitations and test-history disclosure in the paper.

**No additional experiment is recommended by this evidence review.** Reopen experimentation only if peer review or the completed literature comparison identifies a specific, material gap that cannot be addressed from existing evidence.

## 13. Selected literature references

1. Liu, Q., Feng, X., Gu, T., & Liu, X. (2024). *FairCRS: Towards User-oriented Fairness in Conversational Recommendation Systems*. Proceedings of the 18th ACM Conference on Recommender Systems, 126–136. https://doi.org/10.1145/3640457.3688150
2. Zheng, Y., Xie, Z., Wang, G., Liu, Z., Lin, L., & Lam, K.-Y. (2025). *Why Multi-Interest Fairness Matters: Hypergraph Contrastive Multi-Interest Learning for Fair Conversational Recommender System*. Findings of ACL 2025, 25673–25684. https://doi.org/10.18653/v1/2025.findings-acl.1317
3. Mansoury, M., Abdollahpouri, H., Pechenizkiy, M., Mobasher, B., & Burke, R. (2022). *A Graph-Based Approach for Mitigating Multi-Sided Exposure Bias in Recommender Systems*. ACM Transactions on Information Systems, 40(2), Article 32. https://doi.org/10.1145/3470948
4. Lopes, R., Alves, R., Ledent, A., Santos, R. L. T., & Kloft, M. (2024). *Recommendations with Minimum Exposure Guarantees: A Post-processing Framework*. Expert Systems with Applications, Article 121164. https://doi.org/10.1016/j.eswa.2023.121164
5. Wang, J., Ma, W., Li, J., Lu, H., Zhang, M., Li, B., Liu, Y., Jiang, P., & Ma, S. (2022). *Make Fairness More Fair: Fair Item Utility Estimation and Exposure Re-Distribution*. Proceedings of KDD 2022, 1868–1877. https://doi.org/10.1145/3534678.3539354
6. Mansoury, M., & Mobasher, B. (2023). *Fairness of Exposure in Dynamic Recommendation*. arXiv:2309.02322. https://arxiv.org/abs/2309.02322
7. Lioma, C. (2026). *Relevance-aware Individual Item Fairness Measures for Recommender Systems: Limitations and Usage Guidelines*. ACM Transactions on Recommender Systems, 4(3), Article 37. https://doi.org/10.1145/3765624

---

**End of evidence record.** This document consolidates the supplied archive and selected published literature. It is not a claim of exhaustive literature coverage, independent replication from raw data, or guaranteed acceptance at any venue.
