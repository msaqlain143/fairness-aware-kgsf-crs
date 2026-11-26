# KG-FairCRS — Final Experimental Results and Discussion

## Evaluation scope

This synthesis consumes frozen outputs from Experiments 22.35F, 22.35G, 22.35H and 22.35J. It does not retrain KGSF, repeat retrieval, rerank candidates, retune parameters, regenerate recommendation slates, or conduct new inferential tests. R0_filtered is the primary graph configuration; R1_raw is a sensitivity configuration evaluated with the R0-selected parameters.

The independent-test re-evaluation contains **4,003 recommendation events from 1,283 conversations**, with a frozen 500-candidate pool per event. The test split was examined in historical experiments 22.22–22.34; therefore, this is a **frozen-configuration independent-test re-evaluation**, not a never-before-seen test.

## Main independent-test results

For A3_KG_FairCRS (R0_filtered), HR@10 changed by +0.000000, NDCG@10 by +0.000059, and MRR@10 by +0.000068 relative to A0_KGSF. The cumulative category-conditioned fairness objective decreased from 1.769098 to 0.524308, a descriptive reduction of **70.36%**. The fairness value includes reconstructed validation history plus independent-test exposure; it is not a test-only fairness score. No independent-test fairness p-value or confidence interval is claimed.

The independent-test utility bootstrap is preserved from 22.35J in `table_22_35J_test_utility_bootstrap_descriptive.csv`. Its intervals describe utility uncertainty only. Validation-stage inferential evidence remains in the 22.35G/22.35H tables and must not be conflated with test-set inference.

## Category-level heterogeneity and coverage

The unique category contributions reconstruct the reported 22.35J fairness objective for all five system/variant configurations. Each contains 19 categories, with absolute reconstruction differences no greater than 1e-09. In the primary A3 R0 comparison against A0, **16 categories improved, 3 worsened, and 0 were unchanged**. The regressing categories are: musical, action, horror. Aggregate improvement must not be presented as uniform category-level improvement.

Category-supported candidate instances total **100,909 of 2,001,500** (5.04%) under the experiment's reported definition. This is candidate-instance coverage, not catalog-wide item coverage.

## Statistical interpretation

The 22.35G paired bootstrap and 22.35H robustness analyses use conversation as the clustering unit. Report their validation-stage estimates, confidence intervals, multiplicity adjustments and sensitivity findings exactly as recorded in the preserved source tables. Bootstrap tail probabilities are not exact randomization-test p-values. Utility sign-flip inference relies on the conversation-level sign-exchangeability assumption recorded in the 22.35H manifest. The independent-test cumulative fairness objective is descriptive only.

## Reproducibility and limitations

The selected A3 configuration is q_threshold=0.0005, R_min=0.5 and lambda=1.0. R1_raw reuses R0-selected parameters without retuning. The relevance signal is operational (frozen KGSF scores and KG evidence), not a direct semantic judgment of dialogue relevance. Results are limited to ReDial movie recommendation with KGSF and should not be generalized to other datasets or backbones without additional evidence. The replay order is ascending event identifier and is not verified timestamp chronology.

## Manuscript claim boundary

The evidence supports a substantial reduction in the proposal-defined cumulative category-conditioned fairness objective on this frozen-configuration test re-evaluation, with very similar top-10 utility point estimates. It does not establish that every category improves, that the relevance proxy is semantic ground truth, that fairness gains are statistically significant on the independent test, or that the method generalizes beyond the evaluated benchmark/backbone.
