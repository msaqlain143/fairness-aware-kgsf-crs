# KG-FairCRS — Final Experimental Results and Discussion

## Evaluation scope

This synthesis consumes the frozen outputs of Experiments 22.35F, 22.35G, 22.35H and 22.35J. It does not retrain KGSF, repeat retrieval, rerank candidates, retune parameters, or conduct new statistical tests. The primary graph configuration is R0_filtered; R1_raw is a prespecified sensitivity configuration using the R0-selected A3 parameters.

The independent-test re-evaluation contains **4,003 recommendation events from 1,283 conversations**, with a frozen 500-candidate pool per event. The test split had been examined in historical experiments 22.22–22.34; therefore, these results must be described as a **frozen-configuration independent-test re-evaluation**, not as a never-before-seen test.

## Main independent-test results

See `table_independent_test_main_results.csv`. For A3_KG_FairCRS (R0_filtered), HR@10 changed by +0.000000 relative to A0_KGSF, NDCG@10 changed by +0.000059, and cumulative category-conditioned fairness loss decreased from 1.769098 to 0.524308, a reduction of **70.36%**. This is a descriptive independent-test fairness comparison; no independent-test fairness p-value or confidence interval is claimed.

The paired conversation-cluster bootstrap for independent-test utility is retained in `table_22_35J_test_utility_bootstrap_descriptive.csv`. Its intervals are used only to describe utility uncertainty. The validation-stage inferential evidence remains in the 22.35G/22.35H tables and must not be conflated with test-set inference.

## Category-level heterogeneity

The 22.35K reconstruction is reproduced here: unique category contributions sum to the reported 22.35J fairness objective for all five system configurations, with 19 categories and absolute reconstruction differences no greater than 1e-9. In the primary A3 R0 comparison against A0, **16 categories improved, 3 worsened, and 0 were unchanged**. The largest regressions are shown first in `table_category_regressions_R0.csv`; the complete category profile is plotted in `figure_category_contribution_changes.png`. The regressing categories are: musical, action, horror.

The aggregate improvement must not be presented as uniform category-level improvement. These category regressions motivate a limitation and future-work discussion, not post-test parameter changes.

## Statistical interpretation

The 22.35G paired bootstrap and 22.35H robustness analyses use conversation as the clustering unit. Report their validation-stage estimates, confidence intervals, multiplicity adjustments and sensitivity findings exactly as recorded in the source tables. Do not claim an exact randomization p-value for the nonlinear cumulative fairness objective. Utility sign-flip inference depends on the exchangeability/symmetry assumption stated in the 22.35H manifest. The independent-test fairness objective is descriptive only.

## Reproducibility and limitations

The selected A3 configuration is q_threshold=0.0005, R_min=0.5 and lambda=1.0. R1_raw reuses the R0-selected parameters without retuning. The relevance signal is operational (KGSF score and KG support), not a direct semantic judgment of dialogue relevance. Category evidence covers only a subset of catalog/candidate instances; the 22.35J coverage summary reports 100,909 category-supported candidate instances out of 2,001,500 (5.04%). Results are limited to ReDial movie recommendation with KGSF and should not be generalized to other datasets or backbones without additional evidence.

## Manuscript claim boundary

The evidence supports a substantial reduction in the proposal-defined cumulative category-conditioned fairness objective on this frozen-configuration test re-evaluation, with essentially preserved top-10 utility. It does not establish that every category improves, that the relevance score is semantic ground truth, that fairness gains are statistically significant on the independent test, or that the method generalizes beyond the evaluated benchmark/backbone.
