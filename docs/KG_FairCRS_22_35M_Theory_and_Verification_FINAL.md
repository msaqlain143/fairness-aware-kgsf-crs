# KG-FairCRS — Milestone 22.35M
## Formal specification, proof obligations, verified properties, and controlled next experiments

**Status:** Implementation-aligned mathematical audit complete for the inspected source snapshots; the theoretical guarantees below are conditional, not a proof that the full algorithm optimizes its fairness objective.  
**Source basis:** archived `22_35F_validated_fairness_evaluation.py` and `22_35J_integrated_kg_faircrs_independent_test.py` plus ReDial results previously reported in the project.  
**Date:** 22 November 2026

### 1. Executive finding

The implementation uses a frozen KGSF Top-500 candidate list, an operational relevance score, an eligibility-first Top-10 selection rule, and a cumulative category-conditioned exposure ledger. The primary fairness objective is **not** a directly optimized convex function over independently allocated exposure: category and popularity groups share item-level ranking actions; eligibility and minimum-support thresholds create discontinuities; a single ranked movie can contribute to several categories; and relevance opportunity is added before selecting the slate.

We derive **one exact local variance-reduction criterion** for a fixed supported category and one exposure increment, plus conditional ranking and relevance-stability guarantees. These results identify *when* the calibration direction is justified but **do not** prove monotonic end-to-end fairness improvement or publishable novelty by themselves.

### 2. Exact code–equation map

| Object | Source location (inspected snapshot) | Verified behavior |
|:---|:---|:---|
| Candidate probability and KG confidence | `22_35J...py`, `softmax` and `build_q`, lines 123–145 | `p=softmax(scores)`, `kappa=exp(-KG_rank/KG_TAU)` for positive ranks and 0 otherwise; `q=normalize[p*(1+kappa)]` |
| Opportunity accumulation | `22_35F...py`, `ProposalLedger.add_R`, lines 139–144 | Includes only candidates with `q >= qthr`; adds full q to every accepted item category |
| Exposure accumulation | `22_35F...py`, `ProposalLedger.add_E`, lines 145–149 | Adds `1/log2(rank+1)` per displayed Top-10 movie to **every** accepted category, regardless of gate eligibility |
| Active groups/categories | `22_35F...py`, `proposal_terms`, lines 154–174 | Group active iff `R > 1e-12` and `R >= R_min`; category active iff at least two groups active |
| Measured fairness objective | `22_35F...py`, `proposal_terms`, lines 154–174 | Mean across active categories of opportunity-weighted within-category variance of `rho=E/(R+1e-12)` |
| Underexposure direction | `22_35F...py`, `proposal_deficits`, lines 176–194 | Positive part of `category_mean_rho - group_rho` for supported groups |
| A3 reranking | `22_35F...py`, `rank_a3`, lines 196–220 | Adds current-event opportunities **before** deficit calculation; for eligible candidates adds `lambda × mean(accepted category deficits)`; sorts by adjusted score with ID tie-break; selects eligible Top-10 first; fills remaining slots in stable original KGSF score order; then updates exposure |
| Test replay and scope | `22_35J...py`, `reconstruct_history` and `evaluate_one`, lines 288–363 | Reuses system-specific validation history, traverses test events in ascending event ID, computes cumulative test+validation fairness |

**Term distinction:** `q` is model-score + KG-rank evidence, **not** independently validated conversational relevance. Positive KG evidence is not a hard eligibility requirement. `lambda=0` does not necessarily recover A0 because eligibility-first selection remains active. Supported category cells and event eligibility are distinct filters.

### 3. Formal definition

For event e, frozen KGSF candidates `C_e` (size 500), original score `s_ei`, and KG rank `r_ei`, define

\[
p_{ei}=\frac{\exp(s_{ei})}{\sum_{j\in C_e}\exp(s_{ej})},\quad
\kappa_{ei}=\begin{cases}\exp(-r_{ei}/50)&r_{ei}>0\\0&\mathrm{otherwise},\end{cases}\quad
q_{ei}=\frac{p_{ei}(1+\kappa_{ei})}{\sum_{j\in C_e}p_{ej}(1+\kappa_{ej})}.
\]

Eligibility `I_e={i in C_e: q_ei >= t_q}`, with frozen `t_q=0.0005` and `R_min=0.5`. Let category indicator `A_ic` denote accepted multi-label category membership and `g(i)` the training-derived popularity group. The cumulative opportunity and realized exposure ledgers are

\[
R_{cg}=\sum_{e}\sum_{i\in C_e}q_{ei}A_{ic}\,\mathbf1[g(i)=g]\,\mathbf1[q_{ei}\ge t_q],
\]
\[
E_{cg}=\sum_e\sum_{r=1}^{10}w_r A_{\pi_e[r]c}\mathbf1[g(\pi_e[r])=g],\quad w_r=1/\log_2(r+1).
\]

Supported groups satisfy `R_cg >= R_min` and `R_cg > EPS` (EPS=10^−12). An active category has at least two supported groups. Within an active category:

\[
p_{cg}=\frac{R_{cg}}{\sum_{h\in G_c^+}R_{ch}},\quad
\rho_{cg}=\frac{E_{cg}}{R_{cg}+\epsilon},\quad
\bar\rho_c=\sum_{g\in G_c^+}p_{cg}\rho_{cg},
\]
\[
V_c=\sum_{g\in G_c^+}p_{cg}(\rho_{cg}-\bar\rho_c)^2,\quad
L_{fair}=\frac1{|\mathcal C_+|}\sum_{c\in\mathcal C_+}V_c.
\]

The per-category deficit used for adjustment is `d_cg=max(0, mean_rho_c−rho_cg)` for supported groups, otherwise implicitly zero for unsupported cells. Each eligible candidate's adjustment is `lambda × mean_c[d_c,g(i)]` over its accepted category labels (zero with no labels); the adjusted score is used **inside an eligibility-prioritized selection mechanism**, not ordinary global score sorting. Recorded frozen parameters: `(t_q,R_min,lambda)=(0.0005,0.5,1.0)`.

### 4. Theorems and proofs: exact conditional results

#### Theorem 1 — Non-negativity and equality case

If a category has at least two supported groups, strictly positive normalized weights `p_g` and finite rates `rho_g`, then `V_c >= 0`. It is zero iff all `rho_g` are equal. The aggregate loss is nonnegative wherever the active category set is nonempty. If there are no active categories, the implementation returns `NaN`, **not zero**. **Proof:** each squared deviation is nonnegative, and every group weight is positive. **Novelty status:** elementary identity, not a research theorem.

#### Theorem 2 — Pairwise discrepancy identity

For normalized nonnegative weights,

\[
V_c=\frac12\sum_{g,h}p_gp_h(\rho_g-\rho_h)^2=\sum_{g<h}p_gp_h(\rho_g-\rho_h)^2.
\]

**Proof:** expand the double sum to obtain `sum p_g rho_g² − (sum p_g rho_g)²`, the weighted variance. **Novelty status:** standard weighted-variance result.

#### Theorem 3 — Exact one-step exposure-improvement interval (newly explicit analysis; not asserted novel)

Fix a category with its supported groups, opportunity masses `R_g`, smoothing `epsilon`, and weights `p_g=R_g/sum_h R_h`. Suppose exposure to exactly one group `g` is increased by a nonnegative amount `δ`, all other `E_h` are fixed, and no support group or category changes. Write `x=δ/(R_g+epsilon)` and let `mu=sum_h p_h rho_h`. Then

\[
\boxed{V'_c-V_c=2p_g(\rho_g-\mu)x+p_g(1-p_g)x^2.}
\]

**Proof:** only `rho_g` changes to `rho_g+x`. Using `V=sum_h p_h rho_h²−mu²` and `mu'=mu+p_g x`, expand the difference: `p_g(2rho_g x+x²) − [2mu p_g x + p_g² x²]`.

**Consequence:** if `rho_g < mu` and `0<p_g<1`, this single-group promotion **strictly reduces** category variance iff

\[
\boxed{0<\frac{\delta}{R_g+\epsilon}<\frac{2(\mu-\rho_g)}{1-p_g}.}
\]

At the upper boundary disparity is unchanged; above it overshooting **increases** variance. A non-underexposed group (`rho_g>=mu`) never improves the frozen-weight variance by increasing its rate alone. Because A3 adjusts a Top-10 **ranking** that simultaneously changes exposure in multiple groups/categories, `R` is updated before ranking, and active categories can change, this result is **not** a proof that A3's full step always decreases `L_fair`. It provides a precise countermeasure to an unjustified monotonicity claim and motivates an exposure-step safeguard if a future algorithm extension is evaluated.

#### Theorem 4 — Conditional membership preservation under eligibility-first selection

Let original scores be ordered `s_(1)>=...>=s_(n)` and assume (i) every original Top-K candidate is eligible, (ii) at least K eligible candidates exist, (iii) all eligible adjustments are bounded in absolute value by `B`, and (iv) `s_(K)-s_(K+1)>2B`. Then the eligibility-prioritized reranker preserves **Top-K membership**. **Proof:** the weakest original Top-K adjusted score exceeds the strongest outside score under the margin assumption; all original Top-K are eligible, so eligibility-first selection retains them. **Caveat:** internal order may change, so NDCG/MRR need not remain fixed. More strongly, if an original Top-K candidate is **ineligible**, even `lambda=0` can change the slate; the margin theorem does not apply.

#### Theorem 5 — Fixed-support relevance sensitivity

For fixed E, categories, eligibility membership, supported groups, and `R,R' >= R_min > 0`,

\[
\left|\frac{E}{R+\epsilon}-\frac{E}{R'+\epsilon}\right|\le\frac{E|R-R'|}{(R_{min}+\epsilon)^2}.
\]

**Proof:** subtract the two fractions and lower-bound both denominators. **Caveat:** hard `q` and support thresholds can switch sets, so global Lipschitz continuity of `L_fair` is not established. In fact, zero active categories yields `NaN`, demonstrating that additional domain assumptions are necessary.

#### Proposition 6 — Support-limited allocation

If category/group `(c,g)` contains no eligible candidates at all evaluated events, fairness-driven promotion into the group is impossible under the current gate. This does **not** imply its observed exposure is zero: baseline/fallback Top-10 slates can still contain such items. Moreover, the category group may be excluded from the reported objective when `R_cg < R_min`. **Consequence:** the metric can exclude groups facing the strongest opportunity scarcity, so report excluded groups and coverage.

### 5. Critical counterexamples and interpretation

1. **Eligibility-first selection can alter utility at lambda=0.** K=1, scores `(10,9)` and eligibility `(False,True)` lead to item 2 regardless of adjustment, although original KGSF chose item 1. A `lambda=0` ablation is **not** the backbone control.
2. **Overshoot can reverse intended improvement.** With `(rho_low,rho_high)=(0,2)` and equal weights, initial variance is `1`. Raising the low group to `2` gives zero; raising it to `5` gives `2.25` (worse than initially).
3. **Multi-label exposure is not conserved across categories.** A single Top-1 item with labels `{comedy, drama}` contributes a rank-1 exposure of 1 to both, although only one recommendation was delivered.
4. **Active-set change can alter loss even without a change in exposure.** When a previously unsupported group crosses `R_min` or a second group becomes supported, its category may enter the average and change `L_fair` discontinuously.
5. **Repeated category contribution rows are not separate category values.** The exported `category_variance_contribution` is repeated across supported popularity-group rows within the same category; reconstruct loss using one record per distinct `(system,variant,category)`.

### 6. Claims approved vs deferred

| Claim | Status |
|:---|:---|
| Objective equals mean active-category opportunity-weighted variance | Verified in inspected source |
| Relevance proxy uses KGSF softmax and positive KG-rank confidence | Verified in source |
| Eligibility-first, score-adjusted Top-10 with original-score fallback | Verified in source |
| Fairness variance is nonnegative with fixed active set | Proven |
| Single-group update decreases variance iff exact step-size condition holds | Proven under fixed-support assumptions |
| Full A3 reranking monotonically reduces global fairness loss | **Not proven and generally unsafe to claim** |
| A3 yields guaranteed Top-K utility preservation | **Only conditional margin theorem; not a general guarantee** |
| Fairness loss is globally continuous under relevance/support changes | **Not proven; thresholds imply discontinuities** |
| `q` measures human-perceived contextual relevance | **Not validated** |
| Experiment proves CCF-A-level novelty or cross-backbone generality | **Not established** |

### 7. Frozen experiment facts and caveats

- Test: 4,003 recommendation events, 1,283 conversations, identical frozen KGSF Top-500 per event.
- Frozen A3: `q_threshold=0.0005`, `R_min=0.5`, `lambda=1.0`.
- Reported cumulative fairness: A0 `1.7690978032`, A3 R0 `0.5243077807` (70.36% descriptive reduction).
- Test HR@10 remains approximately `0.169623`; NDCG@10 from `0.091053` to `0.091112`; bootstrap intervals for the small utility changes include zero.
- 16 active categories improved, 3 worsened (musical, action, horror); 19 categories active in test+history objective.
- Category-supported candidate instances: `100,909 / 2,001,500 = 5.04%`. This category-evidence rate and `q` eligibility count are different concepts and denominators.
- Evaluation reports cumulative validation-history + test exposure and uses deterministic sorted event IDs rather than verified chronological order.
- Earlier experiments had inspected the test split; present result is a frozen-configuration re-evaluation, not evidence from a never-examined test set.
- Source lineage: 22.35F/22.35C/22.16 archived hashes verified; 22.35J content reconciled via LF/CRLF representations in earlier audit. No experiments are rerun in this milestone.

### 8. Required next scientific milestone (22.35N)

**Name:** *Novelty and Controlled Mechanism Attribution Audit*.

**Before conducting new experiments:** build a concise closest-work matrix with one row per comparable method and columns: protected stakeholder (item/provider/user), category conditioning, relevance definition, eligibility policy, accumulated exposure, KG dependence, model-agnostic integration, mathematical guarantee, benchmark availability. This is a critical publication-decision gate.

**Minimal necessary component checks** (using frozen 22.35F validation events and candidate scores, without test-driven tuning):

- **A0:** frozen KGSF Top-10.
- **E-only:** eligibility-first selection with `lambda=0` (NOT A0), isolates eligibility selection.
- **Bonus-only (diagnostic):** ordinary adjusted-score ranking without eligibility-first selection, with the same frozen bonus and threshold — isolates the effect of prioritizing eligible candidates; label this explicitly as an algorithmic diagnostic, not a comparable published baseline.
- **No-category:** replace per-category deficits with an explicitly defined global group exposure/eligible opportunity computation.
- **No-KG:** compute `q=softmax(KGSF)` using the same frozen scores while preserving all other rules; report eligibility and utility changes; validation-only.
- **No-history:** reset exposure ledger per event, keeping all other components fixed; identifies history dependence.

**Do not immediately execute all variants:** first confirm the closest-work comparison and identify the 2–3 comparisons that reviewers cannot do without. Retain frozen A3 parameters for mechanical ablations but recognize `q` rescales under No-KG, so a fixed numeric threshold tests robustness, not perfectly matched gate selectivity. Match eligibility coverage in a secondary, validation-only sensitivity if needed.

**Completion criteria for 22.35N:** transparent novelty statement, explicit comparison to closest published item-fairness methods, minimal controlled ablations with predeclared outcomes/interpretation, independent-test handling that discloses historical exposure, and no unjustified new numerical claims.

### 9. Repository instructions

Store this source note under `docs/KG_FairCRS_22_35M_Theory_and_Verification_FINAL.md`. Accompany it with `tests/test_22_35M_theory_properties.py` (pure Python standard library) and preserve all frozen historical output. Stage **only** the two added files, review staged changes, commit and push on the current development branch, then merge to `main` after verification. Ignore recovery scratch files and generated test outputs.
