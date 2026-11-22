"""Milestone 22.35M: dependency-free mathematical property tests.

Tests algebraic identities and explicitly bounded counterexamples. Does NOT
claim to test experimental statistical significance or rerun KG-FairCRS.
Run: python -m unittest -v test_22_35M_theory_properties.py
"""
import math
import random
import unittest

EPS = 1e-12


def weighted_variance(rho, p):
    mu = sum(a * b for a, b in zip(rho, p))
    return sum(weight * (r - mu) ** 2 for r, weight in zip(rho, p))


def pairwise_variance(rho, p):
    return sum(p[i] * p[j] * (rho[i] - rho[j]) ** 2
               for i in range(len(rho)) for j in range(i + 1, len(rho)))


def delta_variance(p, rho, idx, delta_ratio):
    mu = sum(a * b for a, b in zip(rho, p))
    pg = p[idx]
    return 2 * pg * (rho[idx] - mu) * delta_ratio + pg * (1 - pg) * delta_ratio ** 2


def gate_first_selection(scores, eligible, bonus, k):
    # Equivalent conceptually to the frozen A3: sort adjusted scores,
    # take eligible candidates first; fill shortfall from original score order.
    adjusted = [s + b if ok else s for s, b, ok in zip(scores, bonus, eligible)]
    ordering = sorted(range(len(scores)), key=lambda i: (-adjusted[i], i))
    chosen = [i for i in ordering if eligible[i]][:k]
    if len(chosen) < k:
        original = sorted(range(len(scores)), key=lambda i: -scores[i])
        chosen.extend(i for i in original if i not in chosen)
    return chosen[:k]


class TestTheoryProperties(unittest.TestCase):
    def test_pairwise_identity(self):
        rng = random.Random(2235)
        for n in (2, 3, 5, 12):
            for _ in range(100):
                weights = [rng.random() + 0.01 for _ in range(n)]
                total = sum(weights)
                p = [x / total for x in weights]
                rho = [rng.uniform(0, 9) for _ in range(n)]
                self.assertAlmostEqual(weighted_variance(rho, p), pairwise_variance(rho, p), places=10)

    def test_nonnegativity_and_equality(self):
        p = [0.2, 0.3, 0.5]
        self.assertGreaterEqual(weighted_variance([1, 2, 3], p), 0)
        self.assertAlmostEqual(weighted_variance([4, 4, 4], p), 0)

    def test_exact_single_group_delta_formula(self):
        rng = random.Random(2235)
        for _ in range(1000):
            p = [rng.random() + 0.05 for _ in range(3)]
            total = sum(p)
            p = [x / total for x in p]
            rho = [rng.uniform(0, 10) for _ in range(3)]
            index = rng.randrange(3)
            x = rng.uniform(0, 10)
            newrho = rho.copy()
            newrho[index] += x
            observed = weighted_variance(newrho, p) - weighted_variance(rho, p)
            formula = delta_variance(p, rho, index, x)
            self.assertAlmostEqual(observed, formula, places=9)

    def test_improvement_and_overshoot(self):
        p = [0.5, 0.5]
        start = weighted_variance([0, 2], p)
        self.assertAlmostEqual(start, 1.0)
        self.assertAlmostEqual(weighted_variance([2, 2], p), 0)
        self.assertGreater(weighted_variance([5, 2], p), start)
        # Here mu = 1, low group = 0, p_low=.5, so x must be <4.
        self.assertLess(delta_variance(p, [0, 2], 0, 3), 0)
        self.assertAlmostEqual(delta_variance(p, [0, 2], 0, 4), 0)
        self.assertGreater(delta_variance(p, [0, 2], 0, 5), 0)

    def test_gate_can_change_ranking_with_zero_bonus(self):
        self.assertEqual(gate_first_selection([10, 9], [False, True], [0, 0], 1), [1])

    def test_margin_condition_preserves_membership(self):
        base_scores = [12, 11, 8, 7]
        elig = [True, True, True, True]
        k, bound = 2, 1.0
        self.assertGreater(base_scores[k - 1] - base_scores[k], 2 * bound)
        for b1 in (-1.0, 0.0, 1.0):
            for b2 in (-1.0, 0.0, 1.0):
                for b3 in (-1.0, 0.0, 1.0):
                    for b4 in (-1.0, 0.0, 1.0):
                        chosen = gate_first_selection(base_scores, elig, [b1, b2, b3, b4], k)
                        self.assertEqual(set(chosen), {0, 1})

    def test_fixed_support_relevance_bound(self):
        E, R, Rp, rmin, eps = 5.0, 1.0, 1.2, 0.5, EPS
        observed = abs(E / (R + eps) - E / (Rp + eps))
        upper = E * abs(R - Rp) / (rmin + eps) ** 2
        self.assertLessEqual(observed, upper)

    def test_multilabel_exposure_double_attribution(self):
        # One rank-1 recommendation can count once in each assigned category.
        categories = ["comedy", "drama"]
        allocations = {c: 1 / math.log2(1 + 1) for c in categories}
        self.assertEqual(sum(allocations.values()), 2.0)
        self.assertEqual(len(categories), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
