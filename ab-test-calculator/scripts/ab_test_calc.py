#!/usr/bin/env python3
"""
A/B Test Calculator - Statistical significance testing for A/B experiments.
"""

import argparse
import json
import math
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy import stats
from scipy.stats import beta as beta_dist


class ABTestCalculator:
    """Calculate statistical significance for A/B tests."""

    def __init__(self, alpha: float = 0.05):
        """
        Initialize calculator.

        Args:
            alpha: Significance level (default 0.05 for 95% confidence)
        """
        self.alpha = alpha

    def test_significance(self, control_visitors: int, control_conversions: int,
                         variant_visitors: int, variant_conversions: int,
                         test: str = "chi_square") -> Dict:
        """
        Test statistical significance between control and variant.

        Args:
            control_visitors: Number of visitors in control group
            control_conversions: Number of conversions in control group
            variant_visitors: Number of visitors in variant group
            variant_conversions: Number of conversions in variant group
            test: Test method ("chi_square", "z_test")

        Returns:
            Dictionary with test results
        """
        control_rate = control_conversions / control_visitors
        variant_rate = variant_conversions / variant_visitors

        lift = (variant_rate - control_rate) / control_rate if control_rate > 0 else 0
        lift_absolute = variant_rate - control_rate

        if test == "chi_square":
            p_value = self._chi_square_test(
                control_visitors, control_conversions,
                variant_visitors, variant_conversions
            )
        elif test == "z_test":
            p_value = self._z_test_proportions(
                control_visitors, control_conversions,
                variant_visitors, variant_conversions
            )
        else:
            raise ValueError(f"Unknown test method: {test}")

        # Calculate confidence interval for lift
        ci = self._lift_confidence_interval(
            control_visitors, control_conversions,
            variant_visitors, variant_conversions
        )

        significant = p_value < self.alpha

        if significant:
            if lift > 0:
                recommendation = "Variant shows significant improvement. Consider implementing."
            else:
                recommendation = "Variant shows significant decrease. Keep control."
        else:
            recommendation = "No significant difference detected. Need more data or larger effect."

        return {
            "significant": significant,
            "p_value": p_value,
            "control_rate": control_rate,
            "variant_rate": variant_rate,
            "lift": lift,
            "lift_absolute": lift_absolute,
            "confidence_interval": ci,
            "test_method": test,
            "alpha": self.alpha,
            "recommendation": recommendation
        }

    def _chi_square_test(self, c_visitors: int, c_conv: int,
                        v_visitors: int, v_conv: int) -> float:
        """Perform chi-square test for independence."""
        # Contingency table
        observed = np.array([
            [c_conv, c_visitors - c_conv],
            [v_conv, v_visitors - v_conv]
        ])

        chi2, p_value, dof, expected = stats.chi2_contingency(observed)
        return p_value

    def _z_test_proportions(self, c_visitors: int, c_conv: int,
                           v_visitors: int, v_conv: int) -> float:
        """Perform Z-test for two proportions."""
        p1 = c_conv / c_visitors
        p2 = v_conv / v_visitors
        n1 = c_visitors
        n2 = v_visitors

        # Pooled proportion
        p_pooled = (c_conv + v_conv) / (n1 + n2)

        # Standard error
        se = math.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))

        if se == 0:
            return 1.0

        # Z statistic
        z = (p2 - p1) / se

        # Two-tailed p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        return p_value

    def _lift_confidence_interval(self, c_visitors: int, c_conv: int,
                                  v_visitors: int, v_conv: int,
                                  confidence: float = 0.95) -> Dict:
        """Calculate confidence interval for relative lift."""
        p1 = c_conv / c_visitors
        p2 = v_conv / v_visitors

        # Standard errors
        se1 = math.sqrt(p1 * (1 - p1) / c_visitors)
        se2 = math.sqrt(p2 * (1 - p2) / v_visitors)

        # SE of difference
        se_diff = math.sqrt(se1**2 + se2**2)

        # Z value for confidence level
        z = stats.norm.ppf((1 + confidence) / 2)

        diff = p2 - p1
        margin = z * se_diff

        # Convert to relative lift if control rate > 0
        if p1 > 0:
            lower = (diff - margin) / p1
            upper = (diff + margin) / p1
        else:
            lower = 0
            upper = 0

        return {
            "lower": lower,
            "upper": upper,
            "confidence_level": confidence
        }

    def calculate_sample_size(self, baseline_rate: float,
                             minimum_detectable_effect: float,
                             power: float = 0.8,
                             alpha: float = None) -> Dict:
        """
        Calculate required sample size per variant.

        Args:
            baseline_rate: Current conversion rate (e.g., 0.05 for 5%)
            minimum_detectable_effect: Relative change to detect (e.g., 0.10 for 10%)
            power: Statistical power (default 0.8 for 80%)
            alpha: Significance level (uses instance default if not specified)

        Returns:
            Dictionary with sample size information
        """
        if alpha is None:
            alpha = self.alpha

        p1 = baseline_rate
        p2 = baseline_rate * (1 + minimum_detectable_effect)

        # Z values
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)

        # Pooled proportion
        p_pooled = (p1 + p2) / 2

        # Sample size formula
        numerator = (z_alpha * math.sqrt(2 * p_pooled * (1 - p_pooled)) +
                    z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
        denominator = (p2 - p1) ** 2

        if denominator == 0:
            sample_size = float('inf')
        else:
            sample_size = math.ceil(numerator / denominator)

        return {
            "sample_size_per_variant": sample_size,
            "total_sample_size": sample_size * 2,
            "baseline_rate": baseline_rate,
            "expected_variant_rate": p2,
            "minimum_detectable_effect": minimum_detectable_effect,
            "power": power,
            "alpha": alpha
        }

    def calculate_power(self, baseline_rate: float,
                       minimum_detectable_effect: float,
                       sample_size: int,
                       alpha: float = None) -> Dict:
        """
        Calculate statistical power given sample size.

        Args:
            baseline_rate: Current conversion rate
            minimum_detectable_effect: Relative change to detect
            sample_size: Sample size per variant
            alpha: Significance level

        Returns:
            Dictionary with power analysis
        """
        if alpha is None:
            alpha = self.alpha

        p1 = baseline_rate
        p2 = baseline_rate * (1 + minimum_detectable_effect)

        # Standard error
        se = math.sqrt(p1 * (1 - p1) / sample_size + p2 * (1 - p2) / sample_size)

        # Z value for alpha
        z_alpha = stats.norm.ppf(1 - alpha/2)

        # Calculate power
        z_beta = (abs(p2 - p1) / se) - z_alpha
        power = stats.norm.cdf(z_beta)

        return {
            "power": power,
            "sample_size_per_variant": sample_size,
            "baseline_rate": baseline_rate,
            "minimum_detectable_effect": minimum_detectable_effect,
            "alpha": alpha,
            "interpretation": f"{power:.0%} chance of detecting the effect if it exists"
        }

    def confidence_interval(self, visitors: int, conversions: int,
                           confidence: float = 0.95) -> Dict:
        """
        Calculate confidence interval for a conversion rate.

        Args:
            visitors: Total visitors
            conversions: Number of conversions
            confidence: Confidence level (default 0.95)

        Returns:
            Dictionary with CI bounds
        """
        rate = conversions / visitors

        # Wilson score interval (better for small samples)
        z = stats.norm.ppf((1 + confidence) / 2)

        denominator = 1 + z**2 / visitors
        center = (rate + z**2 / (2 * visitors)) / denominator
        margin = z * math.sqrt((rate * (1 - rate) + z**2 / (4 * visitors)) / visitors) / denominator

        return {
            "rate": rate,
            "lower": max(0, center - margin),
            "upper": min(1, center + margin),
            "confidence_level": confidence,
            "method": "wilson"
        }

    def bayesian_analysis(self, control_visitors: int, control_conversions: int,
                         variant_visitors: int, variant_conversions: int,
                         simulations: int = 100000) -> Dict:
        """
        Bayesian analysis using Beta distributions.

        Args:
            control_visitors: Number of visitors in control
            control_conversions: Number of conversions in control
            variant_visitors: Number of visitors in variant
            variant_conversions: Number of conversions in variant
            simulations: Number of Monte Carlo simulations

        Returns:
            Dictionary with Bayesian analysis results
        """
        # Beta distribution parameters (using uniform prior)
        alpha_c = control_conversions + 1
        beta_c = control_visitors - control_conversions + 1
        alpha_v = variant_conversions + 1
        beta_v = variant_visitors - variant_conversions + 1

        # Sample from posterior distributions
        control_samples = np.random.beta(alpha_c, beta_c, simulations)
        variant_samples = np.random.beta(alpha_v, beta_v, simulations)

        # Probability variant beats control
        prob_variant_better = np.mean(variant_samples > control_samples)

        # Expected lift
        lift_samples = (variant_samples - control_samples) / control_samples
        expected_lift = np.mean(lift_samples)

        # Credible interval for lift
        ci_lower = np.percentile(lift_samples, 2.5)
        ci_upper = np.percentile(lift_samples, 97.5)

        return {
            "prob_variant_better": prob_variant_better,
            "prob_control_better": 1 - prob_variant_better,
            "expected_lift": expected_lift,
            "credible_interval_95": [ci_lower, ci_upper],
            "control_rate_mean": np.mean(control_samples),
            "variant_rate_mean": np.mean(variant_samples)
        }

    def test_multiple_variants(self, control: Tuple[int, int],
                              variants: List[Tuple[int, int]],
                              correction: str = "bonferroni") -> Dict:
        """
        Test multiple variants against control with correction.

        Args:
            control: (visitors, conversions) for control
            variants: List of (visitors, conversions) tuples for variants
            correction: Multiple testing correction ("bonferroni", "holm", "none")

        Returns:
            Dictionary with all variant comparisons
        """
        c_visitors, c_conversions = control
        c_rate = c_conversions / c_visitors

        n_variants = len(variants)
        results = []
        p_values = []

        # Run tests for each variant
        for i, (v_visitors, v_conversions) in enumerate(variants):
            v_rate = v_conversions / v_visitors
            lift = (v_rate - c_rate) / c_rate if c_rate > 0 else 0

            p_value = self._chi_square_test(
                c_visitors, c_conversions,
                v_visitors, v_conversions
            )
            p_values.append(p_value)

            results.append({
                "name": f"Variant {chr(65 + i)}",
                "visitors": v_visitors,
                "conversions": v_conversions,
                "rate": v_rate,
                "lift": lift,
                "p_value": p_value
            })

        # Apply correction
        if correction == "bonferroni":
            adjusted_alpha = self.alpha / n_variants
            for r in results:
                r["significant"] = r["p_value"] < adjusted_alpha
        elif correction == "holm":
            # Holm-Bonferroni method
            sorted_indices = np.argsort(p_values)
            for rank, idx in enumerate(sorted_indices):
                adjusted_alpha = self.alpha / (n_variants - rank)
                results[idx]["significant"] = p_values[idx] < adjusted_alpha
        else:
            for r in results:
                r["significant"] = r["p_value"] < self.alpha

        # Find winner
        significant_results = [r for r in results if r["significant"] and r["lift"] > 0]
        if significant_results:
            winner = max(significant_results, key=lambda x: x["lift"])["name"]
        else:
            winner = "Control (no significant improvement)"

        return {
            "control": {
                "visitors": c_visitors,
                "conversions": c_conversions,
                "rate": c_rate
            },
            "variants": results,
            "winner": winner,
            "correction_method": correction,
            "alpha": self.alpha
        }

    def estimate_duration(self, daily_visitors: int, baseline_rate: float,
                         minimum_detectable_effect: float,
                         power: float = 0.8) -> Dict:
        """
        Estimate test duration based on traffic.

        Args:
            daily_visitors: Average daily visitors
            baseline_rate: Current conversion rate
            minimum_detectable_effect: Relative change to detect
            power: Desired statistical power

        Returns:
            Dictionary with duration estimates
        """
        sample = self.calculate_sample_size(
            baseline_rate, minimum_detectable_effect, power
        )

        # Assuming 50/50 split
        visitors_per_variant_per_day = daily_visitors / 2

        if visitors_per_variant_per_day > 0:
            days = math.ceil(sample["sample_size_per_variant"] / visitors_per_variant_per_day)
        else:
            days = float('inf')

        return {
            "days": days,
            "weeks": math.ceil(days / 7),
            "sample_size_per_variant": sample["sample_size_per_variant"],
            "daily_visitors": daily_visitors,
            "visitors_per_variant_per_day": visitors_per_variant_per_day
        }


def main():
    parser = argparse.ArgumentParser(
        description="A/B Test Calculator - Statistical significance testing"
    )

    parser.add_argument("--test", nargs=4, type=int, metavar=("CV", "CC", "VV", "VC"),
                       help="Test significance: control_visitors control_conversions variant_visitors variant_conversions")
    parser.add_argument("--sample-size", action="store_true",
                       help="Calculate sample size")
    parser.add_argument("--power-analysis", action="store_true",
                       help="Perform power analysis")
    parser.add_argument("--bayesian", nargs=4, type=int, metavar=("CV", "CC", "VV", "VC"),
                       help="Bayesian analysis: control_visitors control_conversions variant_visitors variant_conversions")
    parser.add_argument("--test-multi", nargs="+", type=int,
                       help="Test multiple variants: c_visitors c_conv v1_visitors v1_conv [v2_visitors v2_conv ...]")

    parser.add_argument("--baseline", type=float, help="Baseline conversion rate (e.g., 0.05)")
    parser.add_argument("--mde", type=float, help="Minimum detectable effect (e.g., 0.10)")
    parser.add_argument("--power", type=float, default=0.8, help="Statistical power (default: 0.8)")
    parser.add_argument("--samples", type=int, help="Sample size per variant")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level (default: 0.05)")
    parser.add_argument("--method", choices=["chi_square", "z_test"], default="chi_square",
                       help="Test method (default: chi_square)")

    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    calc = ABTestCalculator(alpha=args.alpha)

    if args.test:
        result = calc.test_significance(
            args.test[0], args.test[1],
            args.test[2], args.test[3],
            test=args.method
        )

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print("\n=== A/B Test Results ===")
            print(f"Control:  {args.test[1]}/{args.test[0]} ({result['control_rate']:.2%})")
            print(f"Variant:  {args.test[3]}/{args.test[2]} ({result['variant_rate']:.2%})")
            print(f"\nLift: {result['lift']:+.2%} ({result['lift_absolute']:+.4f} absolute)")
            print(f"P-value: {result['p_value']:.4f}")
            print(f"Significant: {'Yes' if result['significant'] else 'No'} (alpha={args.alpha})")
            print(f"\n95% CI for lift: [{result['confidence_interval']['lower']:.2%}, {result['confidence_interval']['upper']:.2%}]")
            print(f"\n{result['recommendation']}")

    elif args.sample_size:
        if not args.baseline or not args.mde:
            parser.error("--sample-size requires --baseline and --mde")

        result = calc.calculate_sample_size(
            args.baseline, args.mde, args.power
        )

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print("\n=== Sample Size Calculation ===")
            print(f"Baseline rate: {result['baseline_rate']:.2%}")
            print(f"Expected variant rate: {result['expected_variant_rate']:.2%}")
            print(f"Minimum detectable effect: {result['minimum_detectable_effect']:.1%}")
            print(f"Power: {result['power']:.0%}")
            print(f"Alpha: {result['alpha']}")
            print(f"\nRequired sample size per variant: {result['sample_size_per_variant']:,}")
            print(f"Total sample size (both variants): {result['total_sample_size']:,}")

    elif args.power_analysis:
        if not args.baseline or not args.mde or not args.samples:
            parser.error("--power-analysis requires --baseline, --mde, and --samples")

        result = calc.calculate_power(
            args.baseline, args.mde, args.samples
        )

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print("\n=== Power Analysis ===")
            print(f"Sample size per variant: {result['sample_size_per_variant']:,}")
            print(f"Baseline rate: {result['baseline_rate']:.2%}")
            print(f"MDE: {result['minimum_detectable_effect']:.1%}")
            print(f"\nStatistical Power: {result['power']:.1%}")
            print(f"({result['interpretation']})")

    elif args.bayesian:
        result = calc.bayesian_analysis(
            args.bayesian[0], args.bayesian[1],
            args.bayesian[2], args.bayesian[3]
        )

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print("\n=== Bayesian Analysis ===")
            print(f"Control rate: {result['control_rate_mean']:.2%}")
            print(f"Variant rate: {result['variant_rate_mean']:.2%}")
            print(f"\nProbability variant beats control: {result['prob_variant_better']:.1%}")
            print(f"Expected lift: {result['expected_lift']:+.2%}")
            print(f"95% Credible interval: [{result['credible_interval_95'][0]:.2%}, {result['credible_interval_95'][1]:.2%}]")

    elif args.test_multi:
        if len(args.test_multi) < 4 or len(args.test_multi) % 2 != 0:
            parser.error("--test-multi requires pairs of (visitors, conversions)")

        control = (args.test_multi[0], args.test_multi[1])
        variants = []
        for i in range(2, len(args.test_multi), 2):
            variants.append((args.test_multi[i], args.test_multi[i+1]))

        result = calc.test_multiple_variants(control, variants)

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print("\n=== Multiple Variant Test ===")
            print(f"Control: {result['control']['conversions']}/{result['control']['visitors']} ({result['control']['rate']:.2%})")
            print(f"\nVariants:")
            for v in result['variants']:
                sig = "*" if v['significant'] else ""
                print(f"  {v['name']}: {v['conversions']}/{v['visitors']} ({v['rate']:.2%}) "
                      f"Lift: {v['lift']:+.2%} p={v['p_value']:.4f}{sig}")
            print(f"\nWinner: {result['winner']}")
            print(f"(Correction: {result['correction_method']})")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
