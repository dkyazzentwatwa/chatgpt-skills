#!/usr/bin/env python3
"""Statistical Power Calculator - Calculate power and sample sizes."""

import argparse
import numpy as np
from scipy import stats
from statsmodels.stats.power import TTestIndPower, TTestPower


class PowerCalculator:
    """Calculate statistical power and sample sizes."""

    def __init__(self):
        pass

    def sample_size_ttest(self, effect_size: float, alpha: float = 0.05,
                         power: float = 0.8, alternative: str = 'two-sided') -> dict:
        """Calculate required sample size for independent t-test."""
        analysis = TTestIndPower()
        n_per_group = analysis.solve_power(
            effect_size=effect_size,
            alpha=alpha,
            power=power,
            alternative=alternative
        )

        return {
            'n_per_group': int(np.ceil(n_per_group)),
            'total_n': int(np.ceil(n_per_group * 2)),
            'effect_size': effect_size,
            'alpha': alpha,
            'power': power,
            'test': 'independent t-test'
        }

    def power_ttest(self, n_per_group: int, effect_size: float,
                   alpha: float = 0.05, alternative: str = 'two-sided') -> float:
        """Calculate statistical power for independent t-test."""
        analysis = TTestIndPower()
        power = analysis.solve_power(
            effect_size=effect_size,
            nobs1=n_per_group,
            alpha=alpha,
            alternative=alternative
        )

        return power

    def effect_size_from_means(self, mean1: float, mean2: float, std: float) -> float:
        """Calculate Cohen's d effect size."""
        return abs(mean1 - mean2) / std


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate statistical power and sample sizes')
    parser.add_argument('--test', default='ttest', choices=['ttest'], help='Statistical test')
    parser.add_argument('--effect-size', type=float, help='Cohen\'s d effect size')
    parser.add_argument('--alpha', type=float, default=0.05, help='Significance level')
    parser.add_argument('--power', type=float, default=0.8, help='Desired power')
    parser.add_argument('--n', type=int, help='Sample size per group (for power calculation)')
    parser.add_argument('--alternative', default='two-sided', choices=['two-sided', 'larger', 'smaller'])

    args = parser.parse_args()

    calc = PowerCalculator()

    if args.n:
        # Calculate power
        if not args.effect_size:
            print("Error: --effect-size required for power calculation")
            sys.exit(1)

        power = calc.power_ttest(
            n_per_group=args.n,
            effect_size=args.effect_size,
            alpha=args.alpha,
            alternative=args.alternative
        )

        print(f"\nStatistical Power Analysis")
        print("=" * 50)
        print(f"Test: Independent t-test ({args.alternative})")
        print(f"Sample size per group: {args.n}")
        print(f"Effect size (Cohen's d): {args.effect_size}")
        print(f"Significance level (α): {args.alpha}")
        print(f"\nStatistical Power: {power:.4f} ({power*100:.2f}%)")

        if power < 0.8:
            print(f"\n⚠ Warning: Power is below recommended 0.80")

    else:
        # Calculate sample size
        if not args.effect_size:
            print("Error: --effect-size required")
            sys.exit(1)

        result = calc.sample_size_ttest(
            effect_size=args.effect_size,
            alpha=args.alpha,
            power=args.power,
            alternative=args.alternative
        )

        print(f"\nSample Size Calculation")
        print("=" * 50)
        print(f"Test: {result['test']} ({args.alternative})")
        print(f"Effect size (Cohen's d): {result['effect_size']}")
        print(f"Significance level (α): {result['alpha']}")
        print(f"Desired power: {result['power']}")
        print(f"\nRequired sample size:")
        print(f"  Per group: {result['n_per_group']}")
        print(f"  Total: {result['total_n']}")
