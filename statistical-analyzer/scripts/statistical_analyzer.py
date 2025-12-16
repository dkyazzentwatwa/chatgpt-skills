#!/usr/bin/env python3
"""
Statistical Analyzer - Guided statistical analysis with plain-English interpretations.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import shapiro, levene, normaltest
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


class StatisticalAnalyzer:
    """Perform statistical analyses with interpretations."""

    def __init__(self):
        """Initialize the analyzer."""
        self.data = None
        self.results = []
        self.plots = []

    def load_data(self, data: pd.DataFrame, **kwargs) -> 'StatisticalAnalyzer':
        """Load data from DataFrame."""
        self.data = data.copy()
        return self

    def load_csv(self, filepath: str, **kwargs) -> 'StatisticalAnalyzer':
        """Load data from CSV."""
        self.data = pd.read_csv(filepath, **kwargs)
        return self

    def t_test(self, group1: str = None, group2: str = None,
               paired: bool = False, alternative: str = 'two-sided',
               group_col: str = None, value_col: str = None) -> Dict:
        """
        Perform t-test comparing two groups.

        Args:
            group1: Name of first group (if using group_col)
            group2: Name of second group (if using group_col)
            paired: Whether to perform paired t-test
            alternative: 'two-sided', 'less', or 'greater'
            group_col: Column containing group labels
            value_col: Column containing values to compare
        """
        if group_col and value_col:
            # Data in long format
            data1 = self.data[self.data[group_col] == group1][value_col]
            data2 = self.data[self.data[group_col] == group2][value_col]
        else:
            # Data in wide format (two columns)
            data1 = self.data[group1]
            data2 = self.data[group2]

        # Check assumptions
        assumptions = self._check_t_test_assumptions(data1, data2, paired)

        # Perform test
        if paired:
            statistic, p_value = stats.ttest_rel(data1, data2, alternative=alternative)
        else:
            # Check equal variances
            equal_var = assumptions['equal_variances']['equal']
            statistic, p_value = stats.ttest_ind(data1, data2,
                                                  equal_var=equal_var,
                                                  alternative=alternative)

        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt((data1.std()**2 + data2.std()**2) / 2)
        cohens_d = (data1.mean() - data2.mean()) / pooled_std

        # Confidence interval
        diff = data1.mean() - data2.mean()
        se = pooled_std * np.sqrt(1/len(data1) + 1/len(data2))
        ci = (diff - 1.96*se, diff + 1.96*se)

        result = {
            'test': 'Paired t-test' if paired else 'Independent samples t-test',
            'statistic': float(statistic),
            'p_value': float(p_value),
            'mean_diff': float(diff),
            'cohens_d': float(cohens_d),
            'ci_95': ci,
            'assumptions': assumptions,
            'interpretation': self._interpret_t_test(p_value, diff, cohens_d, assumptions)
        }

        self.results.append(result)
        return result

    def _check_t_test_assumptions(self, data1, data2, paired) -> Dict:
        """Check t-test assumptions."""
        # Normality tests
        _, p1 = shapiro(data1)
        _, p2 = shapiro(data2)
        normal1 = p1 > 0.05
        normal2 = p2 > 0.05

        assumptions = {
            'normality_group1': {'normal': normal1, 'p_value': float(p1)},
            'normality_group2': {'normal': normal2, 'p_value': float(p2)}
        }

        if not paired:
            # Test equal variances (Levene's test)
            _, p_levene = levene(data1, data2)
            assumptions['equal_variances'] = {
                'equal': p_levene > 0.05,
                'p_value': float(p_levene)
            }

        return assumptions

    def _interpret_t_test(self, p_value, mean_diff, cohens_d, assumptions) -> str:
        """Generate plain-English interpretation."""
        # Significance
        if p_value < 0.001:
            sig = "highly significant"
        elif p_value < 0.01:
            sig = "very significant"
        elif p_value < 0.05:
            sig = "statistically significant"
        else:
            sig = "not statistically significant"

        # Effect size
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            effect = "negligible"
        elif abs_d < 0.5:
            effect = "small"
        elif abs_d < 0.8:
            effect = "medium"
        else:
            effect = "large"

        interp = f"The difference is {sig} (p={p_value:.4f}). "
        interp += f"The mean difference is {mean_diff:.2f} with a {effect} effect size (Cohen's d={cohens_d:.2f}). "

        # Assumptions warnings
        if not assumptions['normality_group1']['normal']:
            interp += "Warning: Group 1 data may not be normally distributed. "
        if not assumptions['normality_group2']['normal']:
            interp += "Warning: Group 2 data may not be normally distributed. "
        if 'equal_variances' in assumptions and not assumptions['equal_variances']['equal']:
            interp += "Note: Welch's t-test was used due to unequal variances. "

        return interp

    def one_sample_t_test(self, column: str, expected_mean: float,
                         alternative: str = 'two-sided') -> Dict:
        """One-sample t-test against expected mean."""
        data = self.data[column].dropna()

        statistic, p_value = stats.ttest_1samp(data, expected_mean, alternative=alternative)

        result = {
            'test': 'One-sample t-test',
            'statistic': float(statistic),
            'p_value': float(p_value),
            'sample_mean': float(data.mean()),
            'expected_mean': expected_mean,
            'interpretation': f"The sample mean ({data.mean():.2f}) is "
                            f"{'significantly' if p_value < 0.05 else 'not significantly'} "
                            f"different from {expected_mean} (p={p_value:.4f})."
        }

        self.results.append(result)
        return result

    def anova(self, group_col: str, value_col: str, post_hoc: bool = True) -> Dict:
        """One-way ANOVA with optional post-hoc tests."""
        groups = self.data[group_col].unique()
        group_data = [self.data[self.data[group_col] == g][value_col].values
                     for g in groups]

        # Perform ANOVA
        statistic, p_value = stats.f_oneway(*group_data)

        result = {
            'test': 'One-way ANOVA',
            'statistic': float(statistic),
            'p_value': float(p_value),
            'groups': list(groups),
            'group_means': {g: float(self.data[self.data[group_col] == g][value_col].mean())
                          for g in groups}
        }

        # Post-hoc tests (Tukey HSD)
        if post_hoc and p_value < 0.05:
            tukey = pairwise_tukeyhsd(self.data[value_col], self.data[group_col])
            result['post_hoc'] = str(tukey)

        result['interpretation'] = self._interpret_anova(p_value, groups, result.get('post_hoc'))

        self.results.append(result)
        return result

    def _interpret_anova(self, p_value, groups, post_hoc) -> str:
        """Interpret ANOVA results."""
        if p_value < 0.05:
            interp = f"There is a statistically significant difference between the {len(groups)} groups (p={p_value:.4f}). "
            if post_hoc:
                interp += "Post-hoc tests (Tukey HSD) identify which specific groups differ. "
        else:
            interp = f"No statistically significant difference found between groups (p={p_value:.4f}). "

        return interp

    def chi_square(self, observed: pd.DataFrame, expected=None) -> Dict:
        """Chi-square test for independence."""
        if expected is None:
            # Test of independence (contingency table)
            chi2, p_value, dof, expected_freq = stats.chi2_contingency(observed)
        else:
            # Goodness of fit test
            chi2, p_value = stats.chisquare(observed.values.flatten(), expected)
            dof = len(observed) - 1

        result = {
            'test': 'Chi-square test',
            'statistic': float(chi2),
            'p_value': float(p_value),
            'degrees_of_freedom': int(dof),
            'interpretation': f"The association is {'statistically significant' if p_value < 0.05 else 'not significant'} "
                            f"(χ²={chi2:.2f}, p={p_value:.4f})."
        }

        self.results.append(result)
        return result

    def linear_regression(self, x: str, y: str) -> Dict:
        """Simple linear regression."""
        X = self.data[x].values
        Y = self.data[y].values

        # Remove NaN
        mask = ~np.isnan(X) & ~np.isnan(Y)
        X, Y = X[mask], Y[mask]

        # Fit model
        X_with_const = sm.add_constant(X)
        model = sm.OLS(Y, X_with_const).fit()

        # Calculate R²
        slope = model.params[1]
        intercept = model.params[0]
        r_squared = model.rsquared
        p_value = model.pvalues[1]

        # Store for plotting
        self.last_regression = {
            'x': x, 'y': y,
            'X': X, 'Y': Y,
            'slope': slope,
            'intercept': intercept,
            'model': model
        }

        result = {
            'test': 'Linear regression',
            'slope': float(slope),
            'intercept': float(intercept),
            'r_squared': float(r_squared),
            'p_value': float(p_value),
            'equation': f"y = {slope:.3f}x + {intercept:.3f}",
            'interpretation': self._interpret_regression(r_squared, p_value)
        }

        self.results.append(result)
        return result

    def _interpret_regression(self, r_squared, p_value) -> str:
        """Interpret regression results."""
        if r_squared < 0.3:
            strength = "weak"
        elif r_squared < 0.7:
            strength = "moderate"
        else:
            strength = "strong"

        sig = "statistically significant" if p_value < 0.05 else "not significant"

        return f"The model shows a {strength} fit (R²={r_squared:.3f}) and is {sig} (p={p_value:.4f}). " \
               f"Approximately {r_squared*100:.1f}% of the variance is explained by the model."

    def correlation(self, method: str = 'pearson', columns: List[str] = None) -> pd.DataFrame:
        """Calculate correlation matrix."""
        if columns:
            data = self.data[columns]
        else:
            data = self.data.select_dtypes(include=[np.number])

        if method == 'pearson':
            corr = data.corr(method='pearson')
        elif method == 'spearman':
            corr = data.corr(method='spearman')
        else:
            corr = data.corr(method='kendall')

        return corr

    def correlation_test(self, var1: str, var2: str, method: str = 'pearson') -> Dict:
        """Test correlation between two variables."""
        x = self.data[var1].dropna()
        y = self.data[var2].dropna()

        # Align indices
        common_idx = x.index.intersection(y.index)
        x = x.loc[common_idx]
        y = y.loc[common_idx]

        if method == 'pearson':
            r, p_value = stats.pearsonr(x, y)
        elif method == 'spearman':
            r, p_value = stats.spearmanr(x, y)
        else:
            raise ValueError("Method must be 'pearson' or 'spearman'")

        # Interpret strength
        abs_r = abs(r)
        if abs_r < 0.3:
            strength = "weak"
        elif abs_r < 0.7:
            strength = "moderate"
        else:
            strength = "strong"

        direction = "positive" if r > 0 else "negative"

        result = {
            'test': f'{method.capitalize()} correlation',
            'correlation': float(r),
            'p_value': float(p_value),
            'interpretation': f"There is a {strength} {direction} correlation (r={r:.3f}, p={p_value:.4f})."
        }

        self.results.append(result)
        return result

    def normality_test(self, column: str, method: str = 'shapiro') -> Dict:
        """Test if data is normally distributed."""
        data = self.data[column].dropna()

        if method == 'shapiro':
            statistic, p_value = shapiro(data)
        elif method == 'normaltest':
            statistic, p_value = normaltest(data)
        else:
            raise ValueError("Method must be 'shapiro' or 'normaltest'")

        result = {
            'test': f'{method.capitalize()} normality test',
            'statistic': float(statistic),
            'p_value': float(p_value),
            'interpretation': f"Data appears {'normally distributed' if p_value > 0.05 else 'not normally distributed'} "
                            f"(p={p_value:.4f})."
        }

        self.results.append(result)
        return result

    def plot_regression(self, output: str, x: str = None, y: str = None) -> str:
        """Plot regression line with data points."""
        if not hasattr(self, 'last_regression'):
            raise ValueError("Run linear_regression() first")

        reg = self.last_regression

        plt.figure(figsize=(10, 6))
        plt.scatter(reg['X'], reg['Y'], alpha=0.6, label='Data')

        # Regression line
        X_line = np.linspace(reg['X'].min(), reg['X'].max(), 100)
        Y_line = reg['slope'] * X_line + reg['intercept']
        plt.plot(X_line, Y_line, 'r-', label=f"y = {reg['slope']:.3f}x + {reg['intercept']:.3f}")

        # Confidence interval
        predict = reg['model'].get_prediction(sm.add_constant(X_line))
        ci = predict.conf_int(alpha=0.05)
        plt.fill_between(X_line, ci[:, 0], ci[:, 1], alpha=0.2, label='95% CI')

        plt.xlabel(reg['x'])
        plt.ylabel(reg['y'])
        plt.title('Linear Regression')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output, dpi=300, bbox_inches='tight')
        plt.close()

        return output

    def plot_boxplot(self, group_col: str, value_col: str, output: str) -> str:
        """Create box plot for groups."""
        plt.figure(figsize=(10, 6))
        self.data.boxplot(column=value_col, by=group_col, figsize=(10, 6))
        plt.suptitle('')
        plt.title(f'{value_col} by {group_col}')
        plt.ylabel(value_col)
        plt.tight_layout()
        plt.savefig(output, dpi=300, bbox_inches='tight')
        plt.close()

        return output

    def plot_distribution(self, column: str, output: str) -> str:
        """Plot distribution with normal curve overlay."""
        data = self.data[column].dropna()

        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=30, density=True, alpha=0.7, edgecolor='black')

        # Overlay normal curve
        mu, sigma = data.mean(), data.std()
        x = np.linspace(data.min(), data.max(), 100)
        plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal')

        plt.xlabel(column)
        plt.ylabel('Density')
        plt.title(f'Distribution of {column}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output, dpi=300, bbox_inches='tight')
        plt.close()

        return output

    def summary(self) -> str:
        """Get summary of all analyses."""
        summary = "Statistical Analysis Summary\n"
        summary += "=" * 50 + "\n\n"

        for i, result in enumerate(self.results, 1):
            summary += f"{i}. {result['test']}\n"
            summary += f"   {result['interpretation']}\n\n"

        return summary


def main():
    parser = argparse.ArgumentParser(description="Statistical Analyzer")

    parser.add_argument("--data", "-d", required=True, help="Input CSV file")
    parser.add_argument("--output", "-o", required=True, help="Output file")

    parser.add_argument("--test", choices=['t-test', 'anova', 'regression', 'correlation'],
                       required=True, help="Type of test")

    # Common arguments
    parser.add_argument("--group", help="Group column")
    parser.add_argument("--value", help="Value column")

    # T-test specific
    parser.add_argument("--group1", help="First group name")
    parser.add_argument("--group2", help="Second group name")
    parser.add_argument("--paired", action="store_true", help="Paired t-test")

    # Regression specific
    parser.add_argument("--x", help="X variable for regression")
    parser.add_argument("--y", help="Y variable for regression")

    args = parser.parse_args()

    analyzer = StatisticalAnalyzer()
    analyzer.load_csv(args.data)

    if args.test == 't-test':
        result = analyzer.t_test(
            group1=args.group1,
            group2=args.group2,
            group_col=args.group,
            value_col=args.value,
            paired=args.paired
        )
        print(result['interpretation'])

    elif args.test == 'anova':
        result = analyzer.anova(group_col=args.group, value_col=args.value)
        print(result['interpretation'])

    elif args.test == 'regression':
        result = analyzer.linear_regression(x=args.x, y=args.y)
        print(result['interpretation'])
        analyzer.plot_regression(args.output)
        print(f"Regression plot saved: {args.output}")

    elif args.test == 'correlation':
        corr_matrix = analyzer.correlation()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=1)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.savefig(args.output, dpi=300, bbox_inches='tight')
        print(f"Correlation matrix saved: {args.output}")


if __name__ == "__main__":
    main()
