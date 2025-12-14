#!/usr/bin/env python3
"""
Financial Calculator Suite - Comprehensive financial calculations
Includes loans, investments, NPV/IRR, retirement planning, and Monte Carlo simulations.
"""

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize

# Try to import numpy_financial
try:
    import numpy_financial as npf
    HAS_NPF = True
except ImportError:
    HAS_NPF = False


class FinanceError(Exception):
    """Custom exception for financial calculation errors."""
    pass


@dataclass
class FinanceConfig:
    """Configuration for financial calculations."""
    default_inflation: float = 2.5
    default_monte_carlo_runs: int = 1000
    currency_symbol: str = '$'
    decimal_places: int = 2


class FinancialCalculator:
    """
    Comprehensive financial calculator with multiple analysis tools.
    """

    def __init__(self):
        """Initialize financial calculator."""
        self.config = FinanceConfig()

    # ==================== LOAN CALCULATIONS ====================

    def loan_payment(
        self,
        principal: float,
        rate: float,
        years: int,
        payments_per_year: int = 12
    ) -> Dict[str, Any]:
        """
        Calculate loan payment and summary.

        Args:
            principal: Loan amount
            rate: Annual interest rate (%)
            years: Loan term in years
            payments_per_year: Number of payments per year (default: 12)

        Returns:
            Dict with payment details
        """
        if principal <= 0:
            raise FinanceError("Principal must be positive")
        if rate < 0:
            raise FinanceError("Rate cannot be negative")
        if years <= 0:
            raise FinanceError("Term must be positive")

        # Convert rate to periodic rate
        periodic_rate = rate / 100 / payments_per_year
        total_payments = years * payments_per_year

        if periodic_rate == 0:
            payment = principal / total_payments
        else:
            # PMT formula
            payment = principal * (
                periodic_rate * (1 + periodic_rate) ** total_payments
            ) / (
                (1 + periodic_rate) ** total_payments - 1
            )

        total_paid = payment * total_payments
        total_interest = total_paid - principal

        return {
            'principal': principal,
            'rate': rate,
            'years': years,
            'monthly_payment': payment,
            'total_payments': total_paid,
            'total_interest': total_interest,
            'interest_ratio': total_interest / principal * 100,
            'payment_breakdown': {
                'avg_monthly_principal': principal / total_payments,
                'avg_monthly_interest': total_interest / total_payments,
            }
        }

    def amortization_schedule(
        self,
        principal: float,
        rate: float,
        years: int,
        extra_payment: float = 0,
        payments_per_year: int = 12
    ) -> List[Dict[str, Any]]:
        """
        Generate full amortization schedule.

        Args:
            principal: Loan amount
            rate: Annual interest rate (%)
            years: Loan term
            extra_payment: Additional payment each period
            payments_per_year: Payments per year

        Returns:
            List of payment records
        """
        loan = self.loan_payment(principal, rate, years, payments_per_year)
        base_payment = loan['monthly_payment']
        payment = base_payment + extra_payment

        periodic_rate = rate / 100 / payments_per_year
        balance = principal
        schedule = []
        total_interest = 0
        total_principal = 0

        month = 0
        while balance > 0.01:  # Account for rounding
            month += 1
            interest = balance * periodic_rate
            principal_paid = min(payment - interest, balance)

            # Handle final payment
            if principal_paid > balance:
                principal_paid = balance
                payment = principal_paid + interest

            balance -= principal_paid
            total_interest += interest
            total_principal += principal_paid

            schedule.append({
                'month': month,
                'payment': payment,
                'principal': principal_paid,
                'interest': interest,
                'balance': max(0, balance),
                'total_interest': total_interest,
                'total_principal': total_principal,
            })

            if balance <= 0:
                break

        return schedule

    def prepayment_comparison(
        self,
        principal: float,
        rate: float,
        years: int,
        extra_monthly: float = 0,
        extra_annual: float = 0,
        lump_sum: float = 0,
        lump_sum_month: int = 12
    ) -> Dict[str, Any]:
        """
        Compare regular payments vs prepayment scenarios.

        Args:
            principal: Loan amount
            rate: Annual interest rate
            years: Loan term
            extra_monthly: Additional monthly payment
            extra_annual: Additional annual payment
            lump_sum: One-time extra payment
            lump_sum_month: Month for lump sum

        Returns:
            Comparison results
        """
        # Original schedule
        original = self.amortization_schedule(principal, rate, years)
        original_months = len(original)
        original_interest = original[-1]['total_interest']

        # With prepayments
        new_schedule = self.amortization_schedule(
            principal, rate, years, extra_payment=extra_monthly
        )
        new_months = len(new_schedule)
        new_interest = new_schedule[-1]['total_interest']

        return {
            'original_months': original_months,
            'original_interest': original_interest,
            'new_months': new_months,
            'new_interest': new_interest,
            'months_saved': original_months - new_months,
            'years_saved': (original_months - new_months) / 12,
            'interest_saved': original_interest - new_interest,
            'new_term_years': new_months / 12,
            'extra_payments_total': extra_monthly * new_months,
        }

    # ==================== INVESTMENT CALCULATIONS ====================

    def future_value(
        self,
        principal: float,
        rate: float,
        years: int,
        compound_frequency: int = 12
    ) -> Dict[str, Any]:
        """
        Calculate future value with compound interest.

        Args:
            principal: Initial investment
            rate: Annual interest rate (%)
            years: Investment period
            compound_frequency: Compounding periods per year

        Returns:
            Future value details
        """
        periodic_rate = rate / 100 / compound_frequency
        periods = years * compound_frequency

        fv = principal * (1 + periodic_rate) ** periods
        growth = fv - principal

        return {
            'principal': principal,
            'rate': rate,
            'years': years,
            'future_value': fv,
            'total_growth': growth,
            'growth_percentage': (growth / principal) * 100,
            'effective_annual_rate': (1 + periodic_rate) ** compound_frequency - 1,
        }

    def investment_growth(
        self,
        principal: float,
        rate: float,
        years: int,
        monthly_contribution: float = 0,
        annual_contribution: float = 0,
        compound_frequency: int = 12
    ) -> Dict[str, Any]:
        """
        Calculate investment growth with recurring contributions.

        Args:
            principal: Initial investment
            rate: Annual return rate (%)
            years: Investment period
            monthly_contribution: Monthly addition
            annual_contribution: Annual addition
            compound_frequency: Compounding frequency

        Returns:
            Growth projection
        """
        monthly_rate = rate / 100 / 12
        total_months = years * 12

        balance = principal
        yearly_data = []
        total_contributions = principal

        for month in range(1, total_months + 1):
            # Monthly growth
            balance *= (1 + monthly_rate)

            # Monthly contribution
            if monthly_contribution > 0:
                balance += monthly_contribution
                total_contributions += monthly_contribution

            # Annual contribution (at year end)
            if annual_contribution > 0 and month % 12 == 0:
                balance += annual_contribution
                total_contributions += annual_contribution

            # Record yearly data
            if month % 12 == 0:
                year = month // 12
                yearly_data.append({
                    'year': year,
                    'balance': balance,
                    'contributions': total_contributions,
                    'growth': balance - total_contributions,
                })

        final_growth = balance - total_contributions

        return {
            'principal': principal,
            'rate': rate,
            'years': years,
            'monthly_contribution': monthly_contribution,
            'final_value': balance,
            'total_contributions': total_contributions,
            'total_growth': final_growth,
            'growth_percentage': (final_growth / total_contributions) * 100,
            'yearly_data': yearly_data,
        }

    def compare_investments(
        self,
        scenarios: List[Dict[str, Any]],
        years: int
    ) -> List[Dict[str, Any]]:
        """
        Compare multiple investment scenarios.

        Args:
            scenarios: List of scenario dicts with 'name', 'rate', 'principal', 'monthly'
            years: Investment period

        Returns:
            Comparison results
        """
        results = []

        for scenario in scenarios:
            growth = self.investment_growth(
                principal=scenario.get('principal', 0),
                rate=scenario.get('rate', 0),
                years=years,
                monthly_contribution=scenario.get('monthly', 0),
            )

            results.append({
                'name': scenario.get('name', 'Unnamed'),
                'rate': scenario.get('rate', 0),
                'final_value': growth['final_value'],
                'total_contributions': growth['total_contributions'],
                'total_growth': growth['total_growth'],
                'growth_percentage': growth['growth_percentage'],
            })

        return sorted(results, key=lambda x: x['final_value'], reverse=True)

    # ==================== NPV/IRR CALCULATIONS ====================

    def npv(
        self,
        cash_flows: List[float],
        discount_rate: float
    ) -> float:
        """
        Calculate Net Present Value.

        Args:
            cash_flows: List of cash flows (first is typically negative investment)
            discount_rate: Annual discount rate (%)

        Returns:
            NPV value
        """
        rate = discount_rate / 100
        npv_value = sum(cf / (1 + rate) ** i for i, cf in enumerate(cash_flows))
        return npv_value

    def irr(self, cash_flows: List[float]) -> float:
        """
        Calculate Internal Rate of Return.

        Args:
            cash_flows: List of cash flows

        Returns:
            IRR as percentage
        """
        if HAS_NPF:
            irr_value = npf.irr(cash_flows)
            return irr_value * 100
        else:
            # Manual IRR calculation
            def npv_func(rate):
                return sum(cf / (1 + rate) ** i for i, cf in enumerate(cash_flows))

            try:
                result = optimize.brentq(npv_func, -0.99, 10.0)
                return result * 100
            except ValueError:
                raise FinanceError("IRR could not be calculated")

    def payback_period(
        self,
        cash_flows: List[float],
        discount_rate: float = 0
    ) -> Dict[str, float]:
        """
        Calculate simple and discounted payback period.

        Args:
            cash_flows: List of cash flows
            discount_rate: Optional discount rate for discounted payback

        Returns:
            Simple and discounted payback periods
        """
        # Simple payback
        cumulative = 0
        simple_payback = None
        for i, cf in enumerate(cash_flows):
            cumulative += cf
            if cumulative >= 0 and simple_payback is None:
                # Interpolate
                simple_payback = i - 1 + (cumulative - cf) / (-cf) if i > 0 else i
                simple_payback = max(0, simple_payback)

        # Discounted payback
        rate = discount_rate / 100
        cumulative = 0
        discounted_payback = None
        for i, cf in enumerate(cash_flows):
            discounted_cf = cf / (1 + rate) ** i
            cumulative += discounted_cf
            if cumulative >= 0 and discounted_payback is None:
                discounted_payback = i - 1 + (cumulative - discounted_cf) / (-discounted_cf) if i > 0 else i
                discounted_payback = max(0, discounted_payback)

        return {
            'simple': simple_payback or len(cash_flows),
            'discounted': discounted_payback or len(cash_flows),
        }

    def compare_projects(
        self,
        projects: List[Dict[str, Any]],
        discount_rate: float
    ) -> List[Dict[str, Any]]:
        """
        Compare multiple projects using NPV, IRR, and payback.

        Args:
            projects: List with 'name' and 'flows' keys
            discount_rate: Discount rate for NPV

        Returns:
            Comparison results
        """
        results = []

        for project in projects:
            flows = project['flows']
            payback = self.payback_period(flows, discount_rate)

            results.append({
                'name': project['name'],
                'initial_investment': abs(flows[0]),
                'npv': self.npv(flows, discount_rate),
                'irr': self.irr(flows),
                'simple_payback': payback['simple'],
                'discounted_payback': payback['discounted'],
            })

        return sorted(results, key=lambda x: x['npv'], reverse=True)

    # ==================== RETIREMENT CALCULATIONS ====================

    def retirement_projection(
        self,
        current_age: int,
        retirement_age: int,
        current_savings: float,
        monthly_contribution: float,
        expected_return: float,
        inflation: float = 2.5
    ) -> Dict[str, Any]:
        """
        Project retirement savings.

        Args:
            current_age: Current age
            retirement_age: Target retirement age
            current_savings: Current retirement savings
            monthly_contribution: Monthly contribution
            expected_return: Expected annual return (%)
            inflation: Expected inflation rate (%)

        Returns:
            Retirement projection
        """
        years_to_retire = retirement_age - current_age
        if years_to_retire <= 0:
            raise FinanceError("Retirement age must be greater than current age")

        # Calculate nominal value
        growth = self.investment_growth(
            principal=current_savings,
            rate=expected_return,
            years=years_to_retire,
            monthly_contribution=monthly_contribution
        )

        # Calculate real value (adjusted for inflation)
        inflation_factor = (1 + inflation / 100) ** years_to_retire
        real_value = growth['final_value'] / inflation_factor

        return {
            'current_age': current_age,
            'retirement_age': retirement_age,
            'years_to_retirement': years_to_retire,
            'nominal_value': growth['final_value'],
            'real_value': real_value,
            'total_contributions': growth['total_contributions'],
            'total_growth': growth['total_growth'],
            'yearly_data': growth['yearly_data'],
        }

    def retirement_withdrawal(
        self,
        savings: float,
        annual_spending: float,
        expected_return: float,
        inflation: float,
        years: int
    ) -> Dict[str, Any]:
        """
        Analyze retirement withdrawal strategy.

        Args:
            savings: Initial retirement savings
            annual_spending: Annual spending requirement
            expected_return: Expected return
            inflation: Expected inflation
            years: Retirement duration

        Returns:
            Withdrawal analysis
        """
        balance = savings
        yearly_data = []
        spending = annual_spending

        for year in range(1, years + 1):
            # Withdraw at start of year
            balance -= spending

            # Growth during year
            if balance > 0:
                balance *= (1 + expected_return / 100)
            else:
                balance = 0

            # Increase spending for inflation
            spending *= (1 + inflation / 100)

            yearly_data.append({
                'year': year,
                'spending': spending,
                'balance': max(0, balance),
            })

            if balance <= 0:
                break

        success = balance > 0

        return {
            'initial_savings': savings,
            'annual_spending': annual_spending,
            'success': success,
            'success_rate': 100 if success else (len([y for y in yearly_data if y['balance'] > 0]) / years) * 100,
            'ending_balance': max(0, balance),
            'years_lasted': len([y for y in yearly_data if y['balance'] > 0]),
            'yearly_data': yearly_data,
        }

    def fire_calculator(
        self,
        annual_expenses: float,
        current_savings: float,
        annual_savings: float,
        expected_return: float,
        safe_withdrawal_rate: float = 4
    ) -> Dict[str, Any]:
        """
        Calculate Financial Independence / Retire Early (FIRE) numbers.

        Args:
            annual_expenses: Annual expenses in retirement
            current_savings: Current savings
            annual_savings: Annual savings amount
            expected_return: Expected investment return
            safe_withdrawal_rate: Safe withdrawal rate (%)

        Returns:
            FIRE analysis
        """
        # FIRE number = annual expenses / SWR
        fire_number = annual_expenses / (safe_withdrawal_rate / 100)

        # Years to FIRE
        if current_savings >= fire_number:
            years_to_fire = 0
        else:
            monthly_return = expected_return / 100 / 12
            monthly_savings = annual_savings / 12
            balance = current_savings
            months = 0

            while balance < fire_number and months < 600:  # Max 50 years
                balance = balance * (1 + monthly_return) + monthly_savings
                months += 1

            years_to_fire = months / 12

        return {
            'annual_expenses': annual_expenses,
            'safe_withdrawal_rate': safe_withdrawal_rate,
            'fire_number': fire_number,
            'current_savings': current_savings,
            'gap': max(0, fire_number - current_savings),
            'years_to_fire': years_to_fire,
            'monthly_savings_needed': (fire_number - current_savings) / (years_to_fire * 12) if years_to_fire > 0 else 0,
        }

    # ==================== MONTE CARLO SIMULATION ====================

    def monte_carlo_investment(
        self,
        principal: float,
        monthly_contribution: float,
        years: int,
        mean_return: float,
        std_dev: float,
        simulations: int = 1000
    ) -> Dict[str, Any]:
        """
        Monte Carlo simulation for investment growth.

        Args:
            principal: Initial investment
            monthly_contribution: Monthly contribution
            years: Investment period
            mean_return: Expected annual return (%)
            std_dev: Annual volatility/standard deviation (%)
            simulations: Number of simulations

        Returns:
            Simulation results
        """
        np.random.seed(42)  # For reproducibility
        months = years * 12
        results = []

        for _ in range(simulations):
            balance = principal
            for month in range(months):
                # Random monthly return
                monthly_return = np.random.normal(
                    mean_return / 100 / 12,
                    std_dev / 100 / np.sqrt(12)
                )
                balance = balance * (1 + monthly_return) + monthly_contribution

            results.append(balance)

        results = np.array(results)

        return {
            'principal': principal,
            'monthly_contribution': monthly_contribution,
            'years': years,
            'mean_return': mean_return,
            'volatility': std_dev,
            'simulations': simulations,
            'mean': float(np.mean(results)),
            'median': float(np.median(results)),
            'std_dev': float(np.std(results)),
            'min': float(np.min(results)),
            'max': float(np.max(results)),
            'p10': float(np.percentile(results, 10)),
            'p25': float(np.percentile(results, 25)),
            'p75': float(np.percentile(results, 75)),
            'p90': float(np.percentile(results, 90)),
            'prob_above_1m': float((results > 1000000).sum() / simulations * 100),
            'prob_double': float((results > principal * 2).sum() / simulations * 100),
            'results': results.tolist(),
        }

    def monte_carlo_retirement(
        self,
        savings: float,
        annual_withdrawal: float,
        years: int,
        mean_return: float,
        std_dev: float,
        inflation_mean: float = 2.5,
        inflation_std: float = 1.0,
        simulations: int = 1000
    ) -> Dict[str, Any]:
        """
        Monte Carlo simulation for retirement withdrawals.

        Args:
            savings: Initial retirement savings
            annual_withdrawal: Initial annual withdrawal
            years: Retirement duration
            mean_return: Expected return
            std_dev: Return volatility
            inflation_mean: Expected inflation
            inflation_std: Inflation volatility
            simulations: Number of simulations

        Returns:
            Simulation results
        """
        np.random.seed(42)
        results = []
        success_count = 0

        for _ in range(simulations):
            balance = savings
            withdrawal = annual_withdrawal

            for year in range(years):
                # Random return
                annual_return = np.random.normal(mean_return / 100, std_dev / 100)

                # Random inflation
                inflation = np.random.normal(inflation_mean / 100, inflation_std / 100)

                # Withdraw and grow
                balance -= withdrawal
                if balance > 0:
                    balance *= (1 + annual_return)
                else:
                    balance = 0
                    break

                # Adjust withdrawal for inflation
                withdrawal *= (1 + inflation)

            results.append(balance)
            if balance > 0:
                success_count += 1

        results = np.array(results)

        return {
            'initial_savings': savings,
            'annual_withdrawal': annual_withdrawal,
            'years': years,
            'simulations': simulations,
            'success_rate': success_count / simulations * 100,
            'mean_ending': float(np.mean(results)),
            'median_ending': float(np.median(results)),
            'p10_ending': float(np.percentile(results, 10)),
            'p90_ending': float(np.percentile(results, 90)),
            'worst_case': float(np.min(results)),
            'best_case': float(np.max(results)),
        }

    # ==================== MORTGAGE CALCULATIONS ====================

    def mortgage_affordability(
        self,
        annual_income: float,
        monthly_debt: float,
        down_payment: float,
        rate: float,
        term_years: int = 30,
        dti_limit: float = 43,
        property_tax_rate: float = 1.2,
        insurance_rate: float = 0.5
    ) -> Dict[str, Any]:
        """
        Calculate maximum affordable home price.

        Args:
            annual_income: Gross annual income
            monthly_debt: Existing monthly debt payments
            down_payment: Available down payment
            rate: Mortgage interest rate
            term_years: Loan term
            dti_limit: Maximum debt-to-income ratio
            property_tax_rate: Annual property tax as % of home value
            insurance_rate: Annual insurance as % of home value

        Returns:
            Affordability analysis
        """
        monthly_income = annual_income / 12
        max_total_debt = monthly_income * (dti_limit / 100)
        max_housing = max_total_debt - monthly_debt

        # Back-calculate max loan from max payment
        monthly_rate = rate / 100 / 12
        n_payments = term_years * 12

        if monthly_rate > 0:
            # Include property tax and insurance in housing payment
            # Assume max ~80% of housing payment goes to P&I
            max_pi = max_housing * 0.75

            max_loan = max_pi * (
                (1 - (1 + monthly_rate) ** (-n_payments)) / monthly_rate
            )
        else:
            max_loan = max_housing * 0.75 * n_payments

        max_price = max_loan + down_payment

        # Actual payment breakdown
        loan = self.loan_payment(max_loan, rate, term_years)
        monthly_tax = (max_price * property_tax_rate / 100) / 12
        monthly_insurance = (max_price * insurance_rate / 100) / 12
        total_payment = loan['monthly_payment'] + monthly_tax + monthly_insurance

        return {
            'max_home_price': max_price,
            'max_loan': max_loan,
            'down_payment': down_payment,
            'down_payment_percent': (down_payment / max_price) * 100,
            'monthly_payment': loan['monthly_payment'],
            'monthly_tax': monthly_tax,
            'monthly_insurance': monthly_insurance,
            'total_monthly': total_payment,
            'dti_ratio': (total_payment + monthly_debt) / monthly_income * 100,
        }

    def refinance_analysis(
        self,
        current_balance: float,
        current_rate: float,
        current_payment: float,
        remaining_months: int,
        new_rate: float,
        new_term_years: int,
        closing_costs: float
    ) -> Dict[str, Any]:
        """
        Analyze refinancing decision.

        Args:
            current_balance: Remaining loan balance
            current_rate: Current interest rate
            current_payment: Current monthly payment
            remaining_months: Remaining months on current loan
            new_rate: New interest rate
            new_term_years: New loan term
            closing_costs: Refinancing costs

        Returns:
            Refinance analysis
        """
        # Current loan remaining cost
        current_total = current_payment * remaining_months

        # New loan
        new_loan = self.loan_payment(current_balance + closing_costs, new_rate, new_term_years)
        new_total = new_loan['monthly_payment'] * new_term_years * 12

        # Monthly savings
        monthly_savings = current_payment - new_loan['monthly_payment']

        # Break-even
        if monthly_savings > 0:
            break_even = closing_costs / monthly_savings
        else:
            break_even = float('inf')

        # Lifetime savings
        lifetime_savings = current_total - new_total - closing_costs

        return {
            'current_balance': current_balance,
            'current_rate': current_rate,
            'current_payment': current_payment,
            'new_rate': new_rate,
            'new_payment': new_loan['monthly_payment'],
            'monthly_savings': monthly_savings,
            'closing_costs': closing_costs,
            'break_even_months': break_even,
            'break_even_years': break_even / 12,
            'current_remaining_cost': current_total,
            'new_total_cost': new_total,
            'lifetime_savings': lifetime_savings,
            'recommend_refinance': lifetime_savings > 0 and break_even < remaining_months,
        }

    # ==================== SAVINGS GOAL ====================

    def savings_goal(
        self,
        target: float,
        current: float,
        rate: float,
        monthly_contribution: float
    ) -> Dict[str, Any]:
        """
        Calculate time to reach savings goal.

        Args:
            target: Target savings amount
            current: Current savings
            rate: Annual return rate
            monthly_contribution: Monthly contribution

        Returns:
            Time to goal
        """
        if current >= target:
            return {
                'target': target,
                'current': current,
                'months': 0,
                'years': 0,
                'already_reached': True,
            }

        monthly_rate = rate / 100 / 12
        balance = current
        months = 0

        while balance < target and months < 1200:  # Max 100 years
            balance = balance * (1 + monthly_rate) + monthly_contribution
            months += 1

        return {
            'target': target,
            'current': current,
            'gap': target - current,
            'rate': rate,
            'monthly_contribution': monthly_contribution,
            'months': months,
            'years': months / 12,
            'final_balance': balance,
        }

    def required_savings(
        self,
        target: float,
        current: float,
        rate: float,
        years: int
    ) -> Dict[str, Any]:
        """
        Calculate required monthly savings to reach goal.

        Args:
            target: Target amount
            current: Current savings
            rate: Annual return rate
            years: Time period

        Returns:
            Required monthly savings
        """
        months = years * 12
        monthly_rate = rate / 100 / 12

        # Future value of current savings
        fv_current = current * (1 + monthly_rate) ** months

        # Gap to fill with monthly payments
        gap = target - fv_current

        if gap <= 0:
            return {
                'target': target,
                'current': current,
                'monthly_needed': 0,
                'already_sufficient': True,
            }

        # PMT formula
        if monthly_rate > 0:
            monthly_needed = gap * monthly_rate / ((1 + monthly_rate) ** months - 1)
        else:
            monthly_needed = gap / months

        return {
            'target': target,
            'current': current,
            'years': years,
            'rate': rate,
            'monthly_needed': monthly_needed,
            'annual_needed': monthly_needed * 12,
            'total_contributions': monthly_needed * months,
            'fv_current': fv_current,
        }

    # ==================== EXPORT FUNCTIONS ====================

    def export_amortization(
        self,
        schedule: List[Dict],
        filepath: Union[str, Path]
    ) -> str:
        """Export amortization schedule to CSV."""
        filepath = Path(filepath)
        df = pd.DataFrame(schedule)
        df.to_csv(filepath, index=False)
        return str(filepath)

    def export_json(
        self,
        data: Dict[str, Any],
        filepath: Union[str, Path]
    ) -> str:
        """Export results to JSON."""
        filepath = Path(filepath)

        # Convert numpy types
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            return obj

        clean_data = json.loads(json.dumps(data, default=convert))
        filepath.write_text(json.dumps(clean_data, indent=2))
        return str(filepath)

    # ==================== VISUALIZATION ====================

    def plot_amortization(
        self,
        schedule: List[Dict],
        filepath: Union[str, Path]
    ) -> str:
        """Plot amortization schedule."""
        df = pd.DataFrame(schedule)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Balance over time
        ax1.plot(df['month'], df['balance'], 'b-', linewidth=2)
        ax1.fill_between(df['month'], df['balance'], alpha=0.3)
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Remaining Balance')
        ax1.set_title('Loan Balance Over Time')
        ax1.grid(True, alpha=0.3)

        # Principal vs Interest
        ax2.stackplot(
            df['month'],
            df['principal'],
            df['interest'],
            labels=['Principal', 'Interest'],
            alpha=0.7
        )
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Payment Amount')
        ax2.set_title('Principal vs Interest Over Time')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        return str(filepath)

    def plot_investment_growth(
        self,
        growth_data: Dict[str, Any],
        filepath: Union[str, Path]
    ) -> str:
        """Plot investment growth over time."""
        yearly = growth_data['yearly_data']
        df = pd.DataFrame(yearly)

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.bar(df['year'], df['contributions'], label='Contributions', alpha=0.7)
        ax.bar(df['year'], df['growth'], bottom=df['contributions'], label='Growth', alpha=0.7)

        ax.set_xlabel('Year')
        ax.set_ylabel('Value')
        ax.set_title('Investment Growth Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        return str(filepath)

    def plot_monte_carlo(
        self,
        simulation: Dict[str, Any],
        filepath: Union[str, Path]
    ) -> str:
        """Plot Monte Carlo simulation results."""
        results = simulation['results']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Histogram
        ax1.hist(results, bins=50, edgecolor='white', alpha=0.7)
        ax1.axvline(simulation['median'], color='red', linestyle='--', label=f"Median: ${simulation['median']:,.0f}")
        ax1.axvline(simulation['p10'], color='orange', linestyle=':', label=f"10th %ile: ${simulation['p10']:,.0f}")
        ax1.axvline(simulation['p90'], color='green', linestyle=':', label=f"90th %ile: ${simulation['p90']:,.0f}")
        ax1.set_xlabel('Final Value')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Outcomes')
        ax1.legend()

        # Box plot
        ax2.boxplot(results, vert=True)
        ax2.set_ylabel('Final Value')
        ax2.set_title('Outcome Range')

        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        return str(filepath)


# ==================== CLI ====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Financial Calculator Suite')
    subparsers = parser.add_subparsers(dest='command', help='Calculator type')

    # Loan subcommand
    loan_parser = subparsers.add_parser('loan', help='Loan calculator')
    loan_parser.add_argument('--principal', type=float, required=True, help='Loan amount')
    loan_parser.add_argument('--rate', type=float, required=True, help='Interest rate (%)')
    loan_parser.add_argument('--years', type=int, required=True, help='Loan term')

    # Investment subcommand
    invest_parser = subparsers.add_parser('invest', help='Investment calculator')
    invest_parser.add_argument('--principal', type=float, required=True, help='Initial investment')
    invest_parser.add_argument('--rate', type=float, required=True, help='Expected return (%)')
    invest_parser.add_argument('--years', type=int, required=True, help='Investment period')
    invest_parser.add_argument('--monthly', type=float, default=0, help='Monthly contribution')

    # NPV subcommand
    npv_parser = subparsers.add_parser('npv', help='NPV/IRR calculator')
    npv_parser.add_argument('--flows', type=str, required=True, help='Cash flows (comma-separated)')
    npv_parser.add_argument('--rate', type=float, required=True, help='Discount rate (%)')

    # Retirement subcommand
    retire_parser = subparsers.add_parser('retire', help='Retirement calculator')
    retire_parser.add_argument('--age', type=int, required=True, help='Current age')
    retire_parser.add_argument('--retire-age', type=int, required=True, help='Retirement age')
    retire_parser.add_argument('--savings', type=float, required=True, help='Current savings')
    retire_parser.add_argument('--monthly', type=float, required=True, help='Monthly contribution')
    retire_parser.add_argument('--return', type=float, default=7, help='Expected return (%)')

    # Monte Carlo subcommand
    mc_parser = subparsers.add_parser('montecarlo', help='Monte Carlo simulation')
    mc_parser.add_argument('--principal', type=float, required=True, help='Initial investment')
    mc_parser.add_argument('--years', type=int, required=True, help='Investment period')
    mc_parser.add_argument('--return', type=float, required=True, help='Expected return (%)')
    mc_parser.add_argument('--volatility', type=float, required=True, help='Volatility (%)')
    mc_parser.add_argument('--monthly', type=float, default=0, help='Monthly contribution')

    args = parser.parse_args()
    calc = FinancialCalculator()

    if args.command == 'loan':
        result = calc.loan_payment(args.principal, args.rate, args.years)
        print(f"Monthly Payment: ${result['monthly_payment']:,.2f}")
        print(f"Total Interest: ${result['total_interest']:,.2f}")
        print(f"Total Payments: ${result['total_payments']:,.2f}")

    elif args.command == 'invest':
        result = calc.investment_growth(
            args.principal, args.rate, args.years, args.monthly
        )
        print(f"Final Value: ${result['final_value']:,.2f}")
        print(f"Total Contributions: ${result['total_contributions']:,.2f}")
        print(f"Total Growth: ${result['total_growth']:,.2f}")

    elif args.command == 'npv':
        flows = [float(x) for x in args.flows.split(',')]
        npv = calc.npv(flows, args.rate)
        irr = calc.irr(flows)
        print(f"NPV: ${npv:,.2f}")
        print(f"IRR: {irr:.2f}%")

    elif args.command == 'retire':
        result = calc.retirement_projection(
            args.age, args.retire_age, args.savings,
            args.monthly, getattr(args, 'return')
        )
        print(f"Projected at retirement: ${result['nominal_value']:,.2f}")
        print(f"Real value (today's $): ${result['real_value']:,.2f}")

    elif args.command == 'montecarlo':
        result = calc.monte_carlo_investment(
            args.principal, args.monthly, args.years,
            getattr(args, 'return'), args.volatility
        )
        print(f"Median outcome: ${result['median']:,.2f}")
        print(f"10th percentile: ${result['p10']:,.2f}")
        print(f"90th percentile: ${result['p90']:,.2f}")

    else:
        parser.print_help()
