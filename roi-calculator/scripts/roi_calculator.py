#!/usr/bin/env python3
"""
ROI Calculator - Calculate ROI for marketing, investments, and business decisions.
"""

import argparse
import json
from typing import Dict, List, Optional, Union

import pandas as pd
import numpy as np


class ROICalculator:
    """Calculate ROI for various business scenarios."""

    def __init__(self):
        """Initialize the calculator."""
        pass

    # Basic ROI Calculations

    def simple_roi(self, investment: float, return_value: float) -> Dict:
        """
        Calculate simple ROI.

        Args:
            investment: Initial investment amount
            return_value: Final value including original investment

        Returns:
            Dict with ROI metrics
        """
        gain = return_value - investment
        roi_percent = (gain / investment) * 100 if investment != 0 else 0
        roi_ratio = gain / investment if investment != 0 else 0

        return {
            "investment": round(investment, 2),
            "return_value": round(return_value, 2),
            "gain": round(gain, 2),
            "roi_percent": round(roi_percent, 2),
            "roi_ratio": round(roi_ratio, 4)
        }

    def net_roi(self, investment: float, gain: float, costs: float = 0) -> Dict:
        """
        Calculate net ROI after additional costs.

        Args:
            investment: Initial investment
            gain: Gross gain
            costs: Additional costs

        Returns:
            Dict with net ROI metrics
        """
        net_gain = gain - costs
        roi_percent = (net_gain / investment) * 100 if investment != 0 else 0

        return {
            "investment": round(investment, 2),
            "gross_gain": round(gain, 2),
            "costs": round(costs, 2),
            "net_gain": round(net_gain, 2),
            "roi_percent": round(roi_percent, 2)
        }

    # Marketing ROI

    def marketing_roi(self, ad_spend: float, revenue_generated: float,
                      cost_of_goods: float = 0) -> Dict:
        """
        Calculate marketing campaign ROI.

        Args:
            ad_spend: Total advertising spend
            revenue_generated: Revenue from campaign
            cost_of_goods: Cost of goods sold

        Returns:
            Dict with marketing ROI metrics
        """
        gross_profit = revenue_generated - cost_of_goods
        net_profit = gross_profit - ad_spend
        roi_percent = (net_profit / ad_spend) * 100 if ad_spend != 0 else 0
        roas = revenue_generated / ad_spend if ad_spend != 0 else 0

        return {
            "ad_spend": round(ad_spend, 2),
            "revenue": round(revenue_generated, 2),
            "cost_of_goods": round(cost_of_goods, 2),
            "gross_profit": round(gross_profit, 2),
            "net_profit": round(net_profit, 2),
            "roi_percent": round(roi_percent, 2),
            "roas": round(roas, 2)
        }

    def roas(self, ad_spend: float, revenue: float) -> Dict:
        """
        Calculate Return on Ad Spend.

        Args:
            ad_spend: Advertising spend
            revenue: Revenue generated

        Returns:
            Dict with ROAS metrics
        """
        roas_value = revenue / ad_spend if ad_spend != 0 else 0

        return {
            "ad_spend": round(ad_spend, 2),
            "revenue": round(revenue, 2),
            "roas": round(roas_value, 2),
            "roas_percent": round(roas_value * 100, 2)
        }

    def campaign_roi(self, campaigns: List[Dict]) -> pd.DataFrame:
        """
        Calculate ROI for multiple campaigns.

        Args:
            campaigns: List of campaign dicts with name, spend, revenue, cogs

        Returns:
            DataFrame with ROI for each campaign
        """
        results = []

        for campaign in campaigns:
            name = campaign.get("name", "Campaign")
            spend = campaign.get("spend", 0)
            revenue = campaign.get("revenue", 0)
            cogs = campaign.get("cogs", 0)

            roi = self.marketing_roi(spend, revenue, cogs)

            results.append({
                "name": name,
                "ad_spend": spend,
                "revenue": revenue,
                "cogs": cogs,
                "gross_profit": roi["gross_profit"],
                "net_profit": roi["net_profit"],
                "roi_percent": roi["roi_percent"],
                "roas": roi["roas"]
            })

        return pd.DataFrame(results)

    # Investment ROI

    def investment_roi(self, initial: float, final: float, years: float) -> Dict:
        """
        Calculate investment ROI with CAGR.

        Args:
            initial: Initial investment
            final: Final value
            years: Time period in years

        Returns:
            Dict with investment ROI metrics
        """
        total_gain = final - initial
        total_roi_percent = (total_gain / initial) * 100 if initial != 0 else 0

        # CAGR calculation
        cagr = self.cagr(initial, final, years)

        return {
            "initial": round(initial, 2),
            "final": round(final, 2),
            "total_gain": round(total_gain, 2),
            "total_roi_percent": round(total_roi_percent, 2),
            "cagr_percent": round(cagr, 2),
            "years": years,
            "annualized_roi_percent": round(total_roi_percent / years, 2) if years > 0 else 0
        }

    def cagr(self, initial: float, final: float, years: float) -> float:
        """
        Calculate Compound Annual Growth Rate.

        Args:
            initial: Starting value
            final: Ending value
            years: Number of years

        Returns:
            CAGR as percentage
        """
        if initial <= 0 or years <= 0:
            return 0

        return ((final / initial) ** (1 / years) - 1) * 100

    def total_return(self, initial: float, final: float, dividends: float = 0) -> Dict:
        """
        Calculate total return including dividends.

        Args:
            initial: Initial investment
            final: Final value
            dividends: Total dividends received

        Returns:
            Dict with total return metrics
        """
        capital_gain = final - initial
        total_return_value = capital_gain + dividends
        total_return_percent = (total_return_value / initial) * 100 if initial != 0 else 0

        return {
            "initial": round(initial, 2),
            "final": round(final, 2),
            "capital_gain": round(capital_gain, 2),
            "capital_gain_percent": round((capital_gain / initial) * 100, 2) if initial != 0 else 0,
            "dividends": round(dividends, 2),
            "dividend_yield": round((dividends / initial) * 100, 2) if initial != 0 else 0,
            "total_return": round(total_return_value, 2),
            "total_return_percent": round(total_return_percent, 2)
        }

    # Break-Even Analysis

    def break_even(self, fixed_costs: float, price_per_unit: float,
                   variable_cost_per_unit: float) -> Dict:
        """
        Calculate break-even point.

        Args:
            fixed_costs: Total fixed costs
            price_per_unit: Selling price per unit
            variable_cost_per_unit: Variable cost per unit

        Returns:
            Dict with break-even metrics
        """
        contribution_margin = price_per_unit - variable_cost_per_unit
        contribution_margin_ratio = contribution_margin / price_per_unit if price_per_unit != 0 else 0

        if contribution_margin <= 0:
            return {
                "error": "Contribution margin must be positive (price > variable cost)",
                "fixed_costs": fixed_costs,
                "price_per_unit": price_per_unit,
                "variable_cost_per_unit": variable_cost_per_unit,
                "contribution_margin": contribution_margin
            }

        break_even_units = fixed_costs / contribution_margin
        break_even_revenue = break_even_units * price_per_unit

        return {
            "fixed_costs": round(fixed_costs, 2),
            "price_per_unit": round(price_per_unit, 2),
            "variable_cost_per_unit": round(variable_cost_per_unit, 2),
            "contribution_margin": round(contribution_margin, 2),
            "contribution_margin_ratio": round(contribution_margin_ratio, 4),
            "break_even_units": round(break_even_units, 0),
            "break_even_revenue": round(break_even_revenue, 2)
        }

    def break_even_revenue(self, fixed_costs: float, contribution_margin_ratio: float) -> Dict:
        """
        Calculate break-even revenue.

        Args:
            fixed_costs: Total fixed costs
            contribution_margin_ratio: Contribution margin as ratio (0-1)

        Returns:
            Dict with break-even revenue
        """
        if contribution_margin_ratio <= 0:
            return {"error": "Contribution margin ratio must be positive"}

        break_even_rev = fixed_costs / contribution_margin_ratio

        return {
            "fixed_costs": round(fixed_costs, 2),
            "contribution_margin_ratio": round(contribution_margin_ratio, 4),
            "break_even_revenue": round(break_even_rev, 2)
        }

    # Payback Period

    def payback_period(self, investment: float, annual_cash_flow: float) -> Dict:
        """
        Calculate simple payback period.

        Args:
            investment: Initial investment
            annual_cash_flow: Expected annual cash flow

        Returns:
            Dict with payback metrics
        """
        if annual_cash_flow <= 0:
            return {"error": "Annual cash flow must be positive"}

        payback_years = investment / annual_cash_flow

        return {
            "investment": round(investment, 2),
            "annual_cash_flow": round(annual_cash_flow, 2),
            "payback_years": round(payback_years, 2),
            "payback_months": round(payback_years * 12, 0)
        }

    def payback_period_uneven(self, investment: float, cash_flows: List[float]) -> Dict:
        """
        Calculate payback period with uneven cash flows.

        Args:
            investment: Initial investment
            cash_flows: List of annual cash flows

        Returns:
            Dict with payback metrics
        """
        cumulative = []
        total = 0
        payback_year = None

        for i, flow in enumerate(cash_flows):
            total += flow
            cumulative.append(total)

            if total >= investment and payback_year is None:
                # Interpolate exact payback point
                prev_cumulative = cumulative[i-1] if i > 0 else 0
                remaining = investment - prev_cumulative
                fraction = remaining / flow if flow != 0 else 0
                payback_year = i + fraction

        return {
            "investment": round(investment, 2),
            "cash_flows": [round(cf, 2) for cf in cash_flows],
            "cumulative_flows": [round(cf, 2) for cf in cumulative],
            "payback_years": round(payback_year, 2) if payback_year is not None else None,
            "fully_recovered": total >= investment,
            "total_return": round(total, 2)
        }

    # Comparative Analysis

    def compare_investments(self, investments: List[Dict]) -> pd.DataFrame:
        """
        Compare multiple investment options.

        Args:
            investments: List of dicts with name, investment, return/annual_return, years

        Returns:
            DataFrame with comparison metrics
        """
        results = []

        for inv in investments:
            name = inv.get("name", "Investment")
            initial = inv.get("investment", 0)
            years = inv.get("years", 1)

            # Handle different return specifications
            if "return" in inv:
                final = inv["return"]
                annual_return = (final - initial) / years if years > 0 else 0
            elif "annual_return" in inv:
                annual_return = inv["annual_return"]
                final = initial + (annual_return * years)
            else:
                final = initial
                annual_return = 0

            # Calculate metrics
            total_roi = ((final - initial) / initial) * 100 if initial != 0 else 0
            cagr_val = self.cagr(initial, final, years)
            payback = initial / annual_return if annual_return > 0 else float('inf')

            results.append({
                "name": name,
                "investment": initial,
                "return": final,
                "gain": final - initial,
                "years": years,
                "roi_percent": round(total_roi, 2),
                "cagr_percent": round(cagr_val, 2),
                "payback_years": round(payback, 2) if payback != float('inf') else None
            })

        return pd.DataFrame(results)

    def rank_by_roi(self, investments: List[Dict]) -> List[Dict]:
        """Rank investments by ROI."""
        comparison = self.compare_investments(investments)
        ranked = comparison.sort_values("roi_percent", ascending=False)
        return ranked.to_dict('records')

    # Sensitivity Analysis

    def sensitivity_analysis(self, base_case: Dict, variables: Dict) -> pd.DataFrame:
        """
        Perform sensitivity analysis on key variables.

        Args:
            base_case: Dict with base case values
            variables: Dict of variable_name -> list of values to test

        Returns:
            DataFrame with results for each combination
        """
        results = []

        # Get variable names and their test values
        var_names = list(variables.keys())

        if len(var_names) == 1:
            # Single variable analysis
            var_name = var_names[0]
            for value in variables[var_name]:
                case = base_case.copy()
                case[var_name] = value

                # Calculate profit
                profit = self._calculate_profit(case)
                results.append({
                    var_name: value,
                    "profit": profit
                })

        elif len(var_names) == 2:
            # Two variable analysis (matrix)
            for val1 in variables[var_names[0]]:
                for val2 in variables[var_names[1]]:
                    case = base_case.copy()
                    case[var_names[0]] = val1
                    case[var_names[1]] = val2

                    profit = self._calculate_profit(case)
                    results.append({
                        var_names[0]: val1,
                        var_names[1]: val2,
                        "profit": profit
                    })

        return pd.DataFrame(results)

    def _calculate_profit(self, case: Dict) -> float:
        """Calculate profit from case parameters."""
        fixed_costs = case.get("fixed_costs", 0)
        price = case.get("price_per_unit", case.get("price", 0))
        variable_cost = case.get("variable_cost_per_unit", case.get("variable_cost", 0))
        units = case.get("units_sold", case.get("units", 0))

        revenue = price * units
        variable_costs = variable_cost * units
        profit = revenue - variable_costs - fixed_costs

        return round(profit, 2)

    def scenario_analysis(self, scenarios: List[Dict], base_params: Dict = None) -> pd.DataFrame:
        """
        Analyze different scenarios.

        Args:
            scenarios: List of scenario dicts with name and parameters
            base_params: Base parameters to merge with scenarios

        Returns:
            DataFrame with scenario results
        """
        results = []

        for scenario in scenarios:
            name = scenario.get("name", "Scenario")

            # Merge with base params
            params = (base_params or {}).copy()
            params.update(scenario)

            profit = self._calculate_profit(params)

            results.append({
                "scenario": name,
                "profit": profit,
                **{k: v for k, v in params.items() if k != "name"}
            })

        return pd.DataFrame(results)

    def generate_report(self, analysis: Dict, output: str = None) -> str:
        """Generate text report from analysis."""
        lines = ["=" * 50, "ROI ANALYSIS REPORT", "=" * 50, ""]

        for key, value in analysis.items():
            if isinstance(value, (int, float)):
                if "percent" in key.lower() or "roi" in key.lower() or "cagr" in key.lower():
                    lines.append(f"{key}: {value:.2f}%")
                elif value >= 1000:
                    lines.append(f"{key}: ${value:,.2f}")
                else:
                    lines.append(f"{key}: {value:.2f}")
            else:
                lines.append(f"{key}: {value}")

        report = "\n".join(lines)

        if output:
            with open(output, 'w') as f:
                f.write(report)

        return report


def main():
    parser = argparse.ArgumentParser(description="ROI Calculator")

    # Mode selection
    parser.add_argument("--marketing", action="store_true", help="Marketing ROI mode")
    parser.add_argument("--breakeven", action="store_true", help="Break-even analysis mode")
    parser.add_argument("--payback", action="store_true", help="Payback period mode")
    parser.add_argument("--compare", help="Compare investments from CSV file")

    # Basic ROI
    parser.add_argument("--investment", "-i", type=float, help="Initial investment")
    parser.add_argument("--return", "-r", type=float, dest="return_value", help="Return value")
    parser.add_argument("--years", "-y", type=float, default=1, help="Time period in years")

    # Marketing ROI
    parser.add_argument("--spend", type=float, help="Ad spend")
    parser.add_argument("--revenue", type=float, help="Revenue generated")
    parser.add_argument("--cogs", type=float, default=0, help="Cost of goods sold")

    # Break-even
    parser.add_argument("--fixed", type=float, help="Fixed costs")
    parser.add_argument("--price", type=float, help="Price per unit")
    parser.add_argument("--variable", type=float, help="Variable cost per unit")

    # Payback
    parser.add_argument("--annual-return", type=float, help="Annual cash flow/return")

    # Output
    parser.add_argument("--output", "-o", help="Output file for report")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    calc = ROICalculator()
    result = None

    if args.marketing:
        # Marketing ROI
        if args.spend and args.revenue:
            result = calc.marketing_roi(args.spend, args.revenue, args.cogs)
            print("\n=== Marketing ROI ===")
            print(f"Ad Spend: ${result['ad_spend']:,.2f}")
            print(f"Revenue: ${result['revenue']:,.2f}")
            print(f"Gross Profit: ${result['gross_profit']:,.2f}")
            print(f"Net Profit: ${result['net_profit']:,.2f}")
            print(f"ROI: {result['roi_percent']:.2f}%")
            print(f"ROAS: {result['roas']:.2f}x")
        else:
            print("Marketing mode requires --spend and --revenue")

    elif args.breakeven:
        # Break-even analysis
        if args.fixed and args.price and args.variable:
            result = calc.break_even(args.fixed, args.price, args.variable)
            if "error" not in result:
                print("\n=== Break-Even Analysis ===")
                print(f"Fixed Costs: ${result['fixed_costs']:,.2f}")
                print(f"Price per Unit: ${result['price_per_unit']:,.2f}")
                print(f"Variable Cost per Unit: ${result['variable_cost_per_unit']:,.2f}")
                print(f"Contribution Margin: ${result['contribution_margin']:,.2f}")
                print(f"Break-Even Units: {result['break_even_units']:,.0f}")
                print(f"Break-Even Revenue: ${result['break_even_revenue']:,.2f}")
            else:
                print(f"Error: {result['error']}")
        else:
            print("Break-even mode requires --fixed, --price, and --variable")

    elif args.payback:
        # Payback period
        if args.investment and args.annual_return:
            result = calc.payback_period(args.investment, args.annual_return)
            print("\n=== Payback Period ===")
            print(f"Investment: ${result['investment']:,.2f}")
            print(f"Annual Return: ${result['annual_cash_flow']:,.2f}")
            print(f"Payback Period: {result['payback_years']:.2f} years ({result['payback_months']:.0f} months)")
        else:
            print("Payback mode requires --investment and --annual-return")

    elif args.compare:
        # Compare investments from CSV
        df = pd.read_csv(args.compare)
        investments = df.to_dict('records')
        comparison = calc.compare_investments(investments)
        print("\n=== Investment Comparison ===")
        print(comparison.to_string(index=False))

    elif args.investment and args.return_value:
        # Simple or investment ROI
        if args.years > 1:
            result = calc.investment_roi(args.investment, args.return_value, args.years)
            print("\n=== Investment ROI ===")
            print(f"Initial: ${result['initial']:,.2f}")
            print(f"Final: ${result['final']:,.2f}")
            print(f"Total Gain: ${result['total_gain']:,.2f}")
            print(f"Total ROI: {result['total_roi_percent']:.2f}%")
            print(f"CAGR: {result['cagr_percent']:.2f}%")
            print(f"Period: {result['years']} years")
        else:
            result = calc.simple_roi(args.investment, args.return_value)
            print("\n=== Simple ROI ===")
            print(f"Investment: ${result['investment']:,.2f}")
            print(f"Return: ${result['return_value']:,.2f}")
            print(f"Gain: ${result['gain']:,.2f}")
            print(f"ROI: {result['roi_percent']:.2f}%")

    else:
        parser.print_help()
        return

    # JSON output
    if args.json and result:
        print("\n" + json.dumps(result, indent=2))

    # Save report
    if args.output and result:
        calc.generate_report(result, args.output)
        print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
