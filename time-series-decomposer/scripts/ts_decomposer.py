#!/usr/bin/env python3
"""
Time Series Decomposer - Extract trend, seasonal, and residual components.
"""

import argparse
import json
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf


class TimeSeriesDecomposer:
    """Decompose time series into trend, seasonal, and residual components."""

    def __init__(self):
        """Initialize the decomposer."""
        self.data: Optional[pd.Series] = None
        self.decomposition = None
        self.period: Optional[int] = None
        self.model: str = "additive"

    def load_csv(self, filepath: str, date_col: str, value_col: str,
                date_format: str = None) -> 'TimeSeriesDecomposer':
        """
        Load time series from CSV file.

        Args:
            filepath: Path to CSV file
            date_col: Name of date column
            value_col: Name of value column
            date_format: Date format string (optional)

        Returns:
            Self for method chaining
        """
        df = pd.read_csv(filepath)
        return self.load_dataframe(df, date_col, value_col, date_format)

    def load_dataframe(self, df: pd.DataFrame, date_col: str, value_col: str,
                      date_format: str = None) -> 'TimeSeriesDecomposer':
        """
        Load time series from DataFrame.

        Args:
            df: Input DataFrame
            date_col: Name of date column
            value_col: Name of value column
            date_format: Date format string (optional)

        Returns:
            Self for method chaining
        """
        df = df.copy()

        # Parse dates
        if date_format:
            df[date_col] = pd.to_datetime(df[date_col], format=date_format)
        else:
            df[date_col] = pd.to_datetime(df[date_col])

        # Create series with datetime index
        df = df.sort_values(date_col)
        self.data = pd.Series(
            df[value_col].values,
            index=pd.DatetimeIndex(df[date_col]),
            name=value_col
        )

        return self

    def load_series(self, series: pd.Series) -> 'TimeSeriesDecomposer':
        """
        Load from pandas Series.

        Args:
            series: Input series with datetime index

        Returns:
            Self for method chaining
        """
        self.data = series.copy()
        if not isinstance(self.data.index, pd.DatetimeIndex):
            self.data.index = pd.to_datetime(self.data.index)
        return self

    def detect_period(self) -> int:
        """
        Auto-detect the seasonal period using autocorrelation.

        Returns:
            Detected period
        """
        if self.data is None:
            raise ValueError("No data loaded")

        # Calculate ACF
        max_lag = min(len(self.data) // 2, 100)
        acf_values = acf(self.data.dropna(), nlags=max_lag)

        # Find peaks in ACF (excluding lag 0)
        peaks, properties = find_peaks(acf_values[1:], height=0.1)

        if len(peaks) > 0:
            # Return first significant peak
            return peaks[0] + 1

        # Fallback: check common periods
        common_periods = [7, 12, 4, 52, 365]
        for p in common_periods:
            if p < len(self.data) // 2 and acf_values[p] > 0.3:
                return p

        # Default
        return 1

    def decompose(self, period: int = None, model: str = "additive") -> Dict:
        """
        Decompose time series into components.

        Args:
            period: Seasonal period (auto-detected if None)
            model: "additive" or "multiplicative"

        Returns:
            Dictionary with decomposition results
        """
        if self.data is None:
            raise ValueError("No data loaded")

        if period is None:
            period = self.detect_period()

        self.period = period
        self.model = model

        # Perform decomposition
        self.decomposition = seasonal_decompose(
            self.data,
            model=model,
            period=period,
            extrapolate_trend='freq'
        )

        # Calculate strength metrics
        trend_strength = self._calculate_trend_strength()
        seasonal_strength = self._calculate_seasonal_strength()

        # Get seasonal pattern
        seasonal_pattern = self._get_seasonal_pattern()

        # Statistics
        trend_stats = self._calculate_trend_stats()

        return {
            "model": model,
            "period": period,
            "trend_strength": trend_strength,
            "seasonal_strength": seasonal_strength,
            "components": {
                "observed": self.data.tolist(),
                "trend": self.decomposition.trend.tolist(),
                "seasonal": self.decomposition.seasonal.tolist(),
                "residual": self.decomposition.resid.tolist()
            },
            "seasonal_pattern": seasonal_pattern,
            "statistics": {
                "trend_slope": trend_stats["slope"],
                "trend_r_squared": trend_stats["r_squared"],
                "residual_std": float(self.decomposition.resid.std()),
                "residual_mean": float(self.decomposition.resid.mean())
            }
        }

    def _calculate_trend_strength(self) -> float:
        """Calculate trend strength (0-1)."""
        if self.decomposition is None:
            return 0

        resid = self.decomposition.resid.dropna()
        detrended = self.data - self.decomposition.trend
        detrended = detrended.dropna()

        var_resid = resid.var()
        var_detrended = detrended.var()

        if var_detrended == 0:
            return 0

        strength = max(0, 1 - var_resid / var_detrended)
        return float(strength)

    def _calculate_seasonal_strength(self) -> float:
        """Calculate seasonal strength (0-1)."""
        if self.decomposition is None:
            return 0

        resid = self.decomposition.resid.dropna()
        deseasoned = self.data - self.decomposition.seasonal
        deseasoned = deseasoned.dropna()

        var_resid = resid.var()
        var_deseasoned = deseasoned.var()

        if var_deseasoned == 0:
            return 0

        strength = max(0, 1 - var_resid / var_deseasoned)
        return float(strength)

    def _get_seasonal_pattern(self) -> Dict[int, float]:
        """Get average seasonal effect by period position."""
        if self.decomposition is None or self.period is None:
            return {}

        seasonal = self.decomposition.seasonal.dropna()

        # Group by period position
        pattern = {}
        for i in range(self.period):
            values = seasonal.iloc[i::self.period]
            if len(values) > 0:
                pattern[i + 1] = float(values.mean())

        return pattern

    def _calculate_trend_stats(self) -> Dict:
        """Calculate trend statistics."""
        if self.decomposition is None:
            return {"slope": 0, "r_squared": 0}

        trend = self.decomposition.trend.dropna()
        x = np.arange(len(trend))

        if len(x) < 2:
            return {"slope": 0, "r_squared": 0}

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, trend)

        return {
            "slope": float(slope),
            "r_squared": float(r_value ** 2)
        }

    def extract_trend(self, method: str = "moving_average",
                     window: int = None) -> pd.Series:
        """
        Extract trend component.

        Args:
            method: "moving_average" or "polynomial"
            window: Window size for moving average

        Returns:
            Trend series
        """
        if self.data is None:
            raise ValueError("No data loaded")

        if window is None:
            window = max(3, len(self.data) // 10)

        if method == "moving_average":
            trend = self.data.rolling(window=window, center=True).mean()
        elif method == "polynomial":
            x = np.arange(len(self.data))
            coeffs = np.polyfit(x, self.data.fillna(method='ffill'), 2)
            trend = pd.Series(np.polyval(coeffs, x), index=self.data.index)
        else:
            raise ValueError(f"Unknown method: {method}")

        return trend

    def analyze_trend(self) -> Dict:
        """
        Analyze trend characteristics.

        Returns:
            Dictionary with trend analysis
        """
        if self.decomposition is None:
            self.decompose()

        trend = self.decomposition.trend.dropna()
        x = np.arange(len(trend))

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, trend)

        # Direction
        if slope > 0.001 * trend.mean():
            direction = "increasing"
        elif slope < -0.001 * trend.mean():
            direction = "decreasing"
        else:
            direction = "flat"

        # Growth rate
        if trend.iloc[0] != 0:
            total_change = (trend.iloc[-1] - trend.iloc[0]) / trend.iloc[0]
            periods = len(trend)
            growth_rate = (1 + total_change) ** (1 / periods) - 1
        else:
            growth_rate = 0

        # Volatility
        volatility = float(trend.pct_change().std())

        return {
            "direction": direction,
            "slope": float(slope),
            "r_squared": float(r_value ** 2),
            "growth_rate": float(growth_rate),
            "volatility": volatility,
            "start_value": float(trend.iloc[0]),
            "end_value": float(trend.iloc[-1])
        }

    def analyze_seasonality(self) -> Dict:
        """
        Analyze seasonal patterns.

        Returns:
            Dictionary with seasonality analysis
        """
        if self.decomposition is None:
            self.decompose()

        pattern = self._get_seasonal_pattern()
        strength = self._calculate_seasonal_strength()

        if not pattern:
            return {"detected_period": 0, "strength": 0, "pattern": {}}

        # Find peak and trough
        peak_period = max(pattern, key=pattern.get)
        trough_period = min(pattern, key=pattern.get)

        # Seasonal range
        seasonal_range = max(pattern.values()) - min(pattern.values())

        # Rank periods
        sorted_periods = sorted(pattern.items(), key=lambda x: x[1], reverse=True)
        ranks = {p: i + 1 for i, (p, v) in enumerate(sorted_periods)}

        detailed_pattern = {}
        for period, value in pattern.items():
            detailed_pattern[period] = {
                "value": value,
                "rank": ranks[period]
            }

        return {
            "detected_period": self.period,
            "strength": strength,
            "pattern": detailed_pattern,
            "peak_period": peak_period,
            "trough_period": trough_period,
            "seasonal_range": float(seasonal_range)
        }

    def analyze_residuals(self) -> Dict:
        """
        Analyze residual component.

        Returns:
            Dictionary with residual analysis
        """
        if self.decomposition is None:
            self.decompose()

        resid = self.decomposition.resid.dropna()

        # Normality test
        if len(resid) >= 8:
            stat, p_value = stats.shapiro(resid[:5000] if len(resid) > 5000 else resid)
            is_normal = p_value > 0.05
        else:
            is_normal = False
            p_value = 0

        # Autocorrelation (Ljung-Box test approximation)
        acf_values = acf(resid, nlags=min(10, len(resid) // 5))
        significant_autocorr = any(abs(acf_values[1:]) > 1.96 / np.sqrt(len(resid)))

        return {
            "mean": float(resid.mean()),
            "std": float(resid.std()),
            "min": float(resid.min()),
            "max": float(resid.max()),
            "is_normal": is_normal,
            "normality_p_value": float(p_value),
            "has_autocorrelation": significant_autocorr,
            "skewness": float(stats.skew(resid)),
            "kurtosis": float(stats.kurtosis(resid))
        }

    def detect_anomalies(self, threshold: float = 2.0) -> pd.DataFrame:
        """
        Detect anomalies in residuals.

        Args:
            threshold: Z-score threshold for anomaly detection

        Returns:
            DataFrame with anomalous points
        """
        if self.decomposition is None:
            self.decompose()

        resid = self.decomposition.resid
        zscore = (resid - resid.mean()) / resid.std()

        anomalies = []
        for idx, (z, r) in enumerate(zip(zscore, resid)):
            if pd.notna(z) and abs(z) > threshold:
                anomalies.append({
                    "date": self.data.index[idx],
                    "value": float(self.data.iloc[idx]),
                    "residual": float(r),
                    "zscore": float(z),
                    "anomaly_type": "high" if z > 0 else "low"
                })

        return pd.DataFrame(anomalies)

    def forecast(self, periods: int, method: str = "combined") -> pd.DataFrame:
        """
        Generate basic forecast.

        Args:
            periods: Number of periods to forecast
            method: "trend", "seasonal_naive", or "combined"

        Returns:
            DataFrame with forecasts
        """
        if self.decomposition is None:
            self.decompose()

        last_date = self.data.index[-1]

        # Determine frequency
        if len(self.data) > 1:
            freq = pd.infer_freq(self.data.index)
            if freq is None:
                # Estimate frequency from average gap
                avg_gap = (self.data.index[-1] - self.data.index[0]) / (len(self.data) - 1)
                future_dates = pd.date_range(
                    start=last_date + avg_gap,
                    periods=periods,
                    freq=avg_gap
                )
            else:
                future_dates = pd.date_range(
                    start=last_date,
                    periods=periods + 1,
                    freq=freq
                )[1:]
        else:
            future_dates = pd.date_range(start=last_date, periods=periods + 1)[1:]

        # Trend extrapolation
        trend = self.decomposition.trend.dropna()
        x = np.arange(len(trend))
        slope, intercept, _, _, _ = stats.linregress(x, trend)

        trend_forecast = []
        for i in range(periods):
            trend_forecast.append(intercept + slope * (len(trend) + i))

        # Seasonal component
        seasonal_pattern = list(self._get_seasonal_pattern().values())
        if not seasonal_pattern:
            seasonal_pattern = [0]

        seasonal_forecast = []
        for i in range(periods):
            seasonal_forecast.append(seasonal_pattern[i % len(seasonal_pattern)])

        # Combine based on method
        forecasts = []
        resid_std = self.decomposition.resid.std()

        for i in range(periods):
            if method == "trend":
                forecast = trend_forecast[i]
            elif method == "seasonal_naive":
                # Last season's value
                idx = len(self.data) - self.period + (i % self.period)
                forecast = self.data.iloc[idx] if idx >= 0 else self.data.mean()
            else:  # combined
                if self.model == "additive":
                    forecast = trend_forecast[i] + seasonal_forecast[i]
                else:
                    forecast = trend_forecast[i] * (1 + seasonal_forecast[i])

            forecasts.append({
                "date": future_dates[i],
                "forecast": float(forecast),
                "lower_bound": float(forecast - 1.96 * resid_std),
                "upper_bound": float(forecast + 1.96 * resid_std)
            })

        return pd.DataFrame(forecasts)

    def plot_components(self, output: str) -> str:
        """
        Plot decomposition components.

        Args:
            output: Output file path

        Returns:
            Output file path
        """
        if self.decomposition is None:
            self.decompose()

        fig, axes = plt.subplots(4, 1, figsize=(12, 10))

        # Original
        axes[0].plot(self.data.index, self.data.values, 'b-', linewidth=0.8)
        axes[0].set_title('Original Series')
        axes[0].set_ylabel('Value')

        # Trend
        axes[1].plot(self.data.index, self.decomposition.trend, 'g-', linewidth=1)
        axes[1].set_title('Trend')
        axes[1].set_ylabel('Value')

        # Seasonal
        axes[2].plot(self.data.index, self.decomposition.seasonal, 'r-', linewidth=0.8)
        axes[2].set_title(f'Seasonal (period={self.period})')
        axes[2].set_ylabel('Value')

        # Residual
        axes[3].plot(self.data.index, self.decomposition.resid, 'purple', linewidth=0.5)
        axes[3].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        axes[3].set_title('Residual')
        axes[3].set_ylabel('Value')
        axes[3].set_xlabel('Date')

        plt.tight_layout()
        plt.savefig(output, dpi=150, bbox_inches='tight')
        plt.close()

        return output

    def plot_acf_pacf(self, output: str, lags: int = 40) -> str:
        """
        Plot ACF and PACF.

        Args:
            output: Output file path
            lags: Number of lags to plot

        Returns:
            Output file path
        """
        if self.data is None:
            raise ValueError("No data loaded")

        lags = min(lags, len(self.data) // 2 - 1)

        fig, axes = plt.subplots(2, 1, figsize=(12, 6))

        # ACF
        acf_values = acf(self.data.dropna(), nlags=lags)
        axes[0].bar(range(len(acf_values)), acf_values, width=0.3)
        axes[0].axhline(y=0, color='black', linewidth=0.5)
        axes[0].axhline(y=1.96/np.sqrt(len(self.data)), color='red', linestyle='--')
        axes[0].axhline(y=-1.96/np.sqrt(len(self.data)), color='red', linestyle='--')
        axes[0].set_title('Autocorrelation Function (ACF)')
        axes[0].set_xlabel('Lag')
        axes[0].set_ylabel('ACF')

        # PACF
        pacf_values = pacf(self.data.dropna(), nlags=lags)
        axes[1].bar(range(len(pacf_values)), pacf_values, width=0.3)
        axes[1].axhline(y=0, color='black', linewidth=0.5)
        axes[1].axhline(y=1.96/np.sqrt(len(self.data)), color='red', linestyle='--')
        axes[1].axhline(y=-1.96/np.sqrt(len(self.data)), color='red', linestyle='--')
        axes[1].set_title('Partial Autocorrelation Function (PACF)')
        axes[1].set_xlabel('Lag')
        axes[1].set_ylabel('PACF')

        plt.tight_layout()
        plt.savefig(output, dpi=150, bbox_inches='tight')
        plt.close()

        return output

    def plot_seasonal(self, output: str) -> str:
        """
        Plot seasonal pattern.

        Args:
            output: Output file path

        Returns:
            Output file path
        """
        if self.decomposition is None:
            self.decompose()

        pattern = self._get_seasonal_pattern()

        fig, ax = plt.subplots(figsize=(10, 5))

        periods = list(pattern.keys())
        values = list(pattern.values())

        colors = ['green' if v >= 0 else 'red' for v in values]
        ax.bar(periods, values, color=colors, alpha=0.7)
        ax.axhline(y=0, color='black', linewidth=0.5)

        ax.set_title(f'Seasonal Pattern (Period = {self.period})')
        ax.set_xlabel('Period')
        ax.set_ylabel('Seasonal Effect')

        plt.tight_layout()
        plt.savefig(output, dpi=150, bbox_inches='tight')
        plt.close()

        return output

    def to_dataframe(self) -> pd.DataFrame:
        """
        Export decomposition to DataFrame.

        Returns:
            DataFrame with all components
        """
        if self.decomposition is None:
            self.decompose()

        return pd.DataFrame({
            "date": self.data.index,
            "observed": self.data.values,
            "trend": self.decomposition.trend.values,
            "seasonal": self.decomposition.seasonal.values,
            "residual": self.decomposition.resid.values
        })

    def summary(self) -> str:
        """
        Generate text summary.

        Returns:
            Summary string
        """
        if self.decomposition is None:
            self.decompose()

        trend = self.analyze_trend()
        seasonal = self.analyze_seasonality()
        resid = self.analyze_residuals()

        lines = [
            "=" * 50,
            "TIME SERIES DECOMPOSITION SUMMARY",
            "=" * 50,
            f"Model: {self.model}",
            f"Period: {self.period}",
            f"Observations: {len(self.data)}",
            "",
            "TREND ANALYSIS",
            "-" * 30,
            f"Direction: {trend['direction']}",
            f"Slope: {trend['slope']:.6f}",
            f"R-squared: {trend['r_squared']:.4f}",
            f"Growth rate: {trend['growth_rate']:.4%}",
            "",
            "SEASONALITY ANALYSIS",
            "-" * 30,
            f"Strength: {seasonal['strength']:.2f}",
            f"Peak period: {seasonal.get('peak_period', 'N/A')}",
            f"Trough period: {seasonal.get('trough_period', 'N/A')}",
            "",
            "RESIDUAL ANALYSIS",
            "-" * 30,
            f"Mean: {resid['mean']:.6f}",
            f"Std Dev: {resid['std']:.6f}",
            f"Normal: {'Yes' if resid['is_normal'] else 'No'}",
            f"Autocorrelation: {'Present' if resid['has_autocorrelation'] else 'Absent'}",
            "=" * 50
        ]

        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Time Series Decomposer - Extract trend, seasonal, and residual components"
    )

    parser.add_argument("--input", "-i", required=True, help="Input CSV file")
    parser.add_argument("--date", "-d", required=True, help="Date column name")
    parser.add_argument("--value", "-v", required=True, help="Value column name")
    parser.add_argument("--period", "-p", type=int, help="Seasonal period (auto-detect if not specified)")
    parser.add_argument("--model", "-m", choices=["additive", "multiplicative"],
                       default="additive", help="Decomposition model")
    parser.add_argument("--auto-period", action="store_true", help="Auto-detect period")
    parser.add_argument("--forecast", "-f", type=int, help="Number of periods to forecast")
    parser.add_argument("--plot", help="Output file for component plot")
    parser.add_argument("--output", "-o", help="Output CSV file for components")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    decomposer = TimeSeriesDecomposer()
    decomposer.load_csv(args.input, args.date, args.value)

    if args.auto_period:
        period = decomposer.detect_period()
        print(f"Detected period: {period}")
    else:
        period = args.period

    result = decomposer.decompose(period=period, model=args.model)

    if args.json:
        output = {
            "decomposition": {
                "model": result["model"],
                "period": result["period"],
                "trend_strength": result["trend_strength"],
                "seasonal_strength": result["seasonal_strength"],
                "statistics": result["statistics"]
            },
            "trend_analysis": decomposer.analyze_trend(),
            "seasonality_analysis": decomposer.analyze_seasonality(),
            "residual_analysis": decomposer.analyze_residuals()
        }

        if args.forecast:
            forecast_df = decomposer.forecast(args.forecast)
            output["forecast"] = forecast_df.to_dict(orient="records")

        print(json.dumps(output, indent=2, default=str))
    else:
        print(decomposer.summary())

        if args.forecast:
            print(f"\nFORECAST ({args.forecast} periods)")
            print("-" * 50)
            forecast_df = decomposer.forecast(args.forecast)
            for _, row in forecast_df.iterrows():
                print(f"{row['date'].strftime('%Y-%m-%d')}: {row['forecast']:.2f} "
                      f"[{row['lower_bound']:.2f}, {row['upper_bound']:.2f}]")

    if args.plot:
        decomposer.plot_components(args.plot)
        print(f"\nPlot saved to: {args.plot}")

    if args.output:
        df = decomposer.to_dataframe()
        df.to_csv(args.output, index=False)
        print(f"Components saved to: {args.output}")


if __name__ == "__main__":
    main()
