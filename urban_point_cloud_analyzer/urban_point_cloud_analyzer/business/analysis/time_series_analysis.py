# urban_point_cloud_analyzer/business/analysis/time_series_analysis.py
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

class TimeSeriesAnalyzer:
    """
    Time series analysis for urban metrics to track changes over time
    and identify trends in urban development.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize time series analyzer.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Configuration for trend analysis
        self.min_data_points = self.config.get('min_data_points', 3)
        self.smoothing_factor = self.config.get('smoothing_factor', 0.3)
    
    def analyze_metric_over_time(self, 
                                timestamps: List[str], 
                                values: List[float]) -> Dict:
        """
        Analyze a single metric over time.
        
        Args:
            timestamps: List of timestamp strings
            values: List of metric values
            
        Returns:
            Dictionary with analysis results
        """
        if len(timestamps) < 2 or len(values) < 2:
            return {
                "error": "Insufficient data for time series analysis",
                "min_required": 2,
                "provided": min(len(timestamps), len(values))
            }
        
        # Convert timestamps to datetime objects
        dates = [datetime.fromisoformat(ts) if isinstance(ts, str) else ts for ts in timestamps]
        
        # Convert to pandas Series for easier analysis
        try:
            series = pd.Series(values, index=dates)
            series = series.sort_index()  # Ensure chronological order
        except Exception as e:
            return {"error": f"Failed to create time series: {str(e)}"}
        
        # Basic statistics
        stats = {
            "start_date": series.index.min().isoformat(),
            "end_date": series.index.max().isoformat(),
            "duration_days": (series.index.max() - series.index.min()).days,
            "num_data_points": len(series),
            "first_value": float(series.iloc[0]),
            "last_value": float(series.iloc[-1]),
            "min_value": float(series.min()),
            "max_value": float(series.max()),
            "mean_value": float(series.mean()),
            "median_value": float(series.median()),
            "std_dev": float(series.std()) if len(series) > 1 else 0
        }
        
        # Calculate absolute and percentage change
        absolute_change = float(series.iloc[-1] - series.iloc[0])
        percentage_change = float((absolute_change / series.iloc[0]) * 100) if series.iloc[0] != 0 else float('inf')
        
        changes = {
            "absolute_change": absolute_change,
            "percentage_change": percentage_change,
            "annualized_change": absolute_change / stats["duration_days"] * 365 if stats["duration_days"] > 0 else 0
        }
        
        # Calculate trend
        if len(series) >= self.min_data_points:
            try:
                # Use linear regression to calculate trend
                x = np.arange(len(series))
                y = series.values
                slope, intercept = np.polyfit(x, y, 1)
                
                trend = {
                    "direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable",
                    "slope": float(slope),
                    "r_squared": float(np.corrcoef(x, y)[0, 1]**2),
                    "is_significant": abs(slope) > self.config.get('significance_threshold', 0.01)
                }
            except Exception as e:
                trend = {"error": f"Failed to calculate trend: {str(e)}"}
        else:
            trend = {"error": f"Insufficient data points for trend analysis (need {self.min_data_points}, got {len(series)})"}
        
        # Generate visualization
        visualization = self._generate_time_series_plot(series)
        
        return {
            "statistics": stats,
            "changes": changes,
            "trend": trend,
            "visualization": visualization
        }
    
    def _generate_time_series_plot(self, series: pd.Series) -> str:
        """
        Generate a base64-encoded time series plot.
        
        Args:
            series: Pandas time series
            
        Returns:
            Base64-encoded image
        """
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(series.index, series.values, marker='o', linestyle='-', color='#4285f4')
            
            # Add trend line if enough data points
            if len(series) >= self.min_data_points:
                x = np.arange(len(series))
                y = series.values
                slope, intercept = np.polyfit(x, y, 1)
                plt.plot(series.index, intercept + slope * x, 'r--', label=f'Trend (slope: {slope:.4f})')
                plt.legend()
            
            plt.title('Metric Trend Over Time')
            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Convert plot to base64 string
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close()
            
            return image_base64
        
        except Exception as e:
            # Return error if visualization fails
            return f"Error generating visualization: {str(e)}"
    
    def compare_metrics_over_time(self, 
                                timestamps: List[str],
                                metrics_data: Dict[str, List[float]]) -> Dict:
        """
        Compare multiple metrics over time.
        
        Args:
            timestamps: List of timestamp strings
            metrics_data: Dictionary mapping metric names to value lists
            
        Returns:
            Dictionary with comparative analysis
        """
        if len(timestamps) < 2:
            return {"error": "Insufficient timestamps for comparison"}
        
        # Analyze each metric
        metric_analyses = {}
        for metric_name, values in metrics_data.items():
            if len(values) == len(timestamps):
                metric_analyses[metric_name] = self.analyze_metric_over_time(timestamps, values)
            else:
                metric_analyses[metric_name] = {
                    "error": f"Length mismatch: {len(values)} values vs {len(timestamps)} timestamps"
                }
        
        # Generate correlation matrix between metrics
        if len(metrics_data) > 1:
            correlation_matrix = self._calculate_correlation_matrix(metrics_data)
        else:
            correlation_matrix = None
        
        # Generate comparative visualization
        comparative_visualization = self._generate_comparative_plot(timestamps, metrics_data)
        
        return {
            "metric_analyses": metric_analyses,
            "correlation_matrix": correlation_matrix,
            "comparative_visualization": comparative_visualization
        }
    
    def _calculate_correlation_matrix(self, metrics_data: Dict[str, List[float]]) -> Dict:
        """
        Calculate correlation matrix between metrics.
        
        Args:
            metrics_data: Dictionary mapping metric names to value lists
            
        Returns:
            Dictionary with correlation matrix
        """
        # Convert to DataFrame for correlation calculation
        try:
            df = pd.DataFrame(metrics_data)
            corr_matrix = df.corr().round(4).to_dict()
            
            # Find strongly correlated metrics (positive or negative)
            strong_correlations = []
            
            for metric1 in corr_matrix:
                for metric2, corr_value in corr_matrix[metric1].items():
                    if metric1 != metric2 and abs(corr_value) > 0.7:  # Threshold for strong correlation
                        relationship = "positive" if corr_value > 0 else "negative"
                        strong_correlations.append({
                            "metric1": metric1,
                            "metric2": metric2,
                            "correlation": corr_value,
                            "relationship": relationship,
                            "strength": "very strong" if abs(corr_value) > 0.9 else "strong"
                        })
            
            return {
                "matrix": corr_matrix,
                "strong_correlations": strong_correlations
            }
            
        except Exception as e:
            return {"error": f"Failed to calculate correlation matrix: {str(e)}"}
    
    def _generate_comparative_plot(self, 
                                 timestamps: List[str],
                                 metrics_data: Dict[str, List[float]]) -> str:
        """
        Generate a comparative plot of multiple metrics.
        
        Args:
            timestamps: List of timestamp strings
            metrics_data: Dictionary mapping metric names to value lists
            
        Returns:
            Base64-encoded image
        """
        try:
            # Convert timestamps to datetime
            dates = [datetime.fromisoformat(ts) if isinstance(ts, str) else ts for ts in timestamps]
            
            plt.figure(figsize=(12, 8))
            
            # Plot each metric
            for metric_name, values in metrics_data.items():
                # Normalize values for better comparison (0-1 scale)
                min_val = min(values)
                max_val = max(values)
                range_val = max_val - min_val
                
                if range_val > 0:
                    normalized_values = [(v - min_val) / range_val for v in values]
                else:
                    normalized_values = [0.5 for _ in values]  # Constant value
                
                plt.plot(dates, normalized_values, marker='o', linestyle='-', label=f"{metric_name} (normalized)")
            
            plt.title('Comparative Metric Trends')
            plt.xlabel('Date')
            plt.ylabel('Normalized Value (0-1)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Convert plot to base64 string
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close()
            
            return image_base64
            
        except Exception as e:
            # Return error if visualization fails
            return f"Error generating comparative visualization: {str(e)}"
    
    def detect_change_points(self, 
                           timestamps: List[str], 
                           values: List[float]) -> Dict:
        """
        Detect significant change points in a time series.
        
        Args:
            timestamps: List of timestamp strings
            values: List of metric values
            
        Returns:
            Dictionary with detected change points
        """
        if len(timestamps) < 4 or len(values) < 4:
            return {
                "error": "Insufficient data for change point detection",
                "min_required": 4,
                "provided": min(len(timestamps), len(values))
            }
        
        try:
            # Convert timestamps to datetime
            dates = [datetime.fromisoformat(ts) if isinstance(ts, str) else ts for ts in timestamps]
            
            # Simple change point detection based on moving average
            window_size = max(2, len(values) // 5)  # 20% of data points
            
            # Calculate moving average
            moving_avg = []
            for i in range(len(values) - window_size + 1):
                window = values[i:i+window_size]
                moving_avg.append(sum(window) / window_size)
            
            # Calculate rate of change in moving average
            rate_of_change = [abs(moving_avg[i] - moving_avg[i-1]) for i in range(1, len(moving_avg))]
            
            # Determine threshold for significant change
            mean_change = sum(rate_of_change) / len(rate_of_change)
            std_change = np.std(rate_of_change)
            threshold = mean_change + 2 * std_change  # 2 standard deviations
            
            # Find change points
            change_points = []
            for i in range(len(rate_of_change)):
                if rate_of_change[i] > threshold:
                    change_point_idx = i + window_size // 2  # Approximate location within window
                    
                    # Ensure index is within bounds
                    if 0 <= change_point_idx < len(dates):
                        change_points.append({
                            "index": change_point_idx,
                            "timestamp": dates[change_point_idx].isoformat() if hasattr(dates[change_point_idx], 'isoformat') else str(dates[change_point_idx]),
                            "value": values[change_point_idx],
                            "change_magnitude": rate_of_change[i]
                        })
            
            # Generate visualization with change points
            visualization = self._generate_change_point_plot(dates, values, change_points)
            
            return {
                "change_points": change_points,
                "threshold": threshold,
                "visualization": visualization
            }
            
        except Exception as e:
            return {"error": f"Failed to detect change points: {str(e)}"}
    
    def _generate_change_point_plot(self, 
                                  dates: List[datetime], 
                                  values: List[float],
                                  change_points: List[Dict]) -> str:
        """
        Generate a plot highlighting change points.
        
        Args:
            dates: List of datetime objects
            values: List of metric values
            change_points: List of detected change points
            
        Returns:
            Base64-encoded image
        """
        try:
            plt.figure(figsize=(12, 6))
            
            # Plot the time series
            plt.plot(dates, values, marker='o', linestyle='-', color='#4285f4', label='Metric Value')
            
            # Highlight change points
            change_point_indices = [cp['index'] for cp in change_points]
            change_point_values = [values[idx] for idx in change_point_indices]
            change_point_dates = [dates[idx] for idx in change_point_indices]
            
            plt.scatter(change_point_dates, change_point_values, color='red', s=100, zorder=5, label='Change Points')
            
            for idx, date, value in zip(change_point_indices, change_point_dates, change_point_values):
                plt.annotate(f"CP {idx}", (date, value), xytext=(10, 10), 
                           textcoords='offset points', color='red', fontweight='bold')
            
            plt.title('Time Series with Detected Change Points')
            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Convert plot to base64 string
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close()
            
            return image_base64
            
        except Exception as e:
            # Return error if visualization fails
            return f"Error generating change point visualization: {str(e)}"
    
    def forecast_future_trends(self, 
                             timestamps: List[str], 
                             values: List[float],
                             forecast_periods: int = 3) -> Dict:
        """
        Forecast future trends based on historical data.
        
        Args:
            timestamps: List of timestamp strings
            values: List of metric values
            forecast_periods: Number of periods to forecast
            
        Returns:
            Dictionary with forecast results
        """
        if len(timestamps) < 3 or len(values) < 3:
            return {
                "error": "Insufficient data for forecasting",
                "min_required": 3,
                "provided": min(len(timestamps), len(values))
            }
        
        try:
            # Convert timestamps to datetime
            dates = [datetime.fromisoformat(ts) if isinstance(ts, str) else ts for ts in timestamps]
            
            # Convert to pandas Series
            series = pd.Series(values, index=dates)
            series = series.sort_index()  # Ensure chronological order
            
            # Calculate average time delta between data points
            if len(dates) > 1:
                time_deltas = [(dates[i+1] - dates[i]).total_seconds() for i in range(len(dates)-1)]
                avg_delta = sum(time_deltas) / len(time_deltas)
                
                # Create forecast dates
                forecast_dates = []
                for i in range(1, forecast_periods + 1):
                    last_date = dates[-1]
                    next_date = last_date + pd.Timedelta(seconds=avg_delta)
                    forecast_dates.append(next_date)
                    dates.append(next_date)  # Add to dates for visualization
            else:
                return {"error": "Need at least two data points to calculate time delta"}
            
            # Use simple linear regression for forecasting
            x = np.arange(len(values))
            y = np.array(values)
            
            # Fit linear model
            slope, intercept = np.polyfit(x, y, 1)
            
            # Generate forecast values
            forecast_x = np.arange(len(values), len(values) + forecast_periods)
            forecast_y = slope * forecast_x + intercept
            
            # Calculate confidence interval (simple approach)
            residuals = y - (slope * x + intercept)
            std_residuals = np.std(residuals)
            confidence_interval = 1.96 * std_residuals  # 95% confidence interval
            
            # Prepare forecast results
            forecast_results = []
            for i in range(forecast_periods):
                forecast_results.append({
                    "period": i + 1,
                    "date": forecast_dates[i].isoformat() if hasattr(forecast_dates[i], 'isoformat') else str(forecast_dates[i]),
                    "forecast_value": float(forecast_y[i]),
                    "lower_bound": float(forecast_y[i] - confidence_interval),
                    "upper_bound": float(forecast_y[i] + confidence_interval)
                })
            
            # Generate visualization
            all_values = list(values) + list(forecast_y)
            visualization = self._generate_forecast_plot(dates, values, forecast_y, forecast_dates, confidence_interval)
            
            return {
                "forecast": forecast_results,
                "model": {
                    "type": "linear",
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "confidence_interval": float(confidence_interval)
                },
                "visualization": visualization
            }
            
        except Exception as e:
            return {"error": f"Failed to generate forecast: {str(e)}"}
    
    def _generate_forecast_plot(self, 
                              dates: List[datetime], 
                              historical_values: List[float],
                              forecast_values: List[float],
                              forecast_dates: List[datetime],
                              confidence_interval: float) -> str:
        """
        Generate a forecast plot.
        
        Args:
            dates: List of all datetime objects (historical + forecast)
            historical_values: List of historical values
            forecast_values: List of forecasted values
            forecast_dates: List of forecasted dates
            confidence_interval: Confidence interval for forecasts
            
        Returns:
            Base64-encoded image
        """
        try:
            plt.figure(figsize=(12, 6))
            
            # Historical data
            historical_dates = dates[:len(historical_values)]
            plt.plot(historical_dates, historical_values, marker='o', linestyle='-', color='#4285f4', label='Historical Data')
            
            # Forecast data
            plt.plot(forecast_dates, forecast_values, marker='x', linestyle='--', color='#34a853', label='Forecast')
            
            # Confidence interval
            lower_bound = [y - confidence_interval for y in forecast_values]
            upper_bound = [y + confidence_interval for y in forecast_values]
            
            plt.fill_between(forecast_dates, lower_bound, upper_bound, color='#34a853', alpha=0.2, label='95% Confidence Interval')
            
            plt.title('Historical Data and Forecast')
            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Convert plot to base64 string
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close()
            
            return image_base64
            
        except Exception as e:
            # Return error if visualization fails
            return f"Error generating forecast visualization: {str(e)}"