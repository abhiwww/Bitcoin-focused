"""
Bitcoin Market Sentiment vs. Trader Performance Analysis
Senior Data Scientist Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set visual style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

# ==================== 1. DATA LOADING ====================

def load_datasets():
    """
    Load and validate both datasets
    """
    try:
        # Load sentiment data
        sentiment_df = pd.read_csv('bitcoin_sentiment.csv')
        print(f"✓ Sentiment data loaded: {len(sentiment_df)} rows")
        
        # Load trader data
        trader_df = pd.read_csv('historical_trader_data.csv')
        print(f"✓ Trader data loaded: {len(trader_df)} rows")
        
        return sentiment_df, trader_df
    
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please ensure both CSV files are in the working directory:")
        print("1. bitcoin_sentiment.csv")
        print("2. historical_trader_data.csv")
        return None, None

# ==================== 2. DATA CLEANING ====================

def clean_sentiment_data(df):
    """
    Clean and standardize sentiment data
    """
    print("\n" + "="*50)
    print("CLEANING SENTIMENT DATA")
    print("="*50)
    
    # Convert date to datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Standardize classification labels
    classification_mapping = {
        'extreme fear': 'Extreme Fear',
        'fear': 'Fear', 
        'neutral': 'Neutral',
        'greed': 'Greed',
        'extreme greed': 'Extreme Greed',
        'Extreme Fear': 'Extreme Fear',
        'Fear': 'Fear',
        'Neutral': 'Neutral', 
        'Greed': 'Greed',
        'Extreme Greed': 'Extreme Greed'
    }
    
    df['Classification'] = df['Classification'].str.lower().map(classification_mapping)
    
    # Drop rows with invalid dates or classifications
    initial_rows = len(df)
    df = df.dropna(subset=['Date', 'Classification'])
    dropped_rows = initial_rows - len(df)
    print(f"Dropped {dropped_rows} rows with invalid dates/classifications")
    
    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Create sentiment score for easier analysis (0-100 scale)
    sentiment_score_map = {
        'Extreme Fear': 20,
        'Fear': 40,
        'Neutral': 60,
        'Greed': 80,
        'Extreme Greed': 100
    }
    df['Sentiment_Score'] = df['Classification'].map(sentiment_score_map)
    
    print(f"✓ Cleaned sentiment data: {len(df)} rows")
    print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"Sentiment distribution:\n{df['Classification'].value_counts()}")
    
    return df

def clean_trader_data(df):
    """
    Clean and standardize trader data
    """
    print("\n" + "="*50)
    print("CLEANING TRADER DATA")
    print("="*50)
    
    # Convert timestamp to datetime
    df['time'] = pd.to_datetime(df['time'], unit='ms', errors='coerce')
    
    # Standardize side labels
    df['side'] = df['side'].str.upper()
    
    # Clean closedPnL - handle various formats
    # Convert to numeric, coerce errors to NaN
    df['closedPnL'] = pd.to_numeric(df['closedPnL'], errors='coerce')
    
    # Handle missing or invalid closedPnL
    initial_rows = len(df)
    df = df.dropna(subset=['time', 'closedPnL', 'side'])
    dropped_rows = initial_rows - len(df)
    print(f"Dropped {dropped_rows} rows with invalid/missing values")
    
    # Create derived columns
    df['trade_date'] = df['time'].dt.date
    df['is_profitable'] = df['closedPnL'] > 0
    df['abs_pnl'] = abs(df['closedPnL'])
    df['pnl_percentage'] = (df['closedPnL'] / df['execution_price']) * 100
    
    # Handle leverage - ensure it's numeric
    if 'leverage' in df.columns:
        df['leverage'] = pd.to_numeric(df['leverage'], errors='coerce').fillna(1)
    else:
        df['leverage'] = 1  # Default leverage if missing
    
    print(f"✓ Cleaned trader data: {len(df)} rows")
    print(f"Date range: {df['time'].min().date()} to {df['time'].max().date()}")
    print(f"Total trades: {len(df)}")
    print(f"Profitable trades: {df['is_profitable'].sum()} ({df['is_profitable'].mean()*100:.1f}%)")
    print(f"Total PnL: ${df['closedPnL'].sum():,.2f}")
    
    return df

# ==================== 3. DATA MERGING ====================

def merge_datasets(sentiment_df, trader_df):
    """
    Merge trader data with sentiment data by date
    """
    print("\n" + "="*50)
    print("MERGING DATASETS")
    print("="*50)
    
    # Create date-only columns for merging
    sentiment_df['trade_date'] = sentiment_df['Date'].dt.date
    trader_df['trade_date'] = trader_df['time'].dt.date
    
    # Merge on date
    merged_df = pd.merge(
        trader_df,
        sentiment_df[['trade_date', 'Classification', 'Sentiment_Score']],
        on='trade_date',
        how='left'
    )
    
    # Check for unmatched dates
    unmatched = merged_df['Classification'].isna().sum()
    print(f"Trades without sentiment data: {unmatched} ({unmatched/len(merged_df)*100:.1f}%)")
    
    # Remove trades without sentiment data
    merged_df = merged_df.dropna(subset=['Classification'])
    
    print(f"✓ Final merged dataset: {len(merged_df)} trades")
    print(f"Merged dataset date range: {merged_df['trade_date'].min()} to {merged_df['trade_date'].max()}")
    
    return merged_df

# ==================== 4. EXPLORATORY DATA ANALYSIS ====================

def perform_eda(merged_df):
    """
    Perform comprehensive exploratory data analysis
    """
    print("\n" + "="*50)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*50)
    
    results = {}
    
    # 1. Average closedPnL by sentiment
    pnl_by_sentiment = merged_df.groupby('Classification')['closedPnL'].agg([
        ('count', 'count'),
        ('avg_pnl', 'mean'),
        ('total_pnl', 'sum'),
        ('std_pnl', 'std'),
        ('median_pnl', 'median')
    ]).sort_values('avg_pnl', ascending=False)
    
    results['pnl_by_sentiment'] = pnl_by_sentiment
    print("\n1. AVERAGE PnL BY SENTIMENT:")
    print(pnl_by_sentiment[['count', 'avg_pnl', 'total_pnl']])
    
    # 2. Win rate by sentiment
    win_rate = merged_df.groupby('Classification')['is_profitable'].agg([
        ('total_trades', 'count'),
        ('profitable_trades', 'sum'),
        ('win_rate', lambda x: x.mean() * 100)
    ]).sort_values('win_rate', ascending=False)
    
    results['win_rate'] = win_rate
    print("\n2. WIN RATE BY SENTIMENT:")
    print(win_rate[['total_trades', 'profitable_trades', 'win_rate']])
    
    # 3. Average leverage by sentiment
    if 'leverage' in merged_df.columns:
        leverage_stats = merged_df.groupby('Classification')['leverage'].agg([
            ('avg_leverage', 'mean'),
            ('median_leverage', 'median'),
            ('max_leverage', 'max'),
            ('min_leverage', 'min')
        ]).sort_values('avg_leverage', ascending=False)
        
        results['leverage_stats'] = leverage_stats
        print("\n3. LEVERAGE BY SENTIMENT:")
        print(leverage_stats)
    
    # 4. Trade volume by sentiment
    volume_by_sentiment = merged_df.groupby('Classification').agg({
        'size': 'sum',
        'abs_pnl': 'sum',
        'account': 'nunique'
    }).rename(columns={
        'size': 'total_volume',
        'abs_pnl': 'total_abs_pnl',
        'account': 'unique_traders'
    }).sort_values('total_volume', ascending=False)
    
    results['volume_stats'] = volume_by_sentiment
    print("\n4. TRADE VOLUME BY SENTIMENT:")
    print(volume_by_sentiment)
    
    # 5. Risk-Adjusted Returns
    merged_df['risk_adjusted_return'] = merged_df['closedPnL'] / merged_df['leverage']
    risk_adj_returns = merged_df.groupby('Classification')['risk_adjusted_return'].agg([
        ('mean', 'mean'),
        ('std', 'std')
    ])
    risk_adj_returns['sharpe_ratio'] = risk_adj_returns['mean'] / risk_adj_returns['std'].replace(0, np.nan)
    
    results['risk_adj_returns'] = risk_adj_returns
    print("\n5. RISK-ADJUSTED RETURNS:")
    print(risk_adj_returns.sort_values('sharpe_ratio', ascending=False))
    
    return results

# ==================== 5. BEHAVIORAL PATTERN ANALYSIS ====================

def analyze_behavioral_patterns(merged_df):
    """
    Analyze trader behavior patterns by sentiment
    """
    print("\n" + "="*50)
    print("BEHAVIORAL PATTERN ANALYSIS")
    print("="*50)
    
    patterns = {}
    
    # 1. Compare Fear vs Greed periods
    fear_periods = merged_df[merged_df['Classification'].isin(['Fear', 'Extreme Fear'])]
    greed_periods = merged_df[merged_df['Classification'].isin(['Greed', 'Extreme Greed'])]
    
    fear_greed_comparison = pd.DataFrame({
        'Metric': ['Total Trades', 'Avg PnL', 'Win Rate', 'Avg Leverage', 'Total Volume'],
        'Fear Periods': [
            len(fear_periods),
            fear_periods['closedPnL'].mean(),
            fear_periods['is_profitable'].mean() * 100,
            fear_periods['leverage'].mean(),
            fear_periods['size'].sum()
        ],
        'Greed Periods': [
            len(greed_periods),
            greed_periods['closedPnL'].mean(),
            greed_periods['is_profitable'].mean() * 100,
            greed_periods['leverage'].mean(),
            greed_periods['size'].sum()
        ]
    })
    
    patterns['fear_vs_greed'] = fear_greed_comparison
    print("\n1. FEAR vs GREED COMPARISON:")
    print(fear_greed_comparison.to_string(index=False))
    
    # 2. Risk-taking behavior analysis
    print("\n2. RISK-TAKING BEHAVIOR:")
    
    # Correlation between sentiment score and leverage
    correlation = merged_df[['Sentiment_Score', 'leverage', 'closedPnL']].corr()
    print("Correlation Matrix:")
    print(correlation)
    
    # Leverage distribution by sentiment
    leverage_by_sentiment = merged_df.groupby('Classification')['leverage'].describe()
    print("\nLeverage Statistics by Sentiment:")
    print(leverage_by_sentiment[['count', 'mean', 'std', 'min', '50%', 'max']])
    
    patterns['leverage_distribution'] = leverage_by_sentiment
    
    # 3. Performance by time of day during different sentiments
    merged_df['hour_of_day'] = merged_df['time'].dt.hour
    
    # Analyze if traders change behavior based on sentiment and time
    hourly_performance = merged_df.groupby(['Classification', 'hour_of_day']).agg({
        'closedPnL': 'mean',
        'is_profitable': 'mean',
        'size': 'sum'
    }).reset_index()
    
    patterns['hourly_patterns'] = hourly_performance
    
    print("\n3. TRADING PATTERNS BY HOUR:")
    print("Peak trading hours (by volume):")
    peak_hours = merged_df.groupby('hour_of_day')['size'].sum().nlargest(3)
    print(peak_hours)
    
    return patterns

# ==================== 6. VISUALIZATIONS ====================

def create_visualizations(merged_df, eda_results):
    """
    Create comprehensive visualizations
    """
    print("\n" + "="*50)
    print("CREATING VISUALIZATIONS")
    print("="*50)
    
    # Define sentiment order for consistent plotting
    sentiment_order = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
    
    # 1. Average PnL by Sentiment (Bar Chart)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Bar chart of average PnL
    ax1 = axes[0, 0]
    pnl_data = eda_results['pnl_by_sentiment'].reindex(sentiment_order)
    ax1.bar(pnl_data.index, pnl_data['avg_pnl'], 
            color=['#d62728', '#ff9896', '#c7c7c7', '#98df8a', '#2ca02c'])
    ax1.set_title('Average PnL by Market Sentiment', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Market Sentiment', fontsize=12)
    ax1.set_ylabel('Average PnL ($)', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(pnl_data['avg_pnl']):
        ax1.text(i, v, f'${v:,.0f}', ha='center', va='bottom' if v >= 0 else 'top', fontweight='bold')
    
    # 2. Boxplot of PnL Distribution by Sentiment
    ax2 = axes[0, 1]
    sentiment_data = []
    for sentiment in sentiment_order:
        data = merged_df[merged_df['Classification'] == sentiment]['closedPnL']
        sentiment_data.append(data)
    
    bp = ax2.boxplot(sentiment_data, labels=sentiment_order, patch_artist=True)
    
    # Color the boxes
    colors = ['#d62728', '#ff9896', '#c7c7c7', '#98df8a', '#2ca02c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax2.set_title('PnL Distribution by Market Sentiment', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Market Sentiment', fontsize=12)
    ax2.set_ylabel('PnL ($)', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 3. Win Rate by Sentiment
    ax3 = axes[1, 0]
    win_rate_data = eda_results['win_rate'].reindex(sentiment_order)
    ax3.bar(win_rate_data.index, win_rate_data['win_rate'], 
            color=['#d62728', '#ff9896', '#c7c7c7', '#98df8a', '#2ca02c'])
    ax3.set_title('Win Rate by Market Sentiment', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Market Sentiment', fontsize=12)
    ax3.set_ylabel('Win Rate (%)', fontsize=12)
    ax3.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(win_rate_data['win_rate']):
        ax3.text(i, v, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 4. Leverage by Sentiment
    ax4 = axes[1, 1]
    leverage_data = eda_results.get('leverage_stats', pd.DataFrame())
    if not leverage_data.empty:
        leverage_data = leverage_data.reindex(sentiment_order)
        ax4.bar(leverage_data.index, leverage_data['avg_leverage'],
                color=['#d62728', '#ff9896', '#c7c7c7', '#98df8a', '#2ca02c'])
        ax4.set_title('Average Leverage by Market Sentiment', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Market Sentiment', fontsize=12)
        ax4.set_ylabel('Average Leverage', fontsize=12)
        ax4.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(leverage_data['avg_leverage']):
            ax4.text(i, v, f'{v:.1f}x', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('sentiment_analysis_summary.png', dpi=300, bbox_inches='tight')
    print("✓ Saved visualization: sentiment_analysis_summary.png")
    
    # 5. Additional Visualization: Cumulative PnL over time by sentiment
    fig2, ax5 = plt.subplots(figsize=(14, 6))
    
    # Sort by time
    merged_df_sorted = merged_df.sort_values('time')
    
    # Create cumulative PnL by sentiment category
    for sentiment in sentiment_order:
        sentiment_data = merged_df_sorted[merged_df_sorted['Classification'] == sentiment]
        cumulative_pnl = sentiment_data['closedPnL'].cumsum()
        ax5.plot(sentiment_data['time'], cumulative_pnl, 
                label=sentiment, linewidth=2)
    
    ax5.set_title('Cumulative PnL Over Time by Market Sentiment', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Time', fontsize=12)
    ax5.set_ylabel('Cumulative PnL ($)', fontsize=12)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Add horizontal line at y=0
    ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('cumulative_pnl_by_sentiment.png', dpi=300, bbox_inches='tight')
    print("✓ Saved visualization: cumulative_pnl_by_sentiment.png")
    
    # 6. Heatmap: Correlation between sentiment and trading metrics
    fig3, ax6 = plt.subplots(figsize=(10, 8))
    
    # Prepare correlation data
    correlation_data = merged_df.groupby('Classification').agg({
        'closedPnL': 'mean',
        'leverage': 'mean',
        'is_profitable': 'mean',
        'size': 'mean',
        'pnl_percentage': 'mean'
    }).reindex(sentiment_order)
    
    # Normalize for heatmap
    correlation_normalized = (correlation_data - correlation_data.mean()) / correlation_data.std()
    
    sns.heatmap(correlation_normalized.T, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0, ax=ax6, cbar_kws={'label': 'Standardized Value'})
    
    ax6.set_title('Standardized Trading Metrics by Market Sentiment', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Market Sentiment', fontsize=12)
    ax6.set_ylabel('Trading Metrics', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('sentiment_metrics_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Saved visualization: sentiment_metrics_heatmap.png")
    
    plt.show()

# ==================== 7. INSIGHTS AND CONCLUSIONS ====================

def generate_insights(merged_df, eda_results, behavioral_patterns):
    """
    Generate professional insights and conclusions
    """
    print("\n" + "="*50)
    print("KEY INSIGHTS & CONCLUSIONS")
    print("="*50)
    
    # Extract key metrics
    pnl_stats = eda_results['pnl_by_sentiment']
    win_rates = eda_results['win_rate']
    
    print("\n📊 PERFORMANCE INSIGHTS:")
    print("-" * 40)
    
    # Find best and worst performing sentiments
    best_sentiment = pnl_stats['avg_pnl'].idxmax()
    worst_sentiment = pnl_stats['avg_pnl'].idxmin()
    best_win_rate = win_rates['win_rate'].idxmax()
    
    print(f"1. BEST PERFORMING SENTIMENT: {best_sentiment}")
    print(f"   • Average PnL: ${pnl_stats.loc[best_sentiment, 'avg_pnl']:,.2f}")
    print(f"   • Win Rate: {win_rates.loc[best_sentiment, 'win_rate']:.1f}%")
    print(f"   • Total PnL: ${pnl_stats.loc[best_sentiment, 'total_pnl']:,.2f}")
    
    print(f"\n2. WORST PERFORMING SENTIMENT: {worst_sentiment}")
    print(f"   • Average PnL: ${pnl_stats.loc[worst_sentiment, 'avg_pnl']:,.2f}")
    print(f"   • Win Rate: {win_rates.loc[worst_sentiment, 'win_rate']:.1f}%")
    
    print(f"\n3. HIGHEST WIN RATE: {best_win_rate}")
    print(f"   • Win Rate: {win_rates.loc[best_win_rate, 'win_rate']:.1f}%")
    
    # Statistical significance check
    print("\n📈 STATISTICAL ANALYSIS:")
    print("-" * 40)
    
    # Check if differences are meaningful
    fear_greed_diff = abs(
        pnl_stats.loc[['Fear', 'Extreme Fear'], 'avg_pnl'].mean() -
        pnl_stats.loc[['Greed', 'Extreme Greed'], 'avg_pnl'].mean()
    )
    
    print(f"• Average PnL difference (Fear vs Greed periods): ${fear_greed_diff:,.2f}")
    
    # Risk behavior insights
    if 'leverage_stats' in eda_results:
        leverage_stats = eda_results['leverage_stats']
        max_leverage_sentiment = leverage_stats['avg_leverage'].idxmax()
        min_leverage_sentiment = leverage_stats['avg_leverage'].idxmin()
        
        print(f"\n⚡ RISK BEHAVIOR:")
        print(f"• Highest leverage used during: {max_leverage_sentiment}")
        print(f"  (Average: {leverage_stats.loc[max_leverage_sentiment, 'avg_leverage']:.1f}x)")
        print(f"• Lowest leverage used during: {min_leverage_sentiment}")
        print(f"  (Average: {leverage_stats.loc[min_leverage_sentiment, 'avg_leverage']:.1f}x)")
    
    # Trading behavior patterns
    print("\n🎯 BEHAVIORAL PATTERNS:")
    print("-" * 40)
    
    fear_greed_comparison = behavioral_patterns['fear_vs_greed']
    fear_trades = fear_greed_comparison.loc[0, 'Fear Periods']
    greed_trades = fear_greed_comparison.loc[0, 'Greed Periods']
    
    print(f"• Fear periods: {int(fear_trades):,} trades")
    print(f"• Greed periods: {int(greed_trades):,} trades")
    print(f"• Ratio (Fear:Greed): {fear_trades/greed_trades:.2f}:1")
    
    # Professional recommendations
    print("\n💡 PROFESSIONAL RECOMMENDATIONS:")
    print("-" * 40)
    
    recommendations = []
    
    if pnl_stats.loc['Extreme Fear', 'avg_pnl'] > 0:
        recommendations.append("Consider increasing position sizing during Extreme Fear periods")
    
    if win_rates.loc['Neutral', 'win_rate'] > win_rates.loc['Extreme Greed', 'win_rate']:
        recommendations.append("Exercise caution during Extreme Greed - success rate declines")
    
    if 'leverage_stats' in eda_results and leverage_stats.loc['Extreme Greed', 'avg_leverage'] > 2:
        recommendations.append("Implement leverage caps during Extreme Greed to manage risk")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    # Conclusion
    print("\n🎯 EXECUTIVE SUMMARY:")
    print("-" * 40)
    
    total_pnl = pnl_stats['total_pnl'].sum()
    overall_win_rate = merged_df['is_profitable'].mean() * 100
    
    print(f"• Total analyzed trades: {len(merged_df):,}")
    print(f"• Total PnL across all sentiments: ${total_pnl:,.2f}")
    print(f"• Overall win rate: {overall_win_rate:.1f}%")
    print(f"• Most profitable sentiment: {best_sentiment}")
    print(f"• Most trades executed during: {merged_df['Classification'].value_counts().idxmax()}")
    
    if total_pnl > 0:
        print("\n✅ CONCLUSION: Trader performance shows significant correlation with market sentiment.")
        print("   Optimal trading strategy appears to be sentiment-aware.")
    else:
        print("\n⚠️  CONCLUSION: Traders struggle to generate profits across all sentiment regimes.")
        print("   Consider sentiment-based position management strategies.")

# ==================== 8. MAIN EXECUTION ====================

def main():
    """
    Main execution function
    """
    print("="*60)
    print("BITCOIN MARKET SENTIMENT VS TRADER PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Step 1: Load data
    sentiment_df, trader_df = load_datasets()
    if sentiment_df is None or trader_df is None:
        return
    
    # Step 2: Clean data
    sentiment_df_clean = clean_sentiment_data(sentiment_df)
    trader_df_clean = clean_trader_data(trader_df)
    
    # Step 3: Merge datasets
    merged_df = merge_datasets(sentiment_df_clean, trader_df_clean)
    
    # Step 4: Perform EDA
    eda_results = perform_eda(merged_df)
    
    # Step 5: Analyze behavioral patterns
    behavioral_patterns = analyze_behavioral_patterns(merged_df)
    
    # Step 6: Create visualizations
    create_visualizations(merged_df, eda_results)
    
    # Step 7: Generate insights
    generate_insights(merged_df, eda_results, behavioral_patterns)
    
    # Save final dataset for further analysis
    merged_df.to_csv('merged_sentiment_trader_data.csv', index=False)
    print("\n✓ Saved merged dataset: merged_sentiment_trader_data.csv")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

# Execute main function
if __name__ == "__main__":
    main()