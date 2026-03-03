import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional

"""
B2B SaaS Funnel Analysis Script
================================
Analyzes marketing and sales funnel performance.
Outputs stage conversion rates, drop-off analysis, 
and revenue attribution by channel.

Usage:
    df = pd.read_csv('your_crm_export.csv')
    analyzer = FunnelAnalyzer(df)
    analyzer.run_full_analysis()
"""


class FunnelAnalyzer:
    """
    Analyzes B2B SaaS conversion funnel from lead to closed-won.
    
    Expected columns in input DataFrame:
    - lead_id: unique identifier
    - lead_date: date lead was created
    - mql_date: date qualified as MQL (nullable)
    - sql_date: date qualified as SQL (nullable)
    - opp_date: date opportunity created (nullable)
    - close_date: date closed won/lost (nullable)
    - status: 'won', 'lost', 'open'
    - channel: acquisition channel
    - acv: annual contract value (for won deals)
    """

    FUNNEL_STAGES = ['lead', 'mql', 'sql', 'opportunity', 'closed_won']
    STAGE_DATE_MAP = {
        'lead': 'lead_date',
        'mql': 'mql_date',
        'sql': 'sql_date',
        'opportunity': 'opp_date',
        'closed_won': 'close_date'
    }

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._preprocess()

    def _preprocess(self):
        """Convert date columns and add derived fields."""
        for stage, col in self.STAGE_DATE_MAP.items():
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
        
        # Add boolean flags for each stage
        self.df['is_mql'] = self.df['mql_date'].notna()
        self.df['is_sql'] = self.df['sql_date'].notna()
        self.df['is_opp'] = self.df['opp_date'].notna()
        self.df['is_won'] = (self.df['status'] == 'won') & self.df['close_date'].notna()

    def conversion_rates(self) -> pd.DataFrame:
        """
        Calculate stage-by-stage conversion rates.
        Returns DataFrame with conversion % for each funnel transition.
        """
        total_leads = len(self.df)
        mqls = self.df['is_mql'].sum()
        sqls = self.df['is_sql'].sum()
        opps = self.df['is_opp'].sum()
        won = self.df['is_won'].sum()

        transitions = [
            ('Lead -> MQL', total_leads, mqls),
            ('MQL -> SQL', mqls, sqls),
            ('SQL -> Opportunity', sqls, opps),
            ('Opportunity -> Closed Won', opps, won),
            ('Lead -> Closed Won (Overall)', total_leads, won),
        ]

        results = []
        for label, numerator_stage, count in transitions:
            rate = (count / numerator_stage * 100) if numerator_stage > 0 else 0
            results.append({
                'Transition': label,
                'From Count': numerator_stage,
                'To Count': count,
                'Conversion Rate (%)': round(rate, 2)
            })

        return pd.DataFrame(results)

    def channel_performance(self) -> pd.DataFrame:
        """
        Break down funnel conversion and revenue by acquisition channel.
        """
        if 'channel' not in self.df.columns:
            raise ValueError("DataFrame must have 'channel' column")

        grouped = self.df.groupby('channel').agg(
            total_leads=('lead_id', 'count'),
            mqls=('is_mql', 'sum'),
            sqls=('is_sql', 'sum'),
            opps=('is_opp', 'sum'),
            won_deals=('is_won', 'sum'),
            total_acv=('acv', lambda x: x[self.df.loc[x.index, 'is_won']].sum())
        ).reset_index()

        grouped['lead_to_close_rate'] = (
            grouped['won_deals'] / grouped['total_leads'] * 100
        ).round(2)

        grouped['avg_deal_size'] = (
            grouped['total_acv'] / grouped['won_deals'].replace(0, np.nan)
        ).round(0)

        return grouped.sort_values('total_acv', ascending=False)

    def velocity_analysis(self) -> pd.DataFrame:
        """
        Calculate average time (in days) spent at each funnel stage.
        """
        self.df['lead_to_mql_days'] = (
            self.df['mql_date'] - self.df['lead_date']
        ).dt.days
        self.df['mql_to_sql_days'] = (
            self.df['sql_date'] - self.df['mql_date']
        ).dt.days
        self.df['sql_to_opp_days'] = (
            self.df['opp_date'] - self.df['sql_date']
        ).dt.days
        self.df['opp_to_close_days'] = (
            self.df['close_date'] - self.df['opp_date']
        ).dt.days

        velocity_metrics = {
            'Lead to MQL': self.df['lead_to_mql_days'].dropna(),
            'MQL to SQL': self.df['mql_to_sql_days'].dropna(),
            'SQL to Opportunity': self.df['sql_to_opp_days'].dropna(),
            'Opportunity to Close': self.df['opp_to_close_days'].dropna(),
        }

        results = []
        for stage, days in velocity_metrics.items():
            if len(days) > 0:
                results.append({
                    'Stage': stage,
                    'Avg Days': round(days.mean(), 1),
                    'Median Days': round(days.median(), 1),
                    'P90 Days': round(days.quantile(0.9), 1),
                    'Sample Size': len(days)
                })

        return pd.DataFrame(results)

    def cohort_conversion(self, period: str = 'month') -> pd.DataFrame:
        """
        Analyze conversion rates by lead creation cohort.
        period: 'month' or 'quarter'
        """
        self.df['cohort'] = self.df['lead_date'].dt.to_period(period[0].upper())

        cohort_data = self.df.groupby('cohort').agg(
            leads=('lead_id', 'count'),
            won=('is_won', 'sum'),
            acv=('acv', lambda x: x[self.df.loc[x.index, 'is_won']].sum())
        ).reset_index()

        cohort_data['close_rate'] = (
            cohort_data['won'] / cohort_data['leads'] * 100
        ).round(2)

        return cohort_data

    def run_full_analysis(self, output_file: Optional[str] = None):
        """Run all analyses and print/save results."""
        print("=" * 60)
        print("B2B SaaS FUNNEL ANALYSIS REPORT")
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"Total Leads Analyzed: {len(self.df):,}")
        print("=" * 60)

        print("\n--- STAGE CONVERSION RATES ---")
        conv = self.conversion_rates()
        print(conv.to_string(index=False))

        print("\n--- CHANNEL PERFORMANCE ---")
        ch = self.channel_performance()
        print(ch.to_string(index=False))

        print("\n--- FUNNEL VELOCITY (Days Per Stage) ---")
        vel = self.velocity_analysis()
        print(vel.to_string(index=False))

        print("\n--- COHORT ANALYSIS (by Month) ---")
        cohort = self.cohort_conversion(period='month')
        print(cohort.to_string(index=False))

        if output_file:
            with pd.ExcelWriter(output_file) as writer:
                self.conversion_rates().to_excel(writer, sheet_name='Conversion Rates', index=False)
                self.channel_performance().to_excel(writer, sheet_name='Channel Performance', index=False)
                self.velocity_analysis().to_excel(writer, sheet_name='Velocity', index=False)
                self.cohort_conversion().to_excel(writer, sheet_name='Cohorts', index=False)
            print(f"\nReport saved to: {output_file}")


if __name__ == '__main__':
    # Example with synthetic data
    np.random.seed(42)
    n = 1000

    lead_dates = pd.date_range('2024-01-01', periods=n, freq='8H')
    channels = np.random.choice(
        ['Google Search', 'LinkedIn', 'Content/SEO', 'Outbound', 'Referral'],
        n, p=[0.30, 0.25, 0.20, 0.15, 0.10]
    )

    # Simulate conversion probabilities by channel
    channel_conversion = {
        'Google Search': 0.08,
        'LinkedIn': 0.06,
        'Content/SEO': 0.05,
        'Outbound': 0.12,
        'Referral': 0.20
    }

    records = []
    for i, (date, channel) in enumerate(zip(lead_dates, channels)):
        p_win = channel_conversion[channel]
        mql = np.random.random() < 0.35
        sql = mql and np.random.random() < 0.45
        opp = sql and np.random.random() < 0.60
        won = opp and np.random.random() < p_win / 0.35 / 0.45 / 0.60

        records.append({
            'lead_id': f'L{i:04d}',
            'lead_date': date,
            'mql_date': date + pd.Timedelta(days=np.random.randint(1, 14)) if mql else None,
            'sql_date': date + pd.Timedelta(days=np.random.randint(7, 30)) if sql else None,
            'opp_date': date + pd.Timedelta(days=np.random.randint(14, 45)) if opp else None,
            'close_date': date + pd.Timedelta(days=np.random.randint(30, 90)) if won else None,
            'status': 'won' if won else ('lost' if opp else 'open'),
            'channel': channel,
            'acv': np.random.choice([12000, 24000, 36000, 60000]) if won else 0
        })

    df = pd.DataFrame(records)
    analyzer = FunnelAnalyzer(df)
    analyzer.run_full_analysis(output_file='funnel_report.xlsx')
