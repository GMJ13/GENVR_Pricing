"""
Visualization module for CVR Greeks
Creates charts and sensitivity surfaces
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from typing import Dict, List
import seaborn as sns

from models.parameters import CVRParams, gbmParams, SimulationParams
from greeks.greeks import CVRGreeksCalculator, GreeksResult

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10


class GreeksVisualizer:
    """Create visualizations for CVR Greeks"""

    def __init__(self,
                 cvr_params: CVRParams,
                 gbm_params: gbmParams,
                 sim_params: SimulationParams):
        self.cvr = cvr_params
        self.gbm = gbm_params
        self.sim = sim_params
        self.calculator = CVRGreeksCalculator(cvr_params, gbm_params, sim_params)

    def plot_greeks_summary(self, greeks: GreeksResult, save_path: str = None):
        """Create comprehensive Greeks summary dashboard"""
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        fig.suptitle('CVR Greeks Dashboard', fontsize=16, fontweight='bold', y=0.98)

        # 1. Greeks bar chart
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_greeks_bars(ax1, greeks)

        # 2. Summary table
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_summary_table(ax2, greeks)

        # 3. Delta profile
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_delta_profile(ax3)

        # 4. Vega profile
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_vega_profile(ax4)

        # 5. Theta decay
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_theta_decay(ax5)

        # 6. Gamma profile
        ax6 = fig.add_subplot(gs[2, 0])
        self._plot_gamma_profile(ax6)

        # 7. Greeks evolution
        ax7 = fig.add_subplot(gs[2, 1:])
        self._plot_greeks_evolution(ax7)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Greeks dashboard saved to {save_path}")

        plt.show()

    def _plot_greeks_bars(self, ax, greeks: GreeksResult):
        """Plot normalized Greeks as bar chart"""
        greeks_dict = {
            'Delta\n(per $1 S)': greeks.delta,
            'Gamma\n(×100)': greeks.gamma * 100,
            'Vega\n(per 1%)': greeks.vega,
            'Theta\n(per day)': greeks.theta,
            'Rho\n(per 100bp)': greeks.rho * 100,
            'Lambda\n(per 5%)': greeks.lambda_greek * 5
        }

        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4B51']

        bars = ax.bar(range(len(greeks_dict)), list(greeks_dict.values()),
                     color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

        ax.set_xticks(range(len(greeks_dict)))
        ax.set_xticklabels(list(greeks_dict.keys()), fontsize=9)
        ax.set_ylabel('Greek Value', fontsize=11, fontweight='bold')
        ax.set_title('First-Order Greeks', fontsize=12, fontweight='bold')
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)

        for bar, val in zip(bars, greeks_dict.values()):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    def _plot_summary_table(self, ax, greeks: GreeksResult):
        """Display summary statistics table"""
        ax.axis('off')

        summary_data = [
            ['CVR Value', f'${greeks.base_value:.2f}'],
            ['', ''],
            ['Delta %', f'{greeks.delta_percent:.1f}%'],
            ['Gamma', f'{greeks.gamma:.6f}'],
            ['Vega %', f'{greeks.vega_percent:.1f}%'],
            ['', ''],
            ['Theta/Day', f'${greeks.theta:.3f}'],
            ['Theta/Week', f'${greeks.theta * 7:.3f}'],
            ['Theta/Month', f'${greeks.theta * 30:.2f}'],
            ['', ''],
            ['Computation', f'{greeks.computation_time:.1f}s']
        ]

        table = ax.table(cellText=summary_data, colWidths=[0.5, 0.5],
                        cellLoc='left', loc='center', bbox=[0, 0, 1, 1])

        table.auto_set_font_size(False)
        table.set_fontsize(9)

        for i in [0, 2, 5, 9]:
            table[(i, 0)].set_facecolor('#E8E8E8')
            table[(i, 1)].set_facecolor('#E8E8E8')
            table[(i, 0)].set_text_props(weight='bold')

        ax.set_title('Summary', fontsize=12, fontweight='bold', pad=10)

    def _plot_delta_profile(self, ax):
        """Plot CVR value vs stock price (Delta profile)"""
        S_range = np.linspace(self.cvr.S0 * 0.7, self.cvr.S0 * 1.4, 15)
        values = []

        print("  Computing Delta profile...")
        for S in S_range:
            cvr_temp = CVRParams(S0=S, barrier=self.cvr.barrier,
                                payoff_shares=self.cvr.payoff_shares,
                                T=self.cvr.T, r=self.cvr.r, q=self.cvr.q,
                                rolling_window=self.cvr.rolling_window,
                                delivery_days=self.cvr.delivery_days)
            value = self.calculator.price_cvr(cvr_temp, self.gbm)
            values.append(value)

        ax.plot(S_range, values, 'b-o', linewidth=2, markersize=4)
        ax.axvline(self.cvr.S0, color='red', linestyle='--',
                  linewidth=1.5, label=f'Current: ${self.cvr.S0:.2f}')
        ax.axvline(self.cvr.barrier, color='green', linestyle='--',
                  linewidth=1.5, label=f'Barrier: ${self.cvr.barrier:.2f}')

        ax.set_xlabel('Stock Price ($)', fontweight='bold')
        ax.set_ylabel('CVR Value ($)', fontweight='bold')
        ax.set_title('Delta Profile', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_vega_profile(self, ax):
        """Plot CVR value vs volatility (Vega profile)"""
        sigma_range = np.linspace(0.20, 0.60, 12)
        values = []

        print("  Computing Vega profile...")
        for sigma in sigma_range:
            gbm_temp = gbmParams(sigma=sigma)
            value = self.calculator.price_cvr(self.cvr, gbm_temp)
            values.append(value)

        ax.plot(sigma_range * 100, values, 'g-o', linewidth=2, markersize=4)
        ax.axvline(self.gbm.sigma * 100, color='red', linestyle='--',
                  linewidth=1.5, label=f'Current: {self.gbm.sigma*100:.0f}%')

        ax.set_xlabel('Volatility (%)', fontweight='bold')
        ax.set_ylabel('CVR Value ($)', fontweight='bold')
        ax.set_title('Vega Profile', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_theta_decay(self, ax):
        """Plot CVR value vs time to expiry (Theta decay)"""
        T_range = np.linspace(0.1, self.cvr.T, 12)
        values = []

        print("  Computing Theta decay...")
        for T in T_range:
            cvr_temp = CVRParams(S0=self.cvr.S0, barrier=self.cvr.barrier,
                                payoff_shares=self.cvr.payoff_shares,
                                T=T, r=self.cvr.r, q=self.cvr.q,
                                rolling_window=self.cvr.rolling_window,
                                delivery_days=self.cvr.delivery_days)
            value = self.calculator.price_cvr(cvr_temp, self.gbm)
            values.append(value)

        ax.plot(T_range * 365, values, 'r-o', linewidth=2, markersize=4)
        ax.axvline(self.cvr.T * 365, color='blue', linestyle='--',
                  linewidth=1.5, label=f'Current: {self.cvr.T*365:.0f} days')

        ax.set_xlabel('Days to Expiry', fontweight='bold')
        ax.set_ylabel('CVR Value ($)', fontweight='bold')
        ax.set_title('Time Decay (Theta)', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()

    def _plot_gamma_profile(self, ax):
        """Plot Gamma (second derivative) profile"""
        S_range = np.linspace(self.cvr.S0 * 0.8, self.cvr.S0 * 1.3, 10)
        gammas = []

        print("  Computing Gamma profile...")
        for S in S_range:
            cvr_temp = CVRParams(S0=S, barrier=self.cvr.barrier,
                                payoff_shares=self.cvr.payoff_shares,
                                T=self.cvr.T, r=self.cvr.r, q=self.cvr.q,
                                rolling_window=self.cvr.rolling_window,
                                delivery_days=self.cvr.delivery_days)
            calc_temp = CVRGreeksCalculator(cvr_temp, self.gbm, self.sim)
            _, gamma, _ = calc_temp.calculate_delta()
            gammas.append(gamma)

        ax.plot(S_range, gammas, 'm-o', linewidth=2, markersize=4)
        ax.axvline(self.cvr.S0, color='red', linestyle='--',
                  linewidth=1.5, label=f'Current: ${self.cvr.S0:.2f}')
        ax.axhline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.3)

        ax.set_xlabel('Stock Price ($)', fontweight='bold')
        ax.set_ylabel('Gamma', fontweight='bold')
        ax.set_title('Gamma Profile (Convexity)', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_greeks_evolution(self, ax):
        """Plot how Greeks evolve as time passes"""
        T_range = np.linspace(self.cvr.T, 0.1, 10)
        deltas, vegas, thetas = [], [], []

        print("  Computing Greeks evolution...")
        for T in T_range:
            cvr_temp = CVRParams(S0=self.cvr.S0, barrier=self.cvr.barrier,
                                payoff_shares=self.cvr.payoff_shares,
                                T=T, r=self.cvr.r, q=self.cvr.q,
                                rolling_window=self.cvr.rolling_window,
                                delivery_days=self.cvr.delivery_days)
            calc_temp = CVRGreeksCalculator(cvr_temp, self.gbm, self.sim)

            delta, _, _ = calc_temp.calculate_delta()
            vega, _, _ = calc_temp.calculate_vega()
            theta = calc_temp.calculate_theta()

            deltas.append(delta)
            vegas.append(vega)
            thetas.append(theta)

        days_to_expiry = T_range * 365

        ax2 = ax.twinx()
        ax3 = ax.twinx()
        ax3.spines['right'].set_position(('outward', 60))

        p1, = ax.plot(days_to_expiry, deltas, 'b-o', linewidth=2,
                     markersize=4, label='Delta')
        p2, = ax2.plot(days_to_expiry, vegas, 'g-s', linewidth=2,
                      markersize=4, label='Vega')
        p3, = ax3.plot(days_to_expiry, thetas, 'r-^', linewidth=2,
                      markersize=4, label='Theta')

        ax.set_xlabel('Days to Expiry', fontweight='bold')
        ax.set_ylabel('Delta', color='b', fontweight='bold')
        ax2.set_ylabel('Vega', color='g', fontweight='bold')
        ax3.set_ylabel('Theta', color='r', fontweight='bold')

        ax.tick_params(axis='y', labelcolor='b')
        ax2.tick_params(axis='y', labelcolor='g')
        ax3.tick_params(axis='y', labelcolor='r')

        ax.set_title('Greeks Evolution Over Time', fontweight='bold')

        lines = [p1, p2, p3]
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left', fontsize=8)

        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()

    def create_sensitivity_heatmap(self, save_path: str = None):
        """Create 2D heatmap of CVR value across stock price and volatility"""
        print("\n" + "="*60)
        print("SENSITIVITY HEATMAP GENERATION")
        print("="*60)

        S_range = np.linspace(self.cvr.S0 * 0.7, self.cvr.S0 * 1.4, 12)
        sigma_range = np.linspace(0.20, 0.55, 12)

        values = np.zeros((len(sigma_range), len(S_range)))

        total = len(S_range) * len(sigma_range)
        count = 0

        print(f"Computing {total} scenarios...")
        for i, sigma in enumerate(sigma_range):
            for j, S in enumerate(S_range):
                count += 1
                if count % 20 == 0:
                    print(f"  Progress: {count}/{total}")

                cvr_temp = CVRParams(S0=S, barrier=self.cvr.barrier,
                                    payoff_shares=self.cvr.payoff_shares,
                                    T=self.cvr.T, r=self.cvr.r, q=self.cvr.q,
                                    rolling_window=self.cvr.rolling_window,
                                    delivery_days=self.cvr.delivery_days)
                gbm_temp = gbmParams(sigma=sigma)

                value = self.calculator.price_cvr(cvr_temp, gbm_temp)
                values[i, j] = value

        fig, ax = plt.subplots(figsize=(12, 8))

        im = ax.imshow(values, cmap='RdYlGn', aspect='auto', origin='lower',
                      extent=[S_range[0], S_range[-1],
                             sigma_range[0]*100, sigma_range[-1]*100])

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('CVR Value ($)', fontsize=12, fontweight='bold')

        contours = ax.contour(S_range, sigma_range*100, values,
                             colors='black', alpha=0.3, linewidths=0.5)
        ax.clabel(contours, inline=True, fontsize=8, fmt='$%.1f')

        ax.plot(self.cvr.S0, self.gbm.sigma*100, 'rx', markersize=15,
               markeredgewidth=3, label='Current Position')

        ax.axvline(self.cvr.barrier, color='blue', linestyle='--',
                  linewidth=2, label=f'Barrier: ${self.cvr.barrier:.2f}')

        ax.set_xlabel('Stock Price ($)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Volatility (%)', fontsize=12, fontweight='bold')
        ax.set_title('CVR Value Sensitivity Heatmap',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.2)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Heatmap saved to {save_path}")

        plt.show()
        print("="*60 + "\n")

    def export_greeks_to_csv(self, greeks: GreeksResult, filepath: str):
        """Export Greeks to CSV for further analysis"""
        data = {
            'Metric': ['CVR Value', 'Delta', 'Gamma', 'Vega', 'Theta',
                      'Rho', 'Lambda', 'Barrier_Sens', 'Vanna', 'Volga'],
            'Value': [greeks.base_value, greeks.delta, greeks.gamma,
                     greeks.vega, greeks.theta, greeks.rho,
                     greeks.lambda_greek, greeks.barrier_sens,
                     greeks.vanna, greeks.volga],
            'Interpretation': [
                'Current fair value',
                f'${greeks.delta:.3f} per $1 stock move',
                'Rate of Delta change',
                f'${greeks.vega:.3f} per 1% vol change',
                f'${greeks.theta:.3f} per day',
                f'${greeks.rho*100:.3f} per 100bp rate change',
                f'${greeks.lambda_greek*5:.3f} per 5% CoC prob change',
                f'${greeks.barrier_sens:.3f} per $1 barrier change',
                'Delta-Vega cross effect',
                'Vol convexity'
            ]
        }

        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        print(f"✓ Greeks exported to {filepath}")


def create_greeks_report(cvr_params: CVRParams,
                        gbm_params: gbmParams,
                        sim_params: SimulationParams,
                        output_dir: str = '.'):
    """Generate complete Greeks report with all visualizations"""
    print("\n" + "="*60)
    print("GENERATING COMPLETE GREEKS REPORT")
    print("="*60)

    from greeks.greeks import get_cvr_greeks

    greeks = get_cvr_greeks(cvr_params, gbm_params, sim_params)

    visualizer = GreeksVisualizer(cvr_params, gbm_params, sim_params)

    print("\n1. Creating Greeks dashboard...")
    visualizer.plot_greeks_summary(greeks,
                                   save_path=f'{output_dir}/greeks_dashboard.png')

    print("\n2. Creating sensitivity heatmap...")
    visualizer.create_sensitivity_heatmap(
        save_path=f'{output_dir}/greeks_heatmap.png')

    print("\n3. Exporting Greeks to CSV...")
    visualizer.export_greeks_to_csv(greeks,
                                   filepath=f'{output_dir}/greeks_data.csv')

    print("\n" + "="*60)
    print("✓ REPORT COMPLETE")
    print("="*60)

    return greeks