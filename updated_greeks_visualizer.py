"""
CORRECTED Greeks Profile Visualization
Shows Greek values vs their parameters (not CVR value vs parameters)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

from models.parameters import CVRParams, gbmParams, SimulationParams
from greeks.greeks import CVRGreeksCalculator, GreeksResult

sns.set_style("whitegrid")


class CorrectedGreeksVisualizer:
    """
    Corrected visualization showing Greeks vs their parameters
    """

    def __init__(self,
                 cvr_params: CVRParams,
                 gbm_params: gbmParams,
                 sim_params: SimulationParams):
        self.cvr = cvr_params
        self.gbm = gbm_params
        self.sim = sim_params
        self.calculator = CVRGreeksCalculator(cvr_params, gbm_params, sim_params)

    def plot_corrected_greeks_dashboard(self, save_path: str = None):
        """
        Create corrected Greeks profile dashboard
        Shows: Greek value vs parameter (NOT CVR value vs parameter)
        """
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)

        fig.suptitle('CVR Greeks Profiles - CORRECTED', fontsize=16, fontweight='bold', y=0.98)

        # Row 1: Delta and Gamma vs Stock Price
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_delta_vs_stock(ax1)

        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_gamma_vs_stock(ax2)

        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_delta_gamma_combined(ax3)

        # Row 2: Vega and Volga vs Volatility
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_vega_vs_volatility(ax4)

        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_volga_vs_volatility(ax5)

        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_vanna_vs_stock(ax6)

        # Row 3: Theta and others
        ax7 = fig.add_subplot(gs[2, 0])
        self._plot_theta_vs_time(ax7)

        ax8 = fig.add_subplot(gs[2, 1])
        self._plot_rho_vs_rate(ax8)

        ax9 = fig.add_subplot(gs[2, 2])
        self._plot_all_greeks_vs_stock(ax9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Corrected Greeks dashboard saved to {save_path}")

        plt.show()

    def _plot_delta_vs_stock(self, ax):
        """Plot Delta (y-axis) vs Stock Price (x-axis)"""
        S_range = np.linspace(self.cvr.S0 * 0.7, self.cvr.S0 * 1.4, 15)
        deltas = []

        print("  Computing Delta profile...")
        for S in S_range:
            cvr_temp = CVRParams(S0=S, barrier=self.cvr.barrier,
                                 payoff_shares=self.cvr.payoff_shares,
                                 T=self.cvr.T, r=self.cvr.r, q=self.cvr.q,
                                 rolling_window=self.cvr.rolling_window,
                                 delivery_days=self.cvr.delivery_days)
            calc_temp = CVRGreeksCalculator(cvr_temp, self.gbm, self.sim)
            delta, _, _ = calc_temp.calculate_delta()
            deltas.append(delta)

        ax.plot(S_range, deltas, 'b-o', linewidth=2.5, markersize=6)
        ax.axvline(self.cvr.S0, color='red', linestyle='--',
                   linewidth=2, label=f'Current: ${self.cvr.S0:.2f}')
        ax.axvline(self.cvr.barrier, color='green', linestyle='--',
                   linewidth=2, label=f'Barrier: ${self.cvr.barrier:.2f}')
        ax.axhline(0, color='black', linewidth=0.8, alpha=0.3)

        ax.set_xlabel('Stock Price ($)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Delta (∂V/∂S)', fontsize=11, fontweight='bold')
        ax.set_title('Delta Profile', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Annotate current delta
        current_delta = deltas[np.argmin(np.abs(S_range - self.cvr.S0))]
        ax.annotate(f'Δ = {current_delta:.4f}',
                    xy=(self.cvr.S0, current_delta),
                    xytext=(self.cvr.S0 + 2, current_delta + 0.05),
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

    def _plot_gamma_vs_stock(self, ax):
        """Plot Gamma (y-axis) vs Stock Price (x-axis)"""
        S_range = np.linspace(self.cvr.S0 * 0.7, self.cvr.S0 * 1.4, 15)
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

        ax.plot(S_range, gammas, 'm-o', linewidth=2.5, markersize=6)
        ax.axvline(self.cvr.S0, color='red', linestyle='--',
                   linewidth=2, label=f'Current: ${self.cvr.S0:.2f}')
        ax.axvline(self.cvr.barrier, color='green', linestyle='--',
                   linewidth=2, label=f'Barrier: ${self.cvr.barrier:.2f}')
        ax.axhline(0, color='black', linewidth=0.8, alpha=0.3)

        ax.set_xlabel('Stock Price ($)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Gamma (∂²V/∂S²)', fontsize=11, fontweight='bold')
        ax.set_title('Gamma Profile (Convexity)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Annotate peak
        max_gamma_idx = np.argmax(gammas)
        ax.annotate(f'Peak Γ = {gammas[max_gamma_idx]:.6f}',
                    xy=(S_range[max_gamma_idx], gammas[max_gamma_idx]),
                    xytext=(S_range[max_gamma_idx] + 2, gammas[max_gamma_idx] * 1.1),
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', color='black'))

    def _plot_delta_gamma_combined(self, ax):
        """Plot Delta and Gamma together with dual y-axis"""
        S_range = np.linspace(self.cvr.S0 * 0.7, self.cvr.S0 * 1.4, 15)
        deltas = []
        gammas = []

        print("  Computing Delta-Gamma combined...")
        for S in S_range:
            cvr_temp = CVRParams(S0=S, barrier=self.cvr.barrier,
                                 payoff_shares=self.cvr.payoff_shares,
                                 T=self.cvr.T, r=self.cvr.r, q=self.cvr.q,
                                 rolling_window=self.cvr.rolling_window,
                                 delivery_days=self.cvr.delivery_days)
            calc_temp = CVRGreeksCalculator(cvr_temp, self.gbm, self.sim)
            delta, gamma, _ = calc_temp.calculate_delta()
            deltas.append(delta)
            gammas.append(gamma)

        ax2 = ax.twinx()

        p1, = ax.plot(S_range, deltas, 'b-o', linewidth=2.5, markersize=5, label='Delta')
        p2, = ax2.plot(S_range, gammas, 'm-s', linewidth=2.5, markersize=5, label='Gamma')

        ax.axvline(self.cvr.S0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axvline(self.cvr.barrier, color='green', linestyle='--', linewidth=1.5, alpha=0.7)

        ax.set_xlabel('Stock Price ($)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Delta', color='b', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Gamma', color='m', fontsize=11, fontweight='bold')

        ax.tick_params(axis='y', labelcolor='b')
        ax2.tick_params(axis='y', labelcolor='m')

        ax.set_title('Delta & Gamma Together', fontsize=12, fontweight='bold')

        # Combined legend
        lines = [p1, p2]
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left', fontsize=9)

        ax.grid(True, alpha=0.3)

    def _plot_vega_vs_volatility(self, ax):
        """Plot Vega (y-axis) vs Volatility (x-axis)"""
        sigma_range = np.linspace(0.20, 0.60, 15)
        vegas = []

        print("  Computing Vega profile...")
        for sigma in sigma_range:
            gbm_temp = gbmParams(sigma=sigma)
            calc_temp = CVRGreeksCalculator(self.cvr, gbm_temp, self.sim)
            vega, _, _ = calc_temp.calculate_vega()
            vegas.append(vega)

        ax.plot(sigma_range * 100, vegas, 'g-o', linewidth=2.5, markersize=6)
        ax.axvline(self.gbm.sigma * 100, color='red', linestyle='--',
                   linewidth=2, label=f'Current: {self.gbm.sigma * 100:.0f}%')
        ax.axhline(0, color='black', linewidth=0.8, alpha=0.3)

        ax.set_xlabel('Volatility (%)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Vega (∂V/∂σ)', fontsize=11, fontweight='bold')
        ax.set_title('Vega Profile', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Annotate current vega
        current_vega = vegas[np.argmin(np.abs(sigma_range - self.gbm.sigma))]
        ax.annotate(f'ν = {current_vega:.2f}',
                    xy=(self.gbm.sigma * 100, current_vega),
                    xytext=(self.gbm.sigma * 100 + 5, current_vega + 2),
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

    def _plot_volga_vs_volatility(self, ax):
        """Plot Volga (y-axis) vs Volatility (x-axis)"""
        sigma_range = np.linspace(0.20, 0.60, 15)
        volgas = []

        print("  Computing Volga profile...")
        for sigma in sigma_range:
            gbm_temp = gbmParams(sigma=sigma)
            calc_temp = CVRGreeksCalculator(self.cvr, gbm_temp, self.sim)
            _, volga, _ = calc_temp.calculate_vega()
            volgas.append(volga)

        ax.plot(sigma_range * 100, volgas, 'c-o', linewidth=2.5, markersize=6)
        ax.axvline(self.gbm.sigma * 100, color='red', linestyle='--',
                   linewidth=2, label=f'Current: {self.gbm.sigma * 100:.0f}%')
        ax.axhline(0, color='black', linewidth=0.8, alpha=0.3)

        ax.set_xlabel('Volatility (%)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Volga (∂²V/∂σ²)', fontsize=11, fontweight='bold')
        ax.set_title('Volga Profile (Vega Convexity)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    def _plot_vanna_vs_stock(self, ax):
        """Plot Vanna (y-axis) vs Stock Price (x-axis)"""
        S_range = np.linspace(self.cvr.S0 * 0.7, self.cvr.S0 * 1.4, 12)
        vannas = []

        print("  Computing Vanna profile...")
        for S in S_range:
            cvr_temp = CVRParams(S0=S, barrier=self.cvr.barrier,
                                 payoff_shares=self.cvr.payoff_shares,
                                 T=self.cvr.T, r=self.cvr.r, q=self.cvr.q,
                                 rolling_window=self.cvr.rolling_window,
                                 delivery_days=self.cvr.delivery_days)
            calc_temp = CVRGreeksCalculator(cvr_temp, self.gbm, self.sim)
            _, _, vanna = calc_temp.calculate_vega()
            vannas.append(vanna)

        ax.plot(S_range, vannas, 'orange', marker='o', linewidth=2.5, markersize=6)
        ax.axvline(self.cvr.S0, color='red', linestyle='--', linewidth=2)
        ax.axvline(self.cvr.barrier, color='green', linestyle='--', linewidth=2)
        ax.axhline(0, color='black', linewidth=0.8, alpha=0.3)

        ax.set_xlabel('Stock Price ($)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Vanna (∂²V/∂S∂σ)', fontsize=11, fontweight='bold')
        ax.set_title('Vanna Profile (Delta-Vega Cross)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    def _plot_theta_vs_time(self, ax):
        """Plot Theta (y-axis) vs Time to Expiry (x-axis)"""
        T_range = np.linspace(0.1, self.cvr.T, 15)
        thetas = []

        print("  Computing Theta profile...")
        for T in T_range:
            cvr_temp = CVRParams(S0=self.cvr.S0, barrier=self.cvr.barrier,
                                 payoff_shares=self.cvr.payoff_shares,
                                 T=T, r=self.cvr.r, q=self.cvr.q,
                                 rolling_window=self.cvr.rolling_window,
                                 delivery_days=self.cvr.delivery_days)
            calc_temp = CVRGreeksCalculator(cvr_temp, self.gbm, self.sim)
            theta = calc_temp.calculate_theta()
            thetas.append(theta)

        ax.plot(T_range * 365, thetas, 'r-o', linewidth=2.5, markersize=6)
        ax.axvline(self.cvr.T * 365, color='blue', linestyle='--',
                   linewidth=2, label=f'Current: {self.cvr.T * 365:.0f} days')
        ax.axhline(0, color='black', linewidth=0.8, alpha=0.3)

        ax.set_xlabel('Days to Expiry', fontsize=11, fontweight='bold')
        ax.set_ylabel('Theta (∂V/∂t per day)', fontsize=11, fontweight='bold')
        ax.set_title('Theta Profile (Time Decay)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()

        # Annotate current theta
        current_theta = thetas[-1]
        ax.annotate(f'θ = {current_theta:.4f}/day',
                    xy=(self.cvr.T * 365, current_theta),
                    xytext=(self.cvr.T * 365 - 100, current_theta - 0.005),
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

    def _plot_rho_vs_rate(self, ax):
        """Plot Rho (y-axis) vs Interest Rate (x-axis)"""
        r_range = np.linspace(0.01, 0.08, 15)
        rhos = []

        print("  Computing Rho profile...")
        for r in r_range:
            cvr_temp = CVRParams(S0=self.cvr.S0, barrier=self.cvr.barrier,
                                 payoff_shares=self.cvr.payoff_shares,
                                 T=self.cvr.T, r=r, q=self.cvr.q,
                                 rolling_window=self.cvr.rolling_window,
                                 delivery_days=self.cvr.delivery_days)
            calc_temp = CVRGreeksCalculator(cvr_temp, self.gbm, self.sim)
            rho = calc_temp.calculate_rho()
            rhos.append(rho)

        ax.plot(r_range * 100, rhos, color='brown', marker='o', linewidth=2.5, markersize=6)
        ax.axvline(self.cvr.r * 100, color='red', linestyle='--',
                   linewidth=2, label=f'Current: {self.cvr.r * 100:.1f}%')
        ax.axhline(0, color='black', linewidth=0.8, alpha=0.3)

        ax.set_xlabel('Interest Rate (%)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Rho (∂V/∂r)', fontsize=11, fontweight='bold')
        ax.set_title('Rho Profile (Rate Sensitivity)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    def _plot_all_greeks_vs_stock(self, ax):
        """Plot multiple normalized Greeks vs Stock Price"""
        S_range = np.linspace(self.cvr.S0 * 0.7, self.cvr.S0 * 1.4, 12)
        deltas, gammas, vegas = [], [], []

        print("  Computing all Greeks vs stock...")
        for S in S_range:
            cvr_temp = CVRParams(S0=S, barrier=self.cvr.barrier,
                                 payoff_shares=self.cvr.payoff_shares,
                                 T=self.cvr.T, r=self.cvr.r, q=self.cvr.q,
                                 rolling_window=self.cvr.rolling_window,
                                 delivery_days=self.cvr.delivery_days)
            calc_temp = CVRGreeksCalculator(cvr_temp, self.gbm, self.sim)
            delta, gamma, _ = calc_temp.calculate_delta()
            vega, _, _ = calc_temp.calculate_vega()
            deltas.append(delta)
            gammas.append(gamma * 10)  # Scale for visibility
            vegas.append(vega / 10)  # Scale for visibility

        ax.plot(S_range, deltas, 'b-o', linewidth=2, markersize=5, label='Delta')
        ax.plot(S_range, gammas, 'm-s', linewidth=2, markersize=5, label='Gamma (×10)')
        ax.plot(S_range, vegas, 'g-^', linewidth=2, markersize=5, label='Vega (÷10)')

        ax.axvline(self.cvr.S0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axvline(self.cvr.barrier, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axhline(0, color='black', linewidth=0.8, alpha=0.3)

        ax.set_xlabel('Stock Price ($)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Normalized Greek Value', fontsize=11, fontweight='bold')
        ax.set_title('All Greeks vs Stock Price', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)


def create_corrected_greeks_report(cvr_params: CVRParams,
                                   gbm_params: gbmParams,
                                   sim_params: SimulationParams,
                                   output_dir: str = '.'):
    """Generate corrected Greeks profile report"""
    print("\n" + "=" * 60)
    print("GENERATING CORRECTED GREEKS PROFILE REPORT")
    print("=" * 60)
    print("Showing: Greek values vs parameters (NOT CVR value)")

    visualizer = CorrectedGreeksVisualizer(cvr_params, gbm_params, sim_params)

    visualizer.plot_corrected_greeks_dashboard(
        save_path=f'{output_dir}/greeks_profiles_corrected.png'
    )

    print("\n" + "=" * 60)
    print("✓ CORRECTED REPORT COMPLETE")
    print("=" * 60)
    print(f"File saved: {output_dir}/greeks_profiles_corrected.png")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    from models.parameters import CVRParams, gbmParams, SimulationParams

    cvr_params = CVRParams(
        S0=26.47,
        barrier=37.50,
        payoff_shares=0.7546,
        T=1.52,
        r=0.0405,
        q=0.0185,
        rolling_window=30,
        delivery_days=12
    )
    # GBM model parameters
    gbm_params = gbmParams(
        sigma=0.262871508582369  # Volatility - UPDATE from calibration or historical
    )

    sim_params = SimulationParams(
        n_paths=50000,  # Number of Monte Carlo paths
        coc_intensity=0,  # 15% annual CoC probability
        coc_premium=1,  # 25% takeover premium
        seed=22,  # Random seed for reproducibility
        n_jobs=-1  # Use all CPU cores
    )
    create_corrected_greeks_report(cvr_params, gbm_params, sim_params, output_dir='.')