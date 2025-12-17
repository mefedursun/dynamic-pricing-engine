"""
Analysis & Visualization Module

This module generates publication-quality plots for simulation results.

Plots Generated:
    1. Cumulative Regret: Algorithm performance comparison
    2. Beta Distributions: Thompson Sampling posterior evolution
    3. Price Selection Heatmap: Price selection behavior over time
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# Set seaborn style - academic appearance
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'


def plot_cumulative_regret(
    results: Dict[str, any],
    shock_step: Optional[int] = 5000,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot cumulative regret over time.
    
    Regret Definition:
        R(T) = Σ_{t=1}^{T} [E[r*] - E[r_t]]
        
    Lower regret = Better performance
    
    Args:
        results: Dictionary of SimulationResult objects
        shock_step: Shock timestep (for vertical line)
        save_path: Path to save figure (None = don't save)
        figsize: Figure dimensions
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color palette
    colors = sns.color_palette("husl", len(results))
    
    for (name, result), color in zip(results.items(), colors):
        ax.plot(
            result.cumulative_regrets,
            label=f"{name} (Final: {result.total_regret:.0f})",
            color=color,
            linewidth=2,
            alpha=0.9
        )
    
    # Shock line
    if shock_step:
        ax.axvline(x=shock_step, color='red', linestyle='--', 
                   linewidth=2, alpha=0.7, label='Market Shock')
    
    ax.set_xlabel('Time Step (t)', fontsize=12)
    ax.set_ylabel('Cumulative Regret', fontsize=12)
    ax.set_title('Cumulative Regret Comparison\n(Lower is Better)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # X-axis format
    ax.ticklabel_format(style='sci', axis='x', scilimits=(3, 3))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', facecolor='white')
        print(f"[SAVED] Regret plot: {save_path}")
    
    return fig


def plot_beta_distributions(
    final_state: Dict[str, np.ndarray],
    true_probs: np.ndarray,
    price_arms: List[float],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 8)
) -> plt.Figure:
    """
    Visualize Thompson Sampling posterior Beta distributions.
    
    Beta Distribution:
        f(θ; α, β) = θ^(α-1) (1-θ)^(β-1) / B(α, β)
        
    Posterior mean: E[θ] = α / (α + β)
    
    Args:
        final_state: {"alphas": [...], "betas": [...]}
        true_probs: True purchase probabilities
        price_arms: Price levels
        save_path: Save path
        figsize: Figure dimensions
        
    Returns:
        matplotlib Figure object
    """
    alphas = final_state["alphas"]
    betas = final_state["betas"]
    n_arms = len(alphas)
    
    # Grid layout
    n_cols = 5
    n_rows = (n_arms + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    x = np.linspace(0, 1, 200)
    
    for i in range(n_arms):
        ax = axes[i]
        a, b = alphas[i], betas[i]
        
        # Beta PDF
        y = stats.beta.pdf(x, a, b)
        ax.fill_between(x, y, alpha=0.3, color='blue')
        ax.plot(x, y, 'b-', linewidth=2)
        
        # True probability
        ax.axvline(true_probs[i], color='red', linestyle='--', 
                   linewidth=2, label=f'True: {true_probs[i]:.2f}')
        
        # Posterior mean
        posterior_mean = a / (a + b)
        ax.axvline(posterior_mean, color='green', linestyle='-', 
                   linewidth=2, label=f'Est: {posterior_mean:.2f}')
        
        ax.set_title(f'Arm {i}: ${price_arms[i]:.0f}\nα={a:.0f}, β={b:.0f}', fontsize=10)
        ax.set_xlabel('θ (Purchase Prob.)', fontsize=8)
        ax.set_xlim(0, 1)
        
        if i == 0:
            ax.legend(fontsize=7, loc='upper right')
    
    # Hide empty subplots
    for i in range(n_arms, len(axes)):
        axes[i].set_visible(False)
    
    fig.suptitle('Thompson Sampling: Posterior Beta Distributions\n(Green=Estimated, Red=True)', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', facecolor='white')
        print(f"[SAVED] Beta distributions: {save_path}")
    
    return fig


def plot_price_selection_heatmap(
    results: Dict[str, any],
    price_arms: List[float],
    n_bins: int = 50,
    shock_step: Optional[int] = 5000,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Plot price selection heatmap - which prices were selected over time.
    
    This visualization shows agent adaptation behavior after shocks.
    
    Args:
        results: Dictionary of SimulationResult objects
        price_arms: Price levels
        n_bins: Number of time bins
        shock_step: Shock timestep
        save_path: Save path
        figsize: Figure dimensions
        
    Returns:
        matplotlib Figure object
    """
    n_agents = len(results)
    fig, axes = plt.subplots(n_agents, 1, figsize=figsize, sharex=True)
    
    if n_agents == 1:
        axes = [axes]
    
    for ax, (name, result) in zip(axes, results.items()):
        selections = result.arm_selections
        n_steps = len(selections)
        n_arms = len(price_arms)
        
        # Create time bins
        bin_size = n_steps // n_bins
        heatmap_data = np.zeros((n_arms, n_bins))
        
        for b in range(n_bins):
            start = b * bin_size
            end = (b + 1) * bin_size if b < n_bins - 1 else n_steps
            bin_selections = selections[start:end]
            
            for arm in range(n_arms):
                heatmap_data[arm, b] = np.sum(bin_selections == arm) / len(bin_selections)
        
        # Draw heatmap
        im = ax.imshow(heatmap_data, aspect='auto', cmap='YlOrRd', 
                       origin='lower', vmin=0, vmax=1)
        
        # Shock line
        if shock_step:
            shock_bin = shock_step // bin_size
            ax.axvline(x=shock_bin, color='cyan', linestyle='--', linewidth=2)
        
        ax.set_ylabel('Price ($)', fontsize=10)
        ax.set_yticks(range(n_arms))
        ax.set_yticklabels([f'${p:.0f}' for p in price_arms])
        ax.set_title(name, fontsize=11, fontweight='bold')
        
        # Colorbar
        plt.colorbar(im, ax=ax, label='Selection Rate', shrink=0.8)
    
    axes[-1].set_xlabel('Time Bins', fontsize=12)
    fig.suptitle('Price Selection Over Time\n(Cyan line = Market Shock)', 
                 fontsize=14, fontweight='bold', y=1.01)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', facecolor='white')
        print(f"[SAVED] Heatmap: {save_path}")
    
    return fig


def generate_all_plots(
    results: Dict[str, any],
    env: any,
    output_dir: str = "output",
    shock_step: Optional[int] = 5000
) -> List[str]:
    """
    Generate and save all analysis plots.
    
    Args:
        results: Simulation results
        env: MarketEnvironment object
        output_dir: Output directory
        shock_step: Shock timestep
        
    Returns:
        List of saved file paths
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    saved_files = []
    
    print("\n" + "="*60)
    print("[PLOT] Generating publication-quality plots...")
    print("="*60)
    
    # 1. Cumulative Regret
    path1 = str(output_path / "cumulative_regret.png")
    plot_cumulative_regret(results, shock_step=shock_step, save_path=path1)
    saved_files.append(path1)
    
    # 2. Beta Distributions (for Thompson Sampling)
    ts_result = None
    for name, result in results.items():
        if "Thompson" in name and result.final_agent_state:
            ts_result = result
            break
    
    if ts_result and ts_result.final_agent_state:
        path2 = str(output_path / "beta_distributions.png")
        plot_beta_distributions(
            final_state=ts_result.final_agent_state,
            true_probs=env.get_true_probabilities(),
            price_arms=env.price_arms,
            save_path=path2
        )
        saved_files.append(path2)
    
    # 3. Price Selection Heatmap
    path3 = str(output_path / "price_selection_heatmap.png")
    plot_price_selection_heatmap(
        results, price_arms=env.price_arms,
        shock_step=shock_step, save_path=path3
    )
    saved_files.append(path3)
    
    print(f"\n[DONE] {len(saved_files)} plots saved to '{output_dir}/'")
    
    return saved_files


if __name__ == "__main__":
    print("Analysis module - Use main.py for direct execution.")
