"""
Real-Time Algorithmic Pricing Engine - Main Entry Point

This script runs the complete MAB algorithm comparison simulation
and generates publication-quality analysis plots.

Usage:
    python main.py
    
Output:
    - Console: Simulation progress and results
    - output/: Saved visualization plots
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pricing_engine.environment import MarketEnvironment
from pricing_engine.agents import BaselineAgent, EpsilonGreedyAgent, ThompsonSamplingAgent
from pricing_engine.simulation import SimulationEngine
from pricing_engine.analysis import generate_all_plots


def main():
    """
    Main simulation and analysis function.
    
    Experiment Design:
    ------------------
    1. 10 price levels: $5, $10, ..., $50
    2. T = 10,000 time steps
    3. Hyperinflation shock at t = 5,000
    4. 3 algorithms: Baseline, Epsilon-Greedy, Thompson Sampling
    
    After the shock, the optimal price shifts upward, testing the
    algorithms' adaptation capacity.
    """
    print("\n" + "="*70)
    print(" REAL-TIME ALGORITHMIC PRICING ENGINE")
    print(" Multi-Armed Bandit Simulation Framework")
    print("="*70)
    
    # ============================================================
    # 1. CREATE ENVIRONMENT
    # ============================================================
    print("\n[INIT] Creating market environment...")
    
    env = MarketEnvironment(
        price_arms=[5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0],
        optimal_price=25.0,      # Initial optimal price
        sensitivity=0.15,        # Demand sensitivity
        noise_std=0.0            # No observation noise
    )
    
    print(f"   {env}")
    print(f"   Initial optimal arm: {env.get_optimal_arm()} "
          f"(${env.price_arms[env.get_optimal_arm()]:.2f})")
    print(f"   Initial optimal expected reward: {env.get_optimal_expected_reward():.2f}")
    
    # ============================================================
    # 2. CREATE AGENTS
    # ============================================================
    print("\n[INIT] Creating agents...")
    
    n_arms = len(env.price_arms)
    
    agents = [
        # Control group - no learning
        BaselineAgent(
            n_arms=n_arms,
            strategy="random",
            name="Baseline (Random)"
        ),
        
        # Classic epsilon-greedy approach
        EpsilonGreedyAgent(
            n_arms=n_arms,
            epsilon=0.1,
            epsilon_decay=0.9999,
            epsilon_min=0.01,
            name="Epsilon-Greedy"
        ),
        
        # Bayesian Thompson Sampling - the star player
        ThompsonSamplingAgent(
            n_arms=n_arms,
            price_arms=env.price_arms,
            name="Thompson Sampling"
        )
    ]
    
    for agent in agents:
        print(f"   + {agent.name}")
    
    # ============================================================
    # 3. CONFIGURE SIMULATION ENGINE
    # ============================================================
    print("\n[INIT] Configuring simulation engine...")
    
    engine = SimulationEngine(
        env=env,
        agents=agents,
        n_steps=10000,           # Total steps
        shock_step=5000,         # Shock at midpoint
        shock_params={
            "price_shift": 15.0,           # Optimal price shifts by +$15
            "sensitivity_change": 0.05,    # Sensitivity increases
            "description": "Hyperinflation Shock"
        },
        seed=42                  # For reproducibility
    )
    
    print(f"   Total steps: {engine.n_steps:,}")
    print(f"   Shock step: {engine.shock_step}")
    
    # ============================================================
    # 4. RUN SIMULATION
    # ============================================================
    results = engine.run(verbose=True)
    
    # ============================================================
    # 5. GENERATE PLOTS
    # ============================================================
    output_dir = project_root / "output"
    saved_files = generate_all_plots(
        results=results,
        env=env,
        output_dir=str(output_dir),
        shock_step=engine.shock_step
    )
    
    # ============================================================
    # 6. DETAILED ANALYSIS
    # ============================================================
    print("\n" + "="*70)
    print("[ANALYSIS] Detailed Performance Breakdown")
    print("="*70)
    
    # Pre-shock vs post-shock comparison
    shock_step = engine.shock_step
    
    print("\n[COMPARE] Pre-Shock vs Post-Shock Performance:")
    print("-"*70)
    
    for name, result in results.items():
        pre_shock_regret = result.cumulative_regrets[shock_step - 1]
        post_shock_regret = result.cumulative_regrets[-1] - pre_shock_regret
        
        pre_optimal_rate = (
            result.arm_selections[:shock_step] == 
            result.optimal_arm_history[:shock_step]
        ).mean()
        
        post_optimal_rate = (
            result.arm_selections[shock_step:] == 
            result.optimal_arm_history[shock_step:]
        ).mean()
        
        print(f"\n{name}:")
        print(f"   Pre-Shock Regret: {pre_shock_regret:,.2f}")
        print(f"   Post-Shock Regret: {post_shock_regret:,.2f}")
        print(f"   Pre-Shock Optimal Rate: {pre_optimal_rate:.2%}")
        print(f"   Post-Shock Optimal Rate: {post_optimal_rate:.2%}")
    
    # Thompson Sampling posterior analysis
    ts_result = results.get("Thompson Sampling")
    if ts_result and ts_result.final_agent_state:
        print("\n" + "-"*70)
        print("[POSTERIOR] Thompson Sampling Final State:")
        print("-"*70)
        
        alphas = ts_result.final_agent_state["alphas"]
        betas = ts_result.final_agent_state["betas"]
        estimates = alphas / (alphas + betas)
        true_probs = env.get_true_probabilities()
        
        print(f"\n{'Arm':<6} {'Price':<8} {'Alpha':<8} {'Beta':<8} {'Est.':<10} {'True':<10} {'Error':<10}")
        print("-"*70)
        
        for i, price in enumerate(env.price_arms):
            error = abs(estimates[i] - true_probs[i])
            print(f"{i:<6} ${price:<7.0f} {alphas[i]:<8.0f} {betas[i]:<8.0f} "
                  f"{estimates[i]:<10.3f} {true_probs[i]:<10.3f} {error:<10.4f}")
    
    print("\n" + "="*70)
    print("[COMPLETE] Simulation and analysis finished")
    print(f"   Plots saved to: {output_dir}/")
    print("="*70 + "\n")
    
    return results


if __name__ == "__main__":
    main()
