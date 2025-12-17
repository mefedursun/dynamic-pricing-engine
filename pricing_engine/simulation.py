"""
Simulation Engine Module - MAB Experiment Orchestration

This module provides the simulation engine for testing Multi-Armed Bandit
algorithms in a controlled environment, collecting metrics for analysis.
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import numpy as np
from tqdm import tqdm
import time

from .environment import MarketEnvironment
from .agents import PricingAgent, BaselineAgent, EpsilonGreedyAgent, ThompsonSamplingAgent


@dataclass
class SimulationResult:
    """
    Data class for storing simulation results.
    
    Attributes:
        agent_name: Name of the agent
        cumulative_rewards: Total reward at each timestep
        cumulative_regrets: Total regret at each timestep
        instant_regrets: Instantaneous regret at each step
        arm_selections: History of selected arms
        sales: Sale outcomes history
        rewards: Instantaneous rewards
        optimal_arm_history: Optimal arm at each step
    """
    agent_name: str
    cumulative_rewards: np.ndarray
    cumulative_regrets: np.ndarray
    instant_regrets: np.ndarray
    arm_selections: np.ndarray
    sales: np.ndarray
    rewards: np.ndarray
    optimal_arm_history: np.ndarray
    total_reward: float = 0.0
    total_regret: float = 0.0
    optimal_selection_rate: float = 0.0
    final_agent_state: Optional[Dict[str, Any]] = None
    
    def compute_statistics(self) -> None:
        """Compute summary statistics."""
        self.total_reward = float(self.cumulative_rewards[-1])
        self.total_regret = float(self.cumulative_regrets[-1])
        optimal_selections = self.arm_selections == self.optimal_arm_history
        self.optimal_selection_rate = float(np.mean(optimal_selections))


@dataclass
class SimulationEngine:
    """
    MAB Simulation Engine - Tests multiple agents in the same environment.
    
    Attributes:
        env: Market environment
        agents: List of agents to test
        n_steps: Total simulation steps (T)
        shock_step: Step at which to inject shock
        shock_params: Shock parameters
    """
    env: MarketEnvironment
    agents: List[PricingAgent] = field(default_factory=list)
    n_steps: int = 10000
    shock_step: Optional[int] = 5000
    shock_params: Dict[str, Any] = field(default_factory=lambda: {
        "price_shift": 15.0,
        "sensitivity_change": 0.05,
        "description": "Hyperinflation Shock"
    })
    seed: int = 42
    
    def add_agent(self, agent: PricingAgent) -> None:
        """Add an agent to the simulation."""
        self.agents.append(agent)
    
    def _create_default_agents(self) -> List[PricingAgent]:
        """Create default agent set."""
        n_arms = len(self.env.price_arms)
        return [
            BaselineAgent(n_arms=n_arms, strategy="random", name="Baseline (Random)"),
            EpsilonGreedyAgent(n_arms=n_arms, epsilon=0.1, name="Epsilon-Greedy"),
            ThompsonSamplingAgent(n_arms=n_arms, price_arms=self.env.price_arms, 
                                  name="Thompson Sampling")
        ]
    
    def run(self, verbose: bool = True) -> Dict[str, SimulationResult]:
        """Run simulation and return results."""
        if not self.agents:
            self.agents = self._create_default_agents()
        
        results: Dict[str, SimulationResult] = {}
        print(f"\n{'='*70}\n[START] SIMULATION - {self.n_steps:,} steps\n{'='*70}\n")
        start_time = time.time()
        
        for agent in self.agents:
            result = self._run_single_agent(agent, verbose)
            results[agent.name] = result
        
        elapsed = time.time() - start_time
        print(f"\n{'='*70}\n[DONE] Completed in {elapsed:.2f}s\n{'='*70}")
        self._print_summary(results)
        return results
    
    def _run_single_agent(self, agent: PricingAgent, verbose: bool) -> SimulationResult:
        """Run simulation for a single agent."""
        self.env.reset(seed=self.seed)
        agent.reset()
        agent.set_seed(self.seed + hash(agent.name) % 1000)
        
        n = self.n_steps
        cumulative_rewards = np.zeros(n)
        cumulative_regrets = np.zeros(n)
        instant_regrets = np.zeros(n)
        arm_selections = np.zeros(n, dtype=int)
        sales = np.zeros(n, dtype=bool)
        rewards = np.zeros(n)
        optimal_arm_history = np.zeros(n, dtype=int)
        
        total_reward, total_regret = 0.0, 0.0
        shock_applied = False
        
        iterator = tqdm(range(n), desc=f"[RUN] {agent.name:<25}", ncols=80) if verbose else range(n)
        
        for t in iterator:
            # Shock check
            if self.shock_step and t == self.shock_step and not shock_applied:
                self.env.inject_shock(timestep=t, **self.shock_params)
                shock_applied = True
            
            optimal_arm = self.env.get_optimal_arm()
            optimal_reward = self.env.get_optimal_expected_reward()
            optimal_arm_history[t] = optimal_arm
            
            arm = agent.select_arm()
            arm_selections[t] = arm
            sale, reward = self.env.get_reward(arm)
            sales[t], rewards[t] = sale, reward
            agent.update(arm, reward, sale)
            
            # Regret = E[optimal] - E[selected]
            selected_exp = self.env.price_arms[arm] * self.env._purchase_probability(self.env.price_arms[arm])
            instant_regrets[t] = optimal_reward - selected_exp
            
            total_reward += reward
            total_regret += instant_regrets[t]
            cumulative_rewards[t] = total_reward
            cumulative_regrets[t] = total_regret
        
        final_state = None
        if isinstance(agent, ThompsonSamplingAgent):
            alphas, betas = agent.get_posterior_params()
            final_state = {"alphas": alphas, "betas": betas}
        
        result = SimulationResult(
            agent_name=agent.name,
            cumulative_rewards=cumulative_rewards,
            cumulative_regrets=cumulative_regrets,
            instant_regrets=instant_regrets,
            arm_selections=arm_selections,
            sales=sales, rewards=rewards,
            optimal_arm_history=optimal_arm_history,
            final_agent_state=final_state
        )
        result.compute_statistics()
        return result
    
    def _print_summary(self, results: Dict[str, SimulationResult]) -> None:
        """Print summary table."""
        print(f"\n[RESULTS] Summary:\n{'-'*70}")
        print(f"{'Agent':<30} {'Total Reward':>12} {'Total Regret':>14} {'Opt. Rate':>12}")
        print("-"*70)
        for name, res in sorted(results.items(), key=lambda x: x[1].total_regret):
            print(f"{name:<30} {res.total_reward:>12,.2f} {res.total_regret:>14,.2f} "
                  f"{res.optimal_selection_rate:>11.2%}")


def run_comparison_experiment(n_steps: int = 10000, shock_step: int = 5000,
                               seed: int = 42, verbose: bool = True) -> Dict[str, SimulationResult]:
    """Run standard comparison experiment."""
    env = MarketEnvironment()
    env.set_seed(seed)
    engine = SimulationEngine(env=env, n_steps=n_steps, shock_step=shock_step, seed=seed)
    return engine.run(verbose=verbose)


if __name__ == "__main__":
    results = run_comparison_experiment(n_steps=1000, shock_step=500, verbose=True)
