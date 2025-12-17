"""
Pricing Agents Module

This module contains Multi-Armed Bandit agents for the dynamic pricing problem.
Each agent implements a different exploration-exploitation strategy.

Agents Implemented:
    1. BaselineAgent: Random or fixed price selection (control group)
    2. EpsilonGreedyAgent: Classic ε-greedy exploration strategy
    3. ThompsonSamplingAgent: Bayesian approach with Beta-Bernoulli model

Mathematical Background:
    The Multi-Armed Bandit (MAB) problem aims to optimize the trade-off 
    between exploration and exploitation:
    
    - Exploration: Try unknown arms to gather information
    - Exploitation: Use the arm that appears best so far
    
    Thompson Sampling solves this balance using Bayesian probability:
    A posterior distribution is maintained for each arm, and decisions
    are made by sampling from these distributions at each step.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Literal, Tuple
import numpy as np
from dataclasses import dataclass, field


@dataclass
class PricingAgent(ABC):
    """
    Abstract Base Class for Pricing Agents.
    
    This abstract class defines the common interface for all MAB agents.
    Subclasses must implement select_arm() and update() methods.
    
    In the Multi-Armed Bandit framework:
        - Arms correspond to discrete price levels
        - Rewards are revenues (price × sale indicator)
        - The agent learns which price maximizes expected revenue
    
    Attributes:
        n_arms: Number of arms (price levels)
        name: Agent name (for visualization)
    """
    
    n_arms: int
    name: str = "BaseAgent"
    
    # Random number generator
    _rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(42),
        repr=False
    )
    
    def set_seed(self, seed: int) -> None:
        """Re-seed the random number generator."""
        self._rng = np.random.default_rng(seed)
    
    @abstractmethod
    def select_arm(self) -> int:
        """
        Select an arm (price).
        
        Returns:
            Index of the selected arm [0, n_arms-1]
        """
        pass
    
    @abstractmethod
    def update(self, arm: int, reward: float, sale: bool) -> None:
        """
        Update agent's beliefs based on observed reward.
        
        Args:
            arm: Index of the selected arm
            reward: Observed reward (0 or price)
            sale: Whether a sale occurred
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset agent to initial state."""
        pass
    
    @abstractmethod
    def get_arm_estimates(self) -> np.ndarray:
        """
        Return estimated value for each arm.
        
        Returns:
            Array of estimates for each arm
        """
        pass


@dataclass
class BaselineAgent(PricingAgent):
    """
    Baseline (Control) Agent.
    
    This agent implements simple strategies for comparison purposes.
    It performs no learning - either random selection or fixed price.
    
    Use Cases:
        - "random": Lower bound benchmark - no learning whatsoever
        - "fixed": Single price strategy - regardless of market conditions
    
    Attributes:
        strategy: Either "random" or "fixed"
        fixed_arm: Arm index to use for the fixed strategy
    """
    
    strategy: Literal["random", "fixed"] = "random"
    fixed_arm: int = 0
    name: str = "Baseline"
    
    # Statistics tracking
    _arm_counts: np.ndarray = field(default=None, repr=False)
    _arm_rewards: np.ndarray = field(default=None, repr=False)
    
    def __post_init__(self):
        """Initialize numpy arrays."""
        self._arm_counts = np.zeros(self.n_arms)
        self._arm_rewards = np.zeros(self.n_arms)
    
    def select_arm(self) -> int:
        """
        Select arm according to strategy.
        
        - random: Uniform random selection
        - fixed: Always the same arm
        
        Returns:
            Index of the selected arm
        """
        if self.strategy == "random":
            # Uniform random selection - uses no information
            return int(self._rng.integers(0, self.n_arms))
        else:
            # Fixed arm - always use the same price
            return self.fixed_arm
    
    def update(self, arm: int, reward: float, sale: bool) -> None:
        """
        Update statistics (no learning, just record-keeping).
        
        Args:
            arm: Selected arm
            reward: Observed reward
            sale: Sale status
        """
        self._arm_counts[arm] += 1
        self._arm_rewards[arm] += reward
    
    def reset(self) -> None:
        """Reset counters."""
        self._arm_counts = np.zeros(self.n_arms)
        self._arm_rewards = np.zeros(self.n_arms)
    
    def get_arm_estimates(self) -> np.ndarray:
        """
        Return average reward estimate for each arm.
        
        Returns:
            Empirical average rewards
        """
        # Prevent division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            estimates = np.where(
                self._arm_counts > 0,
                self._arm_rewards / self._arm_counts,
                0.0
            )
        return estimates


@dataclass
class EpsilonGreedyAgent(PricingAgent):
    """
    Epsilon-Greedy Agent.
    
    Classic exploration-exploitation balancing strategy.
    At each step, explore randomly with probability ε, otherwise
    select the arm with highest estimated value (1-ε probability).
    
    Mathematical Formulation:
    -------------------------
    Action selection at time t:
        
        a_t = { random arm                with probability ε
              { argmax_a Q̂(a)             with probability 1-ε
              
    where Q̂(a) = (Σ rewards from arm a) / (times arm a was pulled)
    
    Epsilon decay:
        ε_t = max(ε_min, ε_0 × decay^t)
        
    This allows more exploration early on and more exploitation later.
    
    Attributes:
        epsilon: Exploration rate ε ∈ [0, 1]
        epsilon_decay: Multiplicative factor for ε decay per step
        epsilon_min: Minimum ε value
    """
    
    epsilon: float = 0.1
    epsilon_decay: float = 0.9999
    epsilon_min: float = 0.01
    name: str = "Epsilon-Greedy"
    
    # Internal state
    _current_epsilon: float = field(default=None, repr=False)
    _arm_counts: np.ndarray = field(default=None, repr=False)
    _arm_rewards: np.ndarray = field(default=None, repr=False)
    _total_pulls: int = field(default=0, repr=False)
    
    def __post_init__(self):
        """Initialize internal state."""
        self._current_epsilon = self.epsilon
        self._arm_counts = np.zeros(self.n_arms)
        self._arm_rewards = np.zeros(self.n_arms)
        self._total_pulls = 0
    
    def select_arm(self) -> int:
        """
        Select arm using ε-greedy strategy.
        
        Algorithm:
            1. Generate random number in [0, 1)
            2. If random < ε: Explore randomly
            3. If random >= ε: Select arm with highest Q̂(a)
            
        Tie-breaking: Random selection among tied arms.
        
        Returns:
            Index of the selected arm
        """
        # Explore vs Exploit decision
        if self._rng.random() < self._current_epsilon:
            # EXPLORATION: Select random arm
            return int(self._rng.integers(0, self.n_arms))
        else:
            # EXPLOITATION: Select best estimated arm
            estimates = self.get_arm_estimates()
            
            # Handle ties with random selection
            max_value = np.max(estimates)
            best_arms = np.where(estimates == max_value)[0]
            
            return int(self._rng.choice(best_arms))
    
    def update(self, arm: int, reward: float, sale: bool) -> None:
        """
        Update Q-value estimates and epsilon.
        
        Update Rule (Incremental Mean):
            n(a) = n(a) + 1
            Q̂(a) = Q̂(a) + (reward - Q̂(a)) / n(a)
            
        Epsilon Decay:
            ε = max(ε_min, ε × decay)
        
        Args:
            arm: Selected arm
            reward: Observed reward
            sale: Sale status (not used by this agent)
        """
        # Update statistics
        self._arm_counts[arm] += 1
        self._arm_rewards[arm] += reward
        self._total_pulls += 1
        
        # Apply epsilon decay
        self._current_epsilon = max(
            self.epsilon_min,
            self._current_epsilon * self.epsilon_decay
        )
    
    def reset(self) -> None:
        """Reset agent state."""
        self._current_epsilon = self.epsilon
        self._arm_counts = np.zeros(self.n_arms)
        self._arm_rewards = np.zeros(self.n_arms)
        self._total_pulls = 0
    
    def get_arm_estimates(self) -> np.ndarray:
        """
        Return Q̂(a) estimate for each arm.
        
        Formula:
            Q̂(a) = Σ(rewards from a) / n(a)
            
        Returns 0 for unpulled arms (extend class for optimistic
        initialization if desired).
        
        Returns:
            Array of average reward estimates for each arm
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            estimates = np.where(
                self._arm_counts > 0,
                self._arm_rewards / self._arm_counts,
                0.0
            )
        return estimates
    
    def get_current_epsilon(self) -> float:
        """Return current epsilon value."""
        return self._current_epsilon


@dataclass
class ThompsonSamplingAgent(PricingAgent):
    """
    Thompson Sampling Agent - Bayesian Multi-Armed Bandit.
    
    This agent maintains a posterior distribution over purchase probability
    for each price. At each step, it samples from these distributions to
    naturally balance exploration and exploitation.
    
    Mathematical Background:
    -------------------------
    Thompson Sampling uses Bayesian inference with conjugate priors.
    For Bernoulli rewards (sale/no-sale), we use the Beta-Bernoulli model:
    
    1. PRIOR:
       We start with a uniform prior for each arm a:
           θ_a ~ Beta(α_a=1, β_a=1)
       where θ_a is the true purchase probability for price a.
       
    2. LIKELIHOOD:
       The observation model is Bernoulli:
           sale ~ Bernoulli(θ_a)
           
    3. POSTERIOR UPDATE:
       Thanks to conjugacy, the posterior is also Beta:
           θ_a | data ~ Beta(α_a + successes, β_a + failures)
           
       Update rules after observing outcome for arm a:
           - If sale:    α_a ← α_a + 1
           - If no sale: β_a ← β_a + 1
           
    4. ACTION SELECTION (Probability Matching):
       For each arm a, sample:
           θ̂_a ~ Beta(α_a, β_a)
       
       Then compute expected reward:
           R̂_a = price_a × θ̂_a
           
       Select arm with highest sampled expected reward:
           a* = argmax_a R̂_a
           
    Why This Works:
    ---------------
    - Arms with high uncertainty → Wide Beta distribution → More exploration
    - Arms with strong evidence → Narrow Beta distribution → More exploitation
    - The algorithm naturally balances exploration and exploitation!
    
    IMPORTANT - Bayesian Inertia:
    -----------------------------
    A key limitation of Thompson Sampling in non-stationary environments:
    After many observations, the posterior becomes very concentrated (narrow).
    When a market shock occurs, the agent struggles to adapt because the
    strong prior beliefs from before the shock overwhelm new observations.
    This phenomenon is called "Bayesian Inertia" or "Posterior Lock-in".
    
    Attributes:
        price_arms: Price levels (for expected reward computation)
        alphas: Beta distribution α parameter for each arm
        betas: Beta distribution β parameter for each arm
    """
    
    price_arms: List[float] = field(default_factory=lambda: [
        5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0
    ])
    name: str = "Thompson Sampling"
    
    # Beta distribution parameters
    # α = number of successes + prior, β = number of failures + prior
    alphas: np.ndarray = field(default=None, repr=False)
    betas: np.ndarray = field(default=None, repr=False)
    
    # History records (for analysis)
    _sample_history: List[np.ndarray] = field(default_factory=list, repr=False)
    
    def __post_init__(self):
        """
        Initialize Beta distributions with uninformative prior.
        
        Prior: Beta(1, 1) = Uniform(0, 1)
        This represents having no prior knowledge.
        """
        # Uniform prior: α=1, β=1 treats all probabilities as equally likely
        self.alphas = np.ones(self.n_arms)
        self.betas = np.ones(self.n_arms)
    
    def select_arm(self) -> int:
        """
        Select arm using Thompson Sampling.
        
        Algorithm:
            1. Sample θ̂ from posterior Beta distribution for each arm
            2. Compute expected reward: R̂ = price × θ̂
            3. Select arm with highest R̂
            
        Mathematical Justification:
            Sampling from the posterior naturally handles the
            exploration-exploitation trade-off:
            - Uncertain arms (wide posterior) → Occasionally high samples
            - Certain poor arms (narrow low posterior) → Rarely selected
            
        Returns:
            Index of the selected arm
        """
        # 1. Sample from posterior for each arm
        # θ̂_a ~ Beta(α_a, β_a)
        sampled_thetas = self._rng.beta(self.alphas, self.betas)
        
        # Save samples for visualization
        self._sample_history.append(sampled_thetas.copy())
        
        # 2. Compute expected rewards: R̂_a = price_a × θ̂_a
        sampled_expected_rewards = np.array(self.price_arms) * sampled_thetas
        
        # 3. Select arm with highest expected reward
        # Random selection among ties
        max_reward = np.max(sampled_expected_rewards)
        best_arms = np.where(sampled_expected_rewards == max_reward)[0]
        
        return int(self._rng.choice(best_arms))
    
    def update(self, arm: int, reward: float, sale: bool) -> None:
        """
        Perform Bayesian posterior update.
        
        Update Rule (Beta-Bernoulli Conjugacy):
            Prior:     θ ~ Beta(α, β)
            Likelihood: X | θ ~ Bernoulli(θ)
            Posterior: θ | X ~ Beta(α + X, β + (1-X))
            
        Simplified:
            - If sale (X=1):    α ← α + 1
            - If no sale (X=0): β ← β + 1
            
        Intuition:
            - α represents "pseudo-count" of successes
            - β represents "pseudo-count" of failures
            - Mean of Beta(α, β) = α / (α + β)
            
        Args:
            arm: Selected arm
            reward: Observed reward (not used, sale is sufficient)
            sale: Whether a sale occurred (True/False)
        """
        if sale:
            # Success: increment α
            self.alphas[arm] += 1
        else:
            # Failure: increment β
            self.betas[arm] += 1
    
    def reset(self) -> None:
        """Reset agent to uninformative prior."""
        self.alphas = np.ones(self.n_arms)
        self.betas = np.ones(self.n_arms)
        self._sample_history = []
    
    def get_arm_estimates(self) -> np.ndarray:
        """
        Return posterior mean probability estimate for each arm.
        
        For Beta(α, β), the posterior mean is:
            E[θ] = α / (α + β)
            
        This adds Bayesian regularization to maximum likelihood estimation.
        
        Returns:
            Array of E[θ] for each arm
        """
        return self.alphas / (self.alphas + self.betas)
    
    def get_expected_reward_estimates(self) -> np.ndarray:
        """
        Return expected reward estimate for each arm.
        
        Formula:
            E[R_a] = price_a × E[θ_a] = price_a × α_a / (α_a + β_a)
        
        Returns:
            Array of estimated expected rewards for each arm
        """
        theta_means = self.get_arm_estimates()
        return np.array(self.price_arms) * theta_means
    
    def get_posterior_params(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return Beta distribution parameters for all arms.
        
        Returns:
            (alphas, betas) tuple
        """
        return self.alphas.copy(), self.betas.copy()
    
    def get_uncertainty(self) -> np.ndarray:
        """
        Compute posterior uncertainty (standard deviation) for each arm.
        
        For Beta(α, β):
            Var[θ] = (α × β) / ((α + β)² × (α + β + 1))
            Std[θ] = √Var[θ]
            
        High uncertainty → Greater exploration potential
        
        Returns:
            Array of posterior standard deviations for each arm
        """
        total = self.alphas + self.betas
        variance = (self.alphas * self.betas) / (total**2 * (total + 1))
        return np.sqrt(variance)


# Module-level test
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy import stats
    
    print("="*60)
    print("PRICING AGENTS TEST")
    print("="*60)
    
    n_arms = 10
    price_arms = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0]
    
    # Create agents
    baseline = BaselineAgent(n_arms=n_arms, strategy="random")
    epsilon_greedy = EpsilonGreedyAgent(n_arms=n_arms, epsilon=0.1)
    thompson = ThompsonSamplingAgent(n_arms=n_arms, price_arms=price_arms)
    
    print(f"\n1. Baseline Agent: {baseline}")
    print(f"2. Epsilon-Greedy Agent: {epsilon_greedy}")
    print(f"3. Thompson Sampling Agent: {thompson}")
    
    # Test Thompson Sampling
    print("\n" + "="*60)
    print("THOMPSON SAMPLING TEST")
    print("="*60)
    
    # Perform some updates
    true_probs = [0.7, 0.5, 0.3, 0.2, 0.1, 0.08, 0.05, 0.03, 0.02, 0.01]
    
    np.random.seed(42)
    for _ in range(100):
        arm = thompson.select_arm()
        # Simulate sale based on true probability
        sale = np.random.random() < true_probs[arm]
        thompson.update(arm, price_arms[arm] if sale else 0, sale)
    
    print("\nPosterior estimates after 100 steps:")
    print("-" * 50)
    estimates = thompson.get_arm_estimates()
    for i, (true_p, est_p) in enumerate(zip(true_probs, estimates)):
        print(f"  Arm {i} (${price_arms[i]:5.2f}): "
              f"True={true_p:.2f}, Estimate={est_p:.3f}")
    
    # Visualize Beta distributions
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    x = np.linspace(0, 1, 100)
    
    alphas, betas_params = thompson.get_posterior_params()
    
    for i, ax in enumerate(axes.flat):
        a, b = alphas[i], betas_params[i]
        y = stats.beta.pdf(x, a, b)
        ax.plot(x, y, 'b-', lw=2)
        ax.axvline(true_probs[i], color='r', linestyle='--', label='True')
        ax.axvline(estimates[i], color='g', linestyle='-', label='Estimate')
        ax.set_title(f'Arm {i} (${price_arms[i]:.0f})\nα={a:.0f}, β={b:.0f}')
        ax.set_xlabel('θ')
        ax.set_ylabel('Density')
        if i == 0:
            ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig('thompson_posteriors_test.png', dpi=150)
    print("\nPosterior distributions saved as 'thompson_posteriors_test.png'.")
