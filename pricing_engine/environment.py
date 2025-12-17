"""
Market Environment Module

This module simulates a non-stationary market environment for Multi-Armed Bandit
algorithms. It includes a Sigmoid demand curve and a market shock mechanism.

Mathematical Model:
    The purchase probability follows a Sigmoid function:
    
    P(buy | price) = 1 / (1 + exp(k * (price - optimal_price)))
    
    where:
        - optimal_price (p₀): The price point at 50% purchase probability
        - k (sensitivity): Controls the steepness of the demand curve
        
    The expected reward for a given price is:
    
    E[reward] = price × P(buy | price)
"""

from typing import Tuple, List, Optional
import numpy as np
from dataclasses import dataclass, field


@dataclass
class ShockEvent:
    """
    Data class representing a market shock event.
    
    A shock event captures a sudden, exogenous change in market conditions
    that shifts the optimal pricing strategy.
    
    Attributes:
        timestep: The simulation step when the shock occurred
        price_shift: Change in the optimal price point (positive = inflation)
        sensitivity_change: Change in customer price sensitivity
        description: Human-readable description of the shock (e.g., "Hyperinflation")
    """
    timestep: int
    price_shift: float
    sensitivity_change: float
    description: str = ""


@dataclass
class MarketEnvironment:
    """
    Non-Stationary Market Environment for Multi-Armed Bandit Pricing.
    
    This class models a dynamic market where customer purchase probability
    depends on the offered price, following a Sigmoid (logistic) function.
    
    Mathematical Background:
    -------------------------
    The demand curve follows the Sigmoid (Logistic) function:
    
        P(buy | p) = σ(-k(p - p₀)) = 1 / (1 + e^(k(p - p₀)))
        
    This models the economic intuition that:
        - Lower prices → Higher purchase probability
        - Higher prices → Lower purchase probability
        - At p = p₀ (optimal price), P(buy) = 0.5
        
    The sensitivity parameter k controls elasticity:
        - High k → Customers are very price-sensitive (elastic demand)
        - Low k → Customers are less price-sensitive (inelastic demand)
    
    Attributes:
        price_arms: Discrete price levels available as bandit arms
        optimal_price: Current price point with 50% purchase probability
        sensitivity: Steepness of the demand curve (k parameter)
        noise_std: Standard deviation of Gaussian noise in observations
        shock_history: Record of applied market shocks
    """
    
    # Price arms - discrete price levels available for selection
    price_arms: List[float] = field(default_factory=lambda: [
        5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0
    ])
    
    # Initial optimal price - the 50% purchase probability point
    optimal_price: float = 25.0
    
    # Demand sensitivity - higher value = steeper curve
    sensitivity: float = 0.15
    
    # Observation noise standard deviation
    noise_std: float = 0.0
    
    # Store shock history for analysis
    shock_history: List[ShockEvent] = field(default_factory=list)
    
    # Internal state: Random number generator
    _rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(42),
        repr=False
    )
    
    def set_seed(self, seed: int) -> None:
        """
        Re-seed the random number generator for reproducibility.
        
        Args:
            seed: Random seed value
        """
        self._rng = np.random.default_rng(seed)
    
    def _purchase_probability(self, price: float) -> float:
        """
        Compute the purchase probability for a given price.
        
        Mathematical Formula:
            P(buy | price) = 1 / (1 + exp(k × (price - optimal_price)))
            
        This Sigmoid function models decreasing purchase probability
        as price increases. At optimal_price, probability equals 0.5.
        
        Args:
            price: Product price
            
        Returns:
            Purchase probability in the range [0, 1]
        """
        # Apply the Sigmoid function
        # Negative exponent ensures probability decreases with price
        exponent = self.sensitivity * (price - self.optimal_price)
        
        # Prevent numerical overflow (numerical stability)
        exponent = np.clip(exponent, -500, 500)
        
        probability = 1.0 / (1.0 + np.exp(exponent))
        
        return float(probability)
    
    def get_true_probabilities(self) -> np.ndarray:
        """
        Return the true purchase probabilities for all price arms.
        
        This function is used to compare Thompson Sampling's learned
        probabilities against the ground truth values.
        
        Returns:
            Array of P(buy | price) for each price arm
        """
        return np.array([self._purchase_probability(p) for p in self.price_arms])
    
    def get_expected_rewards(self) -> np.ndarray:
        """
        Compute the expected reward (revenue) for each price arm.
        
        Mathematical Formula:
            E[reward | price] = price × P(buy | price)
            
        The optimal price is the one that maximizes this expected reward.
        Note: optimal_price and optimal arm are different concepts!
        
        Returns:
            Array of expected rewards for each price arm
        """
        probs = self.get_true_probabilities()
        return np.array(self.price_arms) * probs
    
    def get_optimal_arm(self) -> int:
        """
        Return the arm with the highest expected reward.
        
        This is used for regret calculation:
        Regret = Σ(optimal_reward - actual_reward)
        
        Returns:
            Index of the optimal arm
        """
        expected_rewards = self.get_expected_rewards()
        return int(np.argmax(expected_rewards))
    
    def get_optimal_expected_reward(self) -> float:
        """
        Return the expected reward of the optimal arm.
        
        Returns:
            Maximum expected reward value
        """
        return float(np.max(self.get_expected_rewards()))
    
    def get_reward(self, arm_index: int) -> Tuple[bool, float]:
        """
        Pull an arm and receive a stochastic reward.
        
        This function implements the core bandit interaction:
        1. Agent selects a price (arm)
        2. Environment determines if a purchase occurs (Bernoulli trial)
        3. If purchase occurred, reward = price; otherwise reward = 0
        
        Args:
            arm_index: Index of the selected arm [0, len(price_arms)-1]
            
        Returns:
            (sale_occurred, reward): Whether a sale happened and the revenue
            
        Raises:
            IndexError: Invalid arm index
        """
        if not 0 <= arm_index < len(self.price_arms):
            raise IndexError(
                f"Arm index {arm_index} is invalid. "
                f"Valid range: [0, {len(self.price_arms) - 1}]"
            )
        
        price = self.price_arms[arm_index]
        purchase_prob = self._purchase_probability(price)
        
        # Optional: Add noise to probability (for exploration)
        if self.noise_std > 0:
            noise = self._rng.normal(0, self.noise_std)
            purchase_prob = np.clip(purchase_prob + noise, 0.0, 1.0)
        
        # Bernoulli trial - did purchase occur?
        sale_occurred = self._rng.random() < purchase_prob
        
        # Reward: price if sale, 0 otherwise
        reward = price if sale_occurred else 0.0
        
        return sale_occurred, reward
    
    def inject_shock(
        self,
        price_shift: float = 10.0,
        sensitivity_change: float = 0.05,
        timestep: int = 0,
        description: str = "Market Shock"
    ) -> None:
        """
        Inject a market shock - change optimal price and sensitivity.
        
        This function simulates non-stationary environments:
        - Hyperinflation: Optimal price shifts upward
        - Competitor entry: Sensitivity increases (customers become pickier)
        - Economic crisis: Both parameters may change
        
        Mathematical Effect:
            p₀_new = p₀_old + price_shift
            k_new = k_old + sensitivity_change
            
        This change shifts the entire probability distribution and
        invalidates learned knowledge. Adaptive algorithms (like Thompson
        Sampling) should adapt to this change.
        
        Args:
            price_shift: Change in optimal price (+ = inflation)
            sensitivity_change: Change in sensitivity (+ = more elastic)
            timestep: Simulation step when shock occurs (for logging)
            description: Description of the shock
        """
        # Save old values for debugging
        old_optimal = self.optimal_price
        old_sensitivity = self.sensitivity
        
        # Apply the shock
        self.optimal_price += price_shift
        self.sensitivity = max(0.01, self.sensitivity + sensitivity_change)
        
        # Record shock in history
        shock = ShockEvent(
            timestep=timestep,
            price_shift=price_shift,
            sensitivity_change=sensitivity_change,
            description=description
        )
        self.shock_history.append(shock)
        
        # Print notification message
        print(f"\n{'='*60}")
        print(f"[SHOCK] MARKET SHOCK @ t={timestep}: {description}")
        print(f"   Optimal Price: {old_optimal:.2f} -> {self.optimal_price:.2f}")
        print(f"   Sensitivity: {old_sensitivity:.3f} -> {self.sensitivity:.3f}")
        print(f"   New Optimal Arm: {self.get_optimal_arm()} "
              f"(Price: {self.price_arms[self.get_optimal_arm()]:.2f})")
        print(f"{'='*60}\n")
    
    def reset(self, seed: Optional[int] = None) -> None:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Optional new random seed
        """
        self.optimal_price = 25.0
        self.sensitivity = 0.15
        self.shock_history = []
        
        if seed is not None:
            self.set_seed(seed)
    
    def __repr__(self) -> str:
        return (
            f"MarketEnvironment("
            f"n_arms={len(self.price_arms)}, "
            f"optimal_price={self.optimal_price:.2f}, "
            f"sensitivity={self.sensitivity:.3f}, "
            f"optimal_arm={self.get_optimal_arm()}"
            f")"
        )


# Module-level test (when run directly)
if __name__ == "__main__":
    # Create environment
    env = MarketEnvironment()
    print(f"Environment created: {env}")
    
    # Show probabilities and expected rewards
    print("\nPrice Arms Analysis:")
    print("-" * 50)
    for i, price in enumerate(env.price_arms):
        prob = env._purchase_probability(price)
        exp_reward = price * prob
        print(f"  Arm {i}: Price=${price:5.2f}, "
              f"P(buy)={prob:.3f}, "
              f"E[R]={exp_reward:.2f}")
    
    print(f"\nOptimal Arm: {env.get_optimal_arm()}")
    print(f"Optimal Expected Reward: {env.get_optimal_expected_reward():.2f}")
    
    # Shock test
    print("\n" + "="*50)
    print("TESTING SHOCK INJECTION...")
    env.inject_shock(price_shift=15.0, sensitivity_change=0.05, 
                     timestep=5000, description="Hyperinflation")
    
    print("\nPost-Shock Analysis:")
    print(f"New Optimal Arm: {env.get_optimal_arm()}")
    print(f"New Optimal Expected Reward: {env.get_optimal_expected_reward():.2f}")
