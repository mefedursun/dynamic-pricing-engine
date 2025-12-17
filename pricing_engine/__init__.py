"""
Pricing Engine - Multi-Armed Bandit Simulation Framework

This package provides an academic-grade simulation framework for comparing
Multi-Armed Bandit algorithms in dynamic pricing problems with non-stationary
market conditions.

Key Components:
    - MarketEnvironment: Simulates customer behavior with Sigmoid demand curve
    - PricingAgent: Abstract base class for MAB algorithms
    - BaselineAgent: Random/fixed pricing (control group)
    - EpsilonGreedyAgent: Classic ε-greedy exploration strategy
    - ThompsonSamplingAgent: Bayesian Beta-Bernoulli approach
    - SimulationEngine: Orchestrates experiments with shock injection
    - Analysis functions: Publication-quality visualizations
"""

from .environment import MarketEnvironment
from .agents import BaselineAgent, EpsilonGreedyAgent, ThompsonSamplingAgent
from .simulation import SimulationEngine
from .analysis import (
    plot_cumulative_regret,
    plot_beta_distributions,
    plot_price_selection_heatmap,
)

__version__ = "1.0.0"
__author__ = "Mustafa Efe Dursun"

__all__ = [
    "MarketEnvironment",
    "BaselineAgent",
    "EpsilonGreedyAgent",
    "ThompsonSamplingAgent",
    "SimulationEngine",
    "plot_cumulative_regret",
    "plot_beta_distributions",
    "plot_price_selection_heatmap",
]
