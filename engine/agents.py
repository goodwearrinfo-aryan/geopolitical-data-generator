"""Agent-based micro-foundations for geopolitical scenarios.

Models individual decision-making agents whose aggregate behavior produces
macroscopic geopolitical dynamics. Supports household, firm, government,
elite, and population agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List
import numpy as np


class AgentType(Enum):
    """Types of agents in the micro-foundations model."""

    HOUSEHOLD = "household"
    FIRM = "firm"
    GOVERNMENT = "government"
    ELITE = "elite"
    POPULATION = "population"


@dataclass
class Agent:
    """Base agent class."""

    agent_id: str
    agent_type: AgentType
    parameters: Dict[str, Any] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)

    def update(self, environment: Dict[str, Any]) -> None:
        """Update agent state based on environment. Override in subclasses."""
        pass

    def decide(self, options: Dict[str, Any]) -> Any:
        """Make a decision from given options. Override in subclasses."""
        raise NotImplementedError()


@dataclass
class Household(Agent):
    """Representative household agent with consumption, grievance, and voting."""

    income: float = 1.0
    savings: float = 0.2
    grievance: float = 0.0
    voting_preference: str = "status_quo"

    def update(self, environment: Dict[str, Any]) -> None:
        """Update household state: consumption, grievance accumulation."""
        consumption = self.income * self.savings  # simplified
        self.state["consumption"] = consumption

        # Grievance accumulates based on inequality
        inequality = environment.get("inequality", 0.0)
        self.grievance = np.clip(self.grievance + inequality, 0.0, 1.0)

        # Voting preference shifts with grievance
        if self.grievance > 0.5:
            self.voting_preference = "protest"
        elif self.grievance > 0.2:
            self.voting_preference = "alternative"
        else:
            self.voting_preference = "status_quo"

    def decide(self, options: Dict[str, Any]) -> str:
        """Household voting decision."""
        return self.voting_preference


@dataclass
class Firm(Agent):
    """Representative firm agent with investment, pricing, and export decisions."""

    capital: float = 1.0
    productivity: float = 1.0
    price: float = 1.0
    export_mode: str = "domestic"

    def update(self, environment: Dict[str, Any]) -> None:
        """Firm update: productivity shocks, investment decisions."""
        demand = environment.get("demand", 1.0)
        cost = environment.get("cost", 1.0)
        # Profit-maximizing price
        self.price = np.clip(self.productivity * demand / cost, 0.1, 10.0)
        # Investment decision
        self.state["last_price"] = self.price

    def decide(self, options: Dict[str, Any]) -> str:
        """Firm export/investment decision."""
        market_condition = options.get("market_condition", "stable")
        if market_condition == "favorable":
            self.export_mode = "export"
        else:
            self.export_mode = "domestic"
        return self.export_mode


@dataclass
class Government(Agent):
    """Government agent with policy levers: taxes, subsidies, repression."""

    tax_rate: float = 0.2
    spending: float = 0.3
    repression_level: float = 0.1
    policy_history: List[str] = field(default_factory=list)

    def update(self, environment: Dict[str, Any]) -> None:
        """Government policy response to conditions."""
        stability = environment.get("stability", 0.5)
        unrest = environment.get("unrest", 0.0)
        # Adjust repression based on unrest
        if unrest > 0.5:
            self.repression_level = np.clip(self.repression_level + 0.1, 0.0, 1.0)
            self.policy_history.append("repression_increase")
        elif unrest < 0.2 and self.repression_level > 0.1:
            self.repression_level = np.clip(self.repression_level - 0.05, 0.0, 1.0)
            self.policy_history.append("repression_decrease")

        # Adjust tax rates based on stability
        if stability < 0.3:
            self.tax_rate = np.clip(self.tax_rate - 0.02, 0.0, 1.0)
            self.policy_history.append("tax_reduce")
        else:
            self.policy_history.append("tax_maintain")

    def decide(self, options: Dict[str, Any]) -> str:
        """Government policy decision."""
        policy = options.get("policy_goal", "stability")
        if policy == "stability":
            return "increase_repression"
        elif policy == "growth":
            return "cut_taxes"
        return "maintain"


@dataclass
class Elite(Agent):
    """Elite agent with wealth accumulation and influence peddling."""

    wealth: float = 1.0
    influence: float = 0.1
    tax_avoidance: float = 0.0

    def update(self, environment: Dict[str, Any]) -> None:
        """Elite response to tax policy and market conditions."""
        tax_rate = environment.get("tax_rate", 0.2)
        # Wealth accumulation depends on after-tax returns
        after_tax_return = 0.05 * (1 - tax_rate)  # 5% baseline return
        self.wealth *= 1 + after_tax_return
        # Tax avoidance increases when rates are high
        if tax_rate > 0.3:
            self.tax_avoidance = np.clip(self.tax_avoidance + 0.01, 0.0, 1.0)

    def decide(self, options: Dict[str, Any]) -> str:
        """Elite decision on political support."""
        return "support_ruling" if self.wealth > 5 else "support_opposition"


@dataclass
class Population(Agent):
    """Population aggregate with collective dynamics."""

    size: int = 1000
    protest_probability: float = 0.01
    revolution_probability: float = 0.001

    def update(self, environment: Dict[str, Any]) -> None:
        """Population dynamics: grievance, protest, revolution."""
        grievance = environment.get("grievance", 0.0)
        inequality = environment.get("inequality", 0.0)
        # Aggregate grievance
        self.protest_probability = np.clip(grievance * 0.1, 0.0, 0.5)
        self.revolution_probability = np.clip(grievance * inequality * 0.05, 0.0, 0.3)

    def decide(self, options: Dict[str, Any]) -> str:
        """Population collective decision."""
        r = np.random.random()
        if r < self.revolution_probability:
            return "revolution"
        elif r < self.protest_probability + self.revolution_probability:
            return "protest"
        return "status_quo"
