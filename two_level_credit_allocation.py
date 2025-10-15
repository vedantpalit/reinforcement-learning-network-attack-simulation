import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Any
import matplotlib.pyplot as plt
from collections import deque, defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Beta, Normal
import matplotlib.gridspec as gridspec
import pickle
import json
import os
from pathlib import Path
from datetime import datetime

class CreditAllocationEnvTwoLevel(gym.Env):
    """
    Enhanced Gymnasium environment with two-level design:
    - Level 1: Loan approval decision
    - Level 2: Loan arrival + default modeling
    """

    def __init__(self,
                 mu_R_init: float = 0.6,
                 mu_B_init: float = 0.4,
                 sigma: float = 0.2,
                 alpha_R: float = 0.3,
                 alpha_B: float = 0.3,
                 beta_R: float = 2.0,
                 beta_B: float = 2.0,
                 kappa_R: float = 0.1,
                 kappa_B: float = 0.1,
                 theta_S: float = 0.1,
                 theta_X: float = 2.0,
                 b: float = -1.0,
                 tau: float = 0.5,
                 N_R: int = 1000,
                 N_B: int = 1000,
                 T: int = 100,
                 dt: float = 0.1,
                 loan_amount: float = 0.2,
                 interest_rate: float = 0.05,
                 default_noise_std: float = 0.1):

        super().__init__()

        # Store all parameters
        self.mu_R_init = mu_R_init
        self.mu_B_init = mu_B_init
        self.sigma = sigma
        self.N_R = N_R
        self.N_B = N_B
        self.T = T
        self.dt = dt
        self.tau = tau

        # Hawkes process parameters
        self.alpha_R = alpha_R
        self.alpha_B = alpha_B
        self.beta_R = beta_R
        self.beta_B = beta_B

        # Net-worth update parameters
        self.kappa_R = kappa_R
        self.kappa_B = kappa_B

        # Loan model parameters
        self.theta_S = theta_S
        self.theta_X = theta_X
        self.b = b

        # Two-level design parameters
        self.loan_amount = loan_amount
        self.interest_rate = interest_rate
        self.default_noise_std = default_noise_std

        # Action and observation spaces
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            shape=(1,),
            dtype=np.float32
        )

        # Enhanced observation: [wealth, group, mu_R, mu_B, lambda_R, lambda_B, default_rate_R, default_rate_B]
        self.observation_space = spaces.Box(
            low=np.array([-10.0, 0.0, -10.0, -10.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([10.0, 1.0, 10.0, 10.0, 100.0, 100.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Initialize
        self.reset()

    def _default_probability(self, wealth: float) -> float:
        """
        Monotonic decreasing function for default probability based on wealth.
        Higher wealth = lower default probability
        """
        # Sigmoid-based monotonic function
        base_prob = 1.0 / (1.0 + np.exp(3.0 * wealth))

        # Add noise
        noise = np.random.normal(0, self.default_noise_std)
        prob = np.clip(base_prob + noise, 0.0, 1.0)

        return prob

    def _process_loan_outcome(self, applicant: dict, approved: bool) -> float:
        """
        Process loan outcome including potential default.
        Returns net change in wealth for the individual.
        """
        if not approved:
            return 0.0

        # Check for default
        default_prob = self._default_probability(applicant['wealth'])
        defaults = np.random.random() < default_prob

        if defaults:
            # Default: lose part of the loan amount
            wealth_change = -self.loan_amount * 0.5  # Partial loss
            applicant['defaulted'] = True
        else:
            # Successful repayment: gain from loan usage minus interest
            gross_gain = self.loan_amount * (1 + np.random.uniform(0.1, 0.3))  # Random productivity
            interest_payment = self.loan_amount * self.interest_rate
            wealth_change = gross_gain - self.loan_amount - interest_payment
            applicant['defaulted'] = False

        return wealth_change

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        # Initialize means
        self.mu_R = self.mu_R_init
        self.mu_B = self.mu_B_init

        # Initialize wealth distributions
        self.current_wealth_R = np.random.normal(self.mu_R_init, self.sigma, self.N_R)
        self.current_wealth_B = np.random.normal(self.mu_B_init, self.sigma, self.N_B)

        # Track defaults
        self.total_defaults_R = 0
        self.total_defaults_B = 0
        self.total_loans_R = 0
        self.total_loans_B = 0

        # Calculate initial variances
        self.var_R = np.var(self.current_wealth_R)
        self.var_B = np.var(self.current_wealth_B)

        # Event history for Hawkes process
        self.event_times_R = []
        self.event_times_B = []

        # Enhanced history tracking
        self.history = {
            'time': [0.0],
            'mu_R': [np.mean(self.current_wealth_R)],
            'mu_B': [np.mean(self.current_wealth_B)],
            'lambda_R': [self._f_networth_to_rate(self.mu_R)],
            'lambda_B': [self._f_networth_to_rate(self.mu_B)],
            'loan_arrivals_R': [0],
            'loan_arrivals_B': [0],
            'loan_approvals_R': [0],
            'loan_approvals_B': [0],
            'defaults_R': [0],
            'defaults_B': [0],
            'default_rate_R': [0.0],
            'default_rate_B': [0.0],
            'variance_R': [self.var_R],
            'variance_B': [self.var_B],
            'covariance_R': [0.0],
            'covariance_B': [0.0],
            'correlation_R': [0.0],
            'correlation_B': [0.0],
            'selection_prob_R': [0.0],
            'selection_prob_B': [0.0],
            'profit': [0.0]
        }

        # Time tracking
        self.current_time = 0.0
        self.time_steps = np.arange(0, self.T, self.dt)
        self.time_index = 0

        # For step-by-step decision making
        self.pending_applications = []
        self.current_applicant = None
        self.timestep_data = None
        self.timestep_profit = 0.0

        # Get initial observation
        obs = self._get_observation()
        info = {}

        return obs, info

    def _f_networth_to_rate(self, mu: float) -> float:
        return max(0.1, 1.0 + 2.0 * mu)

    def _phi_R(self, t: float) -> float:
        return self.alpha_R * np.exp(-self.beta_R * t)

    def _phi_B(self, t: float) -> float:
        return self.alpha_B * np.exp(-self.beta_B * t)

    def _compute_lambda_R(self, t: float) -> float:
        """Compute Red group arrival rate with self-excitation"""
        base_rate = self._f_networth_to_rate(self.mu_R)
        excitation = sum(self._phi_R(t - t_i) for t_i in self.event_times_R if t_i < t)
        return base_rate + excitation

    def _compute_lambda_B(self, t: float) -> float:
        """Compute Blue group arrival rate with self-excitation"""
        base_rate = self._f_networth_to_rate(self.mu_B)
        excitation = sum(self._phi_B(t - t_i) for t_i in self.event_times_B if t_i < t)
        return base_rate + excitation

    def _loan_approval_probability(self, X: float, S: int) -> float:
        """Compute loan approval probability ν(h_θ(X,S))"""
        z = self.theta_S * S + self.theta_X * X + self.b
        return 1.0 / (1.0 + np.exp(-z))

    def _wealth_dependent_application_prob(self, wealth: float, base_rate: float) -> float:
        """Application probability depends on wealth"""
        wealth_effect = 1.0 / (1.0 + np.exp(-2.0 * wealth))
        return min(1.0, base_rate * wealth_effect)

    def _compute_empirical_covariance(self, wealth: np.ndarray, apply_select: np.ndarray) -> Tuple[float, float, float]:
        """Compute empirical covariance"""
        if len(wealth) == 0 or len(apply_select) == 0:
            return 0.0, 0.0, 0.0

        cov = np.cov(wealth, apply_select)[0, 1] if len(wealth) > 1 else 0.0
        corr = np.corrcoef(wealth, apply_select)[0, 1] if np.std(apply_select) > 0 and len(wealth) > 1 else 0.0
        selection_prob = np.mean(apply_select)

        return cov, corr, selection_prob

    def _update_wealth_distributions(self):
        """Update group mean wealth from individual wealth updates"""
        self.mu_R = np.mean(self.current_wealth_R)
        self.mu_B = np.mean(self.current_wealth_B)
        self.var_R = np.var(self.current_wealth_R)
        self.var_B = np.var(self.current_wealth_B)

    def _get_observation(self) -> np.ndarray:
        """Get current observation for the agent."""
        # Calculate current default rates
        default_rate_R = self.total_defaults_R / max(self.total_loans_R, 1)
        default_rate_B = self.total_defaults_B / max(self.total_loans_B, 1)

        if self.current_applicant is None:
            return np.array([0.0, 0.5, self.mu_R, self.mu_B,
                           self._compute_lambda_R(self.current_time),
                           self._compute_lambda_B(self.current_time),
                           default_rate_R, default_rate_B], dtype=np.float32)
        else:
            return np.array([
                self.current_applicant['wealth'],
                float(self.current_applicant['group_id']),
                self.mu_R,
                self.mu_B,
                self._compute_lambda_R(self.current_time),
                self._compute_lambda_B(self.current_time),
                default_rate_R,
                default_rate_B
            ], dtype=np.float32)

    def _generate_timestep_applications(self) -> dict:
        """Generate all applications for current timestep."""
        t = self.current_time
        lambda_R = self._compute_lambda_R(t)
        lambda_B = self._compute_lambda_B(t)

        # Base application rates
        base_app_rate_R = lambda_R * self.dt / self.N_R
        base_app_rate_B = lambda_B * self.dt / self.N_B

        # Initialize tracking
        wealth_R = []
        wealth_B = []
        apply_select_R = []
        apply_select_B = []
        applications = []

        # Process Red group (S=1)
        for i in range(self.N_R):
            w_i = self.current_wealth_R[i]
            wealth_R.append(w_i)

            app_prob = self._wealth_dependent_application_prob(w_i, base_app_rate_R)
            applies = np.random.random() < app_prob

            if applies:
                applications.append({
                    'group': 'R',
                    'group_id': 1,
                    'individual_id': i,
                    'wealth': w_i,
                    'selection_prob': self._loan_approval_probability(w_i, 1),
                    'default_prob': self._default_probability(w_i)
                })
                apply_select_R.append(0)
            else:
                apply_select_R.append(0)

        # Process Blue group (S=0)
        for i in range(self.N_B):
            w_i = self.current_wealth_B[i]
            wealth_B.append(w_i)

            app_prob = self._wealth_dependent_application_prob(w_i, base_app_rate_B)
            applies = np.random.random() < app_prob

            if applies:
                applications.append({
                    'group': 'B',
                    'group_id': 0,
                    'individual_id': i,
                    'wealth': w_i,
                    'selection_prob': self._loan_approval_probability(w_i, 0),
                    'default_prob': self._default_probability(w_i)
                })
                apply_select_B.append(0)
            else:
                apply_select_B.append(0)

        return {
            'applications': applications,
            'wealth_R': np.array(wealth_R),
            'wealth_B': np.array(wealth_B),
            'apply_select_R': np.array(apply_select_R),
            'apply_select_B': np.array(apply_select_B),
            'lambda_R': lambda_R,
            'lambda_B': lambda_B
        }

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step in the environment."""
        reward = 0.0

        if isinstance(action, np.ndarray):
            approval_prob = np.clip(action[0], 0.0, 1.0)
        else:
            approval_prob = np.clip(float(action), 0.0, 1.0)

        # Process current application
        if self.current_applicant is not None:
            applicant = self.current_applicant
            approved = np.random.random() < approval_prob

            if approved:
                # Process loan outcome with default possibility
                wealth_change = self._process_loan_outcome(applicant, approved)

                if applicant['group'] == 'R':
                    self.current_wealth_R[applicant['individual_id']] += wealth_change
                    self.event_times_R.append(self.current_time)
                    self.timestep_data['approvals_R'] += 1
                    self.timestep_data['apply_select_R'][applicant['individual_id']] = 1
                    self.total_loans_R += 1

                    if applicant.get('defaulted', False):
                        self.timestep_data['defaults_R'] += 1
                        self.total_defaults_R += 1
                    else:
                        # Profit for lender
                        self.timestep_profit += self.loan_amount * self.interest_rate
                else:
                    self.current_wealth_B[applicant['individual_id']] += wealth_change
                    self.event_times_B.append(self.current_time)
                    self.timestep_data['approvals_B'] += 1
                    self.timestep_data['apply_select_B'][applicant['individual_id']] = 1
                    self.total_loans_B += 1

                    if applicant.get('defaulted', False):
                        self.timestep_data['defaults_B'] += 1
                        self.total_defaults_B += 1
                    else:
                        # Profit for lender
                        self.timestep_profit += self.loan_amount * self.interest_rate

        # Get next application from current timestep
        if self.pending_applications:
            self.current_applicant = self.pending_applications.pop(0)
        else:
            self.current_applicant = None

        # If no more applications in current timestep, move to next timestep
        if self.current_applicant is None and self.time_index < len(self.time_steps):
            # Record current timestep data
            if self.timestep_data is not None:
                # Compute covariances
                cov_R, corr_R, sel_prob_R = self._compute_empirical_covariance(
                    self.timestep_data['wealth_R'], self.timestep_data['apply_select_R']
                )
                cov_B, corr_B, sel_prob_B = self._compute_empirical_covariance(
                    self.timestep_data['wealth_B'], self.timestep_data['apply_select_B']
                )

                # Update wealth distributions
                self._update_wealth_distributions()

                # Calculate default rates
                default_rate_R = self.total_defaults_R / max(self.total_loans_R, 1)
                default_rate_B = self.total_defaults_B / max(self.total_loans_B, 1)

                # Record history
                self.history['time'].append(self.current_time)
                self.history['mu_R'].append(self.mu_R)
                self.history['mu_B'].append(self.mu_B)
                self.history['lambda_R'].append(self.timestep_data['lambda_R'])
                self.history['lambda_B'].append(self.timestep_data['lambda_B'])
                self.history['loan_arrivals_R'].append(self.timestep_data['applications_R'])
                self.history['loan_arrivals_B'].append(self.timestep_data['applications_B'])
                self.history['loan_approvals_R'].append(self.timestep_data['approvals_R'])
                self.history['loan_approvals_B'].append(self.timestep_data['approvals_B'])
                self.history['defaults_R'].append(self.timestep_data['defaults_R'])
                self.history['defaults_B'].append(self.timestep_data['defaults_B'])
                self.history['default_rate_R'].append(default_rate_R)
                self.history['default_rate_B'].append(default_rate_B)
                self.history['variance_R'].append(self.var_R)
                self.history['variance_B'].append(self.var_B)
                self.history['covariance_R'].append(cov_R)
                self.history['covariance_B'].append(cov_B)
                self.history['correlation_R'].append(corr_R)
                self.history['correlation_B'].append(corr_B)
                self.history['selection_prob_R'].append(sel_prob_R)
                self.history['selection_prob_B'].append(sel_prob_B)
                self.history['profit'].append(self.timestep_profit)

            # Move to next timestep
            if self.time_index < len(self.time_steps):
                self.current_time = self.time_steps[self.time_index]
                self.time_index += 1
                self.timestep_profit = 0.0

                # Generate applications for new timestep
                timestep_info = self._generate_timestep_applications()
                self.pending_applications = timestep_info['applications'].copy()

                # Store timestep data
                self.timestep_data = {
                    'applications': timestep_info['applications'],
                    'wealth_R': timestep_info['wealth_R'],
                    'wealth_B': timestep_info['wealth_B'],
                    'apply_select_R': timestep_info['apply_select_R'].copy(),
                    'apply_select_B': timestep_info['apply_select_B'].copy(),
                    'lambda_R': timestep_info['lambda_R'],
                    'lambda_B': timestep_info['lambda_B'],
                    'applications_R': sum(1 for app in timestep_info['applications'] if app['group'] == 'R'),
                    'applications_B': sum(1 for app in timestep_info['applications'] if app['group'] == 'B'),
                    'approvals_R': 0,
                    'approvals_B': 0,
                    'defaults_R': 0,
                    'defaults_B': 0
                }

                # Get first application
                if self.pending_applications:
                    self.current_applicant = self.pending_applications.pop(0)

        # Check termination
        terminated = self.time_index >= len(self.time_steps) and self.current_applicant is None
        truncated = False

        # Get observation and info
        obs = self._get_observation()
        info = {
            'time': self.current_time,
            'mu_R': self.mu_R,
            'mu_B': self.mu_B,
            'pending_applications': len(self.pending_applications),
            'approval_prob_used': approval_prob,
            'applicant': self.current_applicant
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        """Render the environment (optional)."""
        pass

    def close(self):
        """Clean up resources."""
        pass


class RewardFunction:
    """Enhanced reward functions for two-level design."""

    @staticmethod
    def social_welfare(env, action, info):
        """Social welfare: average utility across all individuals."""
        applicant = info.get('applicant')
        if applicant is None:
            return 0.0

        # Consider both approval benefit and default risk
        approval_prob = action[0] if isinstance(action, np.ndarray) else action
        expected_benefit = approval_prob * env.loan_amount * 0.2  # Expected social benefit
        default_cost = approval_prob * applicant.get('default_prob', 0.5) * env.loan_amount * 0.5

        return expected_benefit - default_cost

    @staticmethod
    def rawlsian_maximin(env, action, info):
        """Rawlsian: maximize welfare of worst-off group."""
        applicant = info.get('applicant')
        if applicant is None:
            return 0.0

        approval_prob = action[0] if isinstance(action, np.ndarray) else action

        # Higher reward for helping the disadvantaged group
        if applicant['group'] == 'R' and env.mu_R < env.mu_B:
            return 2.0 * approval_prob * env.loan_amount
        elif applicant['group'] == 'B' and env.mu_B < env.mu_R:
            return 2.0 * approval_prob * env.loan_amount
        else:
            return 0.5 * approval_prob * env.loan_amount

    @staticmethod
    def fairness_lagrangian(env, action, info, lambda_fairness=1.0):
        """Fairness-aware: utilitarian profit with Lagrangian penalty for acceptance rate differences."""
        applicant = info.get('applicant')
        if applicant is None:
            return 0.0

        approval_prob = action[0] if isinstance(action, np.ndarray) else action

        # Utilitarian profit component (same as utilitarian_profit)
        default_prob = applicant.get('default_prob', 0.5)
        expected_return = approval_prob * (1 - default_prob) * env.loan_amount * env.interest_rate
        expected_loss = approval_prob * default_prob * env.loan_amount * 0.5
        utilitarian_reward = expected_return - expected_loss

        # Fairness correction term (penalty for acceptance rate differences)
        total_apps_R = sum(env.history['loan_arrivals_R']) if env.history['loan_arrivals_R'] else 1
        total_apps_B = sum(env.history['loan_arrivals_B']) if env.history['loan_arrivals_B'] else 1
        total_approvals_R = sum(env.history['loan_approvals_R']) if env.history['loan_approvals_R'] else 0
        total_approvals_B = sum(env.history['loan_approvals_B']) if env.history['loan_approvals_B'] else 0

        rate_R = total_approvals_R / max(total_apps_R, 1)
        rate_B = total_approvals_B / max(total_apps_B, 1)
        fairness_penalty = lambda_fairness * abs(rate_R - rate_B)

        # Return: Utilitarian + Correction Term (negative penalty)
        return utilitarian_reward - fairness_penalty

    @staticmethod
    def utilitarian_profit(env, action, info):
        """Utilitarian: maximize expected profit for loan provider."""
        applicant = info.get('applicant')
        if applicant is None:
            return 0.0

        approval_prob = action[0] if isinstance(action, np.ndarray) else action

        # Expected profit calculation
        default_prob = applicant.get('default_prob', 0.5)
        expected_return = approval_prob * (1 - default_prob) * env.loan_amount * env.interest_rate
        expected_loss = approval_prob * default_prob * env.loan_amount * 0.5

        return expected_return - expected_loss


class ValueIteration:
    """Enhanced Value Iteration for discretized state space."""

    def __init__(self, env, gamma=0.95, theta=1e-4, reward_function='social_welfare'):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.reward_func_name = reward_function
        self.reward_function = getattr(RewardFunction, reward_function)

        # Discretization parameters
        self.wealth_bins = np.linspace(-2, 2, 20)
        self.group_bins = [0, 1]
        self.gap_bins = np.linspace(-1, 1, 10)
        self.default_rate_bins = np.linspace(0, 1, 10)
        self.actions = np.linspace(0, 1, 21)

        # Value function and policy
        self.V = defaultdict(float)
        self.Q = defaultdict(lambda: defaultdict(float))
        self.policy = defaultdict(lambda: 0.5)

        # Training history
        self.value_history = []
        self.policy_history = []

    def discretize_obs(self, obs):
        """Convert continuous observation to discrete state."""
        wealth, group, mu_R, mu_B, _, _, default_R, default_B = obs

        wealth_idx = np.digitize(wealth, self.wealth_bins)
        group_idx = int(group)
        gap_idx = np.digitize(mu_R - mu_B, self.gap_bins)
        default_idx = np.digitize((default_R + default_B) / 2, self.default_rate_bins)

        return (wealth_idx, group_idx, gap_idx, default_idx)

    def value_iteration(self, max_iterations=200):
        """Perform value iteration with detailed tracking."""
        print(f"Starting Value Iteration with {self.reward_func_name} reward...")

        for iteration in range(max_iterations):
            delta = 0
            state_updates = {}

            # Iterate over all possible states
            for w_idx in range(len(self.wealth_bins) + 1):
                for g_idx in self.group_bins:
                    for gap_idx in range(len(self.gap_bins) + 1):
                        for def_idx in range(len(self.default_rate_bins) + 1):
                            state = (w_idx, g_idx, gap_idx, def_idx)
                            v_old = self.V[state]

                            # Compute Q-values for all actions
                            for action in self.actions:
                                # Simulate expected reward
                                mock_obs = self._state_to_obs(state)
                                mock_info = self._create_mock_info(mock_obs)
                                reward = self.reward_function(self.env, action, mock_info)

                                # Simplified transition (assume state persistence with decay)
                                next_value = self.gamma * self.V[state]
                                self.Q[state][action] = reward + next_value

                            # Update value function
                            if self.Q[state]:
                                best_action = max(self.Q[state].keys(), key=lambda a: self.Q[state][a])
                                self.V[state] = self.Q[state][best_action]
                                self.policy[state] = best_action
                                delta = max(delta, abs(v_old - self.V[state]))
                                state_updates[state] = self.V[state]

            # Store history
            self.value_history.append(dict(self.V))
            self.policy_history.append(dict(self.policy))

            if iteration % 10 == 0:
                print(f"  Iteration {iteration}: delta = {delta:.6f}, states updated = {len(state_updates)}")

            if delta < self.theta:
                print(f"Value Iteration converged in {iteration + 1} iterations")
                break

        return self.V, self.policy

    def _state_to_obs(self, state):
        """Convert discrete state back to observation format."""
        w_idx, g_idx, gap_idx, def_idx = state

        wealth = self.wealth_bins[min(w_idx, len(self.wealth_bins) - 1)]
        group = float(g_idx)
        gap = self.gap_bins[min(gap_idx, len(self.gap_bins) - 1)]
        mu_R = 0.5 + gap / 2
        mu_B = 0.5 - gap / 2
        default_rate = self.default_rate_bins[min(def_idx, len(self.default_rate_bins) - 1)]

        return np.array([wealth, group, mu_R, mu_B, 1.0, 1.0, default_rate, default_rate])

    def _create_mock_info(self, obs):
        """Create mock info dictionary for reward calculation."""
        wealth, group, _, _, _, _, _, _ = obs
        return {
            'applicant': {
                'wealth': wealth,
                'group': 'R' if group == 1 else 'B',
                'group_id': int(group),
                'default_prob': self.env._default_probability(wealth)
            }
        }

    def get_action(self, obs):
        """Get action from learned policy."""
        state = self.discretize_obs(obs)
        return self.policy.get(state, 0.5)

    def save(self, filepath):
        """Save the value iteration agent."""
        save_dict = {
            'V': dict(self.V),
            'Q': {k: dict(v) for k, v in self.Q.items()},
            'policy': dict(self.policy),
            'value_history': self.value_history,
            'policy_history': self.policy_history,
            'gamma': self.gamma,
            'theta': self.theta,
            'reward_func_name': self.reward_func_name,
            'wealth_bins': self.wealth_bins,
            'group_bins': self.group_bins,
            'gap_bins': self.gap_bins,
            'default_rate_bins': self.default_rate_bins,
            'actions': self.actions
        }
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
        print(f"  Saved Value Iteration agent to {filepath}")

    @classmethod
    def load(cls, filepath, env):
        """Load a saved value iteration agent."""
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)

        agent = cls(env,
                   gamma=save_dict['gamma'],
                   theta=save_dict['theta'],
                   reward_function=save_dict['reward_func_name'])

        agent.V = defaultdict(float, save_dict['V'])
        agent.Q = defaultdict(lambda: defaultdict(float),
                             {k: defaultdict(float, v) for k, v in save_dict['Q'].items()})
        agent.policy = defaultdict(lambda: 0.5, save_dict['policy'])
        agent.value_history = save_dict['value_history']
        agent.policy_history = save_dict['policy_history']
        agent.wealth_bins = save_dict['wealth_bins']
        agent.group_bins = save_dict['group_bins']
        agent.gap_bins = save_dict['gap_bins']
        agent.default_rate_bins = save_dict['default_rate_bins']
        agent.actions = save_dict['actions']

        print(f"  Loaded Value Iteration agent from {filepath}")
        return agent


class PolicyGradient:
    """Policy Gradient with REINFORCE algorithm."""

    def __init__(self, env, hidden_dim=128, lr=1e-3, reward_function='social_welfare'):
        self.env = env
        self.reward_func_name = reward_function
        self.reward_function = getattr(RewardFunction, reward_function)
        self.hidden_dim = hidden_dim
        self.lr = lr

        # Neural network policy
        self.policy_net = PolicyNetwork(8, hidden_dim)  # 8 observation dimensions
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Training parameters
        self.gamma = 0.99

        # History tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.per_step_rewards = []
        self.loss_history = []

    def get_action(self, obs):
        """Sample action from policy network."""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)

        with torch.no_grad():
            alpha, beta = self.policy_net(obs_tensor)

        # Sample from Beta distribution for action in [0, 1]
        dist = Beta(alpha, beta)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob.item()

    def train_episode(self):
        """Train for one episode using REINFORCE."""
        obs, _ = self.env.reset()

        states = []
        actions = []
        log_probs = []
        rewards = []

        done = False
        step_count = 0

        while not done:
            # Get action from policy
            obs_tensor = torch.FloatTensor(obs)
            states.append(obs_tensor)

            alpha, beta = self.policy_net(obs_tensor.unsqueeze(0))
            dist = Beta(alpha, beta)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            actions.append(action)
            log_probs.append(log_prob)

            # Step environment
            next_obs, _, terminated, truncated, info = self.env.step(action.numpy())
            done = terminated or truncated

            # Calculate reward
            reward = self.reward_function(self.env, action.item(), info)
            rewards.append(reward)
            self.per_step_rewards.append(reward)

            obs = next_obs
            step_count += 1

        # Calculate returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # Normalize

        # Calculate loss
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)

        loss = torch.stack(policy_loss).sum()

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Record statistics
        episode_reward = sum(rewards)
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(step_count)
        self.loss_history.append(loss.item())

        return episode_reward

    def train(self, num_episodes=100):
        """Train the policy gradient agent."""
        print(f"Training Policy Gradient with {self.reward_func_name} reward...")

        for episode in range(num_episodes):
            episode_reward = self.train_episode()

            if episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else episode_reward
                print(f"  Episode {episode}: Reward = {episode_reward:.3f}, Avg = {avg_reward:.3f}")

        return self.episode_rewards

    def save(self, filepath):
        """Save the policy gradient agent."""
        save_dict = {
            'policy_net_state': self.policy_net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'per_step_rewards': self.per_step_rewards,
            'loss_history': self.loss_history,
            'gamma': self.gamma,
            'hidden_dim': self.hidden_dim,
            'lr': self.lr,
            'reward_func_name': self.reward_func_name
        }
        torch.save(save_dict, filepath)
        print(f"  Saved Policy Gradient agent to {filepath}")

    @classmethod
    def load(cls, filepath, env):
        """Load a saved policy gradient agent."""
        save_dict = torch.load(filepath)

        agent = cls(env,
                   hidden_dim=save_dict['hidden_dim'],
                   lr=save_dict['lr'],
                   reward_function=save_dict['reward_func_name'])

        agent.policy_net.load_state_dict(save_dict['policy_net_state'])
        agent.optimizer.load_state_dict(save_dict['optimizer_state'])
        agent.episode_rewards = save_dict['episode_rewards']
        agent.episode_lengths = save_dict['episode_lengths']
        agent.per_step_rewards = save_dict['per_step_rewards']
        agent.loss_history = save_dict['loss_history']
        agent.gamma = save_dict['gamma']

        print(f"  Loaded Policy Gradient agent from {filepath}")
        return agent


class PolicyNetwork(nn.Module):
    """Neural network for policy gradient."""

    def __init__(self, input_dim, hidden_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

        # Output alpha and beta for Beta distribution
        self.alpha_head = nn.Linear(hidden_dim, 1)
        self.beta_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # Ensure positive parameters for Beta distribution
        alpha = F.softplus(self.alpha_head(x)) + 1.0
        beta = F.softplus(self.beta_head(x)) + 1.0

        return alpha, beta


class DiscretizedQLearning:
    """Q-Learning with discretized continuous action space."""

    def __init__(self, env, n_action_bins=11, learning_rate=0.1, epsilon=0.1,
                 gamma=0.95, reward_function='social_welfare'):
        self.env = env
        self.n_action_bins = n_action_bins
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.reward_func_name = reward_function
        self.reward_function = getattr(RewardFunction, reward_function)

        # Discretize action space
        self.action_bins = np.linspace(0, 1, n_action_bins)

        # Q-table
        self.q_table = defaultdict(lambda: defaultdict(float))

        # State discretization
        self.state_bins = {
            'wealth': np.linspace(-2, 2, 20),
            'group': [0, 1],
            'mu_gap': np.linspace(-1, 1, 10),
            'default_rate': np.linspace(0, 1, 10)
        }

        # Training history
        self.episode_rewards = []
        self.per_step_rewards = []
        self.q_values_history = []

    def discretize_state(self, obs):
        """Convert continuous observation to discrete state."""
        wealth, group, mu_R, mu_B, _, _, default_R, default_B = obs

        wealth_bin = np.digitize(wealth, self.state_bins['wealth'])
        group_bin = int(group)
        gap_bin = np.digitize(mu_R - mu_B, self.state_bins['mu_gap'])
        default_bin = np.digitize((default_R + default_B) / 2, self.state_bins['default_rate'])

        return (wealth_bin, group_bin, gap_bin, default_bin)

    def get_action(self, state, training=True):
        """Epsilon-greedy action selection."""
        if training and np.random.random() < self.epsilon:
            return np.random.choice(len(self.action_bins))

        q_values = [self.q_table[state][a] for a in range(len(self.action_bins))]
        return np.argmax(q_values)

    def update_q(self, state, action, reward, next_state, done):
        """Update Q-value using Q-learning update rule."""
        if done:
            target = reward
        else:
            next_q_values = [self.q_table[next_state][a] for a in range(len(self.action_bins))]
            target = reward + self.gamma * max(next_q_values)

        current_q = self.q_table[state][action]
        self.q_table[state][action] = current_q + self.learning_rate * (target - current_q)

    def train(self, num_episodes=100):
        """Train the Q-learning agent."""
        print(f"Training Q-Learning with {self.reward_func_name} reward...")

        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            state = self.discretize_state(obs)
            episode_reward = 0
            done = False
            step_count = 0

            while not done:
                action_idx = self.get_action(state, training=True)
                action = np.array([self.action_bins[action_idx]])

                next_obs, _, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # Calculate reward
                reward = self.reward_function(self.env, action[0], info)
                self.per_step_rewards.append(reward)

                next_state = self.discretize_state(next_obs) if not done else None

                self.update_q(state, action_idx, reward, next_state, done)

                state = next_state
                episode_reward += reward
                step_count += 1

            self.episode_rewards.append(episode_reward)

            # Decay epsilon
            self.epsilon = max(0.01, self.epsilon * 0.995)

            # Store Q-values snapshot
            if episode % 10 == 0:
                q_snapshot = {state: dict(actions) for state, actions in self.q_table.items()}
                self.q_values_history.append(q_snapshot)

                avg_reward = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else episode_reward
                print(f"  Episode {episode}: Reward = {episode_reward:.3f}, Avg = {avg_reward:.3f}, Epsilon = {self.epsilon:.3f}")

        return self.episode_rewards

    def save(self, filepath):
        """Save the Q-learning agent."""
        save_dict = {
            'q_table': {k: dict(v) for k, v in self.q_table.items()},
            'action_bins': self.action_bins,
            'state_bins': self.state_bins,
            'episode_rewards': self.episode_rewards,
            'per_step_rewards': self.per_step_rewards,
            'q_values_history': self.q_values_history,
            'n_action_bins': self.n_action_bins,
            'learning_rate': self.learning_rate,
            'epsilon': self.epsilon,
            'gamma': self.gamma,
            'reward_func_name': self.reward_func_name
        }
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
        print(f"  Saved Q-Learning agent to {filepath}")

    @classmethod
    def load(cls, filepath, env):
        """Load a saved Q-learning agent."""
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)

        agent = cls(env,
                   n_action_bins=save_dict['n_action_bins'],
                   learning_rate=save_dict['learning_rate'],
                   epsilon=save_dict['epsilon'],
                   gamma=save_dict['gamma'],
                   reward_function=save_dict['reward_func_name'])

        agent.q_table = defaultdict(lambda: defaultdict(float),
                                   {k: defaultdict(float, v) for k, v in save_dict['q_table'].items()})
        agent.action_bins = save_dict['action_bins']
        agent.state_bins = save_dict['state_bins']
        agent.episode_rewards = save_dict['episode_rewards']
        agent.per_step_rewards = save_dict['per_step_rewards']
        agent.q_values_history = save_dict['q_values_history']

        print(f"  Loaded Q-Learning agent from {filepath}")
        return agent


class CheckpointManager:
    """Manages saving and loading of experiment checkpoints."""

    def __init__(self, base_dir='./checkpoints'):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.base_dir / 'checkpoint_status.json'
        self.load_checkpoint_status()

    def load_checkpoint_status(self):
        """Load checkpoint status from file."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                self.status = json.load(f)
        else:
            self.status = {}

    def save_checkpoint_status(self):
        """Save checkpoint status to file."""
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.status, f, indent=2)

    def is_completed(self, reward_func, method):
        """Check if a specific reward+method combination is completed."""
        key = f"{reward_func}_{method}"
        return self.status.get(key, {}).get('completed', False)

    def mark_completed(self, reward_func, method, timestamp=None):
        """Mark a reward+method combination as completed."""
        key = f"{reward_func}_{method}"
        self.status[key] = {
            'completed': True,
            'timestamp': timestamp or datetime.now().isoformat()
        }
        self.save_checkpoint_status()

    def get_agent_path(self, reward_func, method):
        """Get the filepath for an agent checkpoint."""
        reward_dir = self.base_dir / reward_func
        reward_dir.mkdir(parents=True, exist_ok=True)

        if method == 'policy_gradient':
            return reward_dir / f'{method}_agent.pt'
        else:
            return reward_dir / f'{method}_agent.pkl'

    def get_history_path(self, reward_func, method):
        """Get the filepath for test history."""
        reward_dir = self.base_dir / reward_func
        reward_dir.mkdir(parents=True, exist_ok=True)
        return reward_dir / f'{method}_history.pkl'

    def save_agent(self, agent, reward_func, method):
        """Save an agent to checkpoint."""
        filepath = self.get_agent_path(reward_func, method)
        agent.save(filepath)

    def load_agent(self, env, reward_func, method):
        """Load an agent from checkpoint."""
        filepath = self.get_agent_path(reward_func, method)
        if not filepath.exists():
            return None

        if method == 'q_learning':
            return DiscretizedQLearning.load(filepath, env)
        elif method == 'value_iteration':
            return ValueIteration.load(filepath, env)
        elif method == 'policy_gradient':
            return PolicyGradient.load(filepath, env)
        return None

    def save_history(self, history, reward_func, method):
        """Save test history to checkpoint."""
        filepath = self.get_history_path(reward_func, method)
        with open(filepath, 'wb') as f:
            pickle.dump(history, f)

    def load_history(self, reward_func, method):
        """Load test history from checkpoint."""
        filepath = self.get_history_path(reward_func, method)
        if not filepath.exists():
            return None
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class ComprehensiveRLPipeline:
    """Complete pipeline for training and comparing RL methods with checkpointing."""

    def __init__(self, env_params=None, checkpoint_dir='./checkpoints'):
        self.env_params = env_params or {
            'mu_R_init': 0.6,
            'mu_B_init': 0.4,
            'sigma': 0.2,
            'N_R': 500,
            'N_B': 500,
            'T': 200,
            'dt': 0.5,
            'loan_amount': 0.2,
            'interest_rate': 0.05,
            'default_noise_std': 0.1
        }

        self.reward_functions = ['social_welfare', 'rawlsian_maximin',
                                'fairness_lagrangian', 'utilitarian_profit']
        self.methods = ['q_learning', 'value_iteration', 'policy_gradient']

        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)

    def run_experiment(self, num_episodes=100, skip_completed=True):
        """Run complete experiment with all methods and rewards."""
        results = {}

        for reward_func in self.reward_functions:
            print(f"\n{'='*70}")
            print(f"REWARD FUNCTION: {reward_func.upper()}")
            print(f"{'='*70}")

            results[reward_func] = {}

            for method in self.methods:
                print(f"\n--- Processing {method.upper()} ---")

                # Check if already completed
                if skip_completed and self.checkpoint_manager.is_completed(reward_func, method):
                    print(f"  ✓ Already completed, loading from checkpoint...")

                    # Load agent and history
                    env = CreditAllocationEnvTwoLevel(**self.env_params)
                    agent = self.checkpoint_manager.load_agent(env, reward_func, method)
                    test_history = self.checkpoint_manager.load_history(reward_func, method)

                    if agent and test_history:
                        results[reward_func][method] = {
                            'agent': agent,
                            'test_history': test_history
                        }

                        if hasattr(agent, 'episode_rewards'):
                            results[reward_func][method]['episode_rewards'] = agent.episode_rewards
                        if hasattr(agent, 'per_step_rewards'):
                            results[reward_func][method]['per_step_rewards'] = agent.per_step_rewards
                        if hasattr(agent, 'value_function'):
                            results[reward_func][method]['value_function'] = agent.V
                        if hasattr(agent, 'policy'):
                            results[reward_func][method]['policy'] = agent.policy

                        self._print_method_summary(test_history, method)
                        env.close()
                        continue
                    else:
                        print(f"  ⚠ Checkpoint found but failed to load, retraining...")

                # Train from scratch
                env = CreditAllocationEnvTwoLevel(**self.env_params)

                try:
                    if method == 'q_learning':
                        agent = DiscretizedQLearning(env, reward_function=reward_func)
                        episode_rewards = agent.train(num_episodes=num_episodes)
                        test_history = self._test_agent(env, agent, 'q_learning')

                        results[reward_func][method] = {
                            'agent': agent,
                            'episode_rewards': episode_rewards,
                            'per_step_rewards': agent.per_step_rewards,
                            'test_history': test_history
                        }

                    elif method == 'value_iteration':
                        agent = ValueIteration(env, reward_function=reward_func)
                        V, policy = agent.value_iteration(max_iterations=200)
                        test_history = self._test_agent(env, agent, 'value_iteration')

                        results[reward_func][method] = {
                            'agent': agent,
                            'value_function': V,
                            'policy': policy,
                            'test_history': test_history
                        }

                    elif method == 'policy_gradient':
                        agent = PolicyGradient(env, reward_function=reward_func)
                        episode_rewards = agent.train(num_episodes=num_episodes)
                        test_history = self._test_agent(env, agent, 'policy_gradient')

                        results[reward_func][method] = {
                            'agent': agent,
                            'episode_rewards': episode_rewards,
                            'per_step_rewards': agent.per_step_rewards,
                            'test_history': test_history
                        }

                    # Save checkpoint
                    print(f"  💾 Saving checkpoint...")
                    self.checkpoint_manager.save_agent(agent, reward_func, method)
                    self.checkpoint_manager.save_history(test_history, reward_func, method)
                    self.checkpoint_manager.mark_completed(reward_func, method)
                    print(f"  ✓ Checkpoint saved successfully")

                    # Print summary
                    self._print_method_summary(test_history, method)

                except Exception as e:
                    print(f"  ✗ Error training {method}: {e}")
                    import traceback
                    traceback.print_exc()
                    results[reward_func][method] = None

                env.close()

        return results

    def _test_agent(self, env, agent, agent_type):
        """Test a trained agent and return history."""
        obs, _ = env.reset()
        done = False

        while not done:
            if agent_type == 'q_learning':
                state = agent.discretize_state(obs)
                action_idx = agent.get_action(state, training=False)
                action = np.array([agent.action_bins[action_idx]])
            elif agent_type == 'value_iteration':
                action_val = agent.get_action(obs)
                action = np.array([action_val])
            elif agent_type == 'policy_gradient':
                action_val, _ = agent.get_action(obs)
                action = np.array([action_val])

            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        return env.history.copy()

    def _print_method_summary(self, history, method):
        """Print summary for a method."""
        final_gap = history['mu_R'][-1] - history['mu_B'][-1]
        total_approvals = sum(history['loan_approvals_R']) + sum(history['loan_approvals_B'])
        total_defaults = sum(history['defaults_R']) + sum(history['defaults_B'])
        total_profit = sum(history['profit'])

        print(f"  Results for {method}:")
        print(f"    Final wealth gap: {final_gap:.4f}")
        print(f"    Total approvals: {total_approvals}")
        print(f"    Total defaults: {total_defaults}")
        print(f"    Total profit: {total_profit:.4f}")

    def plot_profit_dynamics(self, results):
        """Create comprehensive profit plots for all reward functions and agents."""
        # Create a large figure with subplots for each combination
        n_rewards = len(self.reward_functions)
        n_methods = len(self.methods)

        fig = plt.figure(figsize=(20, 5 * n_rewards))
        gs = gridspec.GridSpec(n_rewards, n_methods, figure=fig, hspace=0.3, wspace=0.3)

        # Color scheme for methods
        method_colors = {
            'q_learning': 'tab:blue',
            'value_iteration': 'tab:orange',
            'policy_gradient': 'tab:green'
        }

        for i, reward_func in enumerate(self.reward_functions):
            for j, method in enumerate(self.methods):
                ax = fig.add_subplot(gs[i, j])

                # Get data
                node = results.get(reward_func, {}).get(method)
                if not node:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
                    ax.set_title(f'{reward_func.replace("_", " ").title()}\n{method.replace("_", " ").title()}')
                    ax.axis('off')
                    continue

                history = node.get('test_history')
                if not history:
                    ax.text(0.5, 0.5, 'No History', ha='center', va='center', fontsize=14)
                    ax.set_title(f'{reward_func.replace("_", " ").title()}\n{method.replace("_", " ").title()}')
                    ax.axis('off')
                    continue

                # Plot cumulative profit over time
                cum_profit = np.cumsum(history['profit'])
                color = method_colors.get(method, 'black')

                ax.plot(history['time'], cum_profit, color=color, linewidth=2, alpha=0.9)
                ax.fill_between(history['time'], cum_profit, 0, alpha=0.3, color=color)

                # Formatting
                ax.set_xlabel('Time', fontsize=10)
                ax.set_ylabel('Cumulative Profit', fontsize=10)
                ax.set_title(f'{reward_func.replace("_", " ").title()}\n{method.replace("_", " ").title()}',
                           fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3)

                # Add final profit annotation
                final_profit = cum_profit[-1]
                ax.text(0.95, 0.95, f'Final: {final_profit:.2f}',
                       transform=ax.transAxes,
                       ha='right', va='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                       fontsize=9)

                # Color code positive/negative
                if final_profit < 0:
                    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=1)
                else:
                    ax.axhline(y=0, color='green', linestyle='--', alpha=0.7, linewidth=1)

        fig.suptitle('Loan Approver Profit Dynamics: All Reward Functions × All Agents',
                    fontsize=16, fontweight='bold', y=0.995)

        plt.tight_layout()
        return fig

    def plot_profit_comparison_summary(self, results):
        """Create summary comparison plots for profit across all combinations."""
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

        method_colors = {
            'q_learning': 'tab:blue',
            'value_iteration': 'tab:orange',
            'policy_gradient': 'tab:green'
        }

        method_linestyle = {
            'q_learning': '-',
            'policy_gradient': '--',
            'value_iteration': '-.'
        }

        # Plot 1: All profit trajectories overlaid
        ax1 = fig.add_subplot(gs[0, :])
        for reward_func in self.reward_functions:
            for method in self.methods:
                node = results.get(reward_func, {}).get(method)
                if not node:
                    continue
                history = node.get('test_history')
                if not history:
                    continue

                cum_profit = np.cumsum(history['profit'])
                ax1.plot(history['time'], cum_profit,
                        color=method_colors.get(method, 'black'),
                        linestyle=method_linestyle.get(method, '-'),
                        linewidth=1.5,
                        alpha=0.7,
                        label=f"{reward_func.replace('_', ' ')} | {method.replace('_', ' ')}")

        ax1.set_xlabel('Time', fontsize=12)
        ax1.set_ylabel('Cumulative Profit', fontsize=12)
        ax1.set_title('All Profit Trajectories', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax1.legend(fontsize=8, ncol=3, loc='upper left', bbox_to_anchor=(1.01, 1.0))

        # Plot 2: Final profit comparison (bars)
        ax2 = fig.add_subplot(gs[1, 0])
        final_profits = {m: [] for m in self.methods}
        for reward_func in self.reward_functions:
            for method in self.methods:
                node = results.get(reward_func, {}).get(method)
                if not node:
                    final_profits[method].append(0)
                    continue
                history = node.get('test_history')
                if not history:
                    final_profits[method].append(0)
                    continue
                final_profits[method].append(sum(history['profit']))

        x = np.arange(len(self.reward_functions))
        width = 0.25
        for i, method in enumerate(self.methods):
            ax2.bar(x + i*width, final_profits[method], width,
                   label=method.replace('_', ' ').title(),
                   color=method_colors.get(method, 'gray'))

        ax2.set_xlabel('Reward Function', fontsize=11)
        ax2.set_ylabel('Final Cumulative Profit', fontsize=11)
        ax2.set_title('Final Profit Comparison', fontsize=13, fontweight='bold')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels([rf.replace('_', ' ') for rf in self.reward_functions],
                           rotation=30, ha='right', fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax2.legend()

        # Plot 3: Profit per approval
        ax3 = fig.add_subplot(gs[1, 1])
        profit_per_approval = {m: [] for m in self.methods}
        for reward_func in self.reward_functions:
            for method in self.methods:
                node = results.get(reward_func, {}).get(method)
                if not node:
                    profit_per_approval[method].append(0)
                    continue
                history = node.get('test_history')
                if not history:
                    profit_per_approval[method].append(0)
                    continue
                total_approvals = sum(history['loan_approvals_R']) + sum(history['loan_approvals_B'])
                total_profit = sum(history['profit'])
                ppa = total_profit / max(total_approvals, 1)
                profit_per_approval[method].append(ppa)

        for i, method in enumerate(self.methods):
            ax3.bar(x + i*width, profit_per_approval[method], width,
                   label=method.replace('_', ' ').title(),
                   color=method_colors.get(method, 'gray'))

        ax3.set_xlabel('Reward Function', fontsize=11)
        ax3.set_ylabel('Profit per Approval', fontsize=11)
        ax3.set_title('Profit Efficiency (Profit/Approval)', fontsize=13, fontweight='bold')
        ax3.set_xticks(x + width)
        ax3.set_xticklabels([rf.replace('_', ' ') for rf in self.reward_functions],
                           rotation=30, ha='right', fontsize=9)
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax3.legend()

        # Plot 4: Profit rate (profit per time unit)
        ax4 = fig.add_subplot(gs[2, 0])
        for reward_func in self.reward_functions:
            for method in self.methods:
                node = results.get(reward_func, {}).get(method)
                if not node:
                    continue
                history = node.get('test_history')
                if not history:
                    continue

                # Calculate profit rate (derivative approximation)
                profit_rate = np.diff(np.cumsum(history['profit'])) / np.diff(history['time'])
                time_mid = (np.array(history['time'][:-1]) + np.array(history['time'][1:])) / 2

                ax4.plot(time_mid, profit_rate,
                        color=method_colors.get(method, 'black'),
                        linestyle=method_linestyle.get(method, '-'),
                        linewidth=1.2,
                        alpha=0.6,
                        label=f"{reward_func.replace('_', ' ')} | {method.replace('_', ' ')}")

        ax4.set_xlabel('Time', fontsize=11)
        ax4.set_ylabel('Profit Rate (per time unit)', fontsize=11)
        ax4.set_title('Instantaneous Profit Rate', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax4.legend(fontsize=7, ncol=2, loc='upper left', bbox_to_anchor=(1.01, 1.0))

        # Plot 5: Risk-adjusted profit (profit vs defaults)
        ax5 = fig.add_subplot(gs[2, 1])
        for reward_func in self.reward_functions:
            for method in self.methods:
                node = results.get(reward_func, {}).get(method)
                if not node:
                    continue
                history = node.get('test_history')
                if not history:
                    continue

                total_profit = sum(history['profit'])
                total_defaults = sum(history['defaults_R']) + sum(history['defaults_B'])
                total_approvals = sum(history['loan_approvals_R']) + sum(history['loan_approvals_B'])
                default_rate = total_defaults / max(total_approvals, 1)

                ax5.scatter(default_rate, total_profit,
                          s=150,
                          alpha=0.7,
                          color=method_colors.get(method, 'black'),
                          marker='o' if method == 'q_learning' else ('s' if method == 'value_iteration' else 'D'),
                          label=f"{reward_func.replace('_', ' ')} | {method.replace('_', ' ')}")

        ax5.set_xlabel('Default Rate', fontsize=11)
        ax5.set_ylabel('Total Profit', fontsize=11)
        ax5.set_title('Risk-Return Profile (Profit vs Default Rate)', fontsize=13, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax5.legend(fontsize=7, ncol=2, loc='upper left', bbox_to_anchor=(1.01, 1.0))

        fig.suptitle('Comprehensive Profit Analysis: Loan Approver Perspective',
                    fontsize=16, fontweight='bold', y=0.995)

        plt.tight_layout()
        return fig


    def plot_comprehensive_results(self, results):
      """Create comprehensive plots for all reward functions and all agents."""
      # Styling helpers
      method_linestyle = {
          'q_learning': '-',
          'policy_gradient': '--',
          'value_iteration': '-.'
      }
      method_markers = {
          'q_learning': 'o',
          'policy_gradient': 's',
          'value_iteration': 'D'
      }
      # colors per reward function (cycled if needed)
      palette = ['tab:red', 'tab:blue', 'tab:green', 'tab:purple', 'tab:orange', 'tab:brown']
      rf_color = {rf: palette[i % len(palette)] for i, rf in enumerate(self.reward_functions)}

      fig = plt.figure(figsize=(24, 20))
      gs = gridspec.GridSpec(5, 4, figure=fig)

      # ------------------------------
      # Row 1: Learning curves & Per-step reward histograms
      # ------------------------------
      # Plot 1: Learning curves (only agents that have episode_rewards)
      ax = fig.add_subplot(gs[0, :2])
      for rf in self.reward_functions:
          for m in self.methods:
              node = results.get(rf, {}).get(m)
              if not node:
                  continue
              rewards = node.get('episode_rewards')
              if rewards:
                  window = max(1, min(10, len(rewards)//10))
                  if window > 1:
                      smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
                  else:
                      smoothed = np.array(rewards)
                  label = f"{rf.replace('_', ' ')} | {m.replace('_', ' ')}"
                  ax.plot(
                      smoothed,
                      linestyle=method_linestyle.get(m, '-'),
                      marker=None,
                      linewidth=2,
                      color=rf_color[rf],
                      alpha=0.9,
                      label=label
                  )
      ax.set_title('Learning Curves (Episode Rewards)')
      ax.set_xlabel('Episode')
      ax.set_ylabel('Episode Reward')
      ax.grid(True, alpha=0.3)
      ax.legend(fontsize=8, ncol=2, loc='upper left', bbox_to_anchor=(1.02, 1.0))

      # Plot 2: Per-step reward distributions (only agents that have per_step_rewards)
      ax = fig.add_subplot(gs[0, 2:])
      for rf in self.reward_functions:
          for m in self.methods:
              node = results.get(rf, {}).get(m)
              if not node:
                  continue
              per_step = node.get('per_step_rewards')
              if per_step:
                  ax.hist(
                      per_step, bins=30, alpha=0.35,
                      color=rf_color[rf],
                      label=f"{rf.replace('_', ' ')} | {m.replace('_', ' ')}"
                  )
      ax.set_title('Per-Step Reward Distributions')
      ax.set_xlabel('Reward')
      ax.set_ylabel('Frequency')
      ax.grid(True, alpha=0.3)
      ax.legend(fontsize=8, ncol=2, loc='upper left', bbox_to_anchor=(1.02, 1.0))

      # ------------------------------
      # Row 2: Wealth gap over time & Net worth means
      # ------------------------------
      # Plot 3: Wealth Gap Over Time for all methods & reward functions
      ax = fig.add_subplot(gs[1, :2])
      for rf in self.reward_functions:
          for m in self.methods:
              node = results.get(rf, {}).get(m)
              if not node:
                  continue
              history = node.get('test_history')
              if not history:
                  continue
              gap = np.array(history['mu_R']) - np.array(history['mu_B'])
              ax.plot(
                  history['time'], gap,
                  color=rf_color[rf],
                  linestyle=method_linestyle.get(m, '-'),
                  linewidth=1.8,
                  alpha=0.9,
                  label=f"{rf.replace('_', ' ')} | {m.replace('_', ' ')}"
              )
      ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
      ax.set_xlabel('Time')
      ax.set_ylabel('Wealth Gap (μ_R - μ_B)')
      ax.set_title('Wealth Gap Evolution Over Time')
      ax.grid(True, alpha=0.3)
      ax.legend(fontsize=8, ncol=2, loc='upper left', bbox_to_anchor=(1.02, 1.0))

      # Plot 4: Net Worth Distribution Means Over Time for all methods & reward functions
      ax = fig.add_subplot(gs[1, 2:])
      for rf in self.reward_functions:
          for m in self.methods:
              node = results.get(rf, {}).get(m)
              if not node:
                  continue
              history = node.get('test_history')
              if not history:
                  continue
              # Red
              ax.plot(
                  history['time'], history['mu_R'],
                  color=rf_color[rf],
                  linestyle=method_linestyle.get(m, '-'),
                  linewidth=1.8,
                  alpha=0.9,
                  label=f"{rf.replace('_', ' ')} | {m.replace('_', ' ')} | Red"
              )
              # Blue
              ax.plot(
                  history['time'], history['mu_B'],
                  color=rf_color[rf],
                  linestyle=':',
                  linewidth=1.2,
                  alpha=0.9
              )
      ax.set_xlabel('Time')
      ax.set_ylabel('Mean Net Worth')
      ax.set_title('Net Worth Means Over Time (Red solid / Blue dotted)')
      ax.grid(True, alpha=0.3)
      ax.legend(fontsize=8, ncol=2, loc='upper left', bbox_to_anchor=(1.02, 1.0))

      # ------------------------------
      # Row 3: Variances & Default rates
      # ------------------------------
      # Plot 5: Net Worth Variances Over Time for all methods & reward functions
      ax = fig.add_subplot(gs[2, :2])
      for rf in self.reward_functions:
          for m in self.methods:
              node = results.get(rf, {}).get(m)
              if not node:
                  continue
              history = node.get('test_history')
              if not history:
                  continue
              # Red variance
              ax.plot(
                  history['time'], history['variance_R'],
                  color=rf_color[rf],
                  linestyle=method_linestyle.get(m, '-'),
                  linewidth=1.5,
                  alpha=0.85,
                  label=f"{rf.replace('_', ' ')} | {m.replace('_', ' ')} | Red"
              )
              # Blue variance (dotted)
              ax.plot(
                  history['time'], history['variance_B'],
                  color=rf_color[rf],
                  linestyle=':',
                  linewidth=1.2,
                  alpha=0.85
              )
      ax.set_xlabel('Time')
      ax.set_ylabel('Variance')
      ax.set_title('Net Worth Variances Over Time (Red solid / Blue dotted)')
      ax.grid(True, alpha=0.3)
      ax.legend(fontsize=8, ncol=2, loc='upper left', bbox_to_anchor=(1.02, 1.0))

      # Plot 6: Default rates over time for all methods & reward functions
      ax = fig.add_subplot(gs[2, 2:])
      for rf in self.reward_functions:
          for m in self.methods:
              node = results.get(rf, {}).get(m)
              if not node:
                  continue
              history = node.get('test_history')
              if not history:
                  continue
              # Red default rate
              ax.plot(
                  history['time'], history['default_rate_R'],
                  color=rf_color[rf],
                  linestyle=method_linestyle.get(m, '-'),
                  linewidth=1.5,
                  alpha=0.9,
                  label=f"{rf.replace('_', ' ')} | {m.replace('_', ' ')} | Red"
              )
              # Blue default rate (dotted)
              ax.plot(
                  history['time'], history['default_rate_B'],
                  color=rf_color[rf],
                  linestyle=':',
                  linewidth=1.2,
                  alpha=0.9
              )
      ax.set_xlabel('Time')
      ax.set_ylabel('Default Rate')
      ax.set_title('Default Rates Over Time (Red solid / Blue dotted)')
      ax.grid(True, alpha=0.3)
      ax.legend(fontsize=8, ncol=2, loc='upper left', bbox_to_anchor=(1.02, 1.0))

      # ------------------------------
      # Row 4: Final outcomes comparison & Value function dists
      # ------------------------------
      # Plot 7: Final Wealth Gap Comparison (bars) across all methods
      ax = fig.add_subplot(gs[3, :2])
      gaps = {m: [] for m in self.methods}
      for rf in self.reward_functions:
          for m in self.methods:
              node = results.get(rf, {}).get(m)
              if not node:
                  gaps[m].append(np.nan)
                  continue
              history = node.get('test_history')
              if not history:
                  gaps[m].append(np.nan)
                  continue
              final_gap = history['mu_R'][-1] - history['mu_B'][-1]
              gaps[m].append(final_gap)

      x = np.arange(len(self.reward_functions))
      width = 0.25
      for i, m in enumerate(self.methods):
          vals = gaps[m]
          if any(v is not None for v in vals):
              ax.bar(x + i*width, vals, width, label=m.replace('_', ' ').title())
      ax.set_xlabel('Reward Function')
      ax.set_ylabel('Final Wealth Gap')
      ax.set_title('Final Wealth Gap Comparison (All Methods)')
      ax.set_xticks(x + width)
      ax.set_xticklabels([rf.replace('_', ' ') for rf in self.reward_functions], rotation=30, ha='right')
      ax.grid(True, alpha=0.3)
      ax.legend()

      # Plot 8: Value Function Distributions (only for Value Iteration)
      ax = fig.add_subplot(gs[3, 2:])
      for rf in self.reward_functions:
          node = results.get(rf, {}).get('value_iteration')
          if not node:
              continue
          agent = node.get('agent')
          if not agent:
              continue
          vals = [v for v in agent.V.values() if v != 0]
          if vals:
              ax.hist(vals, bins=30, alpha=0.45, color=rf_color[rf], label=rf.replace('_', ' '))
      ax.set_title('Value Function Distributions (Value Iteration)')
      ax.set_xlabel('State Value')
      ax.set_ylabel('Frequency')
      ax.grid(True, alpha=0.3)
      ax.legend(fontsize=8)

      # ------------------------------
      # Row 5: Approvals and Profits comparison
      # ------------------------------
      ax = fig.add_subplot(gs[4, :2])
      approvals = {m: [] for m in self.methods}
      for rf in self.reward_functions:
          for m in self.methods:
              node = results.get(rf, {}).get(m)
              if not node:
                  approvals[m].append(np.nan)
                  continue
              history = node.get('test_history')
              if not history:
                  approvals[m].append(np.nan)
                  continue
              total_approvals = sum(history['loan_approvals_R']) + sum(history['loan_approvals_B'])
              approvals[m].append(total_approvals)

      for i, m in enumerate(self.methods):
          vals = approvals[m]
          if any(v is not None for v in vals):
              ax.bar(x + i*width, vals, width, label=m.replace('_', ' ').title())
      ax.set_xlabel('Reward Function')
      ax.set_ylabel('Total Approvals')
      ax.set_title('Total Approvals Comparison (All Methods)')
      ax.set_xticks(x + width)
      ax.set_xticklabels([rf.replace('_', ' ') for rf in self.reward_functions], rotation=30, ha='right')
      ax.grid(True, alpha=0.3)
      ax.legend()

      ax = fig.add_subplot(gs[4, 2:])
      profits = {m: [] for m in self.methods}
      for rf in self.reward_functions:
          for m in self.methods:
              node = results.get(rf, {}).get(m)
              if not node:
                  profits[m].append(np.nan)
                  continue
              history = node.get('test_history')
              if not history:
                  profits[m].append(np.nan)
                  continue
              profits[m].append(sum(history['profit']))
      for i, m in enumerate(self.methods):
          vals = profits[m]
          if any(v is not None for v in vals):
              ax.bar(x + i*width, vals, width, label=m.replace('_', ' ').title())
      ax.set_xlabel('Reward Function')
      ax.set_ylabel('Total Profit')
      ax.set_title('Total Profit Comparison (All Methods)')
      ax.set_xticks(x + width)
      ax.set_xticklabels([rf.replace('_', ' ') for rf in self.reward_functions], rotation=30, ha='right')
      ax.grid(True, alpha=0.3)
      ax.legend()

      plt.tight_layout()
      return fig


    def plot_detailed_dynamics(self, results, reward_func='social_welfare', method='policy_gradient'):
        """Create detailed plots for a specific reward function and method."""
        if not results.get(reward_func) or not results[reward_func].get(method):
            print(f"No results found for {reward_func} with {method}")
            return None

        history = results[reward_func][method]['test_history']

        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        fig.suptitle(f'Detailed Dynamics: {reward_func.replace("_", " ").title()} - {method.replace("_", " ").title()}',
                     fontsize=16, fontweight='bold')

        # Plot 1: Wealth Gap Evolution
        gap = [r - b for r, b in zip(history['mu_R'], history['mu_B'])]
        axes[0, 0].plot(history['time'], gap, 'purple', linewidth=2)
        axes[0, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[0, 0].fill_between(history['time'], gap, 0, where=[g > 0 for g in gap],
                                color='red', alpha=0.2, label='Red Advantage')
        axes[0, 0].fill_between(history['time'], gap, 0, where=[g < 0 for g in gap],
                                color='blue', alpha=0.2, label='Blue Advantage')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Wealth Gap (μ_R - μ_B)')
        axes[0, 0].set_title('Wealth Gap Evolution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Net Worth Means
        axes[0, 1].plot(history['time'], history['mu_R'], 'r-', label='Red Group', linewidth=2)
        axes[0, 1].plot(history['time'], history['mu_B'], 'b-', label='Blue Group', linewidth=2)
        axes[0, 1].axhline(y=self.env_params['mu_R_init'], color='red', linestyle=':', alpha=0.5)
        axes[0, 1].axhline(y=self.env_params['mu_B_init'], color='blue', linestyle=':', alpha=0.5)
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Mean Net Worth')
        axes[0, 1].set_title('Net Worth Distribution Means')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Net Worth Variances
        axes[0, 2].plot(history['time'], history['variance_R'], 'r:', label='Red Group', linewidth=2)
        axes[0, 2].plot(history['time'], history['variance_B'], 'b:', label='Blue Group', linewidth=2)
        axes[0, 2].axhline(y=self.env_params['sigma']**2, color='gray', linestyle='--',
                          alpha=0.5, label='Initial σ²')
        axes[0, 2].set_xlabel('Time')
        axes[0, 2].set_ylabel('Variance')
        axes[0, 2].set_title('Net Worth Distribution Variances')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # Plot 4: Cumulative Approvals
        cum_approvals_R = np.cumsum(history['loan_approvals_R'])
        cum_approvals_B = np.cumsum(history['loan_approvals_B'])
        axes[1, 0].plot(history['time'], cum_approvals_R, 'r-', label='Red Group', linewidth=2)
        axes[1, 0].plot(history['time'], cum_approvals_B, 'b-', label='Blue Group', linewidth=2)
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Cumulative Approvals')
        axes[1, 0].set_title('Cumulative Loan Approvals')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 5: Default Rates
        axes[1, 1].plot(history['time'], history['default_rate_R'], 'r--', label='Red Group', linewidth=2)
        axes[1, 1].plot(history['time'], history['default_rate_B'], 'b--', label='Blue Group', linewidth=2)
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Default Rate')
        axes[1, 1].set_title('Default Rates Over Time')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # Plot 6: Cumulative Defaults
        cum_defaults_R = np.cumsum(history['defaults_R'])
        cum_defaults_B = np.cumsum(history['defaults_B'])
        axes[1, 2].plot(history['time'], cum_defaults_R, 'r:', label='Red Group', linewidth=2)
        axes[1, 2].plot(history['time'], cum_defaults_B, 'b:', label='Blue Group', linewidth=2)
        axes[1, 2].set_xlabel('Time')
        axes[1, 2].set_ylabel('Cumulative Defaults')
        axes[1, 2].set_title('Cumulative Loan Defaults')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

        # Plot 7: Approval Rates
        approval_rate_R = [a/max(arr, 1) for a, arr in zip(history['loan_approvals_R'], history['loan_arrivals_R'])]
        approval_rate_B = [a/max(arr, 1) for a, arr in zip(history['loan_approvals_B'], history['loan_arrivals_B'])]
        axes[2, 0].plot(history['time'], approval_rate_R, 'r.', alpha=0.7, label='Red Group')
        axes[2, 0].plot(history['time'], approval_rate_B, 'b.', alpha=0.7, label='Blue Group')
        axes[2, 0].set_xlabel('Time')
        axes[2, 0].set_ylabel('Approval Rate')
        axes[2, 0].set_title('Instantaneous Approval Rates')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)

        # Plot 8: Profit Over Time
        cum_profit = np.cumsum(history['profit'])
        axes[2, 1].plot(history['time'], cum_profit, 'g-', linewidth=2)
        axes[2, 1].fill_between(history['time'], cum_profit, 0, alpha=0.3, color='green')
        axes[2, 1].set_xlabel('Time')
        axes[2, 1].set_ylabel('Cumulative Profit')
        axes[2, 1].set_title('Lender Profit Accumulation')
        axes[2, 1].grid(True, alpha=0.3)

        # Plot 9: Covariance Evolution
        axes[2, 2].plot(history['time'], history['covariance_R'], 'r-', label='Red Group', linewidth=2)
        axes[2, 2].plot(history['time'], history['covariance_B'], 'b-', label='Blue Group', linewidth=2)
        axes[2, 2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[2, 2].set_xlabel('Time')
        axes[2, 2].set_ylabel('Covariance')
        axes[2, 2].set_title('Wealth-Selection Covariance')
        axes[2, 2].legend()
        axes[2, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


def run_complete_demo(checkpoint_dir='./checkpoints', skip_completed=True):
    """Run complete demonstration with all methods and detailed analysis."""

    # Set random seed
    np.random.seed(42)
    torch.manual_seed(42)

    # Create pipeline
    pipeline = ComprehensiveRLPipeline(
        env_params={
            'mu_R_init': 0.6,
            'mu_B_init': 0.4,
            'sigma': 0.2,
            'N_R': 300,
            'N_B': 300,
            'T': 2000,
            'dt': 1.0,
            'loan_amount': 0.05,
            'interest_rate': 0.01,
            'default_noise_std': 0.1
        },
        checkpoint_dir=checkpoint_dir
    )

    # Run experiments with checkpointing
    print("\nStarting experiments...")
    results = pipeline.run_experiment(num_episodes=150, skip_completed=skip_completed)

    # Create comprehensive plots
    print("\nGenerating comprehensive plots...")
    fig1 = pipeline.plot_comprehensive_results(results)
    plt.show()

    # Create profit-focused plots
    print("\nGenerating profit dynamics plots for all combinations...")
    fig_profit_grid = pipeline.plot_profit_dynamics(results)
    plt.show()

    print("\nGenerating profit comparison summary...")
    fig_profit_summary = pipeline.plot_profit_comparison_summary(results)
    plt.show()

    # Create detailed dynamics plots
    print("\nGenerating detailed dynamics plots...")

    reward_method_pairs = [
        ('social_welfare', 'q_learning'),
        ('social_welfare', 'value_iteration'),
        ('social_welfare', 'policy_gradient'),
        ('fairness_lagrangian', 'q_learning'),
        ('fairness_lagrangian', 'value_iteration'),
        ('fairness_lagrangian', 'policy_gradient'),
        ('utilitarian_profit', 'q_learning'),
        ('utilitarian_profit', 'value_iteration'),
        ('utilitarian_profit', 'policy_gradient'),
        ('rawlsian_maximin', 'q_learning'),
        ('rawlsian_maximin', 'value_iteration'),
        ('rawlsian_maximin', 'policy_gradient'),
    ]

    for reward_func, method in reward_method_pairs:
        fig = pipeline.plot_detailed_dynamics(results, reward_func, method)
        if fig:
            plt.show()

    # Print final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    for reward_func in pipeline.reward_functions:
        print(f"\n{reward_func.upper()}:")
        print("-"*40)

        for method in pipeline.methods:
            if results[reward_func].get(method) and results[reward_func][method]:
                history = results[reward_func][method]['test_history']

                final_gap = history['mu_R'][-1] - history['mu_B'][-1]
                initial_gap = history['mu_R'][0] - history['mu_B'][0]
                gap_change = final_gap - initial_gap

                final_var_R = history['variance_R'][-1]
                final_var_B = history['variance_B'][-1]
                initial_var_R = history['variance_R'][0]
                initial_var_B = history['variance_B'][0]

                total_approvals_R = sum(history['loan_approvals_R'])
                total_approvals_B = sum(history['loan_approvals_B'])
                total_defaults_R = sum(history['defaults_R'])
                total_defaults_B = sum(history['defaults_B'])
                final_default_rate = (total_defaults_R + total_defaults_B) / max(total_approvals_R + total_approvals_B, 1)
                total_profit = sum(history['profit'])
                profit_per_approval = total_profit / max(total_approvals_R + total_approvals_B, 1)

                print(f"  {method.upper()}:")
                print(f"    Wealth Gap: {initial_gap:.4f} → {final_gap:.4f} (Δ={gap_change:.4f})")
                print(f"    Mean Wealth R: {history['mu_R'][0]:.4f} → {history['mu_R'][-1]:.4f}")
                print(f"    Mean Wealth B: {history['mu_B'][0]:.4f} → {history['mu_B'][-1]:.4f}")
                print(f"    Variance R: {initial_var_R:.4f} → {final_var_R:.4f}")
                print(f"    Variance B: {initial_var_B:.4f} → {final_var_B:.4f}")
                print(f"    Approvals (R/B): {total_approvals_R}/{total_approvals_B}")
                print(f"    Defaults (R/B): {total_defaults_R}/{total_defaults_B}")
                print(f"    Overall default rate: {final_default_rate:.3f}")
                print(f"    Total Profit: {total_profit:.4f}")
                print(f"    Profit per Approval: {profit_per_approval:.4f}")

    return results


if __name__ == "__main__":
    # Run the complete demonstration with checkpointing
    # Set skip_completed=False to retrain everything from scratch
    results = run_complete_demo(checkpoint_dir='/content/drive/MyDrive/RL_fairness/checkpoints', skip_completed=True)
