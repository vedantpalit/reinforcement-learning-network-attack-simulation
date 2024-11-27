import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque

def run_bruteforce_agent(env, step_limit, num_actions, verbose=True):
    """Run brute-force agent on nasim environment.

    Parameters
    ----------
    env : nasim.NASimEnv
        the nasim environment to run agent on
    step_limit : int, optional
        the maximum number of steps to run agent for (default=200)
    verbose : bool, optional
        whether to print out progress messages or not (default=True)

    Returns
    -------
    int
        timesteps agent ran for
    float
        the total reward received by agent
    bool
        whether the goal was reached or not
    """
    LINE_BREAK = "\n"
    monte_carlo=0
    # brute_count+=1
    if verbose:
        print(LINE_BREAK)
        print(f"STARTING EPISODE")
        print(LINE_BREAK)
        print("t: Reward")

    env.reset()  # Reset the environment to start fresh
    total_reward = 0
    done = False
    steps = 0
    cycle_complete = False
    env_step_limit_reached = False

    # Get the number of actions from the action space
    # num_actions = env.action_space.n  # Assumes a discrete action space

    act = 0

    # Main loop: iterate through actions until done or step limit reached
    while not done and steps < step_limit:
        act = (act + 1) % num_actions
        act=int(act)  # Ensure action is within valid range

        # Step through the environment with the chosen action
        obs, rew, done, info, _ = env.step(act)
        total_reward += rew

        cycle_complete = (steps > 0 and act == 0)
        if cycle_complete and verbose:
            print(f"{steps}: {total_reward}")

        steps += 1

    # Print out results at the end
    if (done or env_step_limit_reached) and verbose:
        print(LINE_BREAK)
        print(f"EPISODE FINISHED")
        print(LINE_BREAK)
        print(f"Goal Reached ={env.goal_reached()}")
        print(f"Total steps = {steps}")
        print(f"Total reward = {total_reward}")

    elif verbose:
        print(LINE_BREAK)
        print("STEP LIMIT REACHED")
        monte_carlo+=1
        print(LINE_BREAK)

    if done:
        done = env.goal_reached()


    return steps, total_reward, done,monte_carlo

def run_random_agent(env, step_limit, num_actions, verbose=True):
    LINE_BREAK = "\n"
    monte_carlo=0
    if verbose:
        print(LINE_BREAK)
        print(f"STARTING EPISODE")
        print(LINE_BREAK)
        print(f"t: Reward")

    env.reset()
    total_reward = 0
    done = False
    env_step_limit_reached = False
    t = 0
    a = 0

    while not done and not env_step_limit_reached and t < step_limit:
        a = random.randint(0, num_actions - 1)
        _, r, done, env_step_limit_reached, _ = env.step(a)
        total_reward += r
        if (t+1) % 100 == 0 and verbose:
            print(f"{t}: {total_reward}")
        t += 1

    if (done or env_step_limit_reached) and verbose:
        print(LINE_BREAK)
        print(f"EPISODE FINISHED")
        print(LINE_BREAK)
        print(f"Goal Reached ={env.goal_reached()}")
        print(f"Total steps = {t}")
        print(f"Total reward = {total_reward}")

    elif verbose:
        print(LINE_BREAK)
        print("STEP LIMIT REACHED")
        monte_carlo+=1
        print(LINE_BREAK)

    if done:
        done = env.goal_reached()

    return t, total_reward, done,monte_carlo

def run_q_learning_agent(env, num_eps, step_limit, num_actions, alpha, gamma, epsilon, epsilon_decay, min_epsilon, verbose=True):
    """Run Q-learning agent on the given environment.

    Parameters
    ----------
    env : gym.Env or custom environment
        The environment to run the Q-learning agent on.
    num_eps : int
        Number of episodes to run the agent for.
    step_limit : int
        The maximum number of steps to run each episode for.
    num_actions : int
        The number of actions the agent can take in the environment.
    alpha : float
        The learning rate.
    gamma : float
        The discount factor.
    epsilon : float
        The exploration rate for epsilon-greedy policy.
    epsilon_decay : float
        The factor by which epsilon is multiplied after each episode.
    min_epsilon : float
        The minimum value that epsilon can decay to.
    verbose : bool, optional
        Whether to print out progress messages or not (default=True).

    Returns
    -------
    q_table : dict
        The learned Q-table mapping states to actions.
    cumulative_steps : int
        The total steps taken across all episodes.
    cumulative_reward : float
        The total reward accumulated across all episodes.
    goal_reached : bool
        Whether the goal was reached in any episode.
    """
    monte_carlo=0
    q_table = {}
    cumulative_steps = 0
    cumulative_reward = 0
    goal_reached = False
    episode_arr = [i for i in range(num_eps)]

    for idx, episode in enumerate(episode_arr):
        episode_reward = 0
        episode += 1

        if verbose:
            print(f"STARTING EPISODE NUMBER {episode}")

        # Reset environment and get the initial state
        state, _ = env.reset()
        state = tuple(state)
        done = False
        steps = 0

        while not done and steps < step_limit:
            # If the state is not in the Q-table, initialize it
            if state not in q_table:
                q_table[state] = {action: 0 for action in range(num_actions)}

            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = random.randint(0, num_actions - 1)  # Random action (exploration)
            else:
                action = max(q_table[state], key=q_table[state].get)  # Greedy action (exploitation)

            # Take the chosen action and observe the result
            next_state, reward, done, reached_step_limit, info = env.step(action)
            episode_reward += reward
            cumulative_steps += 1
            cumulative_reward += reward
            steps += 1

            next_state = tuple(next_state)  # Convert next_state to a tuple

            # If next_state is not in the Q-table, initialize it
            if next_state not in q_table:
                q_table[next_state] = {a: 0 for a in range(num_actions)}

            # Q-learning update rule
            best_next_action = max(q_table[next_state], key=q_table[next_state].get)
            q_table[state][action] = q_table[state][action] + alpha * (
                reward + gamma * q_table[next_state][best_next_action] - q_table[state][action]
            )

            # Transition to the next state
            state = next_state

            if steps >= step_limit:
                reached_step_limit = True
                if verbose:
                    print(f"Step limit of {step_limit} reached. Ending episode.")
                    monte_carlo+=1
                break

        # Check if the goal was reached
        if done:
            goal_reached = env.goal_reached()

        # Print episode result
        if verbose:
            print(f"Reward: {episode_reward}, Goal Reached: {goal_reached}, Step Limit Reached: {reached_step_limit}")

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        if verbose:
            print(f"EPISODE NUMBER {episode} FINISHED")
            #env.render_network_graph(show=True)

    return q_table, cumulative_steps, cumulative_reward, goal_reached,monte_carlo


def run_sarsa_agent(env, num_eps, step_limit, num_actions, alpha, gamma, epsilon, epsilon_decay, min_epsilon, verbose=True):
    """Run SARSA agent on the given environment.

    Parameters
    ----------
    env : gym.Env or custom environment
        The environment to run the SARSA agent on.
    num_eps : int
        Number of episodes to run the agent for.
    step_limit : int
        The maximum number of steps to run each episode for.
    num_actions : int
        The number of actions the agent can take in the environment.
    alpha : float
        The learning rate.
    gamma : float
        The discount factor.
    epsilon : float
        The exploration rate for epsilon-greedy policy.
    epsilon_decay : float
        The factor by which epsilon is multiplied after each episode.
    min_epsilon : float
        The minimum value that epsilon can decay to.
    verbose : bool, optional
        Whether to print out progress messages or not (default=True).

    Returns
    -------
    q_table : dict
        The learned Q-table mapping states to actions.
    cumulative_steps : int
        The total steps taken across all episodes.
    cumulative_reward : float
        The total reward accumulated across all episodes.
    goal_reached : bool
        Whether the goal was reached in any episode.
    """
    monte_carlo=0
    q_table = {}
    cumulative_steps = 0
    cumulative_reward = 0
    goal_reached = False
    episode_arr = [i for i in range(num_eps)]

    for idx, episode in enumerate(episode_arr):
        episode_reward = 0
        episode += 1

        if verbose:
            print(f"STARTING EPISODE NUMBER {episode}")

        # Reset environment and get the initial state
        state, _ = env.reset()
        state = tuple(state)
        done = False
        steps = 0

        # Epsilon-greedy action selection for the initial state
        if state not in q_table:
            q_table[state] = {action: 0 for action in range(num_actions)}

        if np.random.rand() < epsilon:
            action = random.randint(0, num_actions - 1)  # Random action (exploration)
        else:
            action = max(q_table[state], key=q_table[state].get)  # Greedy action (exploitation)

        while not done and steps < step_limit:
            # Take the chosen action and observe the result
            next_state, reward, done, reached_step_limit, info = env.step(action)
            episode_reward += reward
            cumulative_steps += 1
            cumulative_reward += reward
            steps += 1

            next_state = tuple(next_state)  # Convert next_state to a tuple

            # Epsilon-greedy action selection for the next state
            if next_state not in q_table:
                q_table[next_state] = {a: 0 for a in range(num_actions)}

            if np.random.rand() < epsilon:
                next_action = random.randint(0, num_actions - 1)  # Random action (exploration)
            else:
                next_action = max(q_table[next_state], key=q_table[next_state].get)  # Greedy action (exploitation)

            # SARSA update rule
            q_table[state][action] = q_table[state][action] + alpha * (
                reward + gamma * q_table[next_state][next_action] - q_table[state][action]
            )

            # Transition to the next state and action
            state = next_state
            action = next_action

            if steps >= step_limit:
                reached_step_limit = True
                if verbose:
                    print(f"Step limit of {step_limit} reached. Ending episode.")
                    monte_carlo+=1
                break

        # Check if the goal was reached
        if done:
            goal_reached = env.goal_reached()

        # Print episode result
        if verbose:
            print(f"Reward: {episode_reward}, Goal Reached: {goal_reached}, Step Limit Reached: {reached_step_limit}")

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        if verbose:
            print(f"EPISODE NUMBER {episode} FINISHED")
            # env.render_network_graph(show=True)

    return q_table, cumulative_steps, cumulative_reward, goal_reached,monte_carlo


def run_sarsa_lambda_agent(env, num_eps, step_limit, num_actions, alpha, gamma, epsilon, epsilon_decay, min_epsilon, lambda_trace=0.9, replay_size=4000, batch_size=32, verbose=True):
    """Run SARSA(λ) agent on the given environment with Experience Replay.

    Parameters
    ----------
    env : gym.Env or custom environment
        The environment to run the SARSA(λ) agent on.
    num_eps : int
        Number of episodes to run the agent for.
    step_limit : int
        The maximum number of steps to run each episode for
    num_actions : int
        The number of actions the agent can take in the environment.
    alpha : float
        The learning rate.
    gamma : float
        The discount factor.
    epsilon : float
        The exploration rate for epsilon-greedy policy.
    epsilon_decay : float
        The factor by which epsilon is multiplied after each episode.
    min_epsilon : float
        The minimum value that epsilon can decay to.
    lambda_trace : float
        The trace decay factor for eligibility traces (default=0.9).
    replay_size : int
        The size of the experience replay buffer (default=1000).
    batch_size : int
        The number of experiences sampled from the buffer for updates (default=32).
    verbose : bool, optional
        Whether to print out progress messages or not (default=True).

    Returns
    -------
    q_table : dict
        The learned Q-table mapping states to actions.
    cumulative_steps : int
        The total steps taken across all episodes.
    cumulative_reward : float
        The total reward accumulated across all episodes.
    goal_reached : bool
        Whether the goal was reached in any episode.
    """

    q_table = {}
    monte_carlo=0
    cumulative_steps = 0
    cumulative_reward = 0
    goal_reached = False
    episode_arr = [i for i in range(num_eps)]

    # Experience replay buffer
    replay_buffer = deque(maxlen=replay_size)

    for idx, episode in enumerate(episode_arr):
        episode_reward = 0
        episode += 1

        if verbose:
            print(f"STARTING EPISODE NUMBER {episode}")

        # Reset environment and get the initial state
        state, _ = env.reset()
        state = tuple(state)
        done = False
        steps = 0

        # Eligibility trace for state-action pairs
        eligibility_trace = {}

        # Epsilon-greedy action selection for the initial state
        if state not in q_table:
            q_table[state] = {action: 0 for action in range(num_actions)}

        if np.random.rand() < epsilon:
            action = random.randint(0, num_actions - 1)  # Random action (exploration)
        else:
            action = max(q_table[state], key=q_table[state].get)  # Greedy action (exploitation)

        while not done and steps < step_limit:
            # Take the chosen action and observe the result
            next_state, reward, done, reached_step_limit, info = env.step(action)
            next_state = tuple(next_state)  # Convert next_state to a tuple
            episode_reward += reward
            cumulative_steps += 1
            cumulative_reward += reward
            steps += 1

            # Epsilon-greedy action selection for the next state
            if next_state not in q_table:
                q_table[next_state] = {a: 0 for a in range(num_actions)}

            if np.random.rand() < epsilon:
                next_action = random.randint(0, num_actions - 1)  # Random action (exploration)
            else:
                next_action = max(q_table[next_state], key=q_table[next_state].get)  # Greedy action (exploitation)

            # Store transition in replay buffer
            replay_buffer.append((state, action, reward, next_state, next_action, done))

            # Initialize eligibility trace for (state, action) if not already done
            if (state, action) not in eligibility_trace:
                eligibility_trace[(state, action)] = 0

            # Update eligibility trace for (state, action)
            eligibility_trace[(state, action)] += 1

            # SARSA(λ) update rule with eligibility traces
            td_error = reward + gamma * q_table[next_state][next_action] - q_table[state][action]
            for s_a in eligibility_trace:
                s, a = s_a
                q_table[s][a] += alpha * td_error * eligibility_trace[s_a]
                eligibility_trace[s_a] *= gamma * lambda_trace  # Decay the eligibility trace

            # Transition to the next state and action
            state = next_state
            action = next_action

            if steps >= step_limit:
                reached_step_limit = True
                if verbose:
                    print(f"Step limit of {step_limit} reached. Ending episode.")
                    monte_carlo+=1
                break

            # Experience replay batch update
            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                for b_state, b_action, b_reward, b_next_state, b_next_action, b_done in batch:
                    # SARSA update for replayed experiences
                    if not b_done:
                        td_target = b_reward + gamma * q_table[b_next_state][b_next_action]
                    else:
                        td_target = b_reward
                    q_table[b_state][b_action] += alpha * (td_target - q_table[b_state][b_action])

        # Check if the goal was reached
        if done:
            goal_reached = env.goal_reached()

        # Print episode result
        if verbose:
            print(f"Reward: {episode_reward}, Goal Reached: {goal_reached}, Step Limit Reached: {reached_step_limit}")

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        if verbose:
            print(f"EPISODE NUMBER {episode} FINISHED")

    return q_table, cumulative_steps, cumulative_reward, goal_reached,monte_carlo



# Neural network for policy (actor) and value function (critic)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import random

# Define Actor-Critic network for PPO
class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_policy = nn.Linear(hidden_dim, action_dim)
        self.fc_value = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        policy_logits = self.fc_policy(x)
        value = self.fc_value(x)
        return policy_logits, value

    def get_action(self, state):
        logits, _ = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def get_value(self, state):
        _, value = self.forward(state)
        return value

# Define the PPO agent
class PPOAgent:
    def __init__(self, input_dim, action_dim, gamma=0.99, lambda_=0.95, epsilon_clip=0.2, policy_lr=3e-4, value_lr=1e-3):
        self.gamma = gamma
        self.lambda_ = lambda_
        self.epsilon_clip = epsilon_clip

        self.actor_critic = ActorCritic(input_dim, action_dim)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=policy_lr)
        self.critic_criterion = nn.MSELoss()

    def compute_advantage(self, rewards, values, masks):
        advantage = 0
        advantages = []
        for reward, value, mask in zip(reversed(rewards), reversed(values), reversed(masks)):
            td_error = reward + self.gamma * value * mask - value
            advantage = td_error + self.gamma * self.lambda_ * mask * advantage
            advantages.insert(0, advantage)
        return advantages

    def ppo_update(self, states, actions, old_log_probs, returns, advantages, epochs=10, batch_size=64):
        for _ in range(epochs):
            for idx in range(0, len(states), batch_size):
                batch_states = states[idx:idx+batch_size]
                batch_actions = actions[idx:idx+batch_size]
                batch_old_log_probs = old_log_probs[idx:idx+batch_size]
                batch_returns = returns[idx:idx+batch_size]
                batch_advantages = advantages[idx:idx+batch_size]

                logits, values = self.actor_critic(batch_states)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # Ratio between new and old policies
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # Clipped PPO objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.epsilon_clip, 1.0 + self.epsilon_clip) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy

                # Critic loss (value function loss)
                value_loss = self.critic_criterion(values, batch_returns)

                # Perform updates
                self.optimizer.zero_grad()
                policy_loss.backward()
                value_loss.backward()
                self.optimizer.step()

    def compute_returns(self, rewards, masks, values, gamma):
        returns = []
        R = 0
        for reward, mask, value in zip(reversed(rewards), reversed(masks), reversed(values)):
            R = reward + gamma * R * mask
            returns.insert(0, R)
        return returns


def run_ppo_agent(env, num_episodes, step_limit, rollout_length=2048, update_epochs=10, batch_size=64, gamma=0.99, lambda_=0.95, epsilon_clip=0.2, verbose=True):
    """Run PPO agent on the given environment.

    Parameters
    ----------
    env : gym.Env or custom environment
        The environment to run the PPO agent on.
    num_episodes : int
        Number of episodes to run the agent for.
    step_limit : int
        The maximum number of steps to run each episode for.
    rollout_length : int
        Number of steps for collecting data before each PPO update.
    update_epochs : int
        Number of epochs for each PPO update.
    batch_size : int
        Batch size for PPO updates.
    gamma : float
        The discount factor.
    lambda_ : float
        The lambda for Generalized Advantage Estimation (GAE).
    epsilon_clip : float
        The clip range for PPO.
    verbose : bool, optional
        Whether to print out progress messages or not (default=True).

    Returns
    -------
    cumulative_steps : int
        The total steps taken across all episodes.
    cumulative_reward : float
        The total reward accumulated across all episodes.
    goal_reached : bool
        Whether the goal was reached in any episode.
    """
    
    input_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPOAgent(input_dim, action_dim, gamma=gamma, lambda_=lambda_, epsilon_clip=epsilon_clip)

    cumulative_steps = 0
    cumulative_reward = 0
    goal_reached = False

    for episode in range(num_episodes):
        states, actions, log_probs, rewards, values, masks = [], [], [], [], [], []
        state, _ = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0)
        episode_reward = 0
        step_limit_reached = False

        for step in range(step_limit):
            with torch.no_grad():
                action, log_prob = agent.actor_critic.get_action(state)
                value = agent.actor_critic.get_value(state)

            next_state, reward, done = env.step(action)[:3]
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            mask = 1 - int(done)

            # Store experience
            states.append(state)
            actions.append(torch.tensor([action]))
            log_probs.append(log_prob)
            rewards.append(torch.tensor([reward]))
            values.append(value)
            masks.append(torch.tensor([mask]))

            state = next_state
            episode_reward += reward
            cumulative_steps += 1
            cumulative_reward += reward

            if done:
                goal_reached = env.goal_reached() if hasattr(env, 'goal_reached') else done
                break

            if step == step_limit - 1:
                step_limit_reached = True

            if step % rollout_length == 0 or step_limit_reached:
                with torch.no_grad():
                    last_value = agent.actor_critic.get_value(state).detach()

                returns = agent.compute_returns(rewards, masks, values + [last_value], gamma)
                advantages = agent.compute_advantage(rewards, values, masks)

                # Convert to tensor
                states = torch.cat(states)
                actions = torch.cat(actions)
                log_probs = torch.cat(log_probs)
                returns = torch.cat(returns)
                advantages = torch.cat(advantages)

                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Update PPO
                agent.ppo_update(states, actions, log_probs, returns, advantages, epochs=update_epochs, batch_size=batch_size)

                states, actions, log_probs, rewards, values, masks = [], [], [], [], [], []

        if verbose:
            print(f"Episode {episode+1}, Reward: {episode_reward}, Goal Reached: {goal_reached}, Step Limit Reached: {step_limit_reached}")
        
    return cumulative_steps, cumulative_reward, goal_reached
