import nasim
from nasim.scenarios.generator import ScenarioGenerator
from nasim.envs import NASimEnv
import matplotlib.pyplot as plt
import numpy as np
from nasim.envs.action import load_action_list
import random
import argparse

from agents import rl_agent,dqn_agent


from itertools import product

from agents.dqn_agent import DQNAgent


# bf_env=nasim.load(scenario_path, flat_actions=True)
# ql_env=nasim.load(scenario_path,flat_actions=True)
# sa_env=nasim.load(scenario_path,flat_actions=True)
# sa_lam_env=nasim.load(scenario_path,flat_actions=True)
# ppo_env=nasim.load(scenario_path,flat_actions=True)



# na_bf=bf_env.action_space.n
# ns_bf=bf_env.observation_space.shape[0]

# na_ql=ql_env.action_space.n
# ns_ql=ql_env.observation_space.shape[0]

# na_sa=sa_env.action_space.n
# ns_sa=sa_env.observation_space.shape[0]


# na_sa_lam=sa_lam_env.action_space.n
# ns_sa_lam=sa_lam_env.observation_space.shape[0]

# na_ppo=ppo_env.action_space.n
# ns_ppo=ppo_env.observation_space.shape[0]







# print("Rendering initial network graph:")


# print("##############################################")














# print("***************RUNNING RANDOM AGENT********************")

# for j in range(num_eps):
#     print(f"EP {j+1}")
#     rand_steps,rand_ep_reward,goal_reach=rl_agent.run_random_agent(rand_env,step_limit=step_limit,num_actions=na_rand,verbose=True)
#     cumulative_random_reward+=rand_ep_reward
#     cumulative_random_step+=rand_steps

# # print(f"RANDOM AGENT HAS REACHED THE GOAL OR NOT? {goal_reach}")

# print("RANDOM AGENT CUMULATIVE REWARD:",cumulative_random_reward/num_eps)
# print("RANDOM AGENT CUMULATIVE STEPS:",cumulative_random_step/num_eps)

# print("***************FINISHED EXECUTION OF RANDOM AGENT********************")


# rand_env.reset()




# print("***************RUNNING BRUTEFORCE AGENT********************")



# # print(f"BRUTE FORCE AGENT HAS REACHED THE GOAL OR NOT? {goal_reach}")

# print("BRUTE FORCE AGENT CUMULATIVE REWARD:",cumulative_brute_reward/num_eps)
# print("BRUTE FORCE AGENT CUMULATIVE STEPS:",cumulative_brute_step/num_eps)

# print("***************FINISHED EXECUTION OF BRUTEFORCE AGENT********************")

# bf_env.render_network_graph(show=True)

# bf_env.reset()

# print("***************RUNNING Q-LEARNING AGENT********************")

# q_table,cumulative_ql_step, cumulative_ql_reward, goal_reach=rl_agent.run_q_learning_agent(ql_env, num_eps, step_limit, na_ql, alpha, gamma, epsilon, epsilon_decay, min_epsilon, verbose=True)

# print("Q-LEARNING AGENT CUMULATIVE REWARD:",cumulative_ql_reward/num_eps)
# print("Q-LEARNING AGENT CUMULATIVE STEPS:",cumulative_ql_step/num_eps)

# print("***************FINISHED EXECUTION OF Q-LEARNING AGENT********************")

# ql_env.render_network_graph(show=True)

# ql_env.reset()


# print("***************RUNNING Q-SARSA AGENT********************")

# qsa_table,cumulative_qsa_step, cumulative_qsa_reward, goal_reach=rl_agent.run_sarsa_agent(sa_env, num_eps, step_limit, na_sa, alpha, gamma, epsilon, epsilon_decay, min_epsilon, verbose=True)

# print("Q-LEARNING SARSA AGENT CUMULATIVE REWARD:",cumulative_qsa_reward/num_eps)
# print("Q-LEARNING SARSA AGENT CUMULATIVE STEPS:",cumulative_qsa_step/num_eps)

# print("***************FINISHED EXECUTION OF Q-SARSA AGENT********************")

# sa_env.render_network_graph(show=True)

# sa_env.reset()

# print("***************RUNNING Q-SARSA(lambda) AGENT********************")

# qsa_table,cumulative_qlam_step, cumulative_qlam_reward, goal_reach=rl_agent.run_sarsa_lambda_agent(sa_lam_env, num_eps, step_limit,na_sa_lam , alpha, gamma, epsilon, epsilon_decay, min_epsilon, lambda_trace=0.9, replay_size=1000, batch_size=32, verbose=True)

# print("Q-LEARNING SARSA(lambda) AGENT CUMULATIVE REWARD:",cumulative_qlam_reward/num_eps)
# print("Q-LEARNING SARSA(lambda) AGENT CUMULATIVE STEPS:",cumulative_qlam_step/num_eps)

# print("***************FINISHED EXECUTION OF Q-SARSA(lambda) AGENT********************")

# sa_lam_env.render_network_graph(show=True)

# sa_lam_env.reset()

# print("***************RUNNING PPO AGENT********************")

# # Example of training
# agent = rl_agent.PPOAgent(input_dim=ns_ppo, action_dim=na_ppo)
# agent.train(ppo_env, num_episodes=500, step_limit=4000)

# print("***************FINISHED EXECUTION OF PPO AGENT********************")

# ppo_env.render_network_graph(show=True)

# ppo_env.reset()

if __name__ == '__main__':

    # scenario_path="/Users/vedantpalit/Desktop/NASIM_BTP/benchmarked_scenarios/scenario_tiny.yaml"
    
    parser = argparse.ArgumentParser(description="Running reinforcement learning attack agents in Network Attack Simulator Environments")
    parser.add_argument("--scenario_path",type=str,required=True,help="Path to the scenario yaml file")
    parser.add_argument("--num_episodes",type=int,required=True,help="Number of episodes to run the scenario for")
    parser.add_argument("--seed_value",type=int,required=True,help="random seed value")
    parser.add_argument("--agent",type=str,required=True,help="The agent to run")
    args = parser.parse_args()

    random.seed(args.seed_value)
    np.random.seed(args.seed_value)

    alpha = 0.5  # Increase learning rate for faster updates
    gamma = 0.99  # Encourage agent to focus more on future rewards
    epsilon = 1.0  # Start with full exploration and decay it
    epsilon_decay = 0.995  # Decay epsilon after each episode
    min_epsilon = 0.1 
    step_limit = 200
    scenario_env = nasim.load(args.scenario_path, fully_obs=True,flat_actions=True)
    na_scenario = scenario_env.action_space.n
    ns_scenario = scenario_env.observation_space.shape[0]

    scenario_env.reset()
    scenario_env.generate_initial_state() #initial state

    num_eps=args.num_episodes

    print(f"Number of Discrete Actions: {na_scenario}")
    print(f"Number of Observable States: {ns_scenario}")

    scenario_env.render_network_graph(show=True)

    cumulative_reward=0
    cumulative_step=0
    total_random_monte=0
    total_bf_monte=0

    if args.agent=="random":
        print("***************RUNNING RANDOM AGENT********************")
        for j in range(num_eps):
            print(f"EPISODE NUMBER: {j+1}")
            rand_steps,rand_ep_reward,goal_reach,monte_carlo=rl_agent.run_random_agent(scenario_env,step_limit=step_limit,num_actions=na_scenario,verbose=True)
            cumulative_reward+=rand_ep_reward
            cumulative_step+=rand_steps
            total_random_monte+=monte_carlo
        scenario_env.render_network_graph(show=True)
        monte_carlo=total_random_monte
    elif args.agent=="bruteforce":
        print("***************RUNNING BRUTEFORCE AGENT********************")
        for j in range(num_eps):
            print(f"EPISODE NUMBER: {j+1}")
            brute_steps,brute_ep_reward,goal_reach,monte_carlo=rl_agent.run_bruteforce_agent(scenario_env,step_limit=step_limit,num_actions=na_scenario, verbose=True)
            
            cumulative_reward+=brute_ep_reward
            cumulative_step+=brute_steps
            total_bf_monte+=monte_carlo
        scenario_env.render_network_graph(show=True)
        monte_carlo=total_bf_monte
    elif args.agent=="qlearning":
        print("***************RUNNING Q-LEARNING AGENT********************")
        q_table,cumulative_step, cumulative_reward, goal_reach,monte_carlo=rl_agent.run_q_learning_agent(scenario_env, num_eps, step_limit, na_scenario, alpha, gamma, epsilon, epsilon_decay, min_epsilon, verbose=True)
        scenario_env.render_network_graph(show=True)
        print("THE Q-TABLE:")
        print(q_table)
    elif args.agent=="sarsa":
        print("***************RUNNING Q-SARSA(lambda) AGENT********************")
        qsa_table,cumulative_step, cumulative_reward, goal_reach,monte_carlo=rl_agent.run_sarsa_lambda_agent(scenario_env, num_eps, step_limit,na_scenario , alpha, gamma, epsilon, epsilon_decay, min_epsilon, lambda_trace=0.99, replay_size=10000, batch_size=64, verbose=True)
        scenario_env.render_network_graph(show=True)
        print("THE Q-SARSA TABLE:")
        print(qsa_table)
    elif args.agent=="dqn":
        print("***************RUNNING DQN AGENT********************")
        agent=DQNAgent(env=scenario_env,seed=args.seed_value,gamma=gamma,verbose=True)
        agent.train()
        for j in range(num_eps):
            print(f"EPISODE NUMBER: {j+1}")
            episode_return, steps, goal_reach=agent.run_eval_episode(render=True)
            cumulative_reward+=episode_return
            cumulative_step+=steps
        scenario_env.render_network_graph(show=True)
    
    print(f"{args.agent} AGENT AVERAGE REWARD:",cumulative_reward/num_eps)
    print(f"{args.agent} AGENT AVERAGE STEPS:",cumulative_step/num_eps)
    print(f"{args.agent} SUCCESS NUMBER:",monte_carlo/num_eps)
    
