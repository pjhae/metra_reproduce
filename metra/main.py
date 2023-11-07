import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from sac import SAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory
from model_metra import Phi, Lambda

from utils_sac import VideoRecorder
from utils_metra import generate_skill, generate_skill_cont


parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="Ant-v3",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=False,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='G',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--alpha', type=float, default=0.01, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                    help='Automaically adjust α (default: True)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=100000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=1024, metavar='N',
                    help='hidden size (default: 1024)')

parser.add_argument('--gradient_steps_per_epoch', type=int, default=200, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--episodes_per_epoch', type=int, default=8, metavar='N',
                    help='model updates per simulator step (default: 1)')

parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--skill_dim', type=int, default=2, metavar='N',
                    help='dimension of skill (default: 8)')
parser.add_argument('--cuda', action="store_false",
                    help='run on CUDA (default: True)')
args = parser.parse_args()

# Environment
env = gym.make(args.env_name)

# Device
device = torch.device("cuda" if args.cuda else "cpu")

# For reproduce
env.seed(args.seed)
env.action_space.seed(args.seed)   
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# For video
video_directory = '/home/jonghae/metra_reproduce/metra/video/{}'.format(datetime.datetime.now().strftime("%H:%M:%S %p"))
video = VideoRecorder(dir_name = video_directory)

# Tesnorboard
writer = SummaryWriter('runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                                             args.policy, "autotune" if args.automatic_entropy_tuning else ""))

# Memory
memory = ReplayMemory(args.replay_size, args.seed)

# Training Loop
total_numsteps = 0
updates = 0
episode_idx = 0

# Skill dim
skill_dim = args.skill_dim

# Agent
agent = SAC(env.observation_space.shape[0] + skill_dim, env.action_space, args)

# phi
phi = Phi(env.observation_space.shape[0], args).to(device)

print("state :", env.observation_space.shape[0])

# lambda
lamb = Lambda(args)
lamb.lambda_value = lamb.lambda_value.to(device)

# Training Loop
for i_epoch in itertools.count(1):
    for i_episode in range(args.episodes_per_epoch):
        episode_reward = 0
        episode_steps = 0
        episode_idx += 1

        done = False
        state = env.reset()
        skill = generate_skill_cont(skill_dim)
        state = np.concatenate([state, skill])
        
        while not done:
            if args.start_steps > total_numsteps:
                action = env.action_space.sample()  # Sample random action
            else:
                action = agent.select_action(state)  # Sample action from policy

            next_state, reward, done, _ = env.step(action) # Step

            next_state = np.concatenate([next_state, skill])
            psuedo_reward = np.dot(phi.forward_np(next_state) - phi.forward_np(state), skill) 

            episode_steps += 1
            total_numsteps += 1
            episode_reward += psuedo_reward

            mask = 1 if episode_steps == env._max_episode_steps else float(not done)
            memory.push(state, action, psuedo_reward, next_state, mask) # Append transition to memory
            state = next_state

        writer.add_scalar('reward/train', episode_reward, episode_idx)
        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(episode_idx, total_numsteps, episode_steps, round(episode_reward, 2)))


    if len(memory) > args.batch_size:
        # Number of updates per step in environment
        for i in range(args.gradient_steps_per_epoch):
            # Update parameters of all the networks
            critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates, phi)
            phi_loss = phi.update_parameters(memory, args.batch_size, lamb.lambda_value)
            lamb_loss = lamb.update_parameters(memory, args.batch_size, phi)
            writer.add_scalar('loss/phi', phi_loss, updates)
            writer.add_scalar('loss/lambda', lamb_loss, updates)
            writer.add_scalar('loss/critic_1', critic_1_loss, updates)
            writer.add_scalar('loss/critic_2', critic_2_loss, updates)
            writer.add_scalar('loss/policy', policy_loss, updates)
            writer.add_scalar('loss/entropy_loss', ent_loss, updates)
            writer.add_scalar('entropy_temprature/alpha', alpha, updates)
            writer.add_scalar('dual_variable/lambda', lamb.lambda_value, updates)
            updates += 1

    if total_numsteps > args.num_steps:
        break   
        
    if episode_idx % 200 == 0:
        agent.save_checkpoint(args.env_name,"{}".format(episode_idx))

    # Evaluate all skill
    if episode_idx % 200 == 0:
        video.init(enabled=True)
        avg_psuedo_reward = 0.
        avg_dist_reward = 0.
        avg_step = 0.
        episodes = 10
        for i in range(episodes):
            skill = generate_skill_cont(skill_dim)

            state = env.reset()
            state = np.concatenate([state, skill])

            episode_steps = 0

            episode_psuedo_reward = 0
            episode_dist_reward = 0

            done = False
            
            while not done:
                action = agent.select_action(state, evaluate=True)
                video.record(env.render(mode='rgb_array', camera_id=0))
                next_state, reward, done, _ = env.step(action)

                next_state = np.concatenate([next_state, skill])
                psuedo_reward = np.dot(phi.forward_np(next_state) - phi.forward_np(state), skill) 

                episode_psuedo_reward += psuedo_reward
                episode_dist_reward += np.linalg.norm(state[:2])
                episode_steps += 1

            avg_psuedo_reward += episode_psuedo_reward
            avg_dist_reward += episode_dist_reward
            avg_step += episode_steps

        avg_psuedo_reward /= episodes
        avg_dist_reward /= episodes
        
        avg_step /= episodes
        
        video.save('test_{}.mp4'.format(episode_idx))
        video.init(enabled=False)

        writer.add_scalar('avg_psuedo_reward/test', avg_psuedo_reward, total_numsteps)
        writer.add_scalar('avg_dist_reward/test', avg_dist_reward, total_numsteps)

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}, Avg. step: {}".format(episodes, round(avg_psuedo_reward, 2), round(avg_step, 2)))
        print("----------------------------------------")

env.close()

