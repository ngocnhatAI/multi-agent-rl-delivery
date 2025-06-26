import argparse
import os
import wandb
import torch
import numpy as np
import cv2
import gymnasium as gym
import torch

from gymnasium import spaces
from gymnasium.utils.save_video import save_video
from env import Environment

from mappo import *
from normalization import *
from replay_buffer import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def convert_state(state, max_time_steps, max_packages=20, max_packages_in_obs=10):
    """
    Convert the raw environment state into per-agent observations and a global state vector.

    Args:
        state (dict): Contains keys 'robots', 'packages', 'time_step', 'map'.
        max_time_steps (int): Used to normalize the time_step.
        max_packages (int): Max packages to include in global state.
        max_packages_in_obs (int): Max packages per-agent observation.

    Returns:
        observations (np.ndarray): Shape (n_agents, obs_dim), each row is an agent’s obs.
        global_state (np.ndarray): Flattened vector of global features for the mixer.
    """
    # 1) Parse and cast raw state arrays
    robots = np.array(state["robots"]).astype(np.float32)
    packages = np.array(state["packages"]).astype(np.float32)
    time_step = np.array([state["time_step"]]).astype(np.float32)
    grid_map = np.array(state["map"]).astype(np.float32)

    # 2) Build global state features
    n_robots = len(robots)
    n_rows, n_cols = len(grid_map), len(grid_map[0])
    global_state = []
    # 2a) normalized time
    global_state.append(time_step / max_time_steps)
    # 2b) entire map flattened
    global_state.append(grid_map.flatten())
    # 2c) robot positions normalized by grid size
    robots_normalized = robots.copy()
    robots_normalized[:, 0] = robots_normalized[:, 0] / n_rows
    robots_normalized[:, 1] = robots_normalized[:, 1] / n_cols
    global_state.append(robots_normalized.flatten())

    # 2d) package features: normalize coords & times, pad or truncate
    if len(packages) > 0:
        packages_normalized = packages.copy()
        # normalize start/target x,y by max dimension
        packages_normalized[:, 1:5] = packages_normalized[:, 1:5] / max(n_rows, n_cols)
        # normalize start_time, deadline by max_time_steps
        packages_normalized[:, 5:7] = packages_normalized[:, 5:7] / max_time_steps
        if len(packages) > max_packages:
            # choose most urgent packages first
            urgency = packages[:, 6] - time_step  # deadline - current time
            most_urgent_indices = np.argsort(urgency)[:max_packages]
            global_state.append(packages_normalized[most_urgent_indices].flatten())
        else:
            # pad with zeros up to max_packages
            padded_packages = np.zeros((max_packages, 7))
            padded_packages[:len(packages)] = packages_normalized
            global_state.append(padded_packages.flatten())
    else:
        # no packages → all zeros
        global_state.append(np.zeros(max_packages * 7))
    
    # flatten the list into one vector
    global_state = np.concatenate(global_state)

    # 3) Build per‐agent observations
    agent_obs = []
    for i in range(n_robots):
        obs_i = []
        # 3a) own position and carrying flag
        robot_x, robot_y, carrying = robots[i]
        obs_i.append(np.array([robot_x/n_rows, robot_y/n_cols, carrying]).astype(np.float32))
        # 3b) current time
        obs_i.append(time_step / max_time_steps)
        # 3c) full map
        obs_i.append(grid_map.flatten())
        # 3d) other robots: relative positions & carrying
        other_robots = np.zeros((n_robots - 1, 3), dtype=np.float32)
        idx = 0
        for j in range(n_robots):
            if i != j:
                other_x, other_y, other_carrying = robots[j]
                # Relative position and carrying status
                other_robots[idx] = [
                    (other_x - robot_x)/n_rows,
                    (other_y - robot_y)/n_cols,
                    other_carrying
                ]
                idx += 1
        obs_i.append(other_robots.flatten())

        # 3e) local package info: relative coords, times, carrying flag
        package_list = []
        carrying_package = None
        for pkg in packages:
            pkg_id, start_x, start_y, target_x, target_y, start_time, deadline = pkg
            normalized_pkg = [
                pkg_id / max_packages,         
                (start_x - robot_x) / n_rows,  
                (start_y - robot_y) / n_cols,  
                (target_x - robot_x) / n_rows, 
                (target_y - robot_y) / n_cols, 
                start_time / max_time_steps,   
                deadline / max_time_steps,
                1.0 if pkg_id == carrying else 0.0
            ]
            if pkg_id == carrying:
                carrying_package = normalized_pkg
            else:
                package_list.append(normalized_pkg)

        # ensure carried package first, pad/truncate to max_packages_in_obs
        if carrying_package:
            final_packages = [carrying_package] + package_list
        else:
            final_packages = package_list
        while len(final_packages) < max_packages_in_obs:
            final_packages.append([0.0] * 8)
        final_packages = final_packages[:max_packages_in_obs]
        obs_i.append(np.array(final_packages, dtype=np.float32).flatten())

        # concatenate all components of observation
        agent_obs.append(np.concatenate([arr.flatten() for arr in obs_i]))

    # stack into array of shape (n_agents, obs_dim)
    observations = np.stack(agent_obs, axis=0)
    return observations, global_state


def reward_shaping(r, env, state, next_state, action):
    """
    Reward shaping function to guide the agent's learning
    r: original reward (10 for on-time delivery, 1 for late delivery)
    env: environment instance
    state: current state before action
    next_state: state after action
    action: current action as tuple (movement, package) for each robot
    """
    # Get state information
    current_robots = np.array(state["robots"])
    next_robots = np.array(next_state["robots"])
    current_packages = np.array(state["packages"])
    packages = np.array(next_state["packages"])
    current_time_step = state["time_step"]
    next_time_step = next_state["time_step"]
    n_robots = len(current_robots)

    # Reward for picking up packages and moving closer to destination
    for i in range(n_robots):
        current_robot_x, current_robot_y, current_carrying = current_robots[i]
        next_robot_x, next_robot_y, next_carrying = next_robots[i]

        # Action for robot i
        movement_action, package_action = action[i]  # Directly unpack the tuple

        # Check for invalid movement actions
        if movement_action != 0:  # Only check if robot attempted to move
            # If position hasn't changed after a movement action, it was invalid
            if current_robot_x == next_robot_x and current_robot_y == next_robot_y:
                r -= 0.1  # Penalty for invalid movement

        # Check for package pickup
        if package_action == 1 and current_carrying == 0 and next_carrying > 0:  # Robot just picked up a package
            # Find the package that was just picked up
            r += 1.0  # Additional reward for picking up
            break

        # Check for package delivery
        if package_action == 2 and current_carrying > 0 and next_carrying == 0:  # Robot just delivered a package
            # Find the package that was just delivered
            for pkg in packages:
                if pkg[0] == current_carrying:  # package was delivered
                    deadline = pkg[6]
                    delivery_time = next_time_step
                    if delivery_time <= deadline:
                        # On-time delivery: keep full reward
                        r += 10.0
                    else:
                        # Late delivery: reduce reward based on how late
                        time_diff = delivery_time - deadline
                        # Reduce reward by 10% for each time step after deadline
                        # Minimum reward is 1.0
                        reduction = min(0.9, time_diff * 0.1)  # 10% reduction per step, max 90%
                        r += max(1.0, 10.0 * (1 - reduction))
                    break

        # If robot is carrying a package, check if it's moving closer to destination
        if current_carrying > 0 and next_carrying > 0:  # Robot is carrying a package in next state
            # Find the package being carried
            for pkg in packages:
                if pkg[0] == current_carrying:  # pkg[0] is package ID
                    target_x, target_y = pkg[3:5]  # target coordinates
                    # Calculate current and previous Manhattan distances to target
                    next_dist = abs(next_robot_x - target_x) + abs(next_robot_y - target_y)
                    current_dist = abs(current_robot_x - target_x) + abs(current_robot_y - target_y)
                    # If robot moved closer to target, give reward
                    if next_dist < current_dist:
                        r += 0.1
                    break

    return r


class Env(gym.Env):
    def __init__(self, *args, **kwargs):
        super(Env, self).__init__()
        self.env = Environment(*args, **kwargs)
        self.action_space = spaces.multi_discrete.MultiDiscrete([5, 3]*self.env.n_robots)
        self.prev_state = self.env.reset()
        obs, global_state = convert_state(self.prev_state, self.env.max_time_steps)
        self.observation_space = spaces.Box(low=-100, high=100, shape=obs.shape, dtype=np.float32)
        self.state_dim = len(global_state)
        self.obs_dim = obs.shape[1]
        self.action_dim = 15
        from sklearn.preprocessing import LabelEncoder
        self.le1, self.le2= LabelEncoder(), LabelEncoder()
        self.le1.fit(['S', 'L', 'R', 'U', 'D'])
        self.le2.fit(['0','1', '2'])

    def reset(self, *args, **kwargs):
        self.prev_state = self.env.reset()
        return convert_state(self.prev_state, self.env.max_time_steps), {}

    def render(self, *args, **kwargs):
        return self.env.render()
    
    def get_state(self):
        return convert_state(self.env.get_state(), self.env.max_time_steps)
    
    def convert_action(self, actions):
        # actions: list of integers in range (0, 14), one per robot
        n_robots = len(actions)
        result = np.zeros(n_robots * 2, dtype=int)
        for i in range(n_robots):
            action_idx = actions[i]
            # Convert from flat index (0-14) to separate movement and package indices
            # Movement: 0-4 (corresponding to 'S', 'L', 'R', 'U', 'D')
            # Package: 0-2 (corresponding to '0', '1', '2')
            movement_idx = action_idx // 3
            package_idx = action_idx % 3
            # Place in the correct positions in the result array
            result[i*2] = movement_idx    # Movement action (even index)
            result[i*2+1] = package_idx   # Package action (odd index)
        return result

    def step(self, action):
        action = self.convert_action(action)
        ret = []
        ret.append(self.le1.inverse_transform(action.reshape(-1, 2).T[0]))
        ret.append(self.le2.inverse_transform(action.reshape(-1, 2).T[1]))
        action = list(zip(*ret))

        # You should not modify the infos object
        s, r, done, infos = self.env.step(action)
        # new_r = reward_shaping(r, self.env, self.prev_state, action)
        self.prev_state = s
        return convert_state(s, self.env.max_time_steps), r, \
            done, False, infos
    
    def render_image(self):
        cell_size = 100 # size of each cell (pixel)
        height = self.env.n_rows * cell_size
        width = self.env.n_cols * cell_size
        img = np.ones((height, width, 3), dtype=np.uint8) * 255
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = cell_size / 100
        # Draw grid
        for r in range(self.env.n_rows):
            for c in range(self.env.n_cols):
                x0 = c * cell_size
                y0 = r * cell_size
                x1 = x0 + cell_size
                y1 = y0 + cell_size

                if self.env.grid[r][c] == 1:  # Obstacle
                    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), -1)
                else:  # Free cell
                    cv2.rectangle(img, (x0, y0), (x1, y1), (200, 200, 200), 1)
        # Draw robots
        for i, robot in enumerate(self.env.robots):
            x = robot.position[1] * cell_size + cell_size // 2
            y = robot.position[0] * cell_size + cell_size // 2
            radius = cell_size // 2 - 5
            cv2.circle(img, (x, y), radius, (255, 0, 0), -1)
            cv2.putText(img, f"R{i}", (x - radius // 3, y + radius // 3),
                        font, font_scale, (255, 255, 255), 2)
            # Draw package inside robot if carrying
            if robot.carrying != 0:
                package_size = radius // 2
                package_color = (0, 255, 0)  # Green for packages
                cv2.rectangle(img, 
                            (x - package_size, y - package_size), 
                            (x + package_size, y + package_size), 
                            package_color, -1)
                cv2.putText(img, f"P{robot.carrying}", (x - radius // 3, y + radius // 3), 
                            font, font_scale, (255, 255, 255), 2)
        # Draw packages
        for package in self.env.packages:
            if package.status == "waiting":
                r, c = package.start
                color = (0, 255, 0)
            elif package.status == "in_transit":
                r, c = package.target
                color = (0, 165, 255)
            else:
                continue
            x = c * cell_size + cell_size // 2
            y = r * cell_size + cell_size // 2
            size = cell_size // 3
            cv2.rectangle(img, (x - size, y - size), (x + size, y + size), color, -1)
            cv2.putText(img, f"P{package.package_id}", (x - size // 2, y + size // 2),
                        font, font_scale, (255, 255, 255), 2)
        return img

class Runner_MAPPO:
    def __init__(self, args, number, seed):
        self.args = args
        self.number = number
        self.seed = seed

        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Create env
        self.env = Env(
            map_file=self.args.map_file, max_time_steps=self.args.max_time_steps,
            n_robots=self.args.n_robots, n_packages=self.args.n_packages,
        )

        self.args.N = self.env.env.n_robots
        self.args.obs_dim = self.env.obs_dim
        self.args.state_dim = self.env.state_dim
        self.args.action_dim = self.env.action_dim
        self.args.episode_limit = self.env.env.max_time_steps

        print(f"map_file={self.args.map_file}")
        print(f"max_time_steps={self.args.max_time_steps}")
        print(f"n_robots={self.args.n_robots}")
        print(f"n_packages={self.args.n_packages}")

        print("obs_dim={}".format(self.args.obs_dim))
        print("state_dim={}".format(self.args.state_dim))
        print("action_dim={}".format(self.args.action_dim))
        print("episode_limit={}".format(self.args.episode_limit))

        # Create N agents
        self.agent_n = MAPPO(self.args)
        self.replay_buffer = ReplayBuffer(self.args)

        self.total_steps = 0
        if self.args.use_reward_norm:
            print("------use reward norm------")
            self.reward_norm = Normalization(shape=1)
        elif self.args.use_reward_scaling:
            print("------use reward scaling------")
            self.reward_scaling = RewardScaling(shape=1, gamma=self.args.gamma)

    def run(self, wandb_logger=None):
        evaluate_num = -1  # Record the number of evaluations
        while self.total_steps < self.args.max_train_steps:
            if self.total_steps // self.args.evaluate_freq > evaluate_num:
                self.evaluate_policy(wandb_logger)  # Evaluate the policy every 'evaluate_freq' steps
                evaluate_num += 1

            _, episode_steps = self.run_episode(evaluate=False)  # Run an episode
            self.total_steps += episode_steps

            if self.replay_buffer.episode_num == self.args.batch_size:
                self.agent_n.train(self.replay_buffer, self.total_steps)  # Training
                self.replay_buffer.reset_buffer()

        self.evaluate_policy(wandb_logger)

        # Save model
        map_file = os.path.basename(self.args.map_file).split('.')[0]
        file_name = f"{self.args.algorithm}-{map_file}-{self.args.n_robots}r-{self.args.n_packages}p-{self.args.max_time_steps}t"
        actor_net_path, critic_net_path = self.agent_n.save_model(custom_name=file_name)

        if wandb_logger:
            actor_artifact = wandb.Artifact(name=f"{file_name}_actor_net", type="model")
            actor_artifact.add_file(actor_net_path)
            wandb_logger.log_artifact(actor_artifact)

            critic_artifact = wandb.Artifact(name=f"{file_name}_critic_net", type="model")
            critic_artifact.add_file(critic_net_path)
            wandb_logger.log_artifact(critic_artifact)

            video_path = self.evaluate_and_record()
            wandb_logger.log({"video_result": wandb.Video(video_path)})
        self.env.close()

    def evaluate_policy(self, wandb_logger):
        evaluate_reward = 0
        for _ in range(self.args.evaluate_times):
            episode_reward, _ = self.run_episode(evaluate=True)
            evaluate_reward += episode_reward

        evaluate_reward = evaluate_reward / self.args.evaluate_times
        print("total_steps:{} \t evaluate_reward:{}".format(self.total_steps, evaluate_reward))
        if wandb_logger:
            wandb_logger.log({
                "evaluate_reward": evaluate_reward,
                "total_steps": self.total_steps
            })

    def run_episode(self, evaluate=False):
        episode_reward = 0
        self.env.reset()
        if self.args.use_reward_scaling:
            self.reward_scaling.reset()
        if self.args.use_rnn:  # If use RNN, before the beginning of each episode，reset the rnn_hidden
            self.agent_n.actor.rnn_hidden = None
            self.agent_n.critic.rnn_hidden = None
        for episode_step in range(self.args.episode_limit):
            obs_n, s = self.env.get_state()  # obs_n.shape=(N,obs_dim) s.shape=(state_dim,)
            avail_a_n = np.ones((self.args.N, self.args.action_dim))
            a_n, a_logprob_n = self.agent_n.choose_action(obs_n, avail_a_n, evaluate=evaluate)  # Get actions and the corresponding log probabilities of N agents
            v_n = self.agent_n.get_value(s, obs_n)  # Get the state values (V(s)) of N agents
            obs, r, done, _, info = self.env.step(a_n)  # Take a step
            episode_reward += r

            if not evaluate:
                if self.args.use_reward_norm:
                    r = self.reward_norm(r)
                elif args.use_reward_scaling:
                    r = self.reward_scaling(r)
                """"
                    When dead or win or reaching the episode_limit, done will be Ture, we need to distinguish them;
                    dw means dead or win,there is no next state s';
                    but when reaching the max_episode_steps,there is a next state s' actually.
                """
                if done and episode_step + 1 != self.args.episode_limit:
                    dw = True
                else:
                    dw = False

                # Store the transition
                self.replay_buffer.store_transition(episode_step, obs_n, s, v_n, avail_a_n, a_n, a_logprob_n, r, dw)

            if done:
                break

        if not evaluate:
            # An episode is over, store obs_n, s and avail_a_n in the last step
            obs_n, s = self.env.get_state()
            v_n = self.agent_n.get_value(s, obs_n)
            self.replay_buffer.store_last_value(episode_step + 1, v_n)

        return episode_reward, episode_step + 1
    
    def evaluate_and_record(self, dir_path="./videos/", prefix_name=""):
        episode_reward = 0
        self.env.reset()
        frames = [self.env.render_image()]
        if self.args.use_reward_scaling:
            self.reward_scaling.reset()
        if self.args.use_rnn:  # If use RNN, before the beginning of each episode，reset the rnn_hidden
            self.agent_n.actor.rnn_hidden = None
            self.agent_n.critic.rnn_hidden = None
        for episode_step in range(self.args.episode_limit):
            obs_n, s = self.env.get_state()  # obs_n.shape=(N,obs_dim), s.shape=(state_dim,)
            avail_a_n = np.ones((self.args.N, self.args.action_dim))
            a_n, a_logprob_n = self.agent_n.choose_action(obs_n, avail_a_n, evaluate=True)  # Get actions and the corresponding log probabilities of N agents
            # v_n = self.agent_n.get_value(s, obs_n)  # Get the state values (V(s)) of N agents
            obs, r, done, _, info = self.env.step(a_n)  # Take a step
            episode_reward += r
            frames.append(self.env.render_image())

        map_file = os.path.basename(self.args.map_file).split('.')[0]
        file_name = f"{self.args.algorithm}-{map_file}-{self.args.n_robots}r-{self.args.n_packages}p-{self.args.max_time_steps}t"
        file_name = prefix_name + file_name
        fps = self.args.max_time_steps // 100
        save_video(frames, dir_path, fps=fps, name_prefix=file_name)
        full_video_path = dir_path + file_name + "-episode-0.mp4"
        print(f"Saved video: {full_video_path}")
        return full_video_path

    def load_weights(self, actor_net_path, critic_net_path):
        self.agent_n.load_model(actor_net_path, critic_net_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for MAPPO")

    # Environment parameters
    parser.add_argument("--map_file", type=str, default="map.txt")
    parser.add_argument("--n_robots", type=int, default=2)
    parser.add_argument("--n_packages", type=int, default=5)
    parser.add_argument("--max_time_steps", type=int, default=100)
    parser.add_argument("--model_save_dir", type=str, default="./weights/")
    parser.add_argument("--algorithm", type=str, default="MAPPO", help="MAPPO")

    # Training parameters
    parser.add_argument("--max_train_steps", type=int, default=int(1e6), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=20000, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=int, default=32, help="Evaluate times")
    parser.add_argument("--save_freq", type=int, default=int(1e5), help="Save frequency")

    # Algorithm parameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (the number of episodes)")
    parser.add_argument("--mini_batch_size", type=int, default=8, help="Minibatch size (the number of episodes)")
    parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of RNN")
    parser.add_argument("--mlp_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of MLP")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GAE parameter")
    parser.add_argument("--K_epochs", type=int, default=15, help="GAE parameter")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")

    # Feature flags
    parser.add_argument("--use_adv_norm",         action="store_true", help="Trick 1: advantage normalization")
    parser.add_argument("--use_reward_norm",      action="store_true", help="Trick 3: reward normalization")
    parser.add_argument("--use_reward_scaling",   action="store_true", help="Trick 4: reward scaling. Here, we do not use it.")
    parser.add_argument("--use_lr_decay",         action="store_true", help="Trick 6: learning rate decay")
    parser.add_argument("--use_grad_clip",        action="store_true", help="Trick 7: gradient clip")
    parser.add_argument("--use_orthogonal_init",  action="store_true", help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps",         action="store_true", help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_relu",             action="store_true", help="Whether to use ReLU (default False)")
    parser.add_argument("--use_rnn",              action="store_true", help="Whether to use RNN")
    parser.add_argument("--add_agent_id",         action="store_true", help="Whether to add agent_id. Here, we do not use it.")
    parser.add_argument("--use_agent_specific",   action="store_true", help="Whether to use agent-specific global state.")
    parser.add_argument("--use_value_clip",       action="store_true", help="Whether to use value clip.")

    args = parser.parse_args()
    
    print(vars(args))

    print(f"Using device: {DEVICE}")

    wandb_run = wandb.init(
        project="RL-MAPD",
        config=vars(args),
    )
    runner = Runner_MAPPO(args, number=1, seed=59)
    runner.run(wandb_logger=wandb_run)
    wandb_run.finish()


        