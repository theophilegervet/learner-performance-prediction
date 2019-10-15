import gym
import pandas as pd

from stable_baselines.common.policies import MlpLstmPolicy, MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO2

import torch
from torch.distributions.binomial import Binomial


class StudentEnv(gym.Env):
    def __init__(self, df, dkt, history_steps=3, simulation_steps=30):
        """Reinforcement learning environment simulating a student.

        Arguments:
            df (pandas Dataframe): output by prepare_data.py
            dkt (torch Module): trained DKT model
            history_steps (int): number of steps from student history before simulating
            simulation_steps (int): number of steps to simulate
        """
        super(StudentEnv, self).__init__()
        """
        self.dkt = dkt.cpu()
        assert dkt.item_in and (not dkt.skill_in)
        assert dkt.item_out and (not dkt.skill_out)
        self.process_data(df)
        self.history_steps = history_steps
        self.simulation_steps = simulation_steps

        self.action_space = gym.spaces.Discrete(dkt.output_size)
        self.observation_space = gym.spaces.Discrete(dkt.input_size)
        """
        self.simulation_steps = simulation_steps
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Discrete(1)

    @torch.no_grad()
    def step(self, item_id):
        """
        self.steps += 1
        success_probs = torch.sigmoid(self.dkt.out(self.hidden[0][-1])).flatten()
        reward = success_probs.mean().item()
        label = Binomial(probs=success_probs[item_id]).sample()
        interaction = (item_id * 2 + label + 1).long().view(1, 1)
        _, self.hidden = self.dkt(item_inputs=interaction,
                                  skill_inputs=None,
                                  hidden=self.hidden)
        done = self.steps > self.simulation_steps
        return interaction.item(), reward, done, {}
        """
        self.steps += 1
        done = self.steps > self.simulation_steps
        return 0, item_id, done, {}

    @torch.no_grad()
    def reset(self):
        """
        #interaction = random.choice(self.data)[:self.history_steps].unsqueeze(0)
        interaction = self.data[0][:self.history_steps].unsqueeze(0)
        _, self.hidden = self.dkt(item_inputs=interaction,
                                  skill_inputs=None,
                                  hidden=None)
        self.steps = 0
        interaction = 0
        return interaction
        """
        self.steps = 0
        return 0


    def process_data(self, df):
        item_ids = [torch.tensor(u_df["item_id"].values, dtype=torch.long)
                    for _, u_df in df.groupby("user_id")]
        labels = [torch.tensor(u_df["correct"].values, dtype=torch.long)
                  for _, u_df in df.groupby("user_id")]
        self.data = [torch.cat((torch.zeros(1, dtype=torch.long), i * 2 + l + 1))[:-1]
                     for (i, l) in zip(item_ids, labels)]


def run_episode(env, model, recurrent=False):
    total_reward = 0
    obs = env.reset()
    done = [False for _ in range(env.num_envs)]
    if recurrent:
        state = None

    while not done[0]:
        if recurrent:
            action, state = model.predict(obs, state=state, mask=done)
        else:
            action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        total_reward += reward[0]

    return total_reward


if __name__ == "__main__":
    df = pd.read_csv('data/assistments09/preprocessed_data.csv', sep="\t")
    dkt = torch.load('save/dkt/assistments09,item_in=True,skill_in=False,item_out=True,skill_out=False')

    env = DummyVecEnv([lambda: StudentEnv(df, dkt)])
    #env = SubprocVecEnv([lambda: StudentEnv(df, dkt) for _ in range(8)])

    model = PPO2(MlpPolicy, env, verbose=0)
    #model = PPO2(MlpLstmPolicy, env, verbose=0)

    reward_before = run_episode(env, model)
    model.learn(total_timesteps=10000)
    reward_after = run_episode(env, model)

    print(reward_before, reward_after)
