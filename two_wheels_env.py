import cv2
import utils
import os
import copy
import matplotlib.pyplot as plt
import gym
import shutil
import Control
import numpy as np
from gym import spaces
from RL import DynamicsModel, RandomPolicy, PPO
from scipy.signal import lfilter
from cpprb import ReplayBuffer
import torch


class TwoWheelsEnv(gym.Env):
    """対向二輪用gym環境
    """

    metadata = {'render.modes': ["human"]}

    def __init__(self, system, controller, u_max, duration):
        super().__init__()
        self.system = system
        self.controller = controller
        self.duration = duration
        self.dt = 0.01
        self.t = 0
        self.u1_max, self.u2_max = u_max

        # 各空間定義
        self.action_space = spaces.Box(
            low=-u_max, high=u_max, shape=(2,), dtype=np.float32
        )
        high = np.array([10, 10, 2 * np.pi], dtype=np.float32)
        low = np.array([0, 0, -2 * np.pi], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.img_dir = "img_rl_mpc"
        self.gif_dir = "gif_rl_mpc"

    def step(self, action):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"
        self.t += self.dt
        u1, u2 = action
        x, y, _ = self.system.state
        x_goal, y_goal = self.goal
        u1 = np.clip(u1, -self.u1_max, self.u1_max)
        u2 = np.clip(u2, -self.u2_max, self.u2_max)
        self.last_u1 = u1
        self.last_u2 = u2
        # costs = (x - x_goal)**2 + (y - y_goal)**2
        costs = (x - x_goal)**2 + (y - y_goal)**2 + .0001 * u1**2 + .0001 * u2**2
        self.system.update_state(action)
        return self.system.state, -costs, False, {}

    def reset(self, is_save_gif=False):
        # 開始位置のみランダム
        self.start = np.random.uniform(low=np.array([3., 3., 0.]), high=np.array([7., 7., np.pi]))
        self.goal = np.array([5., 5.])
        self.system.state = copy.deepcopy(self.start)
        self.system.log_state = []
        self.last_u1 = None
        self.last_u2 = None
        self.t = 0
        self.is_save_gif = is_save_gif
        # ディレクトリ作成
        if is_save_gif:
            if os.path.exists(self.img_dir):
                shutil.rmtree(self.img_dir)
            os.makedirs(self.img_dir)
        return self.system.state

    def save_gif(self):
        # gif保存
        if not os.path.exists(self.gif_dir):
            os.makedirs(self.gif_dir)
        utils.save_gif(self.img_dir, self.gif_dir, "gif_rl_mpc", duration=self.duration)

    def save_log(self, iter_num):
        # ログ保存
        Control.save_log(self.system.log_state, self.controller.log_u, iter_num, self.dt)

    def render(self):
        if int(self.t / self.dt) % int(self.duration / (1000 * self.dt)) == 0:
            fig, ax = plt.subplots()
            # 道筋プロット
            ax.plot(np.array(self.system.log_state)[:, 0], np.array(
                self.system.log_state)[:, 1], color="b", linestyle="--")
            ax.set_aspect("equal")
            # ゴール地点描画
            center = self.system.log_state[-2][:2]
            sign = 1
            r = 0.3
            ax.add_artist(plt.Circle(center, r, fill=False, color="k", linewidth=1.5))
            ax.add_artist(plt.Circle(self.goal, r / 2, fill=False, color="b", linewidth=1.5))
            ax.add_artist(plt.Circle(self.goal, r / 10, fill=True, color="b"))
            # 機体の角度プロット
            line_x = np.array([center[0], center[0] + sign * np.cos(self.system.log_state[-2][2]) * r])
            line_y = np.array([center[1], center[1] + sign * np.sin(self.system.log_state[-2][2]) * r])
            ax.plot(line_x, line_y, color="r", linewidth=1.5)

            plt.xlim(min(self.start[0], self.goal[0]) * 0.8, max(self.start[0], self.goal[0]) * 1.2)
            plt.ylim(min(self.start[1], self.goal[1]) * 0.8, max(self.start[1], self.goal[1]) * 1.2)
            fig.canvas.draw()
            im = np.array(fig.canvas.renderer.buffer_rgba())
            im = cv2.cvtColor(im, cv2.COLOR_RGBA2BGR)
            if self.is_save_gif:
                num_img_files = len(next(os.walk(self.img_dir))[2])
                plt.savefig(f"{self.img_dir}/img_rl_mpc_{str(num_img_files).zfill(6)}.png")
            plt.close(fig)
            # 描画
            cv2.imshow('test', im)
            cv2.waitKey(1)

    def close(self):
        pass


class TwoWheelsSimulator:
    def __init__(self, env, u_max):
        self.env = env
        self.u_max = u_max
        self.batch_size = 100
        self.obs_dim = self.env.observation_space.high.size
        self.act_dim = self.env.action_space.high.size

    def RS_init(self):
        rb_dict = {
            "size": 10000,
            "default_dtype": np.float32,
            "env_dict": {
                "obs": {"shape": self.env.observation_space.shape},
                "next_obs": {"shape": self.env.observation_space.shape},
                "act": {"shape": self.env.action_space.shape}}}
        self.dynamics_buffer = ReplayBuffer(**rb_dict)
        self.dynamics_model = DynamicsModel(input_dim=self.obs_dim + self.act_dim, output_dim=self.obs_dim)
        self.policy = RandomPolicy(max_action=self.env.action_space.high[0], act_dim=self.act_dim)

    def PPO_init(self):
        rb_dict = {
            "size": 10000,
            "default_dtype": np.float32,
            "env_dict": {
                "obs": {"shape": self.env.observation_space.shape},
                "next_obs": {"shape": self.env.observation_space.shape},
                "act": {"shape": self.env.action_space.shape}}}
        self.dynamics_buffer = ReplayBuffer(**rb_dict)
        # ダイナミクスモデルの数
        self.n_dynamics_model = 5
        self.dynamics_model = [DynamicsModel(
            input_dim=self.obs_dim + self.act_dim, output_dim=self.obs_dim) for _ in range(self.n_dynamics_model)]
        # ダイナミクスモデル学習用バッファのクリア
        self.dynamics_buffer.clear()
        self.policy = PPO(self.env.observation_space.shape,
                          self.env.action_space.shape,
                          max_action=self.env.action_space.high[0])
        rb_dict = {
            "size": self.policy.horizon,
            "default_dtype": np.float32,
            "env_dict": {
                "obs": {"shape": self.env.observation_space.shape},
                "act": {"shape": self.env.action_space.shape},
                "done": {},
                "logp": {},
                "ret": {},
                "adv": {}}}
        self.on_policy_buffer = ReplayBuffer(**rb_dict)

        rb_dict = {
            "size": self.iter_num,
            "default_dtype": np.float32,
            "env_dict": {
                "obs": {"shape": self.env.observation_space.shape},
                "act": {"shape": self.env.action_space.shape},
                "next_obs": {"shape": self.env.observation_space.shape},
                "rew": {},
                "done": {},
                "logp": {},
                "val": {}}}
        self.episode_buffer = ReplayBuffer(**rb_dict)

        # 各ダイナミクスモデルの評価エピソード数
        self.n_eval_episodes_per_model = 5

    def save(self, model_dir="model", prefix=""):
        num_model_files = len(next(os.walk(model_dir))[2])
        torch.save(self.dynamics_model.state_dict(), f"{model_dir}/{prefix}_{str(num_model_files).zfill(2)}.pth")
        print(f"Model saved: {model_dir}/{prefix}_{str(num_model_files).zfill(2)}.pth")

    def evaluate_policy(self, total_steps, test_episodes=10):
        # 複数のエピソードで現在の方策を評価し平均リターンを返す
        avg_test_return = 0.
        for i in range(test_episodes):
            episode_return = 0.
            obs = self.env.reset()
            for _ in range(self.iter_num):
                act = self.policy.get_action(obs, test=True)
                next_obs, rew, _, _ = self.env.step(act)
                episode_return += rew
                obs = next_obs
            avg_test_return += episode_return
        return avg_test_return / test_episodes

    def collect_transitions_real_env(self):
        obs = self.env.reset()
        episode_steps = 0
        for _ in range(self.policy.horizon):
            episode_steps += 1
            act = self.policy.get_action(obs)
            # 実環境でロールアウト
            next_obs, *_ = self.env.step(act)
            self.dynamics_buffer.add(obs=obs, act=act, next_obs=next_obs)
            obs = next_obs
            if episode_steps == self.iter_num:
                episode_steps = 0
                obs = self.env.reset()

    def fit_dynamicses(self, n_iter=50):
        mean_losses = np.zeros(shape=(self.n_dynamics_model,), dtype=np.float32)
        for _ in range(n_iter):
            samples = self.dynamics_buffer.sample(self.batch_size)
            inputs = np.concatenate([samples["obs"], samples["act"]], axis=1)
            labels = samples["next_obs"] - samples["obs"]
            # 複数のモデルを学習
            for i, dynamics_model in enumerate(self.dynamics_model):
                mean_losses[i] += dynamics_model.fit(
                    torch.from_numpy(inputs).float(),
                    torch.from_numpy(labels).float())
        return mean_losses

    def collect_transitions_sim_env(self):
        # ダイナミクスモデルを用いた方策学習用サンプルの生成
        self.on_policy_buffer.clear()
        n_episodes = 0
        ave_episode_return = 0
        while self.on_policy_buffer.get_stored_size() < self.policy.horizon:
            # 実環境で初期値を取得
            obs = self.env.reset()
            episode_return = 0.
            for i in range(self.iter_num):
                act, logp, val = self.policy.get_action_and_val(obs)
                act = act.cpu().numpy()[0]
                # ダイナミクスモデルを用いて次状態を予測
                next_obs = self.predict_next_state_ppo(obs, act)
                rew = self.reward_fn(obs, act)
                self.episode_buffer.add(obs=obs, act=act, next_obs=next_obs, rew=rew,
                                        done=False, logp=logp, val=val)
                obs = next_obs
                episode_return += rew
            self.finish_horizon(last_val=val)
            ave_episode_return += episode_return
            n_episodes += 1
        return ave_episode_return / n_episodes

    def finish_horizon(self, last_val=0):
        # PPOの学習のため、エピソード終了時に必要な計算
        samples = self.episode_buffer.get_all_transitions()
        rews = np.append(samples["rew"], last_val)
        vals = np.append(samples["val"], last_val)

        # GAE-Lambda
        deltas = rews[:-1] + self.policy.discount * vals[1:] - vals[:-1]
        advs = discount_cumsum(deltas, self.policy.discount * self.policy.lam)

        # 価値関数学習の際のターゲットとなるリターンを計算
        rets = discount_cumsum(rews, self.policy.discount)[:-1]
        self.on_policy_buffer.add(
            obs=samples["obs"], act=samples["act"], done=samples["done"],
            ret=rets, adv=advs, logp=np.squeeze(samples["logp"]))
        self.episode_buffer.clear()

    def update_policy(self):
        # 前準備としてAdvantageの平均と分散を計算
        samples = self.on_policy_buffer.get_all_transitions()
        mean_adv = np.mean(samples["adv"])
        std_adv = np.std(samples["adv"])

        for _ in range(self.policy.n_epoch):
            samples = self.on_policy_buffer._encode_sample(np.random.permutation(self.policy.horizon))
            adv = (samples["adv"] - mean_adv) / (std_adv + 1e-8)
            # actor_loss, critic_loss = 0., 0.
            for idx in range(int(self.policy.horizon / self.policy.batch_size)):
                target = slice(idx * self.policy.batch_size, (idx + 1) * self.policy.batch_size)
                self.policy.train(
                    states=samples["obs"][target],
                    actions=samples["act"][target],
                    advantages=adv[target],
                    logp_olds=samples["logp"][target],
                    returns=samples["ret"][target])

    def evaluate_current_return(self, init_states):
        # 同じ初期値で評価できるように、関数内で初期値を生成せず引数として与える
        n_episodes = self.n_dynamics_model * self.n_eval_episodes_per_model
        assert init_states.shape[0] == n_episodes

        obses = init_states.copy()
        next_obses = np.zeros_like(obses)
        returns = np.zeros(shape=(n_episodes,), dtype=np.float32)

        for _ in range(self.iter_num):
            # 現在の方策を用いて行動を生成
            acts = self.policy.get_action(obses, test=True)
            for i in range(n_episodes):
                model_idx = i // self.n_eval_episodes_per_model
                env_act = np.clip(acts[i], self.env.action_space.low, self.env.action_space.high)
                next_obses[i] = self.predict_next_state_ppo(obses[i], env_act, idx=model_idx)
            returns += self.reward_fn(obses, acts)
            obses = next_obses

        return returns

    def predict_next_state_ppo(self, obses, acts, idx=None):
        # ダイナミクスモデルを用いた次の状態予測
        is_single_input = obses.ndim == acts.ndim and acts.ndim == 1
        if is_single_input:
            obses = np.expand_dims(obses, axis=0)
            acts = np.expand_dims(acts, axis=0)

        inputs = np.concatenate([obses, acts], axis=1)
        inputs = torch.from_numpy(inputs).float()

        # 次の状態を予測するためのモデルをランダムに選択する
        idx = np.random.randint(self.n_dynamics_model) if idx is None else idx
        obs_diffs = self.dynamics_model[idx].predict(inputs).data.numpy()

        if is_single_input:
            return (obses + obs_diffs)[0]
        return obses + obs_diffs

    def predict_next_state_rs(self, obses, acts):
        inputs = torch.from_numpy(np.concatenate([obses, acts], axis=1)).float()
        next_obses = obses + self.dynamics_model.predict(inputs).data.numpy()
        return next_obses

    def random_shooting(self, init_obs, n_mpc_episodes=64, horizon=20, **kwargs):
        init_actions = self.policy.get_actions(batch_size=n_mpc_episodes)
        returns = np.zeros(shape=(n_mpc_episodes,))
        obses = np.tile(init_obs, (n_mpc_episodes, 1))

        for i in range(horizon):
            acts = init_actions if i == 0 else self.policy.get_actions(batch_size=n_mpc_episodes)
            next_obses = self.predict_next_state_rs(obses, acts)
            rewards = self.reward_fn(obses, acts)
            returns += rewards
            obses = next_obses
        return init_actions[np.argmax(returns)]

    def fit_dynamics(self, n_iter=50):
        mean_loss = 0.
        for _ in range(n_iter):
            samples = self.dynamics_buffer.sample(self.batch_size)
            inputs = np.concatenate([samples["obs"], samples["act"]], axis=1)
            labels = samples["next_obs"] - samples["obs"]
            mean_loss += self.dynamics_model.fit(
                torch.from_numpy(inputs).float(),
                torch.from_numpy(labels).float())
        return mean_loss

    def reward_fn(self, obses, acts):
        # 報酬関数の定義
        is_single_input = obses.ndim == acts.ndim and acts.ndim == 1
        if is_single_input:
            x, y = obses[:2]
        else:
            assert obses.ndim == acts.ndim == 2
            assert obses.shape[0] == acts.shape[0]
            acts = np.squeeze(acts)
            x, y = obses[:, 0], obses[:, 1]
        if acts.shape[0] == 2:
            acts1 = np.clip(acts[0], -self.u_max[0], self.u_max[0])
            acts2 = np.clip(acts[1], -self.u_max[1], self.u_max[1])
        else:
            acts1 = np.clip(acts[:, 0], -self.u_max[0], self.u_max[0])
            acts2 = np.clip(acts[:, 1], -self.u_max[1], self.u_max[1])
        costs = 100 * (x - self.env.goal[0]) ** 2 + 100 * (y - self.env.goal[1]) ** 2 + acts1**2 + acts2**2
        return -costs

    def exec_PPO(self, iter_num):
        self.iter_num = iter_num
        self.PPO_init()
        total_steps = 0
        test_episodes = 10

        while True:
            # 実環境でダイナミクスモデルを学習するためのサンプルを収集
            self.collect_transitions_real_env()
            total_steps += self.policy.horizon
            # ダイナミクスモデルの学習
            self.fit_dynamicses()
            n_updates = 0
            # 方策評価のための初期値の生成
            init_states_for_eval = np.array([
                self.env.reset() for _ in range(self.n_dynamics_model * self.n_eval_episodes_per_model)])
            # 方策更新前の性能評価
            returns_before_update = self.evaluate_current_return(init_states_for_eval)
            while True:
                n_updates += 1
                # ダイナミクスモデルを用いて方策学習用のサンプルを生成
                average_return = self.collect_transitions_sim_env()
                # 方策更新
                self.update_policy()
                # 方策更新後の性能評価
                returns_after_update = self.evaluate_current_return(init_states_for_eval)
                # 方策更新による性能評価の割合を計算
                improved_ratio = np.sum(returns_after_update > returns_before_update) / (
                    self.n_dynamics_model * self.n_eval_episodes_per_model)
                # 方策更新による性能向上があまり見られない場合、ループを抜ける
                if improved_ratio < 0.7:
                    print(
                        "Training total steps: {0: 7} improved ratio: {1: .2f} simulated return: {2: .4f} n_update: {3: 2}".format(
                            total_steps,
                            improved_ratio,
                            average_return,
                            n_updates))
                    break
                returns_before_update = returns_after_update.copy()
            # 実環境での方策評価
            if total_steps // self.policy.horizon % 10 == 0:
                avg_test_return = self.evaluate_policy(total_steps, test_episodes)
                print("Evaluation Total Steps: {0: 7} Average Reward {1: 5.4f} over {2: 2} episodes".format(
                    total_steps, avg_test_return, test_episodes))
            # 100ステップ学習
            if total_steps // self.policy.horizon % 100 == 0:
                break

    def exec(self, func, episode_num, iter_num,
             is_RS=False,
             is_render=True,
             is_obs=True,
             is_save_gif=True,
             is_buffer=False,
             is_fit_dynamics=False):
        if is_RS:
            self.RS_init()
        for episode_idx in range(episode_num):
            obs = self.env.reset(is_save_gif=is_save_gif)
            total_reward = 0.
            for _ in range(iter_num):
                if is_obs:
                    action = func(obs, test=True)
                else:
                    action = func()
                next_obs, reward, done, _ = self.env.step(action)
                total_reward += reward
                if is_render:
                    self.env.render()
                if is_buffer:
                    self.dynamics_buffer.add(obs=obs, act=action, next_obs=next_obs)
                if done:
                    break
                obs = next_obs

            if is_fit_dynamics:
                mean_loss = self.fit_dynamics(n_iter=100)
                print(f"iter={episode_idx: 3d} total reward: {total_reward: .4f} mean loss: {mean_loss:.6f}")
            else:
                print(f"iter={episode_idx:3d} total reward: {total_reward:.4f}")
            if is_save_gif:
                self.env.save_gif()


def discount_cumsum(x, discount):
    return lfilter(
        b=[1],
        a=[1, float(-discount)],
        x=x[::-1],
        axis=0)[::-1]
