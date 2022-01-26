from two_wheels_env import TwoWheelsEnv, TwoWheelsSimulator
from two_wheels_model import TwoWheelsModel
import numpy as np

if __name__ == "__main__":
    # パラメタ定義
    dt = 0.01  # シミュレーション周期
    fps = 10.  # gifのfps
    iter_time = 3.  # イテレーション時間
    duration = int(1000. / fps)
    iter_num = int(iter_time / dt)  # イテレーション回数
    start = np.random.uniform(low=np.array([0., 0.]), high=np.array([5., 5.]))  # ランダムな初期位置
    end = np.array([5., 5.])  # 固定終了位置
    th0 = np.random.uniform(low=-np.pi, high=np.pi)  # 初期角度
    # u_list_0 = np.array([1, 0.1, 0.1, 2.5, 0.8, 0.8])
    u_list_0 = np.array([1., 0.1])
    u_max = np.array([2., 2.])  # 入力数最大値

    # モデル作成
    model = TwoWheelsModel(start=start,
                           end=end,
                           th0=th0,
                           u_list_0=u_list_0,
                           u_max=u_max)
    # Gym環境作成
    env = TwoWheelsEnv(model.system, model.controller, u_max, duration)
    # 実行環境
    sim = TwoWheelsSimulator(env, u_max)

    # テスト
    # ----------------------------
    sim.exec(env.action_space.sample, is_RS=True, episode_num=2, iter_num=iter_num, is_obs=False)

    # Random Shooting
    # ----------------------------
    # 事前学習のための遷移収集
    sim.exec(env.action_space.sample, is_RS=True, episode_num=20, iter_num=200,
             is_obs=False, is_buffer=True, is_save_gif=False)
    # ダイナミクスモデルの事前学習
    sim.fit_dynamics(n_iter=1000)
    # train
    sim.exec(sim.random_shooting, episode_num=200, iter_num=iter_num,
             is_RS=True, is_render=False, is_buffer=True, is_save_gif=False, is_fit_dynamics=True)
    # test
    sim.exec(sim.random_shooting, episode_num=3, iter_num=iter_num)
    # モデル保存
    sim.save()

    # PPO
    # ----------------------------
    # train
    sim.exec_PPO(iter_num=iter_num)
    # test
    sim.exec(sim.policy_get_action, episode_num=3, iter_num=iter_num)
    # モデル保存
    sim.save()
