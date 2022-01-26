import copy
import os
import numpy as np
import matplotlib.pyplot as plt


class System():
    """状態
    """

    def __init__(self, calc_state_dot, state_0=np.array([0., 0.], dtype=np.float32)):
        """
        Args:
            calc_state_dot: 微分計算関数
            state_0 (optional): 初期値
        """
        self._calc_state_dot = calc_state_dot
        self.state = state_0
        self.log_state = []

    def update_state(self, u, dt=0.01):
        """runge kuttaで状態更新, ログ追加
        Args:
            u : 入力
            dt (float, optional): 差分近似幅
        Returns:
            状態の更新値
        """
        k1 = self._calc_state_dot(self.state, u)
        k2 = self._calc_state_dot(self.state + k1 * dt / 2, u)
        k3 = self._calc_state_dot(self.state + k2 * dt / 2, u)
        k4 = self._calc_state_dot(self.state + k3 * dt, u)
        state_dot = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        self.state += dt * state_dot
        self.log_state.append(copy.deepcopy(self.state))
        return self.state

    def set_state(self, state):
        self.state = state


class SimulatorSystem():
    """系列計算
    """

    def __init__(self, calc_state_dot, calc_lam_dot):
        # state_dotとlam_dotを求める関数を入力
        self._calc_state_dot = calc_state_dot
        self._calc_lam_dot = calc_lam_dot

    def predict_adjoint(self, state, u_list, N, dt):
        """各系列の計算
        Args:
            state: 状態
            u_list: [description]
            N: ステップ
            dt: 差分近似幅

        Returns:
            状態系列と随伴係数系列
        """
        state_list = self._predict(state, u_list, N, dt)
        lam_list = self._adjoint(state_list, u_list, N, dt)
        return state_list, lam_list

    def _predict(self, state, u_list, N, dt):
        state_list = [state]
        for i in range(N):
            state_dot = self._calc_state_dot(state_list[i], u_list[i])
            state_next = state_list[i] + dt * state_dot
            state_list.append(state_next)
        return np.array(state_list, dtype=np.float32)

    def _adjoint(self, state_list, u_list, N, dt):
        lam_list = [state_list[-1]]
        for i in range(N - 1, 0, -1):
            lam_dot = self._calc_lam_dot(state_list[i], lam_list[0], u_list[i])
            lam_pre = lam_list[0] + dt * lam_dot
            lam_list.insert(0, lam_pre)
        return np.array(lam_list, dtype=np.float32)


class Controller():
    def __init__(self,
                 u_list_0, calc_f, calc_state_dot, calc_lam_dot,
                 u_num=1,
                 u_max=None,
                 zeta=100.,
                 h=0.01,
                 t_f=1.,
                 alpha=0.5,
                 N=10,
                 threshold=0.001):
        # 一部参考: https://www.mathworks.com/content/dam/mathworks/mathworks-dot-com/solutions/automotive/files/jp-expo-2015/mpc-design.pdf
        self.u_num = u_num
        self.u_max = u_max
        self.h = h  # 差分近似幅
        self.N = N  # 分割数(ステップ数)
        self.t_f = t_f  # 最終時間
        self.zeta = zeta  # 安定化ゲイン
        self.alpha = alpha  # 時間上昇ゲイン
        self.threshold = threshold
        self.calc_f = calc_f  # F値計算関数
        self.input_num = u_list_0.shape[0]
        self.max_iter = self.input_num * self.N

        # simulator
        self.simulator = SimulatorSystem(calc_state_dot, calc_lam_dot)  # state_dotとlam_dotを求める関数を入力

        # 初期値
        self.u_list = np.array([np.ones(self.N) * u_list_0[i] for i in range(self.input_num)])

        # 入力とF値ログ
        self.log_u = []
        self.log_f = []

    def calc_input(self, state, time):
        # "非線形最適制御入門"の式にしたがって入力値を計算
        dt = self.t_f * (1. - np.exp(-self.alpha * time)) / float(self.N)
        state_dot = self.simulator._calc_state_dot(state, self.u_list[:self.u_num, 0])

        state_delta = self.h * state_dot
        u_delta_list = self.h * self.u_list

        state_list, lam_list = self.simulator.predict_adjoint(
            state + state_delta, self.u_list[:self.u_num].T, self.N, dt)
        F_x = self._calc_f(state_list, lam_list, self.u_list, self.N, dt)
        state_list, lam_list = self.simulator.predict_adjoint(state, self.u_list[:self.u_num].T, self.N, dt)
        F = self._calc_f(state_list, lam_list, self.u_list, self.N, dt)

        state_list, lam_list = self.simulator.predict_adjoint(
            state + state_delta, self.u_list[:self.u_num].T + u_delta_list[:self.u_num].T, self.N, dt)
        F_ux = self._calc_f(state_list, lam_list, self.u_list + u_delta_list, self.N, dt)

        r_0 = -self.zeta * F - ((F_x - F) / self.h) - ((F_ux - F_x) / self.h)
        r_0_norm = np.linalg.norm(r_0)

        v_list = np.zeros((self.input_num, self.N, self.max_iter + 1))
        v_list[:, :, 0] = r_0 / r_0_norm  # 最初の基底を算出
        h_list = np.zeros((self.max_iter + 1, self.max_iter + 1))
        e_list = np.zeros((self.max_iter + 1, 1))
        e_list[0] = 1.

        y_list_pre = np.empty(0)
        for i in range(self.max_iter):
            state_delta = self.h * state_dot
            u_list_delta = self.h * v_list[:, :, i]

            state_list, lam_list = self.simulator.predict_adjoint(
                state + state_delta, self.u_list[:self.u_num].T + u_list_delta[:self.u_num].T, self.N, dt)
            F_ux = self._calc_f(state_list, lam_list, self.u_list + u_list_delta, self.N, dt)

            a_list = ((F_ux - F_x) / self.h)
            sum_a_v_list = np.zeros((self.input_num, self.N))
            for j in range(i + 1):
                h_list[j, i] = np.trace(a_list.T @ v_list[:, :, j])
                sum_a_v_list += h_list[j, i] * v_list[:, :, j]

            v_est = a_list - sum_a_v_list
            h_list[i + 1, i] = np.linalg.norm(v_est)
            v_list[:, :, i + 1] = v_est / h_list[i + 1, i]
            h_list_inv = np.linalg.pinv(h_list[:i + 1, :i])
            y_list = np.dot(h_list_inv, r_0_norm * e_list[:i + 1])

            val_judge = r_0_norm * e_list[:i + 1] - np.dot(h_list[:i + 1, :i], y_list[:i])
            if np.linalg.norm(val_judge) < self.threshold or i == self.max_iter - 1:
                val_update = np.dot(v_list[:, :, :i - 1], y_list_pre[:i - 1])[:, :, 0]
                u_list_dot = u_list_delta + val_update
                break
            y_list_pre = y_list

        # 入力値更新
        self.u_list += u_list_dot * self.h

        # 状態系列と随伴係数の計算
        state_list, lam_list = self.simulator.predict_adjoint(state, self.u_list[:self.u_num].T, self.N, dt)
        F = self._calc_f(state_list, lam_list, self.u_list, self.N, dt)
        print(f"F: {np.linalg.norm(F):.6f}")

        # 次の入力値の計算
        if self.u_max is not None:
            if self.u_num > 1:
                u_list_next = [np.clip(self.u_list[i:i + 1, 0], -self.u_max[i], self.u_max[i])[0]
                               for i in range(self.u_num)]
            else:
                u_list_next = np.clip(self.u_list[:self.u_num, 0], -self.u_max, self.u_max)
        else:
            u_list_next = self.u_list[:self.u_num, 0]

        # ログ保存
        self.log_f.append(np.linalg.norm(F))
        self.log_u.append(u_list_next)

        return u_list_next

    def _calc_f(self, state_list, lam_list, u_list, N, dt):
        # F値計算
        F = np.zeros((self.input_num, self.N))
        for i in range(N):
            F[:, i] = self.calc_f(state_list[i], lam_list[i], u_list[:, i], N, dt)
        return F


def plot(system, controller, dt):
    # ログのプロット
    # 状態数と入力数に応じてレイアウトを変えている
    height = max(len(system.log_state[0]), len(controller.log_u[0]))
    fig = plt.figure(figsize=(6.4, height * 2.4))

    for i in range(len(system.log_state[0])):
        ax = plt.subplot2grid((height, 2), (i, 0))
        ax.plot(np.arange(len(system.log_state)) * dt, np.array(system.log_state)[:, i])
        ax.set_xlabel("time [s]")
        ax.set_ylabel(f"state[{i}]")

    for i in range(len(controller.log_u[0])):
        ax = plt.subplot2grid((height, 2), (i, 1))
        ax.plot(np.arange(len(controller.log_u)) * dt, np.array(controller.log_u)[:, i])
        ax.set_xlabel("time [s]")
        ax.set_ylabel(f"u[{i}]")

    fig.tight_layout()
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(len(controller.log_f) - 1) * dt, controller.log_f[1:])
    ax.set_xlabel("time [s]")
    ax.set_ylabel("optimal error")
    plt.show()


def save_log(log_state, log_u, iter_num, dt, fps=10):
    # データの保存
    log_dir = "log/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    num_files = len(next(os.walk(log_dir))[2])
    with open(f"log/log_nmpc_{num_files}.csv", mode="w") as f:
        for i in range(1, iter_num - 1):
            f.write(f"{float(i) * dt}")
            for j in range(len(log_state[i])):
                f.write(f", {log_state[i][j]}")
            for j in range(len(log_u[i])):
                f.write(f", {log_u[i][j]}")
            f.write("\n")
