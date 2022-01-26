import numpy as np
import Control
import shutil
import utils
import os
import matplotlib.pyplot as plt


class TwoWheelsModel:
    """対向二輪モデル定義
    """

    def __init__(self,
                 start=np.array([10., 4.]),
                 end=np.array([0., 0.]),
                 th0=0,
                 u_list_0=np.array([1., .1]),
                 u_max=None):

        def calc_state_dot(state, u):
            """状態方程式そのまま
            """
            x, y, th = state
            u1, u2 = u
            x_dot = np.cos(th) * u1
            y_dot = np.sin(th) * u1
            th_dot = u2
            return np.array([x_dot, y_dot, th_dot], dtype=np.float32)

        def calc_lam_dot(state, lam, u):
            x, y, th = state
            lam1, lam2, lam3 = lam
            u1, u2 = u
            lam1_dot = 1000 * (x - end[0])
            lam2_dot = 1000 * (y - end[1])
            lam3_dot = th - lam1 * np.sin(th) * u1 + lam2 * np.cos(th) * u1
            return np.array([lam1_dot, lam2_dot, lam3_dot], dtype=np.float32)

        def calc_f(state, lam, u, N, dt):
            x, y, th = state
            lam1, lam2, lam3 = lam
            # --------------------
            # 拘束条件なし
            u1, u2 = u
            F = np.zeros(2)
            F[0] = u1 + lam1 * np.cos(th) + lam2 * np.sin(th)
            F[1] = u2 + lam3
            # ----------------------
            # 拘束条件あり
            # u1, u2, v1, v2, rho1, rho2 = u
            # F = np.zeros(6)
            # F[0] = u1 + lam1 * np.cos(th) + lam2 * np.sin(th) + 2 * rho1 * u1
            # F[1] = u2 + lam3 + 2 * rho2 * u2
            # F[2] = -0.01 + 2. * rho1 * v1
            # F[3] = -0.01 + 2. * rho2 * v2
            # F[4] = u1**2 + v1**2 - 2.**2
            # F[5] = u2**2 + v2**2 - 2.**2
            return F

        # 入力数
        u_num = 2

        self.start = start
        self.end = end

        self.system = Control.System(calc_state_dot, state_0=np.r_[start, th0])
        self.controller = Control.Controller(
            u_list_0,
            calc_f,
            calc_state_dot,
            calc_lam_dot,
            u_num,
            u_max=u_max,
            N=5,
            alpha=0.2,
            t_f=0.5)

        self.img_dir = "img_control_mpc"
        self.gif_dir = "gif_control_mpc"

    def save_gif(self, duration):
        # gif保存
        if not os.path.exists(self.gif_dir):
            os.makedirs(self.gif_dir)
        utils.save_gif(self.img_dir, self.gif_dir, "gif_control_mpc", duration=duration)

    def save_img(self):
        fig, ax = plt.subplots()
        # 道筋プロット
        ax.plot(np.array(self.system.log_state)[:, 0], np.array(self.system.log_state)[:, 1], color="b", linestyle="--")
        ax.set_aspect("equal")
        # ゴール地点描画
        center = self.system.log_state[-2][:2]
        center_delta = self.system.log_state[-1][:2] - center
        sign = np.sign(center_delta[0])
        r = 0.3
        ax.add_artist(plt.Circle(center, r, fill=False, color="k", linewidth=1.5))
        ax.add_artist(plt.Circle(self.end, r / 2, fill=False, color="b", linewidth=1.5))
        ax.add_artist(plt.Circle(self.end, r / 10, fill=True, color="b"))
        # 機体の角度プロット
        line_x = np.array([center[0], center[0] + sign * np.cos(self.system.log_state[-2][2]) * r])
        line_y = np.array([center[1], center[1] + sign * np.sin(self.system.log_state[-2][2]) * r])
        ax.plot(line_x, line_y, color="r", linewidth=1.5)

        ax.set_xlim(min(self.start[0], self.end[0], -1), max(self.start[0], self.end[0], 1) * 1.2)
        ax.set_ylim(min(self.start[1], self.end[1], -1), max(self.start[1], self.end[1], 1) * 1.2)
        num_img_files = len(next(os.walk(self.img_dir))[2])
        plt.savefig(f"{self.img_dir}/img_control_mpc_{str(num_img_files).zfill(6)}.png")
        plt.cla()
        plt.close(fig)

    def render(self):
        fig, ax = plt.subplots()
        # 道筋プロット
        ax.plot(np.array(self.system.log_state)[1:, 0], np.array(
            self.system.log_state)[1:, 1], color="b", linestyle="--")
        ax.set_aspect("equal")
        # 途中仮定プロット
        draw_num = 5
        for i in np.arange(1, self.iter_num - 1, int(self.iter_num / draw_num)):
            center = self.system.log_state[i][:2]
            center_delta = self.system.log_state[i + 1][:2] - center
            sign = np.sign(center_delta[0])
            r = 0.5
            ax.add_artist(plt.Circle(center, r, fill=False, color="k", linewidth=1.5))
            line_x = np.array([center[0], center[0] + sign * np.cos(self.system.log_state[i][2]) * r])
            line_y = np.array([center[1], center[1] + sign * np.sin(self.system.log_state[i][2]) * r])
            ax.plot(line_x, line_y, color="r", linewidth=1.5)
        # ゴール地点
        ax.add_artist(plt.Circle(self.end, r / 2, fill=False, color="b", linewidth=1.5))
        ax.add_artist(plt.Circle(self.end, r / 10, fill=True, color="b"))

        ax.set_xlim(min(self.start[0], self.end[0], -1), max(self.start[0], self.end[0], 1) * 1.2)
        ax.set_ylim(min(self.start[1], self.end[1], -1), max(self.start[1], self.end[1], 1) * 1.2)
        plt.show()

    def exec(self, iter_num, dt, is_save_gif, is_save_log, duration=100):
        # 実行
        self.iter_num = iter_num
        if is_save_gif:
            if os.path.exists(self.img_dir):
                shutil.rmtree(self.img_dir)
            os.makedirs(self.img_dir)
        # 状態->入力->状態更新の繰り返し
        for i in range(1, iter_num):
            time = float(i) * dt
            state = self.system.state
            u = self.controller.calc_input(state, time)
            self.system.update_state(u)

            if i % int(duration / 1000 / dt) == 0 and is_save_gif:
                self.save_img()

        if is_save_gif:
            self.save_gif(duration)
        if is_save_log:
            Control.save_log(self.system.log_state, self.controller.log_u, iter_num, dt)
