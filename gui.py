import tkinter as tk
import numpy as np
import Control
from two_wheels_model import TwoWheelsModel


class GUI:
    """tkinterで簡易GUI設計
    """

    def __init__(self, dt, duration, end, u_list_0):
        self.dt = dt
        self.duration = duration
        self.end = end
        self.u_list_0 = u_list_0

        # 描画設定
        self.gif_index = 0
        self.root = tk.Tk()
        self.root.title("TwoWheels")
        self.root.geometry("800x480")

        # 白ベース
        canvas = tk.Canvas(bg="white", width=800, height=480)
        canvas.place(x=0, y=0)
        self.labels = {}
        self.texts = {}

        # gif
        self.filename = "./gif_control_mpc/gif_control_mpc_for_simulate.gif"
        self.gif = tk.PhotoImage(file=self.filename)
        label_gif = tk.Label(self.root, image=self.gif)
        label_gif.place(x=160, y=0)

        # 文字
        self.label("start_x:")
        self.label("start_y:", y=80)
        self.label("th0[deg]:", y=110)
        self.label("umax_x:", y=140)
        self.label("umax_y:", y=170)
        self.label("time[s]:", y=200)

        # 入力
        self.text("start_x", 2)
        self.text("start_y", 2, y=80)
        self.text("th0", 0, y=110)
        self.text("umax_x", 2, y=140)
        self.text("umax_y", 2, y=170)
        self.text("time", 5, y=200)

        # simulateボタン
        self.button_exec = tk.Button(text="Simulate", bg="white")
        self.button_exec.place(x=10, y=270, width=120, height=20)
        self.button_exec["command"] = self.simulate

        # plotボタン
        self.is_plot = tk.BooleanVar()
        self.chk = tk.Checkbutton(variable=self.is_plot, text='plot', bg="white")
        self.chk.place(x=15, y=230)

    def label(self, name, x=10, y=50, w=80, h=20):
        self.labels[name] = tk.Label(text=name, bg="white")
        self.labels[name].place(x=x, y=y, width=w, height=h)

    def text(self, name, default=0, x=80, y=50, w=40, h=20):
        self.texts[name] = tk.Entry()
        self.texts[name].insert(tk.END, str(default))
        self.texts[name].place(x=x, y=y, width=w, height=h)

    def exec(self):
        # 実行
        def next_frame():
            # gif再生用
            try:
                # XXX: 次のフレームに移る
                self.gif.configure(format="gif -index {}".format(self.gif_index))
                self.gif_index += 1
            except tk.TclError:
                self.gif_index = 0
                return next_frame()
            else:
                self.root.after(self.duration, next_frame)  # XXX: アニメーション速度が固定
        self.root.after_idle(next_frame)
        self.root.mainloop()

    def simulate(self):
        # シミュレーション
        try:
            th0 = np.deg2rad(float(self.texts["th0"].get()))
            start = np.array([float(self.texts["start_x"].get()), float(self.texts["start_y"].get())])
            u_max = np.array([float(self.texts["umax_x"].get()), float(self.texts["umax_y"].get())])
            iter_time = float(self.texts["time"].get())
        except BaseException:
            print("Please input valid value")
            return

        # モデル定義
        model = TwoWheelsModel(start=start,
                               end=self.end,
                               th0=th0,
                               u_list_0=self.u_list_0,
                               u_max=u_max)
        # 実行
        model.exec(int(iter_time / self.dt), self.dt, is_save_gif=True, is_save_log=True, duration=self.duration)
        # プロットにチェックがあればプロット
        if self.is_plot.get():
            Control.plot(model.system, model.controller, dt=self.dt)
