from gui import GUI
import numpy as np


if __name__ == "__main__":
    # パラメタ定義
    dt = 0.01
    fps = 6.
    duration = int(1000. / fps)
    end = np.array([5., 5.])
    # u_list_0 = np.array([1, 0.1, 0.1, 2.5, 0.8, 0.8]) # 拘束条件をモデル内に含める場合
    u_list_0 = np.array([1, 0.1])

    # パラメタ設定
    gui = GUI(dt=dt, duration=duration, end=end, u_list_0=u_list_0)
    # 実行
    gui.exec()
