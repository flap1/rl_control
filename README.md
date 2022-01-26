# MPCの比較

## 使い方

目標位置は(x, y)=(5, 5)となっている.

### 制御ベースのMPC

```bash
python control_mpc.py
```

![alt](/img_readme/readme1.png)

`start_x, start_y, th0, umax_x, umax_y, time`に任意の値を入力して`simulate`ボタンを押す.(収束しない場合もある)
`plot`にチェックを入れると収束までの状態(`x, y, th`)推移や`F`(ハミルトン関数の偏微分ベクトルの絶対値)の推移が出力される.
入力値は`u_max`でクリッピングされている.
デフォルトでは入力に対する不等式拘束条件を含めないMPCになっている.
※ 含める場合はコード内のコメント通りに

### 強化学習ベースのMPC

```bash
python rl_mpc.py
```

RSのtrain-->test, PPOのtrain-->testの順に実行される.

## ファイル構成

* `gif_*/`: gif保存ディレクトリ
* `model/`: rlのmodel保存ディレクトリ
* `log/`: ログデータの保存ディレクトリ
* `img_*/`: 画像保存ディレクトリ(一時的)
* `Control.py`: 制御ベースのMPCアルゴリズム
* `RL.py`: 強化学習ベースのMPCアルゴリズム
* `two_wheels_model.py`: `TwoWheelsModel`クラス
* `two_wheels_env.py`: `TwoWheelsEnv`クラスと`TwoWheelsSimulator`クラス
* `gui.py`: `GUI`クラス
* `control_mpc.py`: 制御ベースのMPC実行
* `rl_mpc.py`: 強化学習ベースのMPC実行

## 主要なクラス

* `TwoWheelsModel`: 対向二輪ロボットのモデル
* `TwoWheelsEnv`: 対向二輪ロボットのGymモデル
* `TwoWheelsSimulator`: RLの実行環境
* `GUI`: 制御ベースのMPCのUI
