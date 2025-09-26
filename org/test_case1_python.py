#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
テストケース1: 小規模精密比較 (Python版)
5×5×20格子、5ステップ、Julia版と同一条件
"""

import numpy as np
import time

print("=" * 60)
print("テストケース1: Python版小規模精密比較")
print("=" * 60)

# Julia版と同じ乱数シード
np.random.seed(42)

# =======================================
# テスト設定（Julia版と同一）
# =======================================
TEST_NI, TEST_NJ = 5, 5
TEST_NT = 5
nz = 20
CGM_ITER = 5

print("【テスト設定】")
print(f"格子: {TEST_NI} × {TEST_NJ} × {nz} = {TEST_NI * TEST_NJ * nz} 格子点")
print(f"時間: {TEST_NT} ステップ")
print(f"CGM反復: {CGM_ITER} 回")

# =======================================
# 物理パラメータ（Julia版と同一）
# =======================================
rho = 7823.493962874825
cp_mean = 500.0
k_mean = 15.0
dx = 0.00012
dy = 0.00016712741767680453
dt = 0.001

# =======================================
# テストデータ（Julia版と同一）
# =======================================
print("\n【テストデータ生成】")

# Julia版と同じ温度分布
T_base = 500.0 + 5.0 * np.random.rand(TEST_NT, TEST_NI, TEST_NJ)

# 初期温度
T_init = np.zeros((TEST_NI, TEST_NJ, nz))
for k in range(nz):
    T_init[:, :, k] = T_base[0, :, :]

# 観測データ
Y_obs = T_base[1:, :, :]

# 初期熱流束
q_init = np.zeros((TEST_NT-1, TEST_NI, TEST_NJ))

print(f"温度範囲: {T_base.min():.3f} - {T_base.max():.3f} K")
print(f"初期温度平均: {T_init.mean():.3f} K")

# =======================================
# 簡略化CGM実装（Python版）
# =======================================
def python_cgm_solver(T_init, Y_obs, q_init, max_iter=5):
    """
    Python版簡略化CGMソルバー
    Julia版の主要ロジックを模倣
    """
    ni, nj, nk = T_init.shape
    nt = Y_obs.shape[0] + 1

    q = q_init.copy()
    J_hist = []

    print(f"Python CGM開始: {ni}×{nj}×{nk}, {nt}ステップ")

    # 計算ログ
    iteration_times = []
    beta_values = []

    for iteration in range(max_iter):
        iter_start = time.time()

        # 1. 順問題（簡略化）
        T_cal = forward_solve_python(T_init, q)

        # 2. 目的関数
        res_T = T_cal[1:, :, :, 0] - Y_obs
        J = 0.5 * np.sum(res_T**2)
        J_hist.append(J)

        # 3. 勾配計算（簡略化隨伴）
        grad = adjoint_solve_python(res_T)

        # 4. 共役勾配方向
        if iteration == 0:
            p_n = grad.copy()
        else:
            gamma = 0.01  # 簡略化係数
            p_n = grad + gamma * p_n

        # 5. 感度計算
        dT = sensitivity_solve_python(p_n)

        # 6. ステップサイズ計算
        Sp = dT[1:, :, :, 0]
        numerator = np.sum(res_T * Sp)
        denominator = np.sum(Sp * Sp)

        if denominator < 1e-20:
            print(f"  [警告] 分母が極小 {denominator:.2e} < 1.00e-20 at iter {iteration+1}")
            beta = 1e-6
        else:
            beta = numerator / (denominator + 1e-15)

        # βクリップ（Julia版と同様）
        if iteration == 0 and abs(beta) > 1e6:
            print(f"  [警告] βクリップ: {beta:.2e} => {np.sign(beta)*1e6:.2e}")
            beta = np.clip(beta, -1e6, 1e6)

        beta_values.append(beta)

        # 7. 更新
        q = q - beta * p_n

        iter_time = time.time() - iter_start
        iteration_times.append(iter_time)

        print(f"@ ___ Iter {iteration+1:3d} ___ @ wall_s = {iter_time:.3f}s")
        print(f"J = {J:.5e}, β = {beta:.4e}")

    return q, T_cal, J_hist, iteration_times, beta_values

def forward_solve_python(T_init, q_surface):
    """Python版簡略化順問題"""
    ni, nj, nk = T_init.shape
    nt = q_surface.shape[0] + 1

    T = np.zeros((nt, ni, nj, nk))
    T[0] = T_init.copy()

    alpha = k_mean / (rho * cp_mean)

    for t in range(1, nt):
        T_prev = T[t-1].copy()

        # 内部拡散（簡略化）
        T[t, :, :, 1:-1] = T_prev[:, :, 1:-1] + alpha * dt * (
            (T_prev[:, :, 2:] - 2*T_prev[:, :, 1:-1] + T_prev[:, :, :-2]) / (2.5e-5)**2
        )

        # 境界条件
        T[t, :, :, 0] = T_prev[:, :, 0] + dt * q_surface[t-1] / (rho * cp_mean * 2.5e-5)
        T[t, :, :, -1] = T_prev[:, :, -1]

    return T

def adjoint_solve_python(res_T):
    """Python版簡略化隨伴問題"""
    nt_minus_1, ni, nj = res_T.shape
    grad = np.zeros((nt_minus_1, ni, nj))

    # 表面隨伴変数（残差に基づく勾配）
    for t in range(nt_minus_1):
        grad[t, :, :] = res_T[t, :, :] * 1e-6  # スケーリング

    return grad

def sensitivity_solve_python(p_n):
    """Python版簡略化感度問題"""
    nt_minus_1, ni, nj = p_n.shape
    nk = 20

    dT = np.zeros((nt_minus_1+1, ni, nj, nk))
    alpha = k_mean / (rho * cp_mean)

    for t in range(1, nt_minus_1+1):
        # 表面への摂動
        dT[t, :, :, 0] = dT[t-1, :, :, 0] + dt * p_n[t-1] / (rho * cp_mean * 2.5e-5)

        # 内部拡散
        for k in range(1, nk):
            dT[t, :, :, k] = dT[t-1, :, :, k] + alpha * dt * dT[t-1, :, :, k-1] * 0.01

    return dT

# =======================================
# Python版CGM実行
# =======================================
print("\n【Python版CGM実行】")

start_time = time.time()

try:
    q_opt_python, T_final_python, J_hist_python, iter_times, beta_vals = python_cgm_solver(
        T_init, Y_obs, q_init, max_iter=CGM_ITER
    )

    elapsed_python = time.time() - start_time

    print("✅ Python版CGM成功")

    # =======================================
    # 詳細結果解析（Julia版と同じ）
    # =======================================
    print("\n【Python版詳細結果解析】")

    # 1. 収束解析
    n_iter = len(J_hist_python)
    print("1. 収束解析:")
    print(f"  実行反復数: {n_iter}")
    print(f"  初期目的関数: {J_hist_python[0]:.6e}")
    print(f"  最終目的関数: {J_hist_python[-1]:.6e}")

    if n_iter > 1:
        rel_improve = (J_hist_python[0] - J_hist_python[-1]) / J_hist_python[0]
        print(f"  相対改善率: {rel_improve:.6f}")

    # 2. 熱流束詳細解析
    print("\n2. 熱流束詳細解析:")
    print(f"  最小値: {np.min(q_opt_python):.6e} W/m²")
    print(f"  最大値: {np.max(q_opt_python):.6e} W/m²")
    print(f"  平均値: {np.mean(q_opt_python):.6e} W/m²")
    print(f"  標準偏差: {np.std(q_opt_python):.6e} W/m²")
    print(f"  RMS値: {np.sqrt(np.mean(q_opt_python**2)):.6e} W/m²")

    # 変動解析
    q_spatial_var = np.std([np.std(q_opt_python[t, :, :]) for t in range(TEST_NT-1)])
    q_temporal_var = np.std([np.std(q_opt_python[:, i, j]) for i in range(TEST_NI) for j in range(TEST_NJ)])

    print(f"  空間変動: {q_spatial_var:.6e} W/m²")
    print(f"  時間変動: {q_temporal_var:.6e} W/m²")

    # 3. 温度予測精度
    print("\n3. 温度予測精度:")

    print(f"  T_final_python元形状: {T_final_python.shape}")
    T_true = T_base[-1, :, :]         # 実際の表面温度 (5×5)
    print(f"  T_true形状: {T_true.shape}")

    # T_final_pythonから適切に表面温度を抽出
    if len(T_final_python.shape) == 4:
        # (nt, ni, nj, nk)の場合、最終時刻の表面（z=0）を取得
        T_pred = T_final_python[-1, :, :, 0]  # 最終時刻、表面層
    elif len(T_final_python.shape) == 3:
        # (ni, nj, nk)の場合、表面層を取得
        T_pred = T_final_python[:, :, 0]
    else:
        # その他の場合
        T_pred = T_final_python

    print(f"  抽出後 T_pred形状: {T_pred.shape}")

    # 形状を最終調整
    if T_pred.shape != T_true.shape:
        if len(T_pred.shape) > 2:
            T_pred = T_pred.squeeze()  # 余分な次元を削除

        # まだ形状が合わない場合
        if T_pred.shape != T_true.shape:
            min_dims = min(len(T_pred.shape), len(T_true.shape))
            if min_dims >= 2:
                min_i = min(T_pred.shape[0], T_true.shape[0])
                min_j = min(T_pred.shape[1], T_true.shape[1])
                T_pred = T_pred[:min_i, :min_j] if len(T_pred.shape) >= 2 else T_pred
                T_true = T_true[:min_i, :min_j] if len(T_true.shape) >= 2 else T_true

    print(f"  最終 T_pred形状: {T_pred.shape}")
    print(f"  最終 T_true形状: {T_true.shape}")

    if T_pred.shape == T_true.shape:
        temp_error = T_pred - T_true
        temp_rmse = np.sqrt(np.mean(temp_error**2))
        temp_mae = np.mean(np.abs(temp_error))
        temp_max_error = np.max(np.abs(temp_error))
    else:
        print("  ⚠️ 形状が一致しないため温度予測精度の計算をスキップ")
        temp_rmse = temp_mae = temp_max_error = 0.0

    print(f"  RMSE: {temp_rmse:.6e} K")
    print(f"  MAE: {temp_mae:.6e} K")
    print(f"  最大誤差: {temp_max_error:.6e} K")

    # 4. 計算効率
    print("\n4. 計算効率:")
    total_dofs = TEST_NI * TEST_NJ * nz
    total_operations = total_dofs * (TEST_NT - 1) * n_iter
    throughput = total_operations / elapsed_python

    print(f"  実行時間: {elapsed_python:.6f} 秒")
    print(f"  スループット: {throughput:.0f} 格子点・ステップ・反復/秒")
    print(f"  反復あたり時間: {elapsed_python/n_iter:.6f} 秒")

    # 5. 数値安定性
    print("\n5. 数値安定性:")

    has_nan_q = np.any(np.isnan(q_opt_python)) or np.any(np.isinf(q_opt_python))
    has_nan_T = np.any(np.isnan(T_final_python)) or np.any(np.isinf(T_final_python))

    print(f"  NaN/Inf発生 (熱流束): {'❌' if has_nan_q else '✅'}")
    print(f"  NaN/Inf発生 (温度): {'❌' if has_nan_T else '✅'}")

    significant_q = q_opt_python[np.abs(q_opt_python) > 1e-15]
    if len(significant_q) > 0:
        q_order = np.log10(np.max(np.abs(significant_q)))
        print(f"  有意熱流束桁数: 10^{q_order:.1f}")
    else:
        print("  有意熱流束: 検出されず（極小値のみ）")

    # =======================================
    # 結果保存
    # =======================================
    python_results = {
        'test_case': 1,
        'grid_size': np.array([TEST_NI, TEST_NJ, nz]),
        'time_steps': TEST_NT,
        'iterations': n_iter,
        'objective_initial': J_hist_python[0],
        'objective_final': J_hist_python[-1],
        'relative_improvement': (J_hist_python[0] - J_hist_python[-1]) / J_hist_python[0] if n_iter > 1 else 0.0,
        'q_min': np.min(q_opt_python),
        'q_max': np.max(q_opt_python),
        'q_mean': np.mean(q_opt_python),
        'q_std': np.std(q_opt_python),
        'q_rms': np.sqrt(np.mean(q_opt_python**2)),
        'temp_rmse': temp_rmse,
        'temp_mae': temp_mae,
        'temp_max_error': temp_max_error,
        'elapsed_time': elapsed_python,
        'throughput': throughput,
        'has_numerical_issues': has_nan_q or has_nan_T
    }

    np.savez('test_case1_python_results.npz', **python_results)

    print("\n✅ 結果保存: test_case1_python_results.npz")

except Exception as e:
    print(f"❌ Python版エラー: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("テストケース1 Python版完了")
print("=" * 60)