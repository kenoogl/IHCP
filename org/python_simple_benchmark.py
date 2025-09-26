#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python版 簡略化ベンチマークテスト
Julia版との直接比較用
"""

import numpy as np
import time

print("=" * 60)
print("Python版 簡略化ベンチマーク")
print("=" * 60)

# =======================================
# 設定（Julia版と同一）
# =======================================
EVAL_NI, EVAL_NJ = 10, 15
EVAL_NT = 10
nz = 20

# SUS304物理定数
rho = 7823.493962874825  # kg/m³
cp_mean = 500.0  # J/(kg·K) - 平均値
k_mean = 15.0    # W/(m·K) - 平均値

# 格子パラメータ
dx = 0.00012
dy = 0.00016712741767680453
dt = 0.001

print(f"格子: {EVAL_NI} × {EVAL_NJ} × {nz}")
print(f"物理定数: ρ={rho:.1f}, cp={cp_mean:.1f}, k={k_mean:.1f}")

# =======================================
# 簡略化CGM実装
# =======================================
def simple_cgm_python(T_init, Y_obs, q_init, max_iter=5):
    """
    簡略化CGMアルゴリズム（Python版）
    Julia版と同一のロジック
    """
    ni, nj, nk = T_init.shape
    nt = Y_obs.shape[0]

    q = q_init.copy()
    J_hist = []

    print(f"Python CGM開始: {ni}×{nj}×{nk}, {nt}ステップ")

    start_time = time.time()

    for iteration in range(max_iter):
        iter_start = time.time()

        # 1. 順問題（簡略化）
        T_cal = simple_forward_solve(T_init, q, dt)

        # 2. 目的関数計算
        res_T = T_cal[1:, :, :, 0] - Y_obs  # 表面温度差
        J = 0.5 * np.sum(res_T**2)
        J_hist.append(J)

        # 3. 隨伴問題（簡略化）
        lambda_field = simple_adjoint_solve(res_T, dt)

        # 4. 勾配計算
        grad = lambda_field[:, :, :, -1]  # 上面隨伴変数

        # 5. 共役勾配方向
        if iteration == 0:
            p_n = grad.copy()
        else:
            # 簡単なPR更新
            gamma = 0.1  # 固定値
            p_n = grad + gamma * p_n

        # 6. 感度問題（簡略化）
        dT = simple_sensitivity_solve(p_n, dt)

        # 7. ステップサイズ計算
        Sp = dT[1:, :, :, 0]
        numerator = np.sum(res_T * Sp)
        denominator = np.sum(Sp * Sp)

        if denominator < 1e-20:
            beta = 1e-6
        else:
            beta = numerator / (denominator + 1e-15)

        # ステップサイズ制限
        beta = np.clip(beta, -1e8, 1e8)

        # 8. 更新
        q = q - beta * p_n

        iter_time = time.time() - iter_start
        print(f"  Iter {iteration+1}: J={J:.4e}, β={beta:.4e}, time={iter_time:.3f}s")

    total_time = time.time() - start_time
    print(f"Python CGM完了: {total_time:.3f}秒")

    return q, T_cal, J_hist, total_time

def simple_forward_solve(T_init, q_surface, dt):
    """簡略化順問題ソルバー"""
    ni, nj, nk = T_init.shape
    nt = q_surface.shape[0] + 1

    T = np.zeros((nt, ni, nj, nk))
    T[0] = T_init.copy()

    # 簡単な前進オイラー法
    alpha = k_mean / (rho * cp_mean)  # 熱拡散率

    for t in range(1, nt):
        T_prev = T[t-1].copy()

        # 内部点（簡略化拡散）
        T[t, :, :, 1:-1] = T_prev[:, :, 1:-1] + alpha * dt * (
            (T_prev[:, :, 2:] - 2*T_prev[:, :, 1:-1] + T_prev[:, :, :-2]) / (0.000025)**2
        )

        # 境界条件
        T[t, :, :, 0] = T_prev[:, :, 0] + dt * q_surface[t-1] / (rho * cp_mean * 0.000025)  # 表面
        T[t, :, :, -1] = T_prev[:, :, -1]  # 底面（断熱）

    return T

def simple_adjoint_solve(res_T, dt):
    """簡略化隨伴問題ソルバー"""
    nt_minus_1, ni, nj = res_T.shape
    nk = 20

    lambda_field = np.zeros((nt_minus_1, ni, nj, nk))

    # 簡単な後退計算
    for t in range(nt_minus_1-1, -1, -1):
        # 表面での隨伴変数（残差に比例）
        lambda_field[t, :, :, 0] = res_T[t, :, :]

        # 内部への伝播（簡略化）
        for k in range(1, nk):
            lambda_field[t, :, :, k] = lambda_field[t, :, :, 0] * 0.1**(k-1)

    return lambda_field

def simple_sensitivity_solve(p_n, dt):
    """簡略化感度問題ソルバー"""
    nt_minus_1, ni, nj = p_n.shape
    nk = 20

    dT = np.zeros((nt_minus_1+1, ni, nj, nk))

    # 簡単な前進計算
    alpha = k_mean / (rho * cp_mean)

    for t in range(1, nt_minus_1+1):
        # 表面への影響
        dT[t, :, :, 0] = dT[t-1, :, :, 0] + dt * p_n[t-1] / (rho * cp_mean * 0.000025)

        # 内部への拡散
        for k in range(1, nk):
            dT[t, :, :, k] = dT[t-1, :, :, k] + alpha * dt * dT[t-1, :, :, k-1] * 0.1

    return dT

# =======================================
# テストデータ生成
# =======================================
print("\n【テストデータ生成】")

# Julia版と同様の人工データ
np.random.seed(42)  # 再現性のため
T_base = 500.0 + 10 * np.random.rand(EVAL_NT, EVAL_NI, EVAL_NJ)

# 初期温度
T_init = np.zeros((EVAL_NI, EVAL_NJ, nz))
for k in range(nz):
    T_init[:, :, k] = T_base[0, :, :]

# 観測データ
Y_obs = T_base[1:, :, :]

# 初期熱流束
q_init = np.zeros((EVAL_NT-1, EVAL_NI, EVAL_NJ))

print(f"温度範囲: {T_base.min():.2f} - {T_base.max():.2f} K")

# =======================================
# Python版CGM実行
# =======================================
print("\n【Python版CGM実行】")

try:
    q_opt, T_final, J_hist, elapsed_time = simple_cgm_python(
        T_init, Y_obs, q_init, max_iter=5
    )

    print("\n✅ Python版成功")

    # 結果解析
    print(f"\nPython版結果:")
    print(f"  反復数: {len(J_hist)}")
    print(f"  最終目的関数: {J_hist[-1]:.4e}")
    print(f"  熱流束範囲: {np.min(q_opt):.4e} ~ {np.max(q_opt):.4e} W/m²")
    print(f"  熱流束RMS: {np.sqrt(np.mean(q_opt**2)):.4e} W/m²")
    print(f"  計算時間: {elapsed_time:.3f}秒")

    # スループット
    total_dofs = EVAL_NI * EVAL_NJ * nz
    total_ops = total_dofs * (EVAL_NT - 1) * len(J_hist)
    throughput = total_ops / elapsed_time
    print(f"  スループット: {throughput:.0f} 格子点・ステップ・反復/秒")

    # 結果保存
    python_results = {
        'iterations': len(J_hist),
        'objective_final': J_hist[-1],
        'q_min': np.min(q_opt),
        'q_max': np.max(q_opt),
        'q_rms': np.sqrt(np.mean(q_opt**2)),
        'elapsed_time': elapsed_time,
        'throughput': throughput
    }

    np.savez('python_simple_results.npz', **python_results)
    print(f"\n✅ 結果保存: python_simple_results.npz")

except Exception as e:
    print(f"❌ Python版エラー: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Python版ベンチマーク完了")
print("=" * 60)