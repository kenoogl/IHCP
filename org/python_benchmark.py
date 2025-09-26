#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python版 同一条件ベンチマークテスト
Julia版と同じ条件での定量評価
"""

import numpy as np
import time
import sys
import os

# メインモジュールを読み込み
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 必要な関数をインポート
from IHCP_CGM_Sliding_Window_Calculation_ver2 import (
    thermal_properties_calculator,
    multiple_time_step_solver_DHCP,
    multiple_time_step_solver_Adjoint,
    global_CGM_time
)

print("=" * 60)
print("Python版 定量評価ベンチマーク")
print("=" * 60)

# =======================================
# 設定（Julia版と同一）
# =======================================
EVAL_NI, EVAL_NJ = 10, 15    # 中規模：150格子点
EVAL_NT = 10                  # 10時間ステップ
CGM_ITER = 5                  # CGM反復制限

print("【評価設定】")
print(f"格子: {EVAL_NI} × {EVAL_NJ} × 20 = {EVAL_NI * EVAL_NJ * 20} 格子点")
print(f"時間: {EVAL_NT} ステップ")
print(f"CGM: {CGM_ITER} 反復")

# =======================================
# 物理パラメータ（Julia版と同一）
# =======================================
# SUS304の物理定数
rho = 7823.493962874825  # kg/m³
cp_coeffs = np.array([439.415, 0.191, 0.0, -1.05e-4])  # 比熱多項式係数
k_coeffs = np.array([12.47, 1.28e-2, 0.0, -4.26e-6])   # 熱伝導率多項式係数

# 格子パラメータ
dx = 0.00012  # m
dy = 0.00016712741767680453  # m
Lz = 0.0005  # m
nz = 20

# z方向格子
z_interfaces = np.linspace(0, Lz, nz+1)
dz = np.diff(z_interfaces)
dz_b = dz.copy()
dz_t = dz.copy()

print(f"格子パラメータ: dx={dx*1e3:.3f}mm, dy={dy*1e3:.3f}mm, nz={nz}")

# =======================================
# 実データ読み込み
# =======================================
print("\n【実データ読み込み】")

try:
    # Julia版と同じデータファイル
    T_data = np.load("../julia_version/T_measure_700um_1ms.npy")
    T_test = T_data[:EVAL_NT, :EVAL_NI, :EVAL_NJ]

    print(f"✅ データ読み込み成功: {T_test.shape}")
    print(f"温度範囲: {T_test.min():.2f} - {T_test.max():.2f} K")

    # 初期化
    T0 = np.zeros((EVAL_NI, EVAL_NJ, nz))
    for k in range(nz):
        T0[:, :, k] = T_test[0, :, :]

    q_init = np.zeros((EVAL_NT-1, EVAL_NI, EVAL_NJ))

except Exception as e:
    print(f"❌ データ読み込みエラー: {e}")
    sys.exit(1)

# =======================================
# 熱物性値計算
# =======================================
print("\n【熱物性値計算】")

try:
    # 温度場作成（Julia版と同じ）
    T_calc = np.zeros((EVAL_NT, EVAL_NI, EVAL_NJ, nz))
    for t in range(EVAL_NT):
        for k in range(nz):
            T_calc[t, :, :, k] = T_test[t, :, :] if t < T_test.shape[0] else T_test[-1, :, :]

    # 熱物性値計算
    cp, k = thermal_properties_calculator(T_calc, cp_coeffs, k_coeffs)

    print("✅ 熱物性値計算成功")

except Exception as e:
    print(f"❌ 熱物性値計算エラー: {e}")
    sys.exit(1)

# =======================================
# Python版CGM実行
# =======================================
print("\n【Python版CGM実行】")

dt = 0.001  # 1ms

start_time = time.time()

try:
    # CGM実行（Julia版と同一設定）
    q_opt_python, T_final_python, J_hist_python = global_CGM_time(
        T0, T_test, q_init,
        dx, dy, dz, dz_b, dz_t, dt,
        rho, cp_coeffs, k_coeffs,
        CGM_iteration=CGM_ITER
    )

    elapsed_python = time.time() - start_time

    print("✅ Python版CGM成功")

    # =======================================
    # 結果解析
    # =======================================
    print("\n【Python版結果解析】")

    # 1. 収束性評価
    n_iter = len(J_hist_python)
    print(f"反復数: {n_iter}")
    print(f"初期目的関数: {J_hist_python[0]:.4e}")
    print(f"最終目的関数: {J_hist_python[-1]:.4e}")

    if n_iter > 1:
        rel_improve = (J_hist_python[0] - J_hist_python[-1]) / J_hist_python[0]
        print(f"相対改善率: {rel_improve:.4f}")

    # 2. 熱流束解析
    print(f"\n熱流束解析:")
    print(f"最小値: {np.min(q_opt_python):.4e} W/m²")
    print(f"最大値: {np.max(q_opt_python):.4e} W/m²")
    print(f"平均値: {np.mean(q_opt_python):.4e} W/m²")
    print(f"標準偏差: {np.std(q_opt_python):.4e} W/m²")
    print(f"RMS値: {np.sqrt(np.mean(q_opt_python**2)):.4e} W/m²")

    # 3. 温度予測精度
    temp_prediction = T_final_python[:, :, 0]  # 表面温度予測（bottom_idx=0）
    temp_observed = T_test[-1, :, :]  # 実測表面温度
    temp_error = temp_prediction - temp_observed

    temp_rmse = np.sqrt(np.mean(temp_error**2))
    temp_max_error = np.max(np.abs(temp_error))

    print(f"\n温度予測精度:")
    print(f"RMSE: {temp_rmse:.4e} K")
    print(f"最大誤差: {temp_max_error:.4e} K")

    # 4. 計算効率
    total_dofs = EVAL_NI * EVAL_NJ * nz
    total_operations = total_dofs * (EVAL_NT - 1) * n_iter
    throughput = total_operations / elapsed_python

    print(f"\n計算効率:")
    print(f"総計算時間: {elapsed_python:.2f} 秒")
    print(f"スループット: {throughput:.0f} 格子点・ステップ・反復/秒")

    # フルスケール推定
    full_scale_factor = (80 * 100 * 20) / total_dofs
    full_time_windows = 100
    full_cgm_iters = 20

    estimated_hours = (elapsed_python * full_scale_factor * full_time_windows * full_cgm_iters / n_iter) / 3600
    print(f"フルスケール推定: {estimated_hours:.1f} 時間")

    # =======================================
    # 結果保存（Julia版との比較用）
    # =======================================
    results_python = {
        'iterations': n_iter,
        'objective_initial': J_hist_python[0],
        'objective_final': J_hist_python[-1],
        'q_min': np.min(q_opt_python),
        'q_max': np.max(q_opt_python),
        'q_mean': np.mean(q_opt_python),
        'q_std': np.std(q_opt_python),
        'q_rms': np.sqrt(np.mean(q_opt_python**2)),
        'temp_rmse': temp_rmse,
        'temp_max_error': temp_max_error,
        'elapsed_time': elapsed_python,
        'throughput': throughput,
        'estimated_fullscale_hours': estimated_hours
    }

    # NumPy形式で保存
    np.savez('python_benchmark_results.npz',
             q_opt=q_opt_python,
             T_final=T_final_python,
             J_hist=J_hist_python,
             **results_python)

    print(f"\n✅ 結果を python_benchmark_results.npz に保存")

    print("\n" + "=" * 60)
    print("Python版評価完了")
    print("=" * 60)

except Exception as e:
    print(f"❌ Python版CGMエラー: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)