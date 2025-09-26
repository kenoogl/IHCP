#!/usr/bin/env python3

"""
Python版とJulia版の定量的一致性検証
フルサイズ10ステップでの順問題、随伴問題、感度問題の比較
"""

import numpy as np
import time
from scipy.sparse.linalg import cg
from numba import njit, prange
import pandas as pd

# オリジナルのPython実装を読み込み
exec(open('IHCP_CGM_Sliding_Window_Calculation_ver2.py').read())

print("=" * 80)
print("Python版とJulia版の定量的一致性検証")
print("フルサイズ10ステップ計算での比較テスト")
print("=" * 80)

# 実データ読み込み
print("\n【ステップ1】実データ読み込み")
T_data = np.load("T_measure_700um_1ms.npy")
print(f"データ形状: {T_data.shape}")

# フルサイズ格子設定（実データと同じサイズ）
nx, ny, nz = 80, 100, 20
nt_test = 10  # 10ステップでテスト
dx = 0.12e-3  # 0.12mm

# 格子設定
y_positions = np.array([0.12e-3 * i for i in range(ny)])
z_positions = np.array([0.0001, 0.0002, 0.0003, 0.0005, 0.0008, 0.0013, 0.0021, 0.0034, 0.0055, 0.0089,
                       0.0144, 0.0233, 0.0377, 0.0610, 0.0987, 0.1597, 0.2584, 0.4181, 0.6765, 0.7])

dy_values = np.diff(np.concatenate([[0], y_positions]))
dz_values = np.diff(np.concatenate([[0], z_positions]))

print(f"格子設定: nx={nx}, ny={ny}, nz={nz}")
print(f"テストステップ数: {nt_test}")

# 初期条件設定（室温）
T_initial = np.full((nx, ny, nz), 293.15)
print(f"初期温度: {T_initial[0,0,0]} K")

# 境界温度データ（実データから抽出）
T_boundary = T_data[:nt_test, :, :]  # 10ステップ分
print(f"境界温度範囲: {np.min(T_boundary)} - {np.max(T_boundary)} K")

# 初期熱流束（ゼロ）
q_initial = np.zeros((nx, ny, nt_test))

print("\n【ステップ2】順問題（DHCP）計算")
print("=" * 50)

# Python版順問題計算
print("Python版順問題計算開始...")
python_start_time = time.time()

T_result_python = np.zeros((nx, ny, nz, nt_test + 1))
T_result_python[:, :, :, 0] = T_initial

for time_step in range(nt_test):
    print(f"  時間ステップ {time_step + 1}/{nt_test} 計算中...")

    # 熱物性値計算
    thermal_props = thermal_properties_calculator(T_result_python[:, :, :, time_step])
    rho = thermal_props['rho']
    cp = thermal_props['cp']
    k = thermal_props['k']

    # 係数行列とRHS構築
    A_python, b_python = coeffs_and_rhs_building_DHCP(
        nx, ny, nz, dx, dy_values, dz_values, 0.001,
        rho, cp, k,
        T_result_python[:, :, :, time_step],
        T_boundary[time_step, :, :],
        q_initial[:, :, time_step]
    )

    # 線形システム求解
    T_next_vec, info = cg(A_python, b_python, tol=1e-10, maxiter=10000)
    T_result_python[:, :, :, time_step + 1] = T_next_vec.reshape(nx, ny, nz)

python_dhcp_time = time.time() - python_start_time
print(f"Python版順問題計算時間: {python_dhcp_time:.3f}秒")

# 結果保存（Python版）
np.savez("python_dhcp_results_fullsize.npz",
         T_result=T_result_python,
         computation_time=python_dhcp_time,
         final_temperature_range=[np.min(T_result_python[:,:,:,-1]), np.max(T_result_python[:,:,:,-1])])

print("Python版順問題結果:")
print(f"  最終温度範囲: {np.min(T_result_python[:,:,:,-1]):.2f} - {np.max(T_result_python[:,:,:,-1]):.2f} K")
print(f"  温度上昇: {np.max(T_result_python[:,:,:,-1]) - 293.15:.2f} K")

print("\n【ステップ3】随伴問題計算")
print("=" * 50)

# 測定温度との差（目的関数用）
T_measured = T_boundary  # 境界面の測定温度
T_computed_surface = T_result_python[:, :, 0, 1:]  # 計算された表面温度

# Python版随伴問題計算
print("Python版随伴問題計算開始...")
python_adjoint_start = time.time()

adjoint_result_python = np.zeros((nx, ny, nz, nt_test + 1))

for time_step in range(nt_test - 1, -1, -1):
    print(f"  随伴時間ステップ {nt_test - time_step}/{nt_test} 計算中...")

    # 熱物性値
    thermal_props = thermal_properties_calculator(T_result_python[:, :, :, time_step])
    rho = thermal_props['rho']
    cp = thermal_props['cp']
    k = thermal_props['k']

    # 随伴問題の係数行列とRHS
    A_adj, b_adj = coeffs_and_rhs_building_Adjoint(
        nx, ny, nz, dx, dy_values, dz_values, 0.001,
        rho, cp, k,
        T_result_python[:, :, :, time_step],
        adjoint_result_python[:, :, :, time_step + 1]
    )

    # 境界面での温度差を随伴問題のソースとして追加
    if time_step < nt_test:
        temp_diff = T_computed_surface[:, :, time_step] - T_measured[time_step, :, :]
        # 表面格子点に温度差を追加
        for i in range(nx):
            for j in range(ny):
                idx = i * ny * nz + j * nz + 0  # 表面(z=0)のインデックス
                b_adj[idx] += 2.0 * temp_diff[i, j]

    # 随伴問題求解
    adj_vec, info = cg(A_adj, b_adj, tol=1e-10, maxiter=10000)
    adjoint_result_python[:, :, :, time_step] = adj_vec.reshape(nx, ny, nz)

python_adjoint_time = time.time() - python_adjoint_start
print(f"Python版随伴問題計算時間: {python_adjoint_time:.3f}秒")

# 結果保存（Python版随伴問題）
np.savez("python_adjoint_results_fullsize.npz",
         adjoint_result=adjoint_result_python,
         computation_time=python_adjoint_time,
         adjoint_range=[np.min(adjoint_result_python), np.max(adjoint_result_python)])

print("Python版随伴問題結果:")
print(f"  随伴変数範囲: {np.min(adjoint_result_python):.6f} - {np.max(adjoint_result_python):.6f}")

print("\n【ステップ4】感度計算")
print("=" * 50)

# 感度計算（∂T/∂q）
print("Python版感度計算開始...")
python_sensitivity_start = time.time()

sensitivity_python = np.zeros((nx, ny, nt_test))

for time_step in range(nt_test):
    for i in range(nx):
        for j in range(ny):
            # 表面での熱流束に対する温度の感度
            # 随伴解を用いた感度計算
            sensitivity_python[i, j, time_step] = adjoint_result_python[i, j, 0, time_step] * dz_values[0] / 2.0

python_sensitivity_time = time.time() - python_sensitivity_start
print(f"Python版感度計算時間: {python_sensitivity_time:.3f}秒")

# 結果保存（Python版感度）
np.savez("python_sensitivity_results_fullsize.npz",
         sensitivity=sensitivity_python,
         computation_time=python_sensitivity_time,
         sensitivity_range=[np.min(sensitivity_python), np.max(sensitivity_python)])

print("Python版感度結果:")
print(f"  感度範囲: {np.min(sensitivity_python):.6f} - {np.max(sensitivity_python):.6f}")

print("\n【ステップ5】計算完了サマリー")
print("=" * 50)
print("Python版フルサイズ10ステップ計算完了")
print(f"  順問題計算時間: {python_dhcp_time:.3f}秒")
print(f"  随伴問題計算時間: {python_adjoint_time:.3f}秒")
print(f"  感度計算時間: {python_sensitivity_time:.3f}秒")
print(f"  総計算時間: {python_dhcp_time + python_adjoint_time + python_sensitivity_time:.3f}秒")

print("\n結果ファイル:")
print("  - python_dhcp_results_fullsize.npz")
print("  - python_adjoint_results_fullsize.npz")
print("  - python_sensitivity_results_fullsize.npz")

print("\n次に定量的比較を実行してください。")
print("比較用コマンド: python compare_quantitative_results.py")