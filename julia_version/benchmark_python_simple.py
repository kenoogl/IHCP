#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python版直接問題ソルバー（DHCP）ベンチマーク（簡易版）
Julia版との性能比較用
"""

import numpy as np
import time
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg
from pathlib import Path

print("=" * 80)
print("Python版 vs Julia版 性能比較: 直接問題ソルバー（DHCP）")
print("=" * 80)

# =======================================
# 必要な関数定義（orgから抽出）
# =======================================

def thermal_properties_calculator(Temperature, cp_coeffs, k_coeffs):
    """熱物性値計算"""
    ni, nj, nk = Temperature.shape
    cp = np.zeros_like(Temperature)
    k_array = np.zeros_like(Temperature)

    for i in range(ni):
        for j in range(nj):
            for k_idx in range(nk):
                T_current = Temperature[i, j, k_idx]
                cp[i, j, k_idx] = cp_coeffs[0] + cp_coeffs[1]*T_current + cp_coeffs[2]*T_current**2
                k_array[i, j, k_idx] = k_coeffs[0] + k_coeffs[1]*T_current + k_coeffs[2]*T_current**2

    return cp, k_array

def coeffs_and_rhs_building_DHCP(T, q_surface, t_step, rho, cp, k_array, dx, dy, dz, dz_b, dz_t, dt):
    """DHCP係数行列構築"""
    ni, nj, nk = T.shape

    # Sparse行列用のデータ準備
    row_indices = []
    col_indices = []
    data = []
    rhs = np.zeros(ni * nj * nk)

    def get_index(i, j, k):
        return i * nj * nk + j * nk + k

    for i in range(ni):
        for j in range(nj):
            for k in range(nk):
                idx = get_index(i, j, k)

                # 対角成分
                diagonal_coeff = rho * cp[i, j, k] / dt

                # 拡散項係数
                if k == 0:  # 表面境界
                    # 熱流束境界条件
                    diagonal_coeff += k_array[i, j, k] / (dz[k] * dz_t[k])
                    rhs[idx] = rho * cp[i, j, k] * T[i, j, k] / dt + q_surface[t_step-1, i, j] / dz[k]

                    # z方向の隣接点
                    if k + 1 < nk:
                        neighbor_idx = get_index(i, j, k + 1)
                        coeff = -k_array[i, j, k] / (dz[k] * dz_t[k])
                        row_indices.append(idx)
                        col_indices.append(neighbor_idx)
                        data.append(coeff)
                        diagonal_coeff -= coeff

                elif k == nk - 1:  # 底面境界（断熱）
                    rhs[idx] = rho * cp[i, j, k] * T[i, j, k] / dt

                else:  # 内部点
                    rhs[idx] = rho * cp[i, j, k] * T[i, j, k] / dt

                    # z方向の拡散
                    if k > 0:
                        neighbor_idx = get_index(i, j, k - 1)
                        coeff = -k_array[i, j, k] / (dz[k] * dz_b[k])
                        row_indices.append(idx)
                        col_indices.append(neighbor_idx)
                        data.append(coeff)
                        diagonal_coeff -= coeff

                    if k < nk - 1:
                        neighbor_idx = get_index(i, j, k + 1)
                        coeff = -k_array[i, j, k] / (dz[k] * dz_t[k])
                        row_indices.append(idx)
                        col_indices.append(neighbor_idx)
                        data.append(coeff)
                        diagonal_coeff -= coeff

                # x, y方向の拡散（簡略化版）
                if i > 0:
                    neighbor_idx = get_index(i - 1, j, k)
                    coeff = -k_array[i, j, k] / dx**2
                    row_indices.append(idx)
                    col_indices.append(neighbor_idx)
                    data.append(coeff)
                    diagonal_coeff -= coeff

                if i < ni - 1:
                    neighbor_idx = get_index(i + 1, j, k)
                    coeff = -k_array[i, j, k] / dx**2
                    row_indices.append(idx)
                    col_indices.append(neighbor_idx)
                    data.append(coeff)
                    diagonal_coeff -= coeff

                if j > 0:
                    neighbor_idx = get_index(i, j - 1, k)
                    coeff = -k_array[i, j, k] / dy**2
                    row_indices.append(idx)
                    col_indices.append(neighbor_idx)
                    data.append(coeff)
                    diagonal_coeff -= coeff

                if j < nj - 1:
                    neighbor_idx = get_index(i, j + 1, k)
                    coeff = -k_array[i, j, k] / dy**2
                    row_indices.append(idx)
                    col_indices.append(neighbor_idx)
                    data.append(coeff)
                    diagonal_coeff -= coeff

                # 対角成分を追加
                row_indices.append(idx)
                col_indices.append(idx)
                data.append(diagonal_coeff)

    # Sparse行列組み立て
    A = csr_matrix((data, (row_indices, col_indices)), shape=(ni*nj*nk, ni*nj*nk))

    return A, rhs

def multiple_time_step_solver_DHCP(T_initial, q_surface, nt, rho, cp_coeffs, k_coeffs,
                                  dx, dy, dz, dz_b, dz_t, dt, rtol=1e-6, maxiter=1000):
    """複数時間ステップDHCPソルバー"""
    ni, nj, nk = T_initial.shape
    T_all = np.zeros((nt, ni, nj, nk))
    T_all[0] = T_initial.copy()

    for t in range(1, nt):
        # 現在の温度での熱物性値計算
        cp, k_array = thermal_properties_calculator(T_all[t-1], cp_coeffs, k_coeffs)

        # 係数行列とRHS構築
        A, rhs = coeffs_and_rhs_building_DHCP(
            T_all[t-1], q_surface, t, rho, cp, k_array, dx, dy, dz, dz_b, dz_t, dt
        )

        # 線形システム求解
        T_flat, info = cg(A, rhs, tol=rtol, maxiter=maxiter)

        if info != 0:
            print(f"Warning: CG failed to converge at time step {t}, info = {info}")

        # 3D配列に変換
        T_all[t] = T_flat.reshape((ni, nj, nk))

    return T_all

# =======================================
# Julia版との条件統一
# =======================================
print("\n【データ準備】Julia版との条件統一")

# Julia版で生成したベンチマークデータの読み込み
benchmark_path = Path(__file__).parent / 'benchmark_data.npz'
try:
    benchmark_data = np.load(benchmark_path)
    T0 = benchmark_data['T0']
    q_surface = benchmark_data['q_surface']
    Y_obs = benchmark_data['Y_obs']
    nt_benchmark = int(benchmark_data['nt'])
    dt = float(benchmark_data['dt'])
    julia_avg_time = float(benchmark_data['julia_avg_time'])
    julia_std_time = float(benchmark_data['julia_std_time'])

    print("Julia版ベンチマークデータ読み込み成功")
    print(f"格子サイズ: {T0.shape}")
    print(f"時間ステップ数: {nt_benchmark}")
    print(f"Julia版実行時間: {julia_avg_time:.3f} ± {julia_std_time:.3f} 秒")

except FileNotFoundError:
    print("benchmark_data.npzが見つかりません。")
    print("先にjulia_version/benchmark_comparison.jlを実行してください。")
    exit(1)

# =======================================
# Python版パラメータ設定
# =======================================
print("\n【パラメータ設定】Python版設定値")

ni, nj, nk = T0.shape

# 物理パラメータ（Julia版と同じ値）
rho = 8000.0  # kg/m³
cp_coeffs = np.array([412.93648, 0.21754, -0.000114286])  # J/(kg·K)
k_coeffs = np.array([12.09302, 0.012552, -0.0000067143])  # W/(m·K)

# 格子設定（Julia版と同じ）
dx = 1.2e-4  # m (0.12mm)
dy = 1.2e-4  # m (0.12mm)

# z方向格子（Julia版と同じ設定）
nz = 20
Lz = 20.0e-3  # 20mm
z_faces = np.linspace(0, Lz, nz + 1)
dz = np.diff(z_faces)

# 境界格子間隔
dz_b = np.concatenate([[float('inf')], dz[:-1]])  # 上境界は無限
dz_t = np.concatenate([dz[1:], [float('inf')]])  # 下境界は無限

print(f"密度: ρ={rho:.1f} kg/m³")
print(f"格子間隔: dx={dx:.2e} m, dy={dy:.2e} m")
print(f"z方向格子数: {nz}, 範囲: {z_faces[-1]:.2e} - {z_faces[0]:.2e} m")

# =======================================
# Python版ベンチマーク実行
# =======================================
print("\n【Python版実行】直接問題ソルバー測定")

print("Python版実行中...")
python_times = []

for run in range(3):  # 3回実行して平均取得
    import gc
    gc.collect()

    start_time = time.time()

    # Python版ソルバー実行
    T_all_python = multiple_time_step_solver_DHCP(
        T0, q_surface, nt_benchmark, rho, cp_coeffs, k_coeffs,
        dx, dy, dz, dz_b, dz_t, dt,
        rtol=1e-6, maxiter=20000  # Julia版と同じ設定
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    python_times.append(elapsed_time)

    print(f"Run {run+1}: {elapsed_time:.3f}秒")

# 統計計算
python_avg_time = np.mean(python_times)
python_std_time = np.std(python_times, ddof=1)

print(f"Python版結果:")
print(f"  平均実行時間: {python_avg_time:.3f} ± {python_std_time:.3f} 秒")
print(f"  温度変化: {np.min(T_all_python[0, :, :, :]):.2f} K → {np.max(T_all_python[-1, :, :, :]):.2f} K")

# 計算統計
total_grid_points = ni * nj * nk
total_operations = total_grid_points * (nt_benchmark - 1)
python_rate = total_operations / python_avg_time

print(f"  総格子点数: {total_grid_points}")
print(f"  総計算量: {total_operations} 格子点×時間ステップ")
print(f"  計算レート: {python_rate:.0f} 格子点×ステップ/秒")

# =======================================
# Python vs Julia 性能比較
# =======================================
print("\n【性能比較】Python vs Julia")

speedup_ratio = python_avg_time / julia_avg_time
julia_rate = total_operations / julia_avg_time

print(f"実行時間比較:")
print(f"  Python版: {python_avg_time:.3f} ± {python_std_time:.3f} 秒")
print(f"  Julia版:  {julia_avg_time:.3f} ± {julia_std_time:.3f} 秒")
print(f"  スピードアップ比: {speedup_ratio:.2f}x ({'Julia' if speedup_ratio > 1 else 'Python'}が高速)")

print(f"計算レート比較:")
print(f"  Python版: {python_rate:.0f} 格子点×ステップ/秒")
print(f"  Julia版:  {julia_rate:.0f} 格子点×ステップ/秒")
print(f"  効率向上: {(julia_rate/python_rate):.2f}x")

# =======================================
# 結果サマリー
# =======================================
print("\n" + "=" * 80)
print("Python版ベンチマーク完了")
print("=" * 80)

print("📊 Python版性能サマリー:")
print(f"⏱️  実行時間: {python_avg_time:.3f} ± {python_std_time:.3f} 秒")
print(f"🖥️  計算レート: {python_rate:.0f} 格子点×ステップ/秒")
print(f"🧮 問題規模: {total_grid_points}格子点 × {nt_benchmark-1}時間ステップ")

print("\n🆚 Python vs Julia 比較結果:")
if speedup_ratio > 1.1:
    print(f"🏆 Julia版が{speedup_ratio:.1f}倍高速")
elif speedup_ratio < 0.9:
    print(f"🏆 Python版が{1/speedup_ratio:.1f}倍高速")
else:
    print("🤝 両版がほぼ同等の性能")

print(f"📈 計算効率: Julia版は{julia_rate/python_rate:.1f}倍の処理レート")

# =======================================
# 結果保存
# =======================================
print("\n【結果保存】")

# 比較結果を保存
comparison_data = {
    'python_avg_time': python_avg_time,
    'python_std_time': python_std_time,
    'python_rate': python_rate,
    'julia_avg_time': julia_avg_time,
    'julia_std_time': julia_std_time,
    'julia_rate': julia_rate,
    'speedup_ratio': speedup_ratio,
    'efficiency_ratio': julia_rate / python_rate,
    'T_final_python': T_all_python[-1],
    'problem_size': total_grid_points,
    'time_steps': nt_benchmark - 1
}

comparison_path = Path(__file__).parent / 'python_julia_comparison.npz'
np.savez(comparison_path, **comparison_data)
print("python_julia_comparison.npz に比較結果を保存しました")

print("\n✅ Python vs Julia ベンチマーク比較完了")