#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Numba最適化Python版直接問題ソルバー（DHCP）ベンチマーク
Julia版との性能比較用（公平な比較のためNumba並列化適用）
"""

import numpy as np
import time
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg
from pathlib import Path
from numba import njit, prange

print("=" * 80)
print("Python版（Numba最適化） vs Julia版 性能比較: 直接問題ソルバー（DHCP）")
print("=" * 80)

# =======================================
# Numba最適化関数定義
# =======================================

@njit(parallel=True)
def thermal_properties_calculator_numba(Temperature, cp_coeffs, k_coeffs):
    """熱物性値計算（Numba並列最適化版）"""
    ni, nj, nk = Temperature.shape
    cp = np.zeros_like(Temperature)
    k_array = np.zeros_like(Temperature)

    for i in prange(ni):
        for j in range(nj):
            for k_idx in range(nk):
                T_current = Temperature[i, j, k_idx]
                cp[i, j, k_idx] = cp_coeffs[0] + cp_coeffs[1]*T_current + cp_coeffs[2]*T_current**2
                k_array[i, j, k_idx] = k_coeffs[0] + k_coeffs[1]*T_current + k_coeffs[2]*T_current**2

    return cp, k_array

@njit
def get_index_numba(i, j, k, nj, nk):
    """1D配列インデックス計算（Numba最適化）"""
    return i * nj * nk + j * nk + k

@njit
def coeffs_and_rhs_building_DHCP_numba(T, q_surface_2d, t_step, rho, cp, k_array,
                                       dx, dy, dz, dz_b, dz_t, dt):
    """DHCP係数行列構築（Numba対応版）"""
    ni, nj, nk = T.shape

    # 最大非ゼロ要素数を推定（7-point stencil）
    max_nnz = ni * nj * nk * 7

    # 配列を事前確保
    row_indices = np.empty(max_nnz, dtype=np.int32)
    col_indices = np.empty(max_nnz, dtype=np.int32)
    data = np.empty(max_nnz, dtype=np.float64)
    rhs = np.zeros(ni * nj * nk, dtype=np.float64)

    nnz_count = 0

    for i in range(ni):
        for j in range(nj):
            for k in range(nk):
                idx = get_index_numba(i, j, k, nj, nk)

                # 対角成分
                diagonal_coeff = rho * cp[i, j, k] / dt

                # 拡散項係数
                if k == 0:  # 表面境界
                    # 熱流束境界条件
                    diagonal_coeff += k_array[i, j, k] / (dz[k] * dz_t[k])
                    rhs[idx] = (rho * cp[i, j, k] * T[i, j, k] / dt +
                               q_surface_2d[i, j] / dz[k])

                    # z方向の隣接点
                    if k + 1 < nk:
                        neighbor_idx = get_index_numba(i, j, k + 1, nj, nk)
                        coeff = -k_array[i, j, k] / (dz[k] * dz_t[k])
                        row_indices[nnz_count] = idx
                        col_indices[nnz_count] = neighbor_idx
                        data[nnz_count] = coeff
                        nnz_count += 1
                        diagonal_coeff -= coeff

                elif k == nk - 1:  # 底面境界（断熱）
                    rhs[idx] = rho * cp[i, j, k] * T[i, j, k] / dt

                else:  # 内部点
                    rhs[idx] = rho * cp[i, j, k] * T[i, j, k] / dt

                    # z方向の拡散
                    if k > 0:
                        neighbor_idx = get_index_numba(i, j, k - 1, nj, nk)
                        coeff = -k_array[i, j, k] / (dz[k] * dz_b[k])
                        row_indices[nnz_count] = idx
                        col_indices[nnz_count] = neighbor_idx
                        data[nnz_count] = coeff
                        nnz_count += 1
                        diagonal_coeff -= coeff

                    if k < nk - 1:
                        neighbor_idx = get_index_numba(i, j, k + 1, nj, nk)
                        coeff = -k_array[i, j, k] / (dz[k] * dz_t[k])
                        row_indices[nnz_count] = idx
                        col_indices[nnz_count] = neighbor_idx
                        data[nnz_count] = coeff
                        nnz_count += 1
                        diagonal_coeff -= coeff

                # x, y方向の拡散
                if i > 0:
                    neighbor_idx = get_index_numba(i - 1, j, k, nj, nk)
                    coeff = -k_array[i, j, k] / dx**2
                    row_indices[nnz_count] = idx
                    col_indices[nnz_count] = neighbor_idx
                    data[nnz_count] = coeff
                    nnz_count += 1
                    diagonal_coeff -= coeff

                if i < ni - 1:
                    neighbor_idx = get_index_numba(i + 1, j, k, nj, nk)
                    coeff = -k_array[i, j, k] / dx**2
                    row_indices[nnz_count] = idx
                    col_indices[nnz_count] = neighbor_idx
                    data[nnz_count] = coeff
                    nnz_count += 1
                    diagonal_coeff -= coeff

                if j > 0:
                    neighbor_idx = get_index_numba(i, j - 1, k, nj, nk)
                    coeff = -k_array[i, j, k] / dy**2
                    row_indices[nnz_count] = idx
                    col_indices[nnz_count] = neighbor_idx
                    data[nnz_count] = coeff
                    nnz_count += 1
                    diagonal_coeff -= coeff

                if j < nj - 1:
                    neighbor_idx = get_index_numba(i, j + 1, k, nj, nk)
                    coeff = -k_array[i, j, k] / dy**2
                    row_indices[nnz_count] = idx
                    col_indices[nnz_count] = neighbor_idx
                    data[nnz_count] = coeff
                    nnz_count += 1
                    diagonal_coeff -= coeff

                # 対角成分を追加
                row_indices[nnz_count] = idx
                col_indices[nnz_count] = idx
                data[nnz_count] = diagonal_coeff
                nnz_count += 1

    # 有効なデータのみを返す
    return row_indices[:nnz_count], col_indices[:nnz_count], data[:nnz_count], rhs

def multiple_time_step_solver_DHCP_numba(T_initial, q_surface, nt, rho, cp_coeffs, k_coeffs,
                                        dx, dy, dz, dz_b, dz_t, dt, rtol=1e-6, maxiter=1000):
    """複数時間ステップDHCPソルバー（Numba最適化版）"""
    ni, nj, nk = T_initial.shape
    T_all = np.zeros((nt, ni, nj, nk))
    T_all[0] = T_initial.copy()

    for t in range(1, nt):
        # 現在の温度での熱物性値計算（Numba並列）
        cp, k_array = thermal_properties_calculator_numba(T_all[t-1], cp_coeffs, k_coeffs)

        # 係数行列とRHS構築（Numba最適化）
        q_surface_2d = q_surface[t-1]  # 2D slice for Numba
        row_indices, col_indices, data, rhs = coeffs_and_rhs_building_DHCP_numba(
            T_all[t-1], q_surface_2d, t, rho, cp, k_array, dx, dy, dz, dz_b, dz_t, dt
        )

        # Sparse行列構築
        A = csr_matrix((data, (row_indices, col_indices)), shape=(ni*nj*nk, ni*nj*nk))

        # 線形システム求解
        T_flat, info = cg(A, rhs, rtol=rtol, maxiter=maxiter)

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
# 標準Python版結果の読み込み
# =======================================
comparison_path = Path(__file__).parent / 'python_julia_comparison.npz'
try:
    comparison_data = np.load(comparison_path)
    python_std_time = float(comparison_data['python_avg_time'])
    python_std_rate = float(comparison_data['python_rate'])
    print(f"Python標準版実行時間: {python_std_time:.3f} 秒")
except FileNotFoundError:
    python_std_time = 0.0
    python_std_rate = 0.0
    print("標準Python版結果が見つかりません")

# =======================================
# パラメータ設定
# =======================================
print("\n【パラメータ設定】Numba最適化Python版")

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
print(f"Numba並列化: 有効")

# =======================================
# Numba JIT コンパイル（ウォームアップ）
# =======================================
print("\n【Numbaウォームアップ】JITコンパイル実行")

# 小規模データでコンパイル
T_small = T0[:2, :2, :2].copy()
q_small = q_surface[:1, :2, :2].copy()
cp_small_coeffs = cp_coeffs.copy()
k_small_coeffs = k_coeffs.copy()

print("Numba JITコンパイル中...")
start_warmup = time.time()

# ウォームアップ実行
cp_test, k_test = thermal_properties_calculator_numba(T_small, cp_small_coeffs, k_small_coeffs)
warmup_time = time.time() - start_warmup
print(f"JITコンパイル完了: {warmup_time:.3f}秒")

# =======================================
# Numba最適化Python版ベンチマーク実行
# =======================================
print("\n【Numba最適化Python版実行】直接問題ソルバー測定")

print("Numba最適化Python版実行中...")
numba_times = []

for run in range(3):  # 3回実行して平均取得
    import gc
    gc.collect()

    start_time = time.time()

    # Numba最適化Python版ソルバー実行
    T_all_numba = multiple_time_step_solver_DHCP_numba(
        T0, q_surface, nt_benchmark, rho, cp_coeffs, k_coeffs,
        dx, dy, dz, dz_b, dz_t, dt,
        rtol=1e-6, maxiter=20000  # Julia版と同じ設定
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    numba_times.append(elapsed_time)

    print(f"Run {run+1}: {elapsed_time:.3f}秒")

# 統計計算
numba_avg_time = np.mean(numba_times)
numba_std_time = np.std(numba_times, ddof=1)

print(f"Numba最適化Python版結果:")
print(f"  平均実行時間: {numba_avg_time:.3f} ± {numba_std_time:.3f} 秒")
print(f"  温度変化: {np.min(T_all_numba[0, :, :, :]):.2f} K → {np.max(T_all_numba[-1, :, :, :]):.2f} K")

# 計算統計
total_grid_points = ni * nj * nk
total_operations = total_grid_points * (nt_benchmark - 1)
numba_rate = total_operations / numba_avg_time

print(f"  総格子点数: {total_grid_points}")
print(f"  総計算量: {total_operations} 格子点×時間ステップ")
print(f"  計算レート: {numba_rate:.0f} 格子点×ステップ/秒")

# =======================================
# 3-way 性能比較
# =======================================
print("\n【3-way性能比較】Julia vs Python(Numba) vs Python(標準)")

julia_rate = total_operations / julia_avg_time
numba_vs_julia = julia_avg_time / numba_avg_time
numba_vs_python = python_std_time / numba_avg_time if python_std_time > 0 else 0

print(f"実行時間比較:")
print(f"  Julia版:           {julia_avg_time:.3f} ± {julia_std_time:.3f} 秒")
print(f"  Python(Numba)版:   {numba_avg_time:.3f} ± {numba_std_time:.3f} 秒")
if python_std_time > 0:
    print(f"  Python(標準)版:    {python_std_time:.3f} 秒")

print(f"計算レート比較:")
print(f"  Julia版:           {julia_rate:.0f} 格子点×ステップ/秒")
print(f"  Python(Numba)版:   {numba_rate:.0f} 格子点×ステップ/秒")
if python_std_rate > 0:
    print(f"  Python(標準)版:    {python_std_rate:.0f} 格子点×ステップ/秒")

print(f"性能比較:")
if numba_vs_julia > 1:
    print(f"  Julia vs Numba:    Julia版が{numba_vs_julia:.2f}倍高速")
else:
    print(f"  Julia vs Numba:    Numba版が{1/numba_vs_julia:.2f}倍高速")

if numba_vs_python > 0:
    print(f"  Numba vs 標準:     Numba版が{numba_vs_python:.2f}倍高速")

# =======================================
# 結果サマリー
# =======================================
print("\n" + "=" * 80)
print("Numba最適化Python版ベンチマーク完了")
print("=" * 80)

print("📊 Numba最適化Python版性能サマリー:")
print(f"⏱️  実行時間: {numba_avg_time:.3f} ± {numba_std_time:.3f} 秒")
print(f"🖥️  計算レート: {numba_rate:.0f} 格子点×ステップ/秒")
print(f"🧮 問題規模: {total_grid_points}格子点 × {nt_benchmark-1}時間ステップ")
print(f"🚀 JITコンパイル: {warmup_time:.3f}秒")

print("\n🏆 最終性能ランキング:")
results = [
    ("Julia版", julia_avg_time, julia_rate),
    ("Python(Numba)版", numba_avg_time, numba_rate)
]
if python_std_time > 0:
    results.append(("Python(標準)版", python_std_time, python_std_rate))

results.sort(key=lambda x: x[1])  # 実行時間でソート

for i, (name, exec_time, rate) in enumerate(results, 1):
    print(f"  {i}位: {name} - {exec_time:.3f}秒 ({rate:.0f} 格子点×ステップ/秒)")

# =======================================
# 結果保存
# =======================================
print("\n【結果保存】")

# 3-way比較結果を保存
comparison_3way_data = {
    'julia_avg_time': julia_avg_time,
    'julia_std_time': julia_std_time,
    'julia_rate': julia_rate,
    'numba_avg_time': numba_avg_time,
    'numba_std_time': numba_std_time,
    'numba_rate': numba_rate,
    'python_std_time': python_std_time,
    'python_std_rate': python_std_rate,
    'numba_vs_julia_ratio': numba_vs_julia,
    'numba_vs_python_ratio': numba_vs_python,
    'T_final_numba': T_all_numba[-1],
    'problem_size': total_grid_points,
    'time_steps': nt_benchmark - 1,
    'warmup_time': warmup_time
}

comparison_3way_path = Path(__file__).parent / 'python_numba_3way_comparison.npz'
np.savez(comparison_3way_path, **comparison_3way_data)
print("python_numba_3way_comparison.npz に3-way比較結果を保存しました")

print("\n✅ Numba最適化Python版ベンチマーク比較完了")
print("🎯 Julia vs Python(Numba) vs Python(標準) の性能評価が完了しました")