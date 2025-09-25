#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python版直接問題ソルバー（DHCP）ベンチマーク
Julia版との性能比較用
"""

import numpy as np
import time
import sys
import os
from pathlib import Path

# orgディレクトリのPython実装をインポート
org_path = Path(__file__).parent.parent / 'org'
sys.path.insert(0, str(org_path))

# パス設定修正（相対パス使用）
import os
os.chdir(org_path)

try:
    from IHCP_CGM_Sliding_Window_Calculation_ver2 import (
        thermal_properties_calculator,
        coeffs_and_rhs_building_DHCP,
        multiple_time_step_solver_DHCP
    )
    print("Python版ソルバー関数のインポート成功")
except ImportError as e:
    print(f"Python版ソルバー関数のインポートエラー: {e}")
    sys.exit(1)

print("=" * 80)
print("Python版 vs Julia版 性能比較: 直接問題ソルバー（DHCP）")
print("=" * 80)

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
    sys.exit(1)

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
python_iterations = []

for run in range(3):  # 3回実行して平均取得
    # ガベージコレクション（可能な限り）
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
# メモリ使用量推定
# =======================================
print("\n【メモリ使用量推定】")

# 主要配列のメモリ使用量計算
T_all_memory = T_all_python.nbytes / 1024**2  # MB
sparse_elements = total_grid_points * 7  # 7-point stencil
sparse_memory = sparse_elements * 16 / 1024**2  # MB (Float64 + index)
temp_arrays_memory = total_grid_points * 8 * 5 / 1024**2  # MB (cp, k, coeffs etc.)

total_memory_estimate = T_all_memory + sparse_memory + temp_arrays_memory

print(f"メモリ使用量推定:")
print(f"  温度配列: {T_all_memory:.1f} MB")
print(f"  Sparse行列: {sparse_memory:.1f} MB")
print(f"  一時配列: {temp_arrays_memory:.1f} MB")
print(f"  推定総計: {total_memory_estimate:.1f} MB")

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
# 数値精度比較
# =======================================
print("\n【数値精度比較】")

# Julia版結果と比較するため、Julia版も再実行が必要
# ここでは温度場の統計的比較のみ実行
print(f"Python版温度統計:")
print(f"  最終温度範囲: {np.min(T_all_python[-1]):.2f} - {np.max(T_all_python[-1]):.2f} K")
print(f"  平均温度上昇: {np.mean(T_all_python[-1]) - np.mean(T_all_python[0]):.2f} K")
print(f"  最大温度勾配: {np.max(np.diff(T_all_python[-1], axis=2)):.2f} K/m")

# =======================================
# 結果サマリー
# =======================================
print("\n" + "=" * 80)
print("Python版ベンチマーク完了")
print("=" * 80)

print("📊 Python版性能サマリー:")
print(f"⏱️  実行時間: {python_avg_time:.3f} ± {python_std_time:.3f} 秒")
print(f"🖥️  計算レート: {python_rate:.0f} 格子点×ステップ/秒")
print(f"💾 推定メモリ: {total_memory_estimate:.1f} MB")
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

# 比較結果をNPZファイルに保存
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

print("\n次のステップ:")
print("1. 数値精度の詳細比較（必要に応じて）")
print("2. 大規模問題での性能スケーリング評価")
print("3. メモリ効率の詳細分析")