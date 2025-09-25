#!/usr/bin/env julia
# -*- coding: utf-8 -*-
"""
Python vs Julia 性能比較テスト
実データを使った直接問題ソルバー（DHCP）の計算時間測定
"""

include("dhcp_solver.jl")

using Printf

println("=" ^ 80)
println("Python vs Julia 性能比較: 直接問題ソルバー（DHCP）")
println("=" ^ 80)

# =======================================
# 実データ読み込み
# =======================================
println("\n【データ準備】実データ読み込み")

T_measure_K = npzread("T_measure_700um_1ms.npy")
println(@sprintf("実データサイズ: %s", string(size(T_measure_K))))

# Python版と同じ設定
dt = 0.001  # 1ms

# 初期温度分布の抽出（下面温度をz方向全体に複製）
T_measure_init_K = T_measure_K[1, :, :]
T0 = repeat(reshape(T_measure_init_K, size(T_measure_init_K)..., 1), 1, 1, nz)

# 測定条件: Python版と同じ500フレーム切り出し
Y_obs = T_measure_K[1:500, :, :]
nt_benchmark = 11  # 10 time steps (nt-1 = 10 heat flux steps)
Y_obs_benchmark = Y_obs[1:nt_benchmark, :, :]

ni, nj, nk = size(T0)
println(@sprintf("ベンチマーク条件:"))
println(@sprintf("  格子サイズ: %d × %d × %d", ni, nj, nk))
println(@sprintf("  時間ステップ数: %d", nt_benchmark))
println(@sprintf("  温度範囲: %.2f - %.2f K", minimum(Y_obs_benchmark), maximum(Y_obs_benchmark)))

# 初期熱流束（ゼロ）
q_surface_benchmark = zeros(nt_benchmark-1, ni, nj)
T_all_julia = nothing  # 変数の初期化

# =======================================
# Julia版ベンチマーク実行
# =======================================
println("\n【Julia版実行】直接問題ソルバー測定")

println("Julia版実行中...")
julia_times = Float64[]
julia_iterations = Int[]

for run in 1:3  # 3回実行して平均取得
    GC.gc()  # ガベージコレクション実行

    start_time = time()

    global T_all_julia = multiple_time_step_solver_DHCP(
        T0, q_surface_benchmark, nt_benchmark, rho, cp_coeffs, k_coeffs,
        dx, dy, dz, dz_b, dz_t, dt,
        1e-6, 20000  # rtol, maxiter
    )

    end_time = time()
    elapsed_time = end_time - start_time
    push!(julia_times, elapsed_time)

    println(@sprintf("Run %d: %.3f秒", run, elapsed_time))
end

julia_avg_time = sum(julia_times) / length(julia_times)
julia_std_time = sqrt(sum((julia_times .- julia_avg_time).^2) / (length(julia_times) - 1))

println(@sprintf("Julia版結果:"))
println(@sprintf("  平均実行時間: %.3f ± %.3f 秒", julia_avg_time, julia_std_time))
println(@sprintf("  温度変化: %.2f K → %.2f K",
                minimum(T_all_julia[1, :, :, :]), maximum(T_all_julia[end, :, :, :])))

# 計算統計
total_grid_points = ni * nj * nk
total_operations = total_grid_points * (nt_benchmark - 1)
julia_rate = total_operations / julia_avg_time

println(@sprintf("  総格子点数: %d", total_grid_points))
println(@sprintf("  総計算量: %d 格子点×時間ステップ", total_operations))
println(@sprintf("  計算レート: %.0f 格子点×ステップ/秒", julia_rate))

# =======================================
# メモリ使用量推定
# =======================================
println("\n【メモリ使用量推定】")

# 主要配列のメモリ使用量計算
T_all_memory = sizeof(T_all_julia) / 1024^2  # MB
sparse_elements = total_grid_points * 7  # 7-point stencil
sparse_memory = sparse_elements * 16 / 1024^2  # MB (Float64 + index)
temp_arrays_memory = total_grid_points * 8 * 5 / 1024^2  # MB (cp, k, coeffs etc.)

total_memory_estimate = T_all_memory + sparse_memory + temp_arrays_memory

println(@sprintf("メモリ使用量推定:"))
println(@sprintf("  温度配列: %.1f MB", T_all_memory))
println(@sprintf("  Sparse行列: %.1f MB", sparse_memory))
println(@sprintf("  一時配列: %.1f MB", temp_arrays_memory))
println(@sprintf("  推定総計: %.1f MB", total_memory_estimate))

# =======================================
# Python版比較用データ生成
# =======================================
println("\n【Python版比較用】データエクスポート")

# Python版で同じ条件で実行するためのデータ保存
npzwrite("benchmark_data.npz", Dict(
    "T0" => T0,
    "q_surface" => q_surface_benchmark,
    "Y_obs" => Y_obs_benchmark,
    "nt" => nt_benchmark,
    "dt" => dt,
    "julia_avg_time" => julia_avg_time,
    "julia_std_time" => julia_std_time
))

println("benchmark_data.npz にデータ保存完了")
println("Python版での比較実行用データを準備しました")

# =======================================
# 結果サマリー
# =======================================
println("\n" * "=" ^ 80)
println("Julia版ベンチマーク完了")
println("=" ^ 80)

println("📊 Julia版性能サマリー:")
println(@sprintf("⏱️  実行時間: %.3f ± %.3f 秒", julia_avg_time, julia_std_time))
println(@sprintf("🖥️  計算レート: %.0f 格子点×ステップ/秒", julia_rate))
println(@sprintf("💾 推定メモリ: %.1f MB", total_memory_estimate))
println(@sprintf("🧮 問題規模: %d格子点 × %d時間ステップ", total_grid_points, nt_benchmark-1))

println("\n次のステップ:")
println("1. Python版で同じ条件のベンチマークを実行")
println("2. 'benchmark_data.npz' を使用して条件を統一")
println("3. 実行時間・精度・メモリ使用量を比較")

println("\nPython版実行コマンド例:")
println("python benchmark_python.py")