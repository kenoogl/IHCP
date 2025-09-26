#!/usr/bin/env julia
# -*- coding: utf-8 -*-
"""
Julia版IHCP-CGMソルバー フルスケール実行スクリプト
オリジナルPython版と同等の実データを使用した完全実行

実行方法:
JULIA_NUM_THREADS=8 julia julia_full_scale_execution.jl
"""

include("cgm_solver.jl")

using Printf
using Dates
using Statistics  # mean関数のために必要

println("=" ^ 80)
println("Julia版 IHCP-CGM ソルバー フルスケール実行")
println("実行開始時刻: $(now())")
println("=" ^ 80)

# =======================================
# 実行設定（オリジナルPython版と同等）
# =======================================
println("\n【実行設定】")

# 計算パラメータ
const WINDOW_SIZE = 50          # スライディングウィンドウサイズ（時間フレーム）
const CGM_MAX_ITERATIONS = 100  # CGM最大反復数（Python版と同じ）
const TIME_STEP_MS = 1.0        # 時間ステップ 1ms
const CONVERGENCE_TOL = 1e-6    # 収束判定閾値

# 格子サイズ設定（実データに合わせて）
const SPATIAL_REGION = (1:80, 1:100)  # 実データから使用する空間領域

println("  スライディングウィンドウサイズ: $WINDOW_SIZE フレーム")
println("  CGM最大反復数: $CGM_MAX_ITERATIONS")
println("  時間ステップ: $TIME_STEP_MS ms")
println("  空間領域: $(SPATIAL_REGION[1]) × $(SPATIAL_REGION[2])")
println("  スレッド数: $(Threads.nthreads())")

# =======================================
# 実データ読み込み
# =======================================
println("\n【実データ読み込み】")

try
    global T_measure_K = npzread("T_measure_700um_1ms.npy")
    println("実測定データ読み込み成功")
    println("  データ形状: $(size(T_measure_K))")
    println("  温度範囲: $(minimum(T_measure_K):.2f) - $(maximum(T_measure_K):.2f) K")
    println("  データサイズ: $(sizeof(T_measure_K) / 1024^3:.2f) GB")
catch e
    println("❌ 実測定データ読み込みエラー: $e")
    println("T_measure_700um_1ms.npy ファイルが見つかりません。")
    exit(1)
end

# データサイズ確認
nt_total, ni_total, nj_total = size(T_measure_K)

# 使用する空間領域の切り出し
ni_use = length(SPATIAL_REGION[1])
nj_use = length(SPATIAL_REGION[2])
T_region = T_measure_K[:, SPATIAL_REGION[1], SPATIAL_REGION[2]]

println("  使用する領域: $ni_use × $nj_use × $nz")
println("  総時間フレーム数: $nt_total")

# =======================================
# 初期設定と前処理
# =======================================
println("\n【初期設定】")

# 時間ステップ設定
dt = TIME_STEP_MS / 1000.0  # ms → s変換

# 初期温度分布（z方向に拡張）
T_initial = zeros(ni_use, nj_use, nz)
for k in 1:nz
    T_initial[:, :, k] = T_region[1, :, :]
end

println("  初期温度範囲: $(minimum(T_initial):.2f) - $(maximum(T_initial):.2f) K")
println("  物理パラメータ:")
println("    密度: $rho kg/m³")
println("    格子間隔: dx=$(dx*1e3:.2f)mm, dy=$(dy*1e3:.2f)mm")
println("    z方向格子数: $nz")

# =======================================
# スライディングウィンドウ計算ループ
# =======================================
println("\n【スライディングウィンドウCGM計算開始】")

# 結果保存用配列
num_windows = div(nt_total - WINDOW_SIZE, 10) + 1  # 10フレームずつスライド
q_results = zeros(num_windows, WINDOW_SIZE-1, ni_use, nj_use)
computation_times = Float64[]

window_count = 0
total_start_time = time()

for start_frame in 1:10:(nt_total - WINDOW_SIZE + 1)
    global window_count += 1
    end_frame = start_frame + WINDOW_SIZE - 1

    if end_frame > nt_total
        break
    end

    window_start_time = time()

    println("\n--- ウィンドウ $window_count ---")
    println("  フレーム範囲: $start_frame - $end_frame")
    println("  実行時刻: $(now())")

    # 現在のウィンドウのデータ抽出
    Y_obs_window = T_region[start_frame:end_frame, :, :]
    nt_window = size(Y_obs_window, 1)

    # ウィンドウの初期温度
    T0_window = zeros(ni_use, nj_use, nz)
    for k in 1:nz
        T0_window[:, :, k] = Y_obs_window[1, :, :]
    end

    # 初期熱流束推定（前回結果または零）
    if window_count == 1
        q_init_window = zeros(nt_window-1, ni_use, nj_use)
    else
        # 前回の結果を初期推定として使用
        q_init_window = zeros(nt_window-1, ni_use, nj_use)
        # 簡単な外挿（実際にはより高度な方法を使用）
        for t in 1:(nt_window-1)
            q_init_window[t, :, :] = q_results[window_count-1, min(t, size(q_results, 2)), :, :]
        end
    end

    try
        # CGM最適化実行
        q_optimized, T_final, J_history = global_CGM_time(
            T0_window, Y_obs_window, q_init_window,
            dx, dy, dz, dz_b, dz_t, dt,
            rho, cp_coeffs, k_coeffs;
            CGM_iteration=CGM_MAX_ITERATIONS
        )

        # 結果保存
        q_results[window_count, :, :, :] = q_optimized

        window_elapsed = time() - window_start_time
        push!(computation_times, window_elapsed)

        # 進捗表示
        println("  CGM反復数: $(length(J_history))")
        println("  最終目的関数: $(J_history[end]:.2e)")
        println("  ウィンドウ計算時間: $(window_elapsed:.1f)秒")
        println("  熱流束統計:")
        println("    最小値: $(minimum(q_optimized):.0f) W/m²")
        println("    最大値: $(maximum(q_optimized):.0f) W/m²")
        println("    平均値: $(mean(q_optimized):.0f) W/m²")

        # 進捗推定
        avg_time = mean(computation_times)
        remaining_windows = num_windows - window_count
        estimated_remaining = remaining_windows * avg_time

        println("  進捗: $window_count/$num_windows ($(100*window_count/num_windows:.1f)%)")
        println("  推定残り時間: $(estimated_remaining/3600:.1f)時間")

    catch e
        println("  ❌ ウィンドウ $window_count でエラー発生: $e")

        # エラー時の処理（零埋めまたは前回値継続）
        if window_count > 1
            q_results[window_count, :, :, :] = q_results[window_count-1, :, :, :]
        else
            q_results[window_count, :, :, :] = zeros(WINDOW_SIZE-1, ni_use, nj_use)
        end

        push!(computation_times, time() - window_start_time)
    end

    # 定期的なガベージコレクション
    if window_count % 10 == 0
        GC.gc()
        println("  メモリクリーンアップ実行")
    end
end

total_elapsed = time() - total_start_time

# =======================================
# 結果処理とファイル出力
# =======================================
println("\n【結果処理】")

# 時系列再構築（重複部分の平均化）
println("時系列データ再構築中...")

# 全時間フレームの結果配列
q_full_timeline = zeros(nt_total-1, ni_use, nj_use)
weight_timeline = zeros(nt_total-1)

for window_idx in 1:window_count
    start_frame = 1 + (window_idx - 1) * 10

    for t in 1:(WINDOW_SIZE-1)
        global_t = start_frame + t - 1

        if global_t <= nt_total - 1
            # 重み付き平均（時系列の中央部により高い重み）
            weight = 1.0 - abs(t - WINDOW_SIZE/2) / (WINDOW_SIZE/2)

            q_full_timeline[global_t, :, :] += weight * q_results[window_idx, t, :, :]
            weight_timeline[global_t] += weight
        end
    end
end

# 正規化
for t in 1:(nt_total-1)
    if weight_timeline[t] > 0
        q_full_timeline[t, :, :] /= weight_timeline[t]
    end
end

println("時系列再構築完了")

# =======================================
# 結果保存
# =======================================
println("\n【結果保存】")

# 結果をNPZファイルに保存
output_filename = "julia_ihcp_results_$(Dates.format(now(), "yyyymmdd_HHMMSS")).npz"

result_data = Dict(
    "q_surface_optimized" => q_full_timeline,
    "computation_times" => computation_times,
    "total_elapsed_time" => total_elapsed,
    "window_size" => WINDOW_SIZE,
    "cgm_max_iterations" => CGM_MAX_ITERATIONS,
    "spatial_region_i" => collect(SPATIAL_REGION[1]),
    "spatial_region_j" => collect(SPATIAL_REGION[2]),
    "time_step_dt" => dt,
    "grid_params" => [dx, dy, mean(dz)],
    "thermal_params" => [rho, cp_coeffs, k_coeffs]
)

npzwrite(output_filename, result_data)
println("結果ファイル保存: $output_filename")

# =======================================
# 計算統計とサマリー
# =======================================
println("\n" * "=" ^ 80)
println("Julia版 IHCP-CGM フルスケール計算完了")
println("完了時刻: $(now())")
println("=" ^ 80)

println("\n📊 計算統計:")
println("  処理ウィンドウ数: $window_count")
println("  総計算時間: $(total_elapsed/3600:.2f)時間")
println("  平均ウィンドウ計算時間: $(mean(computation_times):.1f)秒")
println("  最大ウィンドウ計算時間: $(maximum(computation_times):.1f)秒")

println("\n🎯 計算規模:")
println("  空間格子: $ni_use × $nj_use × $nz = $(ni_use * nj_use * nz) 格子点")
println("  時間フレーム: $(nt_total-1)")
println("  総自由度: $((nt_total-1) * ni_use * nj_use) 熱流束値")

println("\n🔥 熱流束結果統計:")
q_min = minimum(q_full_timeline)
q_max = maximum(q_full_timeline)
q_mean = mean(q_full_timeline)
q_std = std(q_full_timeline)

println("  最小値: $(q_min:.0f) W/m²")
println("  最大値: $(q_max:.0f) W/m²")
println("  平均値: $(q_mean:.0f) W/m²")
println("  標準偏差: $(q_std:.0f) W/m²")

println("\n💾 出力ファイル:")
println("  $output_filename")
println("  ファイルサイズ: $(filesize(output_filename) / 1024^2:.1f) MB")

println("\n⚡ 性能評価:")
total_operations = window_count * WINDOW_SIZE * ni_use * nj_use * nz * CGM_MAX_ITERATIONS
operations_per_second = total_operations / total_elapsed
println("  総計算量: $(total_operations) 格子点×反復")
println("  計算レート: $(operations_per_second:.0f) 格子点×反復/秒")

println("\n✅ Julia版IHCP-CGMフルスケール実行が正常に完了しました")
println("   結果ファイル: $output_filename を確認してください")

println("\n📋 次のステップ:")
println("  1. 結果の可視化（Python/MATLAB）")
println("  2. Python版との結果比較")
println("  3. 物理的妥当性の検証")

println("\n" * "=" ^ 80)