#!/usr/bin/env julia
"""
julia_full_scale_execution.jlの1ウィンドウテスト版
"""

include("cgm_solver.jl")

using Printf
using Dates
using Statistics

println("単一ウィンドウテスト開始")

# 基本設定
const WINDOW_SIZE = 50
const CGM_MAX_ITERATIONS = 10  # テスト用に短縮
const TIME_STEP_MS = 1.0
const SPATIAL_REGION = (1:80, 1:100)

# データ読み込み
println("データ読み込み中...")
T_measure_K = npzread("T_measure_700um_1ms.npy")
println("✅ データ読み込み成功: $(size(T_measure_K))")

# 領域切り出し
nt_total, ni_total, nj_total = size(T_measure_K)
ni_use = length(SPATIAL_REGION[1])
nj_use = length(SPATIAL_REGION[2])
T_region = T_measure_K[:, SPATIAL_REGION[1], SPATIAL_REGION[2]]

# 時間ステップ設定
dt = TIME_STEP_MS / 1000.0

# 最初のウィンドウのみテスト
start_frame = 1
end_frame = start_frame + WINDOW_SIZE - 1

println("ウィンドウテスト:")
println("  フレーム範囲: $start_frame - $end_frame")

# データ抽出
Y_obs_window = T_region[start_frame:end_frame, :, :]
nt_window = size(Y_obs_window, 1)

# 初期温度設定
T0_window = zeros(ni_use, nj_use, nz)
for k in 1:nz
    T0_window[:, :, k] = Y_obs_window[1, :, :]
end

# 初期熱流束（零）
q_init_window = zeros(nt_window-1, ni_use, nj_use)

println("CGM最適化開始...")
start_time = time()

try
    # CGM最適化実行
    q_optimized, T_final, J_history = global_CGM_time(
        T0_window, Y_obs_window, q_init_window,
        dx, dy, dz, dz_b, dz_t, dt,
        rho, cp_coeffs, k_coeffs;
        CGM_iteration=CGM_MAX_ITERATIONS
    )

    elapsed = time() - start_time

    println("✅ CGM最適化成功")
    println("  反復数: $(length(J_history))")
    println("  最終目的関数: $(J_history[end]:.2e)")
    println("  計算時間: $(elapsed:.1f)秒")
    println("  熱流束統計:")
    println("    最小値: $(minimum(q_optimized):.2e) W/m²")
    println("    最大値: $(maximum(q_optimized):.2e) W/m²")
    println("    平均値: $(mean(q_optimized):.2e) W/m²")

catch e
    println("❌ CGM最適化エラー: $e")
end

println("単一ウィンドウテスト完了")