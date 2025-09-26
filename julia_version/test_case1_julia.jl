#!/usr/bin/env julia
"""
テストケース1: 小規模精密比較 (Julia版)
5×5×20格子、5ステップ、詳細ログ出力
"""

include("cgm_solver.jl")
using Printf, Statistics, Random

println("="^60)
println("テストケース1: Julia版小規模精密比較")
println("="^60)

# 再現性のため乱数固定
Random.seed!(42)

# =======================================
# テスト設定
# =======================================
const TEST_NI, TEST_NJ = 5, 5
const TEST_NT = 5
const CGM_ITER = 5

println("【テスト設定】")
println("格子: $TEST_NI × $TEST_NJ × $nz = $(TEST_NI * TEST_NJ * nz) 格子点")
println("時間: $TEST_NT ステップ")
println("CGM反復: $CGM_ITER 回")

# =======================================
# 制御された人工データ生成
# =======================================
println("\n【テストデータ生成】")

# 制御された温度分布（再現可能）
T_base = 500.0 .+ 5.0 * rand(TEST_NT, TEST_NI, TEST_NJ)

# 初期温度（z方向均一）
T_init = zeros(TEST_NI, TEST_NJ, nz)
for k in 1:nz
    T_init[:, :, k] = T_base[1, :, :]
end

# 観測データ
Y_obs = T_base[2:end, :, :]

# 初期熱流束（ゼロ）
q_init = zeros(TEST_NT-1, TEST_NI, TEST_NJ)

dt_test = 0.001

temp_range = (minimum(T_base), maximum(T_base))
temp_min_str = @sprintf("%.3f", temp_range[1])
temp_max_str = @sprintf("%.3f", temp_range[2])
println("温度範囲: $temp_min_str - $temp_max_str K")
println("初期温度平均: $(@sprintf("%.3f", mean(T_init))) K")

# =======================================
# Julia版CGM実行（詳細ログ）
# =======================================
println("\n【Julia版CGM実行】")

start_time = time()

try
    # 詳細ログを取得するためのCGM実行
    q_opt_julia, T_final_julia, J_hist_julia = global_CGM_time(
        T_init, T_base, q_init,
        dx, dy, dz, dz_b, dz_t, dt_test,
        rho, cp_coeffs, k_coeffs;
        CGM_iteration=CGM_ITER
    )

    elapsed_julia = time() - start_time

    println("✅ Julia版CGM成功")

    # =======================================
    # 詳細結果解析
    # =======================================
    println("\n【Julia版詳細結果解析】")

    # 1. 収束解析
    n_iter = length(J_hist_julia)
    J_init_str = @sprintf("%.6e", J_hist_julia[1])
    J_final_str = @sprintf("%.6e", J_hist_julia[end])

    println("1. 収束解析:")
    println("  実行反復数: $n_iter")
    println("  初期目的関数: $J_init_str")
    println("  最終目的関数: $J_final_str")

    if n_iter > 1
        rel_improve = (J_hist_julia[1] - J_hist_julia[end]) / J_hist_julia[1]
        rel_improve_str = @sprintf("%.6f", rel_improve)
        println("  相対改善率: $rel_improve_str")
    end

    # 2. 熱流束詳細解析
    println("\n2. 熱流束詳細解析:")
    q_min_str = @sprintf("%.6e", minimum(q_opt_julia))
    q_max_str = @sprintf("%.6e", maximum(q_opt_julia))
    q_mean_str = @sprintf("%.6e", mean(q_opt_julia))
    q_std_str = @sprintf("%.6e", std(q_opt_julia))
    q_rms_str = @sprintf("%.6e", sqrt(mean(q_opt_julia.^2)))

    println("  最小値: $q_min_str W/m²")
    println("  最大値: $q_max_str W/m²")
    println("  平均値: $q_mean_str W/m²")
    println("  標準偏差: $q_std_str W/m²")
    println("  RMS値: $q_rms_str W/m²")

    # 空間分布の詳細
    q_spatial_var = std([std(q_opt_julia[t, :, :]) for t in 1:(TEST_NT-1)])
    q_temporal_var = std([std(q_opt_julia[:, i, j]) for i in 1:TEST_NI, j in 1:TEST_NJ])

    spatial_var_str = @sprintf("%.6e", q_spatial_var)
    temporal_var_str = @sprintf("%.6e", q_temporal_var)
    println("  空間変動: $spatial_var_str W/m²")
    println("  時間変動: $temporal_var_str W/m²")

    # 3. 温度予測精度
    println("\n3. 温度予測精度:")
    T_pred = T_final_julia[:, :, 1]  # 表面温度予測
    T_true = T_base[end, :, :]       # 実際の表面温度

    temp_error = T_pred - T_true
    temp_rmse = sqrt(mean(temp_error.^2))
    temp_mae = mean(abs.(temp_error))
    temp_max_error = maximum(abs.(temp_error))

    rmse_str = @sprintf("%.6e", temp_rmse)
    mae_str = @sprintf("%.6e", temp_mae)
    max_err_str = @sprintf("%.6e", temp_max_error)

    println("  RMSE: $rmse_str K")
    println("  MAE: $mae_str K")
    println("  最大誤差: $max_err_str K")

    # 4. 計算効率
    println("\n4. 計算効率:")
    total_dofs = TEST_NI * TEST_NJ * nz
    total_operations = total_dofs * (TEST_NT - 1) * n_iter
    throughput = total_operations / elapsed_julia

    elapsed_str = @sprintf("%.6f", elapsed_julia)
    throughput_str = @sprintf("%.0f", throughput)

    println("  実行時間: $elapsed_str 秒")
    println("  スループット: $throughput_str 格子点・ステップ・反復/秒")
    println("  反復あたり時間: $(@sprintf("%.6f", elapsed_julia/n_iter)) 秒")

    # 5. 数値安定性指標
    println("\n5. 数値安定性:")

    # NaN/Inf チェック
    has_nan_q = any(isnan.(q_opt_julia)) || any(isinf.(q_opt_julia))
    has_nan_T = any(isnan.(T_final_julia)) || any(isinf.(T_final_julia))

    println("  NaN/Inf発生 (熱流束): $(has_nan_q ? "❌" : "✅")")
    println("  NaN/Inf発生 (温度): $(has_nan_T ? "❌" : "✅")")

    # 数値の桁数評価
    significant_q = q_opt_julia[abs.(q_opt_julia) .> 1e-15]
    if length(significant_q) > 0
        q_order = log10(maximum(abs.(significant_q)))
        q_order_str = @sprintf("%.1f", q_order)
        println("  有意熱流束桁数: 10^$q_order_str")
    else
        println("  有意熱流束: 検出されず（極小値のみ）")
    end

    # =======================================
    # 結果保存
    # =======================================
    julia_results = Dict(
        "test_case" => 1,
        "grid_size" => [TEST_NI, TEST_NJ, nz],
        "time_steps" => TEST_NT,
        "iterations" => n_iter,
        "objective_initial" => J_hist_julia[1],
        "objective_final" => J_hist_julia[end],
        "relative_improvement" => n_iter > 1 ? (J_hist_julia[1] - J_hist_julia[end]) / J_hist_julia[1] : 0.0,
        "q_min" => minimum(q_opt_julia),
        "q_max" => maximum(q_opt_julia),
        "q_mean" => mean(q_opt_julia),
        "q_std" => std(q_opt_julia),
        "q_rms" => sqrt(mean(q_opt_julia.^2)),
        "temp_rmse" => temp_rmse,
        "temp_mae" => temp_mae,
        "temp_max_error" => temp_max_error,
        "elapsed_time" => elapsed_julia,
        "throughput" => throughput,
        "has_numerical_issues" => has_nan_q || has_nan_T
    )

    # NPZ形式で保存（Python互換）
    using NPZ
    npzwrite("test_case1_julia_results.npz", julia_results)

    println("\n✅ 結果保存: test_case1_julia_results.npz")

    global julia_case1_success = true

catch e
    println("❌ Julia版エラー: $e")
    global julia_case1_success = false
end

println("\n" * "="^60)
println("テストケース1 Julia版完了")
println("="^60)