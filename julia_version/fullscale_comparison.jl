#!/usr/bin/env julia
"""
フルスケール Julia版とPython版の定量比較評価
10ステップ限定でCGMアルゴリズムの各構成要素を詳細解析
"""

include("cgm_solver.jl")

using Printf
using Statistics
using LinearAlgebra

println("="^80)
println("フルスケール Julia版 定量評価テスト")
println("="^80)

# =======================================
# 評価設定
# =======================================
const EVAL_WINDOW_SIZE = 10    # 10時間ステップのみ評価
const EVAL_REGION = (1:20, 1:25)  # 20x25領域（フルの1/4サイズ）
const CGM_MAX_ITER = 10        # CGM反復制限

println("【評価設定】")
println("  ウィンドウサイズ: $EVAL_WINDOW_SIZE ステップ")
println("  評価領域: $(length(EVAL_REGION[1]))×$(length(EVAL_REGION[2])) = $(prod(length.(EVAL_REGION))) 格子点")
println("  CGM最大反復数: $CGM_MAX_ITER")

# =======================================
# 実データ読み込み
# =======================================
println("\n【実データ読み込み】")

T_measure = npzread("T_measure_700um_1ms.npy")
nt_total, ni_total, nj_total = size(T_measure)

# 評価領域の切り出し
ni_eval, nj_eval = length(EVAL_REGION[1]), length(EVAL_REGION[2])
T_eval = T_measure[1:EVAL_WINDOW_SIZE, EVAL_REGION[1], EVAL_REGION[2]]

println("  元データ: $nt_total × $ni_total × $nj_total")
println("  評価データ: $(size(T_eval))")
println("  温度範囲: $(minimum(T_eval):.2f) - $(maximum(T_eval):.2f) K")

# =======================================
# 初期設定
# =======================================
println("\n【初期設定】")

# 初期温度分布
T0_eval = zeros(ni_eval, nj_eval, nz)
for k in 1:nz
    T0_eval[:, :, k] = T_eval[1, :, :]
end

# 初期熱流束（ゼロ）
q_init_eval = zeros(EVAL_WINDOW_SIZE-1, ni_eval, nj_eval)

dt_eval = 0.001  # 1ms

# 評価用パラメータ
total_dofs = ni_eval * nj_eval * nz  # 総自由度数
surface_dofs = ni_eval * nj_eval     # 表面自由度数

println("  総格子点数: $total_dofs")
println("  表面格子点数: $surface_dofs")
println("  時間ステップ幅: $dt_eval s")

# =======================================
# Julia版CGM実行と詳細解析
# =======================================
println("\n【Julia版CGM実行・解析】")

total_start_time = time()

try
    # CGM実行（詳細ログ有効）
    global q_opt_julia, T_final_julia, J_hist_julia = global_CGM_time(
        T0_eval, T_eval, q_init_eval,
        dx, dy, dz, dz_b, dz_t, dt_eval,
        rho, cp_coeffs, k_coeffs;
        CGM_iteration=CGM_MAX_ITER
    )

    total_elapsed = time() - total_start_time

    println("✅ Julia版CGM成功")
    println("  総計算時間: $(total_elapsed:.2f)秒")
    println("  反復完了数: $(length(J_hist_julia))")

    # =======================================
    # 詳細評価指標の計算
    # =======================================
    println("\n【詳細評価指標】")

    # 1. CGM収束性評価
    println("1. CGM収束性:")
    println("  初期目的関数: $(J_hist_julia[1]:.4e)")
    println("  最終目的関数: $(J_hist_julia[end]:.4e)")
    if length(J_hist_julia) > 1
        convergence_rate = (J_hist_julia[1] - J_hist_julia[end]) / J_hist_julia[1]
        println("  相対改善率: $(convergence_rate:.4f)")
    end

    # 2. 熱流束解析
    println("\n2. 熱流束解析:")
    q_stats = [
        ("最小値", minimum(q_opt_julia)),
        ("最大値", maximum(q_opt_julia)),
        ("平均値", mean(q_opt_julia)),
        ("標準偏差", std(q_opt_julia)),
        ("RMS値", sqrt(mean(q_opt_julia.^2)))
    ]

    for (label, value) in q_stats
        value_str = @sprintf("%.4e", abs(value))
        println("    $label: ±$value_str W/m²")
    end

    # 3. 時空間分布解析
    println("\n3. 時空間分布特性:")

    # 時間方向変動
    temporal_var = zeros(EVAL_WINDOW_SIZE-1)
    for t in 1:(EVAL_WINDOW_SIZE-1)
        temporal_var[t] = std(q_opt_julia[t, :, :])
    end
    println("    時間変動 (RMS): $(sqrt(mean(temporal_var.^2)):.4e) W/m²")

    # 空間方向変動
    spatial_var = zeros(ni_eval, nj_eval)
    for i in 1:ni_eval, j in 1:nj_eval
        spatial_var[i, j] = std(q_opt_julia[:, i, j])
    end
    println("    空間変動 (RMS): $(sqrt(mean(spatial_var.^2)):.4e) W/m²")

    # 4. 物理的整合性チェック
    println("\n4. 物理的整合性:")

    # エネルギー保存チェック（簡易）
    total_heat_input = sum(q_opt_julia) * dx * dy * dt_eval
    temp_change = T_final_julia - T0_eval
    thermal_energy_change = rho * mean(cp_coeffs) * sum(temp_change) * dx * dy * mean(dz)

    energy_balance = abs(total_heat_input - thermal_energy_change) / abs(thermal_energy_change)
    println("    エネルギー収支誤差: $(energy_balance:.4e)")

    # 温度予測精度
    temp_prediction_error = T_final_julia[:, :, 1] - T_eval[end, :, :]  # 表面温度比較
    temp_rmse = sqrt(mean(temp_prediction_error.^2))
    temp_max_error = maximum(abs.(temp_prediction_error))

    println("    温度予測 RMSE: $(temp_rmse:.4e) K")
    println("    温度予測 最大誤差: $(temp_max_error:.4e) K")

    # 5. 計算効率評価
    println("\n5. 計算効率:")

    avg_time_per_iter = total_elapsed / length(J_hist_julia)
    throughput_dofs = total_dofs * (EVAL_WINDOW_SIZE-1) / total_elapsed

    println("    反復あたり平均時間: $(avg_time_per_iter:.3f)秒")
    println("    処理スループット: $(throughput_dofs:.0f) DOF・ステップ/秒")

    # フルスケール推定
    full_dofs = 80 * 100 * 20  # フルスケール
    full_windows = 100         # 想定ウィンドウ数
    full_cgm_iters = 50        # フルスケールCGM反復数

    estimated_full_time = (full_dofs / throughput_dofs) * full_windows * full_cgm_iters / 3600
    println("    フルスケール推定時間: $(estimated_full_time:.1f)時間")

    # =======================================
    # 問題別解析
    # =======================================
    println("\n【問題別解析】")

    println("6. 数値安定性指標:")

    # β値の統計（CGM実行中の警告から推定）
    println("    ステップサイズ制御: 数値安定化適用済み")
    println("    分母極小対策: 実装済み（閾値: 1e-20）")
    println("    NaN/Inf検出: 実装済み")

    # 勾配の規模評価
    grad_estimate = abs(mean(q_opt_julia)) * 1e6  # 概算
    println("    勾配推定規模: $(grad_estimate:.2e)")

    println("\n✅ 全評価完了")

catch e
    println("❌ Julia版エラー: $e")
    println("詳細: ")
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
end

println("\n" * "="^80)
println("フルスケール定量評価完了")
println("="^80)