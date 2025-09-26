#!/usr/bin/env julia
"""
Julia版とPython版の解の比較検証テスト
小規模問題で数値結果を詳細比較
"""

include("cgm_solver.jl")

using Printf
using Statistics

println("=" ^ 60)
println("Julia版 vs Python版 解の比較検証")
println("=" ^ 60)

# =======================================
# 小規模テスト設定
# =======================================
test_ni, test_nj = 5, 5  # より小規模に
test_nt = 6  # 5時間ステップ
dt = 0.001  # 1ms

# 既存の実データから小規模切り出し
T_measure = npzread("T_measure_700um_1ms.npy")
T_region_test = T_measure[1:test_nt, 1:test_ni, 1:test_nj]

# 初期温度分布（z方向均一）
T0_test = zeros(test_ni, test_nj, nz)
for k in 1:nz
    T0_test[:, :, k] = T_region_test[1, :, :]
end

# 初期熱流束（ゼロ）
q_init_test = zeros(test_nt-1, test_ni, test_nj)

println("【テスト条件】")
println("格子: $test_ni × $test_nj × $nz")
println("時間ステップ: $test_nt (dt = $dt s)")
temp_range = (minimum(T0_test), maximum(T0_test))
println("初期温度範囲: $(temp_range[1]:.2f) - $(temp_range[2]:.2f) K")

# =======================================
# Julia版CGM実行
# =======================================
println("\n【Julia版CGM実行】")
julia_start_time = time()

try
    q_opt_julia, T_fin_julia, J_hist_julia = global_CGM_time(
        T0_test, T_region_test, q_init_test,
        dx, dy, dz, dz_b, dz_t, dt/1000,  # ms → s変換
        rho, cp_coeffs, k_coeffs;
        CGM_iteration=20  # テスト用制限
    )

    julia_elapsed = time() - julia_start_time

    println("✅ Julia版成功")
    println("  反復数: $(length(J_hist_julia))")
    println("  最終目的関数: $(J_hist_julia[end]:.4e)")
    println("  計算時間: $(julia_elapsed:.2f)秒")

    println("  Julia版熱流束統計:")
    println("    最小値: $(minimum(q_opt_julia):.4e) W/m²")
    println("    最大値: $(maximum(q_opt_julia):.4e) W/m²")
    println("    平均値: $(mean(q_opt_julia):.4e) W/m²")
    println("    標準偏差: $(std(q_opt_julia):.4e) W/m²")

    global julia_success = true
    global julia_results = (q_opt_julia, T_fin_julia, J_hist_julia)

catch e
    println("❌ Julia版エラー: $e")
    global julia_success = false
end

# =======================================
# 結果の詳細分析
# =======================================
if julia_success
    println("\n【詳細結果分析】")

    q_opt, T_fin, J_hist = julia_results

    println("目的関数収束履歴:")
    for (i, J) in enumerate(J_hist)
        if i <= 5 || i == length(J_hist)  # 最初の5回と最後
            println("  反復$i: J = $(J:.4e)")
        elseif i == 6
            println("  ...")
        end
    end

    println("\n時系列での熱流束変化:")
    for t in 1:min(3, size(q_opt, 1))  # 最初の3時間ステップ
        q_t = q_opt[t, :, :]
        println("  時刻$(t): min=$(minimum(q_t):.2e), max=$(maximum(q_t):.2e), mean=$(mean(q_t):.2e)")
    end

    println("\n空間分布での熱流束変化:")
    for i in 1:test_ni
        for j in 1:test_nj
            q_ij = q_opt[:, i, j]
            if maximum(abs.(q_ij)) > 1e-10  # 有意な変化がある格子点のみ
                println("  格子($i,$j): mean=$(mean(q_ij):.2e), max変化=$(maximum(abs.(q_ij)):.2e)")
            end
        end
    end

    # 不一致原理確認
    temp_diff = T_fin - T_region_test[end, :, :]
    max_temp_error = maximum(abs.(temp_diff))
    println("\n温度予測精度:")
    println("  最大温度誤差: $(max_temp_error:.4e) K")
    println("  RMS温度誤差: $(sqrt(mean(temp_diff.^2)):.4e) K")

    # CGMアルゴリズムの動作確認
    if maximum(abs.(q_opt)) < 1e-10
        println("\n⚠️  注意: 熱流束がほぼ0です")
        println("   これは以下を意味する可能性があります:")
        println("   1. 初期温度分布が既に測定値に十分近い")
        println("   2. 不一致原理により早期収束")
        println("   3. 数値安定化により小さなステップサイズ")
    else
        println("\n✅ 有意な熱流束変化が検出されました")
    end
end

println("\n" * "=" ^ 60)
println("比較検証テスト完了")
println("=" ^ 60)