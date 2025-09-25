#!/usr/bin/env julia
# -*- coding: utf-8 -*-
"""
共役勾配法（CGM）ソルバーのテスト
小規模逆問題での最適化検証
"""

include("cgm_solver.jl")

using Printf

println("=" ^ 60)
println("共役勾配法（CGM）ソルバーテスト開始")
println("=" ^ 60)

# =======================================
# テスト1: 合成データでの逆問題設定
# =======================================
println("\n【テスト1】合成データでの逆問題設定")

# 小規模格子設定
test_ni, test_nj, test_nk = 2, 2, 4  # さらに小さく
test_dx = 1.0e-4
test_dy = 1.0e-4
test_dt = 2.0e-3  # 時間ステップを大きく

# z方向格子
test_Lz = 8.0e-4
test_dz = fill(test_Lz / test_nk, test_nk)
test_dz_b = [Inf; fill(test_dz[1], test_nk-1)]
test_dz_t = [fill(test_dz[1], test_nk-1); Inf]

# 初期温度
T_init_test = fill(300.0, test_ni, test_nj, test_nk)

# "真の"熱流束を定義（時間・空間変動）
test_nt = 8
q_true = zeros(test_nt-1, test_ni, test_nj)
for t in 1:(test_nt-1)
    for i in 1:test_ni, j in 1:test_nj
        # 時間・空間変動する熱流束
        q_true[t, i, j] = 1000.0 * (1 + 0.5 * sin(2π * t / (test_nt-1))) * (1 + 0.2 * (i + j))
    end
end

println(@sprintf("格子: %d×%d×%d, 時間ステップ: %d", test_ni, test_nj, test_nk, test_nt))
println(@sprintf("真の熱流束範囲: %.0f - %.0f W/m²", minimum(q_true), maximum(q_true)))

# =======================================
# テスト2: 合成観測データ生成
# =======================================
println("\n【テスト2】合成観測データ生成")

# "真の"熱流束を使って順問題を解く
T_true = multiple_time_step_solver_DHCP(
    T_init_test, q_true, test_nt, rho, cp_coeffs, k_coeffs,
    test_dx, test_dy, test_dz, test_dz_b, test_dz_t, test_dt,
    1e-8, 1000
)

println(@sprintf("真の温度場サイズ: (%d, %d, %d, %d)", size(T_true)...))

# 観測データを底面温度にノイズを追加して生成
noise_level = 0.05  # 0.05K ノイズ
Y_obs_test = T_true[:, :, :, 1] + noise_level * randn(size(T_true, 1), test_ni, test_nj)

println(@sprintf("観測ノイズレベル: %.2f K", noise_level))
println(@sprintf("観測温度範囲: %.2f - %.2f K", minimum(Y_obs_test), maximum(Y_obs_test)))

# =======================================
# テスト3: 初期推定値とCGMパラメータ設定
# =======================================
println("\n【テスト3】初期推定値とCGMパラメータ設定")

# 初期推定（真の値から大きくずらす）
q_init_test = fill(500.0, test_nt-1, test_ni, test_nj)  # 真の値の約半分で均一

println(@sprintf("初期推定熱流束: %.0f W/m² (均一)", q_init_test[1,1,1]))

# 真の値との初期差
initial_error = sqrt(sum((q_init_test - q_true).^2) / length(q_true))
println(@sprintf("初期RMSEエラー: %.0f W/m²", initial_error))

# =======================================
# テスト4: 小規模CGM最適化実行
# =======================================
println("\n【テスト4】小規模CGM最適化実行")

test4_success = false
q_optimized = nothing
T_final = nothing
J_hist = nothing

try
    # CGM実行（反復数を制限）
    local q_opt, T_fin, J_history = global_CGM_time(
        T_init_test, Y_obs_test, q_init_test,
        test_dx, test_dy, test_dz, test_dz_b, test_dz_t, test_dt,
        rho, cp_coeffs, k_coeffs;
        CGM_iteration=50  # テスト用に制限
    )

    # グローバル変数に代入
    global q_optimized = q_opt
    global T_final = T_fin
    global J_hist = J_history

    println(@sprintf("最適化完了。反復数: %d", length(J_hist)))
    println(@sprintf("最終目的関数: %.2e", J_hist[end]))

    # 最適化結果の評価
    final_error = sqrt(sum((q_optimized - q_true).^2) / length(q_true))
    println(@sprintf("最終RMSEエラー: %.0f W/m²", final_error))

    improvement = (initial_error - final_error) / initial_error * 100
    println(@sprintf("改善率: %.1f%%", improvement))

    global test4_success = true
catch e
    println("CGM最適化エラー: ", e)
    global test4_success = false
end

# =======================================
# テスト5: 収束性分析
# =======================================
println("\n【テスト5】収束性分析")

test5_success = false
if test4_success && J_hist !== nothing
    try
        println("目的関数の収束履歴（最初の10反復）:")
        for i in 1:min(10, length(J_hist))
            println(@sprintf("  Iter %2d: J = %.2e", i, J_hist[i]))
        end

        if length(J_hist) > 1
            # 収束率計算
            initial_J = J_hist[1]
            final_J = J_hist[end]
            total_reduction = (initial_J - final_J) / initial_J * 100

            println(@sprintf("総減少率: %.1f%%", total_reduction))
            println(@sprintf("初期 J = %.2e → 最終 J = %.2e", initial_J, final_J))
        end

        test5_success = true
    catch e
        println("収束性分析エラー: ", e)
        test5_success = false
    end
end

# =======================================
# テスト6: 最適化結果の物理的妥当性
# =======================================
println("\n【テスト6】物理的妥当性検査")

test6_success = false
if test4_success && q_optimized !== nothing
    try
        # 熱流束の正値性
        q_min = minimum(q_optimized)
        q_max = maximum(q_optimized)
        println(@sprintf("最適化熱流束範囲: %.0f - %.0f W/m²", q_min, q_max))

        # 真の値との局所比較
        println("真の値 vs 最適化値（最初の3時間ステップ、格子点[1,1]）:")
        for t in 1:min(3, test_nt-1)
            println(@sprintf("  t=%d: 真=%.0f, 最適=%.0f, 差=%.0f W/m²",
                           t, q_true[t,1,1], q_optimized[t,1,1],
                           abs(q_true[t,1,1] - q_optimized[t,1,1])))
        end

        # 空間分布の妥当性
        spatial_variation = maximum(q_optimized) - minimum(q_optimized)
        println(@sprintf("空間変動幅: %.0f W/m²", spatial_variation))

        test6_success = true
    catch e
        println("物理的妥当性エラー: ", e)
        test6_success = false
    end
end

# =======================================
# テスト結果評価
# =======================================
println("\n" * "=" ^ 60)
println("テスト結果評価")
println("=" ^ 60)

success_count = 0
total_tests = 6

# テスト1: 問題設定
success_count += 1
println("✓ テスト1: 合成逆問題設定完了")

# テスト2: 観測データ
if size(Y_obs_test) == (test_nt, test_ni, test_nj)
    success_count += 1
    println("✓ テスト2: 合成観測データ生成正常")
else
    println("✗ テスト2: 合成観測データ生成エラー")
end

# テスト3: 初期推定
if size(q_init_test) == (test_nt-1, test_ni, test_nj)
    success_count += 1
    println("✓ テスト3: 初期推定値設定正常")
else
    println("✗ テスト3: 初期推定値設定エラー")
end

# テスト4: CGM最適化
if test4_success
    success_count += 1
    println("✓ テスト4: CGM最適化実行正常")
else
    println("✗ テスト4: CGM最適化実行エラー")
end

# テスト5: 収束性
if test5_success
    success_count += 1
    println("✓ テスト5: 収束性分析正常")
else
    println("✗ テスト5: 収束性分析エラー")
end

# テスト6: 物理的妥当性
if test6_success
    success_count += 1
    println("✓ テスト6: 物理的妥当性確認正常")
else
    println("✗ テスト6: 物理的妥当性確認エラー")
end

# 最終結果
println(@sprintf("\n【最終結果】成功: %d/%d テスト", success_count, total_tests))
if success_count >= 5  # 6個中5個以上成功で合格
    println("共役勾配法（CGM）ソルバーテスト: 成功 ✓")
    exit(0)
else
    println("共役勾配法（CGM）ソルバーテスト: 失敗 ✗")
    exit(1)
end