#!/usr/bin/env julia
# -*- coding: utf-8 -*-
"""
直接問題ソルバー（DHCP）のテスト
小規模問題での動作確認と基本検証
"""

include("dhcp_solver.jl")

using Printf

println("=" ^ 60)
println("直接問題ソルバー（DHCP）テスト開始")
println("=" ^ 60)

# =======================================
# テスト1: 小規模問題設定
# =======================================
println("\n【テスト1】小規模問題設定")

# 小規模格子設定
test_ni, test_nj, test_nk = 3, 3, 5
test_dx = 1.0e-4  # 0.1mm
test_dy = 1.0e-4  # 0.1mm
test_dt = 1.0e-3  # 1ms

# 簡易z方向格子（等間隔）
test_Lz = 1.0e-3  # 1mm
test_dz = fill(test_Lz / test_nk, test_nk)
test_dz_b = [Inf; fill(test_dz[1], test_nk-1)]
test_dz_t = [fill(test_dz[1], test_nk-1); Inf]

println(@sprintf("格子: %d × %d × %d", test_ni, test_nj, test_nk))
println(@sprintf("格子間隔: dx=%.1e, dy=%.1e", test_dx, test_dy))
println(@sprintf("時間ステップ: dt=%.1e s", test_dt))

# =======================================
# テスト2: 初期温度と境界条件設定
# =======================================
println("\n【テスト2】初期温度と境界条件設定")

# 初期温度分布（均一）
T_init_test = fill(300.0, test_ni, test_nj, test_nk)  # 300K
println(@sprintf("初期温度: %.1f K (均一)", T_init_test[1,1,1]))

# 表面熱流束（時間依存、空間均一）
test_nt = 5
q_surface_test = zeros(test_nt-1, test_ni, test_nj)

for t in 1:(test_nt-1)
    # 段階的な熱流束印加（1000 W/m² × ステップ）
    q_surface_test[t, :, :] .= 1000.0 * t
end

println(@sprintf("熱流束範囲: %.0f - %.0f W/m²",
                minimum(q_surface_test), maximum(q_surface_test)))

# =======================================
# テスト3: 熱物性値計算
# =======================================
println("\n【テスト3】熱物性値計算")

# テスト用熱物性値（実際の係数使用）
cp_test, k_test = thermal_properties_calculator(T_init_test, cp_coeffs, k_coeffs)

println(@sprintf("比熱範囲: %.1f - %.1f J/(kg·K)", minimum(cp_test), maximum(cp_test)))
println(@sprintf("熱伝導率範囲: %.2f - %.2f W/(m·K)", minimum(k_test), maximum(k_test)))
println(@sprintf("密度: %.1f kg/m³", rho))

# =======================================
# テスト4: 係数行列構築テスト
# =======================================
println("\n【テスト4】係数行列構築テスト")

# 係数計算
a_w, a_e, a_s, a_n, a_b, a_t, a_p, b = coeffs_and_rhs_building_DHCP(
    T_init_test, q_surface_test[1, :, :], rho, cp_test, k_test,
    test_dx, test_dy, test_dz, test_dz_b, test_dz_t, test_dt
)

println(@sprintf("係数配列サイズ: %d (= %d × %d × %d)", length(a_p), test_ni, test_nj, test_nk))
println(@sprintf("主対角項範囲: %.2e - %.2e", minimum(a_p), maximum(a_p)))
println(@sprintf("右辺項範囲: %.2e - %.2e", minimum(b), maximum(b)))

# sparse行列構築
A_test = assemble_A_DHCP(test_ni, test_nj, test_nk, a_w, a_e, a_s, a_n, a_b, a_t, a_p)

println(@sprintf("sparse行列サイズ: %d × %d", size(A_test, 1), size(A_test, 2)))
println(@sprintf("非ゼロ要素数: %d", nnz(A_test)))
println(@sprintf("充填率: %.2f%%", nnz(A_test) / (size(A_test, 1) * size(A_test, 2)) * 100))

# =======================================
# テスト5: 単一時間ステップ解法テスト
# =======================================
println("\n【テスト5】単一時間ステップ解法テスト")

# 前処理行列
diag_vec = diag(A_test)
inv_diag = [d != 0.0 ? 1.0/d : 0.0 for d in diag_vec]

# 初期解ベクトル
x0_test = reshape(T_init_test, :)

# CG法でシステム求解
x_result_test = cg(A_test, b; Pl=Diagonal(inv_diag), reltol=1e-8, maxiter=1000)

println("CG法実行: 完了")

# 解の妥当性チェック
T_solution = reshape(x_result_test, (test_ni, test_nj, test_nk))
println(@sprintf("解温度範囲: %.2f - %.2f K", minimum(T_solution), maximum(T_solution)))

# 温度上昇の確認
temp_increase = T_solution .- T_init_test
println(@sprintf("温度上昇範囲: %.2f - %.2f K", minimum(temp_increase), maximum(temp_increase)))

# =======================================
# テスト6: 複数時間ステップ解法テスト
# =======================================
println("\n【テスト6】複数時間ステップ解法テスト")

# 複数時間ステップソルバー実行
test6_success = false
try
    T_all_test = multiple_time_step_solver_DHCP(
        T_init_test, q_surface_test, test_nt, rho, cp_coeffs, k_coeffs,
        test_dx, test_dy, test_dz, test_dz_b, test_dz_t, test_dt,
        1e-8, 1000
    )

    println(@sprintf("時間発展解のサイズ: (%d, %d, %d, %d)", size(T_all_test)...))

    # 各時間ステップでの温度範囲
    for t in 1:test_nt
        T_min = minimum(T_all_test[t, :, :, :])
        T_max = maximum(T_all_test[t, :, :, :])
        println(@sprintf("t=%d: 温度範囲 %.2f - %.2f K", t, T_min, T_max))
    end

    # 最終時刻での表面温度（z方向最大）
    T_surface_final = T_all_test[end, :, :, end]
    println(@sprintf("最終表面温度範囲: %.2f - %.2f K",
                    minimum(T_surface_final), maximum(T_surface_final)))

    global test6_success = true
catch e
    println("エラー: ", e)
    global test6_success = false
end

# =======================================
# テスト結果評価
# =======================================
println("\n" * "=" ^ 60)
println("テスト結果評価")
println("=" ^ 60)

success_count = 0
total_tests = 6

# テスト1: 基本設定
success_count += 1
println("✓ テスト1: 小規模問題設定完了")

# テスト2: 境界条件
if size(q_surface_test) == (test_nt-1, test_ni, test_nj)
    success_count += 1
    println("✓ テスト2: 境界条件設定正常")
else
    println("✗ テスト2: 境界条件設定エラー")
end

# テスト3: 熱物性値
if all(isfinite.(cp_test)) && all(isfinite.(k_test))
    success_count += 1
    println("✓ テスト3: 熱物性値計算正常")
else
    println("✗ テスト3: 熱物性値計算エラー")
end

# テスト4: 係数行列
if size(A_test, 1) == test_ni * test_nj * test_nk && nnz(A_test) > 0
    success_count += 1
    println("✓ テスト4: 係数行列構築正常")
else
    println("✗ テスト4: 係数行列構築エラー")
end

# テスト5: 単一ステップ解法
if all(isfinite.(T_solution)) && minimum(T_solution) > 0
    success_count += 1
    println("✓ テスト5: 単一時間ステップ解法正常")
else
    println("✗ テスト5: 単一時間ステップ解法エラー")
end

# テスト6: 複数ステップ解法
if test6_success
    success_count += 1
    println("✓ テスト6: 複数時間ステップ解法正常")
else
    println("✗ テスト6: 複数時間ステップ解法エラー")
end

# 最終結果
println(@sprintf("\n【最終結果】成功: %d/%d テスト", success_count, total_tests))
if success_count == total_tests
    println("直接問題ソルバー（DHCP）テスト: すべて成功 ✓")
    exit(0)
else
    println("直接問題ソルバー（DHCP）テスト: 一部失敗 ✗")
    exit(1)
end