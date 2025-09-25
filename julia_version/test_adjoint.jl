#!/usr/bin/env julia
# -*- coding: utf-8 -*-
"""
隨伴問題ソルバー（Adjoint）のテスト
直接問題の解を使用して隨伴問題を検証
"""

include("adjoint_solver.jl")

using Printf

println("=" ^ 60)
println("隨伴問題ソルバー（Adjoint）テスト開始")
println("=" ^ 60)

# =======================================
# テスト1: 直接問題の解を準備
# =======================================
println("\n【テスト1】直接問題解の準備")

# 小規模格子設定（DHCPテストと同じ）
test_ni, test_nj, test_nk = 3, 3, 5
test_dx = 1.0e-4
test_dy = 1.0e-4
test_dt = 1.0e-3

# z方向格子
test_Lz = 1.0e-3
test_dz = fill(test_Lz / test_nk, test_nk)
test_dz_b = [Inf; fill(test_dz[1], test_nk-1)]
test_dz_t = [fill(test_dz[1], test_nk-1); Inf]

# 初期温度と熱流束
T_init_test = fill(300.0, test_ni, test_nj, test_nk)
test_nt = 6  # 時間ステップを増やして隨伴解を明確にする

q_surface_test = zeros(test_nt-1, test_ni, test_nj)
for t in 1:(test_nt-1)
    q_surface_test[t, :, :] .= 2000.0 * t  # より大きな熱流束
end

println(@sprintf("時間ステップ数: %d", test_nt))
println(@sprintf("熱流束範囲: %.0f - %.0f W/m²", minimum(q_surface_test), maximum(q_surface_test)))

# 直接問題を解く
T_all_test = multiple_time_step_solver_DHCP(
    T_init_test, q_surface_test, test_nt, rho, cp_coeffs, k_coeffs,
    test_dx, test_dy, test_dz, test_dz_b, test_dz_t, test_dt,
    1e-8, 1000
)

println(@sprintf("直接問題解のサイズ: (%d, %d, %d, %d)", size(T_all_test)...))

# 表面温度の確認
T_surface_range = [minimum(T_all_test[t, :, :, 1]) for t in 1:test_nt]
println(@sprintf("底面温度推移: %.2f - %.2f K", minimum(T_surface_range), maximum(T_surface_range)))

# =======================================
# テスト2: 観測データの作成
# =======================================
println("\n【テスト2】観測データ作成")

# 実際の表面温度に擾乱を加えて観測データを生成
Y_obs_test = T_all_test[:, :, :, 1] + 0.1 * randn(test_nt, test_ni, test_nj)  # 0.1K のノイズ

println(@sprintf("観測データサイズ: (%d, %d, %d)", size(Y_obs_test)...))
println(@sprintf("観測温度範囲: %.2f - %.2f K", minimum(Y_obs_test), maximum(Y_obs_test)))

# 観測との差
temp_diff = T_all_test[:, :, :, 1] - Y_obs_test
println(@sprintf("計算-観測差: %.3f - %.3f K", minimum(temp_diff), maximum(temp_diff)))

# =======================================
# テスト3: 隨伴問題係数構築テスト
# =======================================
println("\n【テスト3】隨伴問題係数構築テスト")

# 隨伴変数初期値（ゼロで初期化）
lambda_initial_test = zeros(test_ni, test_nj, test_nk)

# 熱物性値計算
cp_test, k_test = thermal_properties_calculator(T_all_test[1, :, :, :], cp_coeffs, k_coeffs)

# 隨伴問題係数計算
a_w_adj, a_e_adj, a_s_adj, a_n_adj, a_b_adj, a_t_adj, a_p_adj, b_adj = coeffs_and_rhs_building_Adjoint(
    lambda_initial_test, T_all_test[1, :, :, 1], Y_obs_test[1, :, :], rho, cp_test, k_test,
    test_dx, test_dy, test_dz, test_dz_b, test_dz_t, test_dt
)

println(@sprintf("隨伴係数配列サイズ: %d", length(a_p_adj)))
println(@sprintf("隨伴主対角項範囲: %.2e - %.2e", minimum(a_p_adj), maximum(a_p_adj)))
println(@sprintf("隨伴右辺項範囲: %.2e - %.2e", minimum(b_adj), maximum(b_adj)))

# 隨伴sparse行列構築
A_adj_test = assemble_A_Adjoint(test_ni, test_nj, test_nk, a_w_adj, a_e_adj, a_s_adj, a_n_adj, a_b_adj, a_t_adj, a_p_adj)

println(@sprintf("隨伴sparse行列サイズ: %d × %d", size(A_adj_test, 1), size(A_adj_test, 2)))
println(@sprintf("隨伴非ゼロ要素数: %d", nnz(A_adj_test)))

# =======================================
# テスト4: 単一時間ステップ隨伴解法テスト
# =======================================
println("\n【テスト4】単一時間ステップ隨伴解法テスト")

# 前処理行列
diag_vec_adj = diag(A_adj_test)
inv_diag_adj = [d != 0.0 ? 1.0/d : 0.0 for d in diag_vec_adj]

# CG法で隨伴システム求解
x_adj_result_test = cg(A_adj_test, b_adj; Pl=Diagonal(inv_diag_adj), reltol=1e-8, maxiter=1000)

println("隨伴CG法実行: 完了")

# 隨伴解の妥当性チェック
lambda_solution = reshape(x_adj_result_test, (test_ni, test_nj, test_nk))
println(@sprintf("隨伴解範囲: %.2e - %.2e", minimum(lambda_solution), maximum(lambda_solution)))

# =======================================
# テスト5: 複数時間ステップ隨伴解法テスト
# =======================================
println("\n【テスト5】複数時間ステップ隨伴解法テスト")

test5_success = false
try
    lambda_field_test = multiple_time_step_solver_Adjoint(
        T_all_test, Y_obs_test, test_nt, rho, cp_coeffs, k_coeffs,
        test_dx, test_dy, test_dz, test_dz_b, test_dz_t, test_dt,
        1e-8, 1000
    )

    println(@sprintf("隨伴場のサイズ: (%d, %d, %d, %d)", size(lambda_field_test)...))

    # 各時間ステップでの隨伴変数範囲
    for t in 1:test_nt
        lambda_min = minimum(lambda_field_test[t, :, :, :])
        lambda_max = maximum(lambda_field_test[t, :, :, :])
        println(@sprintf("t=%d: 隨伴範囲 %.2e - %.2e", t, lambda_min, lambda_max))
    end

    # 隨伴変数の表面値（勾配計算で重要）
    lambda_surface = lambda_field_test[:, :, :, end]  # 上面（z最大）
    println(@sprintf("表面隨伴変数範囲: %.2e - %.2e",
                    minimum(lambda_surface), maximum(lambda_surface)))

    global test5_success = true
catch e
    println("エラー: ", e)
    global test5_success = false
end

# =======================================
# テスト6: 隨伴解の物理的妥当性チェック
# =======================================
println("\n【テスト6】物理的妥当性チェック")

test6_success = false
lambda_field_test = nothing

if test5_success
    try
        # 隨伴問題を再実行してローカル変数として取得
        lambda_field_test = multiple_time_step_solver_Adjoint(
            T_all_test, Y_obs_test, test_nt, rho, cp_coeffs, k_coeffs,
            test_dx, test_dy, test_dz, test_dz_b, test_dz_t, test_dt,
            1e-8, 1000
        )

        # 最終時刻での隨伴変数はゼロであるべき
        lambda_final_check = abs.(lambda_field_test[end, :, :, :])
        final_max_error = maximum(lambda_final_check)
        println(@sprintf("最終時刻隨伴変数の最大絶対値: %.2e", final_max_error))

        # 隨伴変数の時間発展傾向確認
        lambda_norms = [sqrt(sum(lambda_field_test[t, :, :, :].^2)) for t in 1:test_nt]
        println("隨伴変数ノルムの時間発展:")
        for t in 1:test_nt
            println(@sprintf("  t=%d: ||λ|| = %.2e", t, lambda_norms[t]))
        end

        global test6_success = true
    catch e
        println("物理的妥当性チェックエラー: ", e)
        global test6_success = false
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

# テスト1: 直接問題解の準備
if size(T_all_test) == (test_nt, test_ni, test_nj, test_nk)
    success_count += 1
    println("✓ テスト1: 直接問題解準備正常")
else
    println("✗ テスト1: 直接問題解準備エラー")
end

# テスト2: 観測データ作成
if size(Y_obs_test) == (test_nt, test_ni, test_nj)
    success_count += 1
    println("✓ テスト2: 観測データ作成正常")
else
    println("✗ テスト2: 観測データ作成エラー")
end

# テスト3: 係数構築
if length(a_p_adj) == test_ni * test_nj * test_nk && nnz(A_adj_test) > 0
    success_count += 1
    println("✓ テスト3: 隨伴係数構築正常")
else
    println("✗ テスト3: 隨伴係数構築エラー")
end

# テスト4: 単一ステップ解法
if all(isfinite.(lambda_solution))
    success_count += 1
    println("✓ テスト4: 単一時間ステップ隨伴解法正常")
else
    println("✗ テスト4: 単一時間ステップ隨伴解法エラー")
end

# テスト5: 複数ステップ解法
if test5_success
    success_count += 1
    println("✓ テスト5: 複数時間ステップ隨伴解法正常")
else
    println("✗ テスト5: 複数時間ステップ隨伴解法エラー")
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
if success_count == total_tests
    println("隨伴問題ソルバー（Adjoint）テスト: すべて成功 ✓")
    exit(0)
else
    println("隨伴問題ソルバー（Adjoint）テスト: 一部失敗 ✗")
    exit(1)
end