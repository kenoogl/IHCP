#!/usr/bin/env julia
"""
簡潔なJulia版動作確認テスト
"""

include("cgm_solver.jl")
using NPZ, Statistics

println("簡潔比較テスト開始")

# 最小規模テスト
test_ni, test_nj = 3, 3
test_nt = 4

# テストデータ生成（実データの代わりに人工データ）
T_test = 500.0 .+ 10 * rand(test_nt, test_ni, test_nj)

# 初期温度
T0 = zeros(test_ni, test_nj, nz)
for k in 1:nz
    T0[:, :, k] = T_test[1, :, :]
end

# 初期熱流束
q_init = zeros(test_nt-1, test_ni, test_nj)

dt = 0.001

println("CGM実行開始...")
try
    q_opt, T_final, J_hist = global_CGM_time(
        T0, T_test, q_init,
        dx, dy, dz, dz_b, dz_t, dt,
        rho, cp_coeffs, k_coeffs;
        CGM_iteration=5
    )

    println("✅ 成功")
    J_final = @sprintf("%.2e", J_hist[end])
    q_min = @sprintf("%.2e", minimum(q_opt))
    q_max = @sprintf("%.2e", maximum(q_opt))
    println("目的関数: $J_final")
    println("熱流束範囲: $q_min ~ $q_max")

catch e
    println("❌ エラー: $e")
end

println("テスト完了")