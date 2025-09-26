#!/usr/bin/env julia
"""
フルスケールテストのデバッグ版
"""

include("cgm_solver.jl")
using Printf, Statistics

println("フルスケールデバッグテスト開始")

# 段階的実行
try
    println("1. データ読み込み")
    T_data = npzread("T_measure_700um_1ms.npy")
    println("✅ データ読み込み成功: $(size(T_data))")

    println("2. 小領域切り出し")
    T_test = T_data[1:5, 1:5, 1:5]  # 最小規模
    println("✅ 切り出し成功: $(size(T_test))")

    println("3. 初期化")
    ni, nj = 5, 5
    nt = size(T_test, 1)

    T0 = zeros(ni, nj, nz)
    for k in 1:nz
        T0[:, :, k] = T_test[1, :, :]
    end

    q_init = zeros(nt-1, ni, nj)
    println("✅ 初期化完了")

    println("4. CGM実行")
    q_opt, T_final, J_hist = global_CGM_time(
        T0, T_test, q_init,
        dx, dy, dz, dz_b, dz_t, 0.001,
        rho, cp_coeffs, k_coeffs;
        CGM_iteration=3
    )

    println("✅ CGM成功")
    println("結果:")
    J_str = @sprintf("%.2e", J_hist[end])
    q_min_str = @sprintf("%.2e", minimum(q_opt))
    q_max_str = @sprintf("%.2e", maximum(q_opt))
    println("  目的関数: $J_str")
    println("  熱流束範囲: $q_min_str ~ $q_max_str")

catch err
    println("❌ エラー発生: $err")
    println("詳細:")
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
end

println("デバッグテスト完了")