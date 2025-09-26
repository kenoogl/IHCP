#!/usr/bin/env julia
# -*- coding: utf-8 -*-
"""
Julia版クイック実行スクリプト
"""

println("=" ^ 70)
println("Julia版 IHCP-CGM ソルバー クイック実行")
println("=" ^ 70)

# スレッド数確認
println("Julia実行環境:")
println("  Juliaバージョン: ", VERSION)
println("  スレッド数: ", Threads.nthreads())
println("  最大スレッドID: ", Threads.maxthreadid())

# 基本チェック
println("\n【基本チェック】")

# 必要ファイル確認
required_files = [
    "IHCP_CGM_Julia.jl",
    "dhcp_solver.jl",
    "adjoint_solver.jl",
    "cgm_solver.jl",
    "metal_thermal_properties.csv",
    "T_measure_700um_1ms.npy"
]

all_files_exist = true
for file in required_files
    if isfile(file)
        println("  ✅ $file")
    else
        println("  ❌ $file (見つかりません)")
        global all_files_exist = false
    end
end

if !all_files_exist
    println("\n❌ 必要ファイルが不足しています。")
    println("必要なファイルを配置してから再実行してください。")
    exit(1)
end

# パッケージチェック
println("\n【パッケージチェック】")
required_packages = ["NPZ", "IterativeSolvers", "SparseArrays", "LinearAlgebra", "Printf"]

for pkg in required_packages
    try
        eval(Meta.parse("using $pkg"))
        println("  ✅ $pkg")
    catch e
        println("  ❌ $pkg (インストールが必要)")
        println("  実行: julia -e \"using Pkg; Pkg.add(\\\"$pkg\\\")\"")
    end
end

println("\n【実行可能テスト一覧】")
println("1. julia --threads 8 test_basic.jl      # 基本機能テスト")
println("2. julia --threads 8 test_dhcp.jl       # 直接問題ソルバー")
println("3. julia --threads 8 test_adjoint.jl    # 隨伴問題ソルバー")
println("4. julia --threads 8 test_cgm.jl        # CGMアルゴリズム")
println("5. julia --threads 8 real_data_test.jl  # 実データ統合テスト")
println("6. julia --threads 8 benchmark_comparison.jl # 性能ベンチマーク")

println("\n【推奨実行順序】")
println("初回:")
println("  JULIA_NUM_THREADS=8 julia test_basic.jl")
println("実データテスト:")
println("  JULIA_NUM_THREADS=8 julia real_data_test.jl")
println("性能測定:")
println("  JULIA_NUM_THREADS=8 julia benchmark_comparison.jl")

# 環境変数確認
println("\n【環境変数】")
julia_threads = get(ENV, "JULIA_NUM_THREADS", "未設定")
omp_threads = get(ENV, "OMP_NUM_THREADS", "未設定")
println("  JULIA_NUM_THREADS: $julia_threads")
println("  OMP_NUM_THREADS: $omp_threads")

if julia_threads == "未設定"
    println("  💡 推奨: export JULIA_NUM_THREADS=8")
end

println("\n【簡易実行テスト】")
if all_files_exist
    try
        # 基本機能の簡易テスト
        println("基本機能テスト実行中...")
        include("IHCP_CGM_Julia.jl")

        # 小規模データでテスト
        T_test = fill(500.0, 2, 2, 4)  # 500K, 2x2x4格子
        cp_test, k_test = thermal_properties_calculator(T_test, cp_coeffs, k_coeffs)

        println("  ✅ 熱物性値計算: 正常")
        println("  比熱範囲: $(minimum(cp_test):.1f) - $(maximum(cp_test):.1f) J/(kg·K)")
        println("  熱伝導率範囲: $(minimum(k_test):.2f) - $(maximum(k_test):.2f) W/(m·K)")

        println("\n🎉 Julia版が正常に動作しています！")
        println("\n次のステップ:")
        println("JULIA_NUM_THREADS=8 julia real_data_test.jl")

    catch e
        println("  ❌ エラー発生: $e")
        println("\n解決方法:")
        println("1. パッケージインストール確認")
        println("2. ファイルパス確認")
        println("3. データファイル配置確認")
    end
else
    println("必要ファイルが不足しているため、テストをスキップします。")
end

println("\n" * "=" ^ 70)
println("Julia版実行準備完了")
println("=" ^ 70)