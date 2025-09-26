#!/usr/bin/env julia
# -*- coding: utf-8 -*-
"""
Julia版クイックテストのデバッグ版
"""

println("=" ^ 60)
println("Julia版デバッグテスト")
println("=" ^ 60)

println("Julia実行環境:")
println("  Juliaバージョン: ", VERSION)
println("  スレッド数: ", Threads.nthreads())

# ステップ1: 基本機能テスト
println("\n【ステップ1】基本機能テスト")
try
    include("IHCP_CGM_Julia.jl")
    println("  ✅ IHCP_CGM_Julia.jl 読み込み成功")
catch e
    println("  ❌ IHCP_CGM_Julia.jl 読み込みエラー: $e")
    exit(1)
end

# ステップ2: NPZパッケージテスト
println("\n【ステップ2】NPZパッケージテスト")
try
    using NPZ
    println("  ✅ NPZパッケージ読み込み成功")
catch e
    println("  ❌ NPZパッケージエラー: $e")
    exit(1)
end

# ステップ3: ファイル存在確認
println("\n【ステップ3】データファイル確認")
data_file = "T_measure_700um_1ms.npy"
if isfile(data_file)
    file_size = stat(data_file).size
    println("  ✅ $data_file 存在確認")
    println("    ファイルサイズ: $(file_size / 1024^2:.1f) MB")
else
    println("  ❌ $data_file が見つかりません")
    exit(1)
end

# ステップ4: NPZ読み込みテスト
println("\n【ステップ4】NPZ読み込みテスト")
try
    println("  NPZ読み込み開始...")
    T_data = npzread(data_file)
    println("  ✅ NPZ読み込み成功")

    data_shape = size(T_data)
    println("    データ形状: $data_shape")
    println("    データ型: $(eltype(T_data))")

    if length(data_shape) == 3
        nt, ni, nj = data_shape
        println("    時間フレーム: $nt")
        println("    空間サイズ: $ni × $nj")
    end

    temp_min = minimum(T_data)
    temp_max = maximum(T_data)
    println("    温度範囲: $temp_min - $temp_max K")

catch e
    println("  ❌ NPZ読み込みエラー詳細:")
    println("    エラー型: $(typeof(e))")
    println("    メッセージ: $e")

    # スタックトレース表示
    println("\n    スタックトレース:")
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
    exit(1)
end

# ステップ5: 小規模データテスト
println("\n【ステップ5】小規模データ処理テスト")
try
    # 小さなデータ切り出し
    test_data = T_data[1:10, 1:5, 1:5]
    println("  ✅ データ切り出し成功: $(size(test_data))")

    # 統計計算
    mean_temp = sum(test_data) / length(test_data)
    println("  ✅ 統計計算成功: 平均温度 $mean_temp K")

catch e
    println("  ❌ データ処理エラー: $e")
    exit(1)
end

# ステップ6: 基本計算テスト
println("\n【ステップ6】基本計算機能テスト")
try
    # 熱物性値計算テスト
    T_test = fill(500.0, 2, 2, 4)
    cp_test, k_test = thermal_properties_calculator(T_test, cp_coeffs, k_coeffs)

    println("  ✅ 熱物性値計算成功")
    println("    比熱範囲: $(minimum(cp_test):.1f) - $(maximum(cp_test):.1f) J/(kg·K)")
    println("    熱伝導率範囲: $(minimum(k_test):.2f) - $(maximum(k_test):.2f) W/(m·K)")

catch e
    println("  ❌ 基本計算エラー: $e")
    exit(1)
end

println("\n" * "=" ^ 60)
println("✅ すべてのデバッグテストが成功しました")
println("Julia版の基本機能は正常に動作しています")
println("=" ^ 60)