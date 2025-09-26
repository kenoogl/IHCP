#!/usr/bin/env julia
"""
julia_full_scale_execution.jlのデバッグ用テストスクリプト
"""

println("デバッグテスト開始")

# 段階的にテスト
try
    println("1. cgm_solver.jl読み込みテスト")
    include("cgm_solver.jl")
    println("✅ cgm_solver.jl読み込み成功")
catch e
    println("❌ cgm_solver.jlエラー: $e")
    exit(1)
end

try
    println("2. NPZパッケージテスト")
    using NPZ
    println("✅ NPZパッケージ読み込み成功")
catch e
    println("❌ NPZパッケージエラー: $e")
    exit(1)
end

try
    println("3. データファイル存在確認")
    if isfile("T_measure_700um_1ms.npy")
        println("✅ データファイル存在")
    else
        println("❌ データファイル不存在")
        exit(1)
    end
catch e
    println("❌ ファイル確認エラー: $e")
    exit(1)
end

try
    println("4. データ読み込みテスト")
    T_data = npzread("T_measure_700um_1ms.npy")
    println("✅ データ読み込み成功: $(size(T_data))")
catch e
    println("❌ データ読み込みエラー: $e")
    exit(1)
end

println("✅ 全デバッグテスト完了")