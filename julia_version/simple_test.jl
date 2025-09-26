#!/usr/bin/env julia
"""
簡易Julia動作確認テスト
"""

println("Julia版簡易動作確認テスト")
println("スレッド数: $(Threads.nthreads())")

# 基本機能テスト
try
    include("IHCP_CGM_Julia.jl")
    println("✅ 基本機能読み込み成功")
catch e
    println("❌ 基本機能エラー: $e")
    exit(1)
end

# NPZ読み込みテスト
try
    using NPZ
    data_file = "T_measure_700um_1ms.npy"

    if isfile(data_file)
        println("✅ データファイル存在確認")

        # データ読み込み
        T_data = npzread(data_file)
        data_shape = size(T_data)
        println("✅ データ読み込み成功: $data_shape")

        # 小規模テスト
        test_size = min(10, data_shape[1]), min(10, data_shape[2]), min(10, data_shape[3])
        T_test = T_data[1:test_size[1], 1:test_size[2], 1:test_size[3]]

        temp_range = (minimum(T_test), maximum(T_test))
        println("✅ データ処理成功: 温度範囲 $temp_range K")

        # 熱物性計算テスト
        T_calc_test = zeros(2, 2, 4)
        fill!(T_calc_test, 500.0)

        cp_result, k_result = thermal_properties_calculator(T_calc_test, cp_coeffs, k_coeffs)
        println("✅ 熱物性値計算成功")

        println("\n🎉 Julia版が正常に動作しています！")
        println("フルスケール実行の準備が完了しました")

    else
        println("❌ データファイルが見つかりません: $data_file")
        exit(1)
    end

catch e
    println("❌ テストエラー: $e")
    exit(1)
end