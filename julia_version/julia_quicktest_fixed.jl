#!/usr/bin/env julia
# -*- coding: utf-8 -*-
"""
Julia版フルスケール実行前動作確認テスト（修正版）
"""

println("=" ^ 60)
println("Julia版フルスケール実行前動作確認")
println("=" ^ 60)

println("Julia実行環境:")
println("  Juliaバージョン: ", VERSION)
println("  スレッド数: ", Threads.nthreads())

# =======================================
# ファイル存在確認
# =======================================
println("\n【必要ファイル確認】")
required_files = [
    "cgm_solver.jl",
    "adjoint_solver.jl",
    "dhcp_solver.jl",
    "IHCP_CGM_Julia.jl",
    "metal_thermal_properties.csv",
    "T_measure_700um_1ms.npy"
]

all_files_ok = true
for file in required_files
    if isfile(file)
        println("  ✅ $file")
    else
        println("  ❌ $file")
        all_files_ok = false
    end
end

if !all_files_ok
    println("\n❌ 必要ファイルが不足しています")
    exit(1)
end

# =======================================
# 基本機能読み込み
# =======================================
try
    include("IHCP_CGM_Julia.jl")
    println("\n✅ 基本機能読み込み成功")
catch e
    println("\n❌ 基本機能読み込みエラー: $e")
    exit(1)
end

# =======================================
# 実データ読み込みテスト
# =======================================
println("\n【実データ読み込みテスト】")
try
    using NPZ
    T_measure = npzread("T_measure_700um_1ms.npy")
    nt_total, ni_total, nj_total = size(T_measure)

    println("  ✅ 実データ読み込み成功")
    println("    データ形状: $nt_total × $ni_total × $nj_total")
    println("    温度範囲: $(minimum(T_measure):.2f) - $(maximum(T_measure):.2f) K")

    file_size_mb = stat("T_measure_700um_1ms.npy").size / 1024^2
    println("    ファイルサイズ: $(file_size_mb:.1f) MB")

    global T_test_data = T_measure
    global data_loaded = true
    global nt_data = nt_total
    global ni_data = ni_total
    global nj_data = nj_total
catch e
    println("  ❌ 実データ読み込みエラー: $e")
    global data_loaded = false
end

if !data_loaded
    println("\n❌ 実データが読み込めません")
    exit(1)
end

# =======================================
# 小規模CGMテスト準備
# =======================================
println("\n【小規模CGMテスト】")

# テスト用パラメータ
test_ni, test_nj = 10, 10  # 小さな領域
test_nt = 11  # 10時間ステップ

# テスト用データ切り出し
T_region_test = T_test_data[1:test_nt, 1:test_ni, 1:test_nj]

# 初期温度設定
T0_test = zeros(test_ni, test_nj, nz)
for k in 1:nz
    T0_test[:, :, k] = T_region_test[1, :, :]
end

# 初期熱流束（ゼロ）
q_init_test = zeros(test_nt-1, test_ni, test_nj)

println("  テスト条件:")
println("    格子: $test_ni × $test_nj × $nz")
println("    時間ステップ: $test_nt")
println("    初期温度範囲: $(minimum(T0_test):.2f) - $(maximum(T0_test):.2f) K")

# =======================================
# CGM関数読み込み
# =======================================
try
    include("cgm_solver.jl")
    println("  ✅ CGMソルバー読み込み成功")
catch e
    println("  ❌ CGMソルバー読み込みエラー: $e")
    exit(1)
end

# CGMテスト実行
println("\n  CGM最適化テスト実行中...")
test_start_time = time()

try
    q_opt_test, T_fin_test, J_hist_test = global_CGM_time(
        T0_test, T_region_test, q_init_test,
        dx, dy, dz, dz_b, dz_t, dt/1000,  # ms → s変換
        rho, cp_coeffs, k_coeffs;
        CGM_iteration=10  # テスト用に制限
    )

    test_elapsed = time() - test_start_time

    println("  ✅ CGMテスト成功")
    println("    実行時間: $(test_elapsed:.2f)秒")
    println("    反復数: $(length(J_hist_test))")
    println("    最終目的関数: $(J_hist_test[end]:.2e)")
    println("    熱流束統計:")
    println("      最小値: $(minimum(q_opt_test):.0f) W/m²")
    println("      最大値: $(maximum(q_opt_test):.0f) W/m²")
    println("      平均値: $(mean(q_opt_test):.0f) W/m²")

    global cgm_test_ok = true

catch e
    println("  ❌ CGMテストエラー: $e")
    global cgm_test_ok = false
end

# =======================================
# 性能推定
# =======================================
if cgm_test_ok
    println("\n【フルスケール性能推定】")

    # テスト性能から推定
    test_points = test_ni * test_nj * nz
    test_operations = test_points * (test_nt - 1)
    test_rate = test_operations / test_elapsed

    # フルスケールパラメータ
    full_ni, full_nj = 80, 100
    full_window_size = 50
    num_windows = div(nt_data - full_window_size, 10) + 1
    cgm_iterations_full = 100

    full_operations_per_window = full_ni * full_nj * nz * (full_window_size - 1) * cgm_iterations_full
    estimated_time_per_window = full_operations_per_window / test_rate
    total_estimated_time = estimated_time_per_window * num_windows

    println("  テスト性能:")
    println("    処理レート: $(test_rate:.0f) 格子点×ステップ/秒")
    println("    テスト規模: $test_points 格子点")

    println("  フルスケール推定:")
    println("    処理ウィンドウ数: $num_windows")
    println("    ウィンドウあたり推定時間: $(estimated_time_per_window/60:.1f)分")
    println("    総推定実行時間: $(total_estimated_time/3600:.1f)時間")

    if total_estimated_time < 3600
        println("    ⏱️  推定: 1時間以内で完了")
    elseif total_estimated_time < 8*3600
        println("    ⏰ 推定: $(total_estimated_time/3600:.1f)時間で完了（実用的）")
    else
        println("    ⚠️  推定: $(total_estimated_time/3600:.1f)時間（長時間実行）")
    end
end

# =======================================
# 結果とアドバイス
# =======================================
println("\n" * "=" ^ 60)
println("動作確認完了")
println("=" ^ 60)

if cgm_test_ok
    println("✅ すべてのテストに成功しました")
    println("\n🚀 フルスケール実行準備完了")

    println("\n実行コマンド:")
    println("  ./run_julia_fullscale.sh")
    println("  または")
    println("  JULIA_NUM_THREADS=8 julia julia_full_scale_execution.jl")

    println("\n💡 推奨:")
    println("  - バックグラウンド実行を使用")
    println("  - 十分なディスク容量を確保")
    println("  - 実行中はシステム負荷に注意")

else
    println("❌ テストに失敗しました")
    println("\n🔧 解決方法:")
    println("  - パッケージインストール確認")
    println("  - データファイル配置確認")
    println("  - メモリ容量確認")
end

println("\n" * "=" ^ 60)