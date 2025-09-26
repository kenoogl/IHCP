#!/usr/bin/env julia

"""
Python版とJulia版の定量的一致性検証
フルサイズ10ステップでの順問題、随伴問題、感度問題の比較
"""

using LinearAlgebra, SparseArrays, NPZ

# 必要なモジュールを読み込み
include("IHCP_CGM_Julia.jl")
include("dhcp_solver.jl")
include("adjoint_solver.jl")
include("cgm_solver.jl")

println("="^80)
println("Python版とJulia版の定量的一致性検証")
println("フルサイズ10ステップ計算での比較テスト")
println("="^80)

# 実データ読み込み
println("\n【ステップ1】実データ読み込み")
T_data = npzread("T_measure_700um_1ms.npy")
println("データ形状: $(size(T_data))")

# フルサイズ格子設定（実データと同じサイズ）
nx, ny, nz = 80, 100, 20
nt_test = 10  # 10ステップでテスト
dx = 0.12e-3  # 0.12mm

# 格子設定
y_positions = [0.12e-3 * (i-1) for i in 1:ny]
z_positions = [0.0001, 0.0002, 0.0003, 0.0005, 0.0008, 0.0013, 0.0021, 0.0034, 0.0055, 0.0089,
               0.0144, 0.0233, 0.0377, 0.0610, 0.0987, 0.1597, 0.2584, 0.4181, 0.6765, 0.7]

dy_values = [y_positions[i] - y_positions[max(1, i-1)] for i in 1:ny]
dz_values = [z_positions[i] - z_positions[max(1, i-1)] for i in 1:nz]

println("格子設定: nx=$nx, ny=$ny, nz=$nz")
println("テストステップ数: $nt_test")

# 初期条件設定（室温）
T_initial = fill(293.15, nx, ny, nz)
println("初期温度: $(T_initial[1,1,1]) K")

# 境界温度データ（実データから抽出）
T_boundary = T_data[1:nt_test, :, :]  # 10ステップ分
println("境界温度範囲: $(minimum(T_boundary)) - $(maximum(T_boundary)) K")

# 初期熱流束（ゼロ）
q_initial = zeros(nx, ny, nt_test)

println("\n【ステップ2】順問題（DHCP）計算")
println("="^50)

# Julia版順問題計算
println("Julia版順問題計算開始...")
julia_start_time = time()

T_result_julia = zeros(nx, ny, nz, nt_test + 1)
T_result_julia[:, :, :, 1] = T_initial

for time_step in 1:nt_test
    println("  時間ステップ $time_step/$nt_test 計算中...")

    # 熱物性値計算
    cp, k = thermal_properties_calculator(T_result_julia[:, :, :, time_step], cp_coeffs, k_coeffs)
    rho_val = fill(rho, nx, ny, nz)  # 一定密度

    # 係数行列とRHS構築
    A_julia, b_julia = coeffs_and_rhs_building_DHCP(
        T_result_julia[:, :, :, time_step],
        q_initial[:, :, time_step],
        rho,
        cp, k,
        dx, mean(dy_values), dz_values,
        dz_b, dz_t, 0.001
    )

    # 線形システム求解
    T_next_vec = A_julia \ b_julia
    T_result_julia[:, :, :, time_step + 1] = reshape(T_next_vec, nx, ny, nz)
end

julia_dhcp_time = time() - julia_start_time
println("Julia版順問題計算時間: $(round(julia_dhcp_time, digits=3))秒")

# 結果保存（Julia版）
npzwrite("julia_dhcp_results_fullsize.npz", Dict(
    "T_result" => T_result_julia,
    "computation_time" => julia_dhcp_time,
    "final_temperature_range" => [minimum(T_result_julia[:,:,:,end]), maximum(T_result_julia[:,:,:,end])]
))

println("Julia版順問題結果:")
println("  最終温度範囲: $(round(minimum(T_result_julia[:,:,:,end]), digits=2)) - $(round(maximum(T_result_julia[:,:,:,end]), digits=2)) K")
println("  温度上昇: $(round(maximum(T_result_julia[:,:,:,end]) - 293.15, digits=2)) K")

println("\n【ステップ3】随伴問題計算")
println("="^50)

# 測定温度との差（目的関数用）
T_measured = T_boundary  # 境界面の測定温度
T_computed_surface = T_result_julia[:, :, 1, 2:end]  # 計算された表面温度

# Julia版随伴問題計算
println("Julia版随伴問題計算開始...")
julia_adjoint_start = time()

adjoint_result_julia = zeros(nx, ny, nz, nt_test + 1)

for time_step in nt_test:-1:1
    println("  随伴時間ステップ $(nt_test - time_step + 1)/$nt_test 計算中...")

    # 熱物性値
    cp, k = thermal_properties_calculator(T_result_julia[:, :, :, time_step], cp_coeffs, k_coeffs)
    rho_val = fill(rho, nx, ny, nz)  # 一定密度

    # 随伴問題の係数行列とRHS（引数順序は後で確認して修正）
    A_adj, b_adj = coeffs_and_rhs_building_Adjoint(
        T_result_julia[:, :, :, time_step],
        adjoint_result_julia[:, :, :, time_step + 1],
        rho,
        cp, k,
        dx, mean(dy_values), dz_values,
        dz_b, dz_t, 0.001
    )

    # 境界面での温度差を随伴問題のソースとして追加
    if time_step <= nt_test
        temp_diff = T_computed_surface[:, :, time_step] - T_measured[time_step, :, :]
        # 表面格子点に温度差を追加
        for i in 1:nx, j in 1:ny
            idx = (i-1)*ny*nz + (j-1)*nz + 1  # 表面(z=1)のインデックス
            b_adj[idx] += 2.0 * temp_diff[i, j]
        end
    end

    # 随伴問題求解
    adj_vec = A_adj \ b_adj
    adjoint_result_julia[:, :, :, time_step] = reshape(adj_vec, nx, ny, nz)
end

julia_adjoint_time = time() - julia_adjoint_start
println("Julia版随伴問題計算時間: $(round(julia_adjoint_time, digits=3))秒")

# 結果保存（Julia版随伴問題）
npzwrite("julia_adjoint_results_fullsize.npz", Dict(
    "adjoint_result" => adjoint_result_julia,
    "computation_time" => julia_adjoint_time,
    "adjoint_range" => [minimum(adjoint_result_julia), maximum(adjoint_result_julia)]
))

println("Julia版随伴問題結果:")
println("  随伴変数範囲: $(round(minimum(adjoint_result_julia), digits=6)) - $(round(maximum(adjoint_result_julia), digits=6))")

println("\n【ステップ4】感度計算")
println("="^50)

# 感度計算（∂T/∂q）
println("Julia版感度計算開始...")
julia_sensitivity_start = time()

sensitivity_julia = zeros(nx, ny, nt_test)

for time_step in 1:nt_test
    for i in 1:nx, j in 1:ny
        # 表面での熱流束に対する温度の感度
        # 随伴解を用いた感度計算
        sensitivity_julia[i, j, time_step] = adjoint_result_julia[i, j, 1, time_step] * dz_values[1] / 2.0
    end
end

julia_sensitivity_time = time() - julia_sensitivity_start
println("Julia版感度計算時間: $(round(julia_sensitivity_time, digits=3))秒")

# 結果保存（Julia版感度）
npzwrite("julia_sensitivity_results_fullsize.npz", Dict(
    "sensitivity" => sensitivity_julia,
    "computation_time" => julia_sensitivity_time,
    "sensitivity_range" => [minimum(sensitivity_julia), maximum(sensitivity_julia)]
))

println("Julia版感度結果:")
println("  感度範囲: $(round(minimum(sensitivity_julia), digits=6)) - $(round(maximum(sensitivity_julia), digits=6))")

println("\n【ステップ5】計算完了サマリー")
println("="^50)
println("Julia版フルサイズ10ステップ計算完了")
println("  順問題計算時間: $(round(julia_dhcp_time, digits=3))秒")
println("  随伴問題計算時間: $(round(julia_adjoint_time, digits=3))秒")
println("  感度計算時間: $(round(julia_sensitivity_time, digits=3))秒")
println("  総計算時間: $(round(julia_dhcp_time + julia_adjoint_time + julia_sensitivity_time, digits=3))秒")

println("\n結果ファイル:")
println("  - julia_dhcp_results_fullsize.npz")
println("  - julia_adjoint_results_fullsize.npz")
println("  - julia_sensitivity_results_fullsize.npz")

println("\n次はPython版で同じ計算を実行してください。")
println("比較用コマンド: python quantitative_comparison_python.py")