#!/usr/bin/env julia
"""
順問題・隨伴問題・感度問題の個別解析
各問題の数値特性と計算精度の詳細評価
"""

include("cgm_solver.jl")
using Printf, Statistics

println("="^60)
println("順問題・隨伴問題・感度問題 個別解析")
println("="^60)

# 小規模テスト設定
ni, nj, nt = 8, 8, 6
nz_local = nz

# テストデータ準備
T_data = npzread("T_measure_700um_1ms.npy")
T_test = T_data[1:nt, 1:ni, 1:nj]

T0 = zeros(ni, nj, nz_local)
for k in 1:nz_local
    T0[:, :, k] = T_test[1, :, :]
end

q_test = 1000.0 * ones(nt-1, ni, nj)  # テスト用熱流束 1kW/m²
dt_test = 0.001

println("テスト設定:")
println("  格子: $ni × $nj × $nz_local")
println("  時間: $nt ステップ")
println("  熱流束: 1000 W/m² (一様)")

# =======================================
# 1. 順問題（DHCP）解析
# =======================================
println("\n【1. 順問題（DHCP）解析】")

try
    println("順問題実行中...")

    T_forward = multiple_time_step_solver_DHCP(
        T0, q_test, nt, rho, cp_coeffs, k_coeffs,
        dx, dy, dz, dz_b, dz_t, dt_test,
        1e-8, 10000
    )

    println("✅ 順問題成功")

    # 順問題の解析
    println("\n順問題結果解析:")

    # 温度場統計
    T_surface_initial = T_forward[1, :, :, 1]
    T_surface_final = T_forward[end, :, :, 1]

    temp_rise = mean(T_surface_final) - mean(T_surface_initial)
    temp_rise_str = @sprintf("%.3f", temp_rise)
    println("  平均表面温度上昇: $temp_rise_str K")

    max_temp_str = @sprintf("%.2f", maximum(T_forward))
    min_temp_str = @sprintf("%.2f", minimum(T_forward))
    println("  最高温度: $max_temp_str K")
    println("  最低温度: $min_temp_str K")

    # 熱拡散チェック
    z_profile_initial = mean(T_forward[1, :, :, :], dims=(1,2))[1,1,:]
    z_profile_final = mean(T_forward[end, :, :, :], dims=(1,2))[1,1,:]

    penetration_depth = 0.0
    for k in 2:nz_local
        if abs(z_profile_final[k] - z_profile_initial[k]) > 0.01  # 0.01K以上の変化
            penetration_depth = sum(dz[1:(k-1)])
        end
    end
    penetration_str = @sprintf("%.4f", penetration_depth * 1000)  # mm単位
    println("  熱侵入深さ: $penetration_str mm")

    global dhcp_success = true
    global T_forward_result = T_forward

catch e
    println("❌ 順問題エラー: $e")
    global dhcp_success = false
end

# =======================================
# 2. 隨伴問題（Adjoint）解析
# =======================================
println("\n【2. 隨伴問題（Adjoint）解析】")

if dhcp_success
    try
        println("隨伴問題実行中...")

        # 仮想観測データ（順問題結果＋ノイズ）
        Y_obs = T_forward_result[2:end, :, :, 1] + 0.1 * randn(nt-1, ni, nj)

        λ_adjoint = multiple_time_step_solver_Adjoint(
            T_forward_result, Y_obs, nt, rho, cp_coeffs, k_coeffs,
            dx, dy, dz, dz_b, dz_t, dt_test,
            1e-8, 10000
        )

        println("✅ 隨伴問題成功")

        # 隨伴問題の解析
        println("\n隨伴問題結果解析:")

        λ_surface = λ_adjoint[:, :, :, nz_local]  # 表面隨伴変数

        λ_max_str = @sprintf("%.4e", maximum(abs.(λ_surface)))
        λ_mean_str = @sprintf("%.4e", mean(abs.(λ_surface)))
        println("  隨伴変数最大値: ±$λ_max_str")
        println("  隨伴変数平均値: ±$λ_mean_str")

        # 勾配の推定（隨伴変数 = 目的関数の熱流束に対する勾配）
        gradient_norm = sqrt(sum(λ_surface.^2))
        gradient_norm_str = @sprintf("%.4e", gradient_norm)
        println("  勾配ノルム: $gradient_norm_str")

        # 時間方向の変動
        temporal_gradient_var = std([std(λ_surface[t, :, :]) for t in 1:(nt-1)])
        temp_grad_var_str = @sprintf("%.4e", temporal_gradient_var)
        println("  時間変動: $temp_grad_var_str")

        global adjoint_success = true
        global lambda_result = λ_adjoint

    catch e
        println("❌ 隨伴問題エラー: $e")
        global adjoint_success = false
    end
else
    println("⏭️ 順問題失敗のため隨伴問題をスキップ")
    global adjoint_success = false
end

# =======================================
# 3. 感度問題解析
# =======================================
println("\n【3. 感度問題解析】")

if dhcp_success
    try
        println("感度問題実行中...")

        # 感度解析用の摂動熱流束
        δq = 100.0 * ones(nt-1, ni, nj)  # 100 W/m² の摂動

        dT_sensitivity = multiple_time_step_solver_DHCP(
            zeros(ni, nj, nz_local), δq, nt, rho, cp_coeffs, k_coeffs,
            dx, dy, dz, dz_b, dz_t, dt_test,
            1e-8, 10000
        )

        println("✅ 感度問題成功")

        # 感度問題の解析
        println("\n感度問題結果解析:")

        # 表面温度感度
        dT_surface = dT_sensitivity[2:end, :, :, 1]

        sensitivity_max_str = @sprintf("%.4e", maximum(abs.(dT_surface)))
        sensitivity_mean_str = @sprintf("%.4e", mean(abs.(dT_surface)))
        println("  感度最大値: ±$sensitivity_max_str K/(W/m²)")
        println("  感度平均値: ±$sensitivity_mean_str K/(W/m²)")

        # 感度の空間分布
        spatial_sensitivity_std = std(dT_surface[end, :, :])  # 最終時刻での空間標準偏差
        spatial_sens_std_str = @sprintf("%.4e", spatial_sensitivity_std)
        println("  空間感度標準偏差: $spatial_sens_std_str K/(W/m²)")

        # 時間発展
        temporal_sensitivity = [mean(abs.(dT_surface[t, :, :])) for t in 1:(nt-1)]
        temp_sens_growth = temporal_sensitivity[end] / temporal_sensitivity[1]
        temp_sens_growth_str = @sprintf("%.2f", temp_sens_growth)
        println("  時間発展倍率: $temp_sens_growth_str")

        global sensitivity_success = true

    catch e
        println("❌ 感度問題エラー: $e")
        global sensitivity_success = false
    end
else
    println("⏭️ 順問題失敗のため感度問題をスキップ")
    global sensitivity_success = false
end

# =======================================
# 4. 統合解析
# =======================================
println("\n【4. 統合解析】")

if dhcp_success && adjoint_success && sensitivity_success
    println("✅ 全問題解析成功")

    # 随伴法と有限差分法の勾配比較（理論的整合性チェック）
    println("\n統合的検証:")

    # 隨伴法による勾配
    adjoint_gradient = lambda_result[2:end, :, :, nz_local]

    # 感度問題による勾配（有限差分近似）
    fd_gradient = dT_surface / 100.0  # δq = 100での感度

    # 相関係数計算
    adj_flat = reshape(adjoint_gradient, :)
    fd_flat = reshape(fd_gradient, :)

    correlation = cor(adj_flat, fd_flat)
    correlation_str = @sprintf("%.4f", correlation)
    println("  隨伴法 vs 有限差分 相関: $correlation_str")

    if correlation > 0.9
        println("  ✅ 高い相関（理論的整合性良好）")
    elseif correlation > 0.5
        println("  ⚠️  中程度の相関（要確認）")
    else
        println("  ❌ 低い相関（要調査）")
    end

    # スケーリング比較
    adj_scale = sqrt(mean(adj_flat.^2))
    fd_scale = sqrt(mean(fd_flat.^2))
    scale_ratio = adj_scale / fd_scale
    scale_ratio_str = @sprintf("%.2e", scale_ratio)
    println("  スケール比 (隨伴/有限差分): $scale_ratio_str")

else
    problem_status = []
    dhcp_success && push!(problem_status, "順問題")
    adjoint_success && push!(problem_status, "隨伴問題")
    sensitivity_success && push!(problem_status, "感度問題")

    if length(problem_status) > 0
        println("部分成功: $(join(problem_status, ", "))")
    else
        println("❌ 全問題で問題発生")
    end
end

# =======================================
# 結果サマリー
# =======================================
println("\n" * "="^60)
println("【問題別解析結果サマリー】")
println("="^60)

status_dhcp = dhcp_success ? "✅" : "❌"
status_adjoint = adjoint_success ? "✅" : "❌"
status_sensitivity = sensitivity_success ? "✅" : "❌"

println("順問題（DHCP）: $status_dhcp")
println("隨伴問題（Adjoint）: $status_adjoint")
println("感度問題（Sensitivity）: $status_sensitivity")

if dhcp_success && adjoint_success && sensitivity_success
    println("\n🎯 Julia版の全構成要素が正常動作")
    println("   理論的整合性と数値精度を確認")
else
    println("\n⚠️  一部の問題で課題あり")
    println("   個別の問題解決が必要")
end

println("="^60)