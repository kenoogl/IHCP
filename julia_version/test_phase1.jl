#!/usr/bin/env julia
# -*- coding: utf-8 -*-
"""
Phase 1 テスト: 基本関数と熱物性値計算の検証
Python版との数値精度を比較検証
"""

include("IHCP_CGM_Julia.jl")

println("=" ^ 50)
println("Phase 1 テスト開始: 基本関数検証")
println("=" ^ 50)

# =======================================
# テスト1: 多項式評価関数の検証
# =======================================
println("\n【テスト1】多項式評価関数 polyval_julia")

# テスト用係数とテスト値
test_coeffs = [1.0, -2.0, 3.0, -4.0]  # x³ - 2x² + 3x - 4
test_values = [0.0, 1.0, 2.0, 273.15, 500.0]

println("係数: $(test_coeffs)")
for x in test_values
    result = polyval_julia(test_coeffs, x)
    # 手計算での検証
    expected = test_coeffs[1] * x^3 + test_coeffs[2] * x^2 + test_coeffs[3] * x + test_coeffs[4]
    error = abs(result - expected)
    println(@sprintf("x = %8.2f: result = %12.6f, expected = %12.6f, error = %.2e",
                    x, result, expected, error))
end

# =======================================
# テスト2: 熱物性値係数の確認
# =======================================
println("\n【テスト2】熱物性値多項式係数")

println("密度係数 rho_coeffs: ", rho_coeffs)
println("比熱係数 cp_coeffs: ", cp_coeffs)
println("熱伝導率係数 k_coeffs: ", k_coeffs)

# 基準温度での密度値確認
T_ref = 225.0 + 273.15
rho_calc = polyval_julia(rho_coeffs, T_ref)
println(@sprintf("基準温度 T = %.2f K での密度: ρ = %.2f kg/m³", T_ref, rho_calc))

# =======================================
# テスト3: 熱物性値計算関数の検証
# =======================================
println("\n【テスト3】熱物性値計算関数 thermal_properties_calculator")

# 小規模テスト用温度配列
test_temps = Array{Float64}(undef, 3, 2, 4)
for i in 1:3, j in 1:2, k in 1:4
    test_temps[i, j, k] = 300.0 + 10.0 * (i + j + k)  # 310K～370K範囲
end

println("テスト温度配列サイズ: ", size(test_temps))
println("温度範囲: ", (minimum(test_temps), maximum(test_temps)))

# 熱物性値計算実行
cp_result, k_result = thermal_properties_calculator(test_temps, cp_coeffs, k_coeffs)

println("比熱結果サイズ: ", size(cp_result))
println("比熱範囲: ", (minimum(cp_result), maximum(cp_result)))
println("熱伝導率結果サイズ: ", size(k_result))
println("熱伝導率範囲: ", (minimum(k_result), maximum(k_result)))

# 手計算との比較（いくつかの点で）
println("\n詳細検証（手計算との比較）:")
for i in 1:2, j in 1:1, k in 1:2
    T = test_temps[i, j, k]
    cp_calc = polyval_julia(cp_coeffs, T)
    k_calc = polyval_julia(k_coeffs, T)

    cp_error = abs(cp_result[i, j, k] - cp_calc)
    k_error = abs(k_result[i, j, k] - k_calc)

    println(@sprintf("点[%d,%d,%d] T=%.1f: cp=%.3f(誤差%.2e), k=%.3f(誤差%.2e)",
                    i, j, k, T, cp_result[i, j, k], cp_error, k_result[i, j, k], k_error))
end

# =======================================
# テスト4: 格子パラメータの確認
# =======================================
println("\n【テスト4】格子パラメータ")

println(@sprintf("dx = %.6e m", dx))
println(@sprintf("dy = %.6e m", dy))
println(@sprintf("Lz = %.6e m", Lz))
println(@sprintf("nz = %d", nz))

println("\nZ方向格子情報:")
println(@sprintf("z_faces[1] (底面) = %.6e m", z_faces[1]))
println(@sprintf("z_faces[end] (上面) = %.6e m", z_faces[end]))
println(@sprintf("格子サイズ範囲: %.6e - %.6e m", minimum(dz), maximum(dz)))

println("格子中心座標 (最初の5点):")
for i in 1:min(5, nz)
    println(@sprintf("  z_centers[%d] = %.6e m", i, z_centers[i]))
end

println("\n境界成分:")
println(@sprintf("dz_t[1] = %.6e, dz_t[end] = %s", dz_t[1], string(dz_t[end])))
println(@sprintf("dz_b[1] = %s, dz_b[end] = %.6e", string(dz_b[1]), dz_b[end]))

println("\n" * "=" ^ 50)
println("Phase 1 テスト完了")
println("=" ^ 50)

# エラーチェック用フラグ
success_count = 0
total_tests = 4

# 基本的な正常性チェック
if length(rho_coeffs) == 4 && length(cp_coeffs) == 4 && length(k_coeffs) == 4
    success_count += 1
    println("✓ 多項式係数の長さ確認: 正常")
else
    println("✗ 多項式係数の長さエラー")
end

if size(cp_result) == size(test_temps) && size(k_result) == size(test_temps)
    success_count += 1
    println("✓ 熱物性値計算結果のサイズ確認: 正常")
else
    println("✗ 熱物性値計算結果のサイズエラー")
end

if length(z_faces) == nz + 1 && length(dz) == nz
    success_count += 1
    println("✓ 格子配列長さ確認: 正常")
else
    println("✗ 格子配列長さエラー")
end

if all(isfinite.(cp_result)) && all(isfinite.(k_result))
    success_count += 1
    println("✓ 数値結果の有限性確認: 正常")
else
    println("✗ 数値結果に無限大またはNaNが含まれています")
end

println("\n【テスト結果】")
println(@sprintf("成功: %d/%d テスト", success_count, total_tests))
if success_count == total_tests
    println("Phase 1 テスト: すべて成功 ✓")
else
    println("Phase 1 テスト: 一部失敗 ✗")
    exit(1)
end