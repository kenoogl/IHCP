#!/usr/bin/env julia
# -*- coding: utf-8 -*-
"""
基本テスト: パッケージを使わずに基本機能を検証
"""

using LinearAlgebra
using CSV
using DataFrames
using Polynomials
using Printf

println("=" ^ 50)
println("基本機能テスト開始")
println("=" ^ 50)

# =======================================
# テスト1: 熱物性値データ読み込み
# =======================================
println("\n【テスト1】熱物性値データ読み込み")

thermal_properties_file_path = "metal_thermal_properties.csv"
sus304_data = CSV.read(thermal_properties_file_path, DataFrame)

sus304_temp = sus304_data[!, "Temperature/K"]
sus304_rho = sus304_data[!, "Density"]
sus304_cp = sus304_data[!, "Specific_Heat"]
sus304_k = sus304_data[!, "Thermal_Conductivity"]

println("データ行数: ", nrow(sus304_data))
println("温度範囲: ", (minimum(sus304_temp), maximum(sus304_temp)))
println("密度範囲: ", (minimum(sus304_rho), maximum(sus304_rho)))

# =======================================
# テスト2: 多項式フィッティング
# =======================================
println("\n【テスト2】多項式フィッティング")

rho_poly = fit(sus304_temp, sus304_rho, 3)
cp_poly = fit(sus304_temp, sus304_cp, 3)
k_poly = fit(sus304_temp, sus304_k, 3)

# 係数をPythonと同じ順序で取得
rho_coeffs = reverse(coeffs(rho_poly))
cp_coeffs = reverse(coeffs(cp_poly))
k_coeffs = reverse(coeffs(k_poly))

println("密度係数: ", rho_coeffs)
println("比熱係数: ", cp_coeffs)
println("熱伝導率係数: ", k_coeffs)

# =======================================
# テスト3: 多項式評価関数
# =======================================
println("\n【テスト3】多項式評価関数")

function polyval_julia(coeffs::Vector{Float64}, x::Float64)
    result = 0.0
    n = length(coeffs)
    for i in 1:n
        result += coeffs[i] * x^(n - i)
    end
    return result
end

# 基準温度での計算
T_ref = 225.0 + 273.15
rho_calc = polyval_julia(rho_coeffs, T_ref)
cp_calc = polyval_julia(cp_coeffs, T_ref)
k_calc = polyval_julia(k_coeffs, T_ref)

println(@sprintf("基準温度 T = %.2f K:", T_ref))
println(@sprintf("  密度 ρ = %.2f kg/m³", rho_calc))
println(@sprintf("  比熱 cp = %.2f J/(kg·K)", cp_calc))
println(@sprintf("  熱伝導率 k = %.2f W/(m·K)", k_calc))

# =======================================
# テスト4: 格子パラメータ計算
# =======================================
println("\n【テスト4】格子パラメータ")

nz = 20
dx = 0.12e-3
dy = dx * sind(80) / sind(45)
Lz = 0.5e-3
stretch_factor = 3

z_faces_temp = LinRange(1.0, 0.0, nz + 1)
z_faces = Lz .- (Lz / (exp(stretch_factor) - 1)) .* (exp.(stretch_factor .* z_faces_temp) .- 1)

z_centers = zeros(nz)
z_centers[1] = z_faces[1]
z_centers[end] = z_faces[end]
z_centers[2:end-1] = (z_faces[2:end-2] + z_faces[3:end-1]) / 2

dz = diff(z_faces)

println(@sprintf("格子数 nz = %d", nz))
println(@sprintf("dx = %.6e m, dy = %.6e m", dx, dy))
println(@sprintf("z方向範囲: %.6e - %.6e m", z_faces[1], z_faces[end]))
println(@sprintf("格子サイズ範囲: %.6e - %.6e m", minimum(dz), maximum(dz)))

# =======================================
# テスト5: 小規模熱物性値計算
# =======================================
println("\n【テスト5】熱物性値計算（並列処理なし）")

# 小規模テスト配列
test_temps = Array{Float64}(undef, 2, 2, 3)
for i in 1:2, j in 1:2, k in 1:3
    test_temps[i, j, k] = 300.0 + 20.0 * (i + j + k)
end

println("テスト温度配列:")
println(test_temps)

# 逐次処理版
cp_result = Array{Float64}(undef, size(test_temps))
k_result = Array{Float64}(undef, size(test_temps))

for i in 1:2, j in 1:2, k in 1:3
    T = test_temps[i, j, k]
    cp_result[i, j, k] = polyval_julia(cp_coeffs, T)
    k_result[i, j, k] = polyval_julia(k_coeffs, T)
end

println("比熱結果:")
println(cp_result)
println("熱伝導率結果:")
println(k_result)

# =======================================
# 成功判定
# =======================================
success_count = 0
total_tests = 5

println("\n" * "=" ^ 50)
println("テスト結果検証")
println("=" ^ 50)

# テスト1: データ読み込み成功
if nrow(sus304_data) > 0
    success_count += 1
    println("✓ テスト1: データ読み込み成功")
else
    println("✗ テスト1: データ読み込み失敗")
end

# テスト2: 係数の妥当性
if length(rho_coeffs) == 4 && all(isfinite.(rho_coeffs))
    success_count += 1
    println("✓ テスト2: 多項式フィッティング成功")
else
    println("✗ テスト2: 多項式フィッティング失敗")
end

# テスト3: 基準値の妥当性
if 7000 < rho_calc < 8000 && 400 < cp_calc < 700 && 10 < k_calc < 30
    success_count += 1
    println("✓ テスト3: 基準値計算が妥当範囲")
else
    println("✗ テスト3: 基準値計算が範囲外")
end

# テスト4: 格子パラメータ
if length(z_faces) == nz + 1 && length(dz) == nz
    success_count += 1
    println("✓ テスト4: 格子パラメータ正常")
else
    println("✗ テスト4: 格子パラメータエラー")
end

# テスト5: 熱物性値計算
if all(isfinite.(cp_result)) && all(isfinite.(k_result))
    success_count += 1
    println("✓ テスト5: 熱物性値計算成功")
else
    println("✗ テスト5: 熱物性値計算失敗")
end

println(@sprintf("\n【最終結果】成功: %d/%d テスト", success_count, total_tests))
if success_count == total_tests
    println("すべてのテストに成功しました ✓")
    exit(0)
else
    println("一部のテストに失敗しました ✗")
    exit(1)
end