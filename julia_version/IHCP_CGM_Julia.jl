# -*- coding: utf-8 -*-
"""
Julia版 逆熱伝導問題（IHCP）共役勾配法（CGM）ソルバー
Pythonオリジナルからの変換版

@author: SHI ZHENGQI (Python版)
@translator: Claude Code (Julia変換)

逆熱伝導問題のスライディングウィンドウ計算プログラム：
1. IRカメラから取得したMATLABファイルの温度データ読み込み
2. SUS304の熱物性値（密度、比熱、熱伝導率）の多項式フィッティング
3. 熱伝導方程式の直接ソルバー（DHCP）
4. 共役勾配法による随伴ソルバー（Adjoint）
5. 表面熱流束の逆解析計算
6. 全時間領域でのスライディングウィンドウ計算
"""

using LinearAlgebra
using SparseArrays
using IterativeSolvers
using CSV
using DataFrames
using MAT
using NPZ
using Polynomials
using Printf

# =======================================
# 熱物性値データ読み込みと多項式フィッティング
# =======================================

"""
SUS304熱物性値データの読み込み
現在は相対パスを使用（元のハードコードパスから変更）
"""
function load_thermal_properties_data()
    # オリジナルのパスをJulia環境用に調整
    thermal_properties_file_path = "metal_thermal_properties.csv"

    sus304_data = CSV.read(thermal_properties_file_path, DataFrame)

    sus304_temp = sus304_data[!, "Temperature/K"]
    sus304_rho = sus304_data[!, "Density"]
    sus304_cp = sus304_data[!, "Specific_Heat"]
    sus304_k = sus304_data[!, "Thermal_Conductivity"]

    return sus304_temp, sus304_rho, sus304_cp, sus304_k
end

"""
3次多項式フィッティング係数の計算
戻り値: [a,b,c,d] for y = ax³ + bx² + cx + d
"""
function fit_thermal_properties()
    sus304_temp, sus304_rho, sus304_cp, sus304_k = load_thermal_properties_data()

    # Julia の fit 関数を使用（Polynomials.jl）
    rho_poly = fit(sus304_temp, sus304_rho, 3)
    cp_poly = fit(sus304_temp, sus304_cp, 3)
    k_poly = fit(sus304_temp, sus304_k, 3)

    # 係数を Python と同じ順序で取得 [a, b, c, d]
    rho_coeffs = reverse(coeffs(rho_poly))
    cp_coeffs = reverse(coeffs(cp_poly))
    k_coeffs = reverse(coeffs(k_poly))

    return rho_coeffs, cp_coeffs, k_coeffs
end

# グローバル変数として多項式係数を定義
const rho_coeffs, cp_coeffs, k_coeffs = fit_thermal_properties()

"""
多項式評価関数（Numbaのpolyval_numba相当）
coeffs: [a,b,c,d] for y = ax³ + bx² + cx + d
"""
function polyval_julia(coeffs::Vector{Float64}, x::Float64)
    result = 0.0
    n = length(coeffs)
    for i in 1:n
        result += coeffs[i] * x^(n - i)
    end
    return result
end

"""
温度依存熱物性値計算（並列化版）
Pythonのthermal_properties_calculator相当
"""
function thermal_properties_calculator(Temperature::Array{Float64,3}, cp_coeffs::Vector{Float64}, k_coeffs::Vector{Float64})
    ni, nj, nk = size(Temperature)
    cp = Array{Float64}(undef, ni, nj, nk)
    k = Array{Float64}(undef, ni, nj, nk)

    # Julia の並列処理 (@threads macro)
    Threads.@threads for i in 1:ni
        for j in 1:nj
            for k_ijk in 1:nk
                T_current = Temperature[i, j, k_ijk]
                cp[i, j, k_ijk] = polyval_julia(cp_coeffs, T_current)
                k[i, j, k_ijk] = polyval_julia(k_coeffs, T_current)
            end
        end
    end

    return cp, k
end

# 基準温度での密度計算
const rho = polyval_julia(rho_coeffs, 225.0 + 273.15)

# =======================================
# 物理パラメータ定義
# =======================================

# 格子パラメータ
const nz = 20
const dx = 0.12e-3
const dy = dx * sind(80) / sind(45)  # Julia の sind は度単位
const Lz = 0.5e-3

# 格子ストレッチファクター
const stretch_factor = 3

# Z方向格子面生成
z_faces_temp = LinRange(1.0, 0.0, nz + 1)
const z_faces = Lz .- (Lz / (exp(stretch_factor) - 1)) .* (exp.(stretch_factor .* z_faces_temp) .- 1)

# 格子中心座標
z_centers = zeros(nz)
z_centers[1] = z_faces[1]      # 最下面格子中心
z_centers[end] = z_faces[end]  # 最上面格子中心
z_centers[2:end-1] = (z_faces[2:end-2] + z_faces[3:end-1]) / 2  # 中間格子

# 格子サイズ計算
const dz = diff(z_faces)

# 境界成分計算
dz_t = zeros(nz)
dz_t[end] = Inf  # 最上面CV
dz_t[1:end-1] = z_centers[2:end] - z_centers[1:end-1]

dz_b = zeros(nz)
dz_b[1] = Inf    # 最下面CV
dz_b[2:end] = z_centers[2:end] - z_centers[1:end-1]

println("Julia版 IHCP-CGM ソルバー初期化完了")
println("格子パラメータ:")
println("  nz = $nz, dx = $dx, dy = $dy, Lz = $Lz")
println("  rho = $rho")
println("熱物性値多項式係数準備完了")