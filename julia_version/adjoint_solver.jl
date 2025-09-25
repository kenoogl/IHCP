# -*- coding: utf-8 -*-
"""
隨伴問題ソルバー（Adjoint Problem Solver）
共役勾配法（CGM）用の隨伴方程式求解
Python版からの変換: 係数行列構築と隨伴問題求解
"""

include("dhcp_solver.jl")

using Printf

"""
隨伴問題の係数と右辺項の構築（並列化版）
Python の coeffs_and_rhs_building_Adjoint 相当

引数:
- lambda_initial: 隨伴変数の初期値 (ni, nj, nk)
- T_cal: 計算温度 (2D表面温度) (ni, nj)
- Y_obs: 観測温度 (2D表面温度) (ni, nj)
- rho: 密度
- cp: 比熱分布 (ni, nj, nk)
- k: 熱伝導率分布 (ni, nj, nk)
- dx, dy: 格子間隔
- dz: z方向格子サイズ配列
- dz_b, dz_t: 境界成分配列
- dt: 時間ステップ

戻り値:
- a_w, a_e, a_s, a_n, a_b, a_t, a_p: 係数配列
- b: 右辺項配列
"""
function coeffs_and_rhs_building_Adjoint(lambda_initial::Array{Float64,3}, T_cal::Array{Float64,2}, Y_obs::Array{Float64,2},
                                          rho::Float64, cp::Array{Float64,3}, k::Array{Float64,3},
                                          dx::Float64, dy::Float64, dz::Vector{Float64},
                                          dz_b::Vector{Float64}, dz_t::Vector{Float64}, dt::Float64)

    ni, nj, nk = size(lambda_initial)
    N = ni * nj * nk

    # 係数配列を初期化
    a_w = zeros(N)
    a_e = zeros(N)
    a_s = zeros(N)
    a_n = zeros(N)
    a_b = zeros(N)
    a_t = zeros(N)
    a_p = zeros(N)
    b = zeros(N)

    # 並列処理で係数計算
    Threads.@threads for p in 1:N
        # 3D配列インデックスの計算 (Julia 1-based)
        i = ((p - 1) % ni) + 1
        j = (((p - 1) ÷ ni) % nj) + 1
        k_ijk = ((p - 1) ÷ (ni * nj)) + 1

        dz_k = dz[k_ijk]
        dz_t_k = dz_t[k_ijk]
        dz_b_k = dz_b[k_ijk]

        k_p = k[i, j, k_ijk]

        # 時間項係数（隨伴問題）
        a_p_0 = rho * cp[i, j, k_ijk] * dx * dy * dz_k / dt

        # 境界条件を考慮した係数計算（直接問題と同じ）
        a_w[p] = j == 1 ? 0.0 : (2 * k_p * k[i, j-1, k_ijk] / (k_p + k[i, j-1, k_ijk])) * dy * dz_k / dx
        a_e[p] = j == nj ? 0.0 : (2 * k_p * k[i, j+1, k_ijk] / (k_p + k[i, j+1, k_ijk])) * dy * dz_k / dx
        a_s[p] = i == 1 ? 0.0 : (2 * k_p * k[i-1, j, k_ijk] / (k_p + k[i-1, j, k_ijk])) * dx * dz_k / dy
        a_n[p] = i == ni ? 0.0 : (2 * k_p * k[i+1, j, k_ijk] / (k_p + k[i+1, j, k_ijk])) * dx * dz_k / dy
        a_b[p] = k_ijk == 1 ? 0.0 : (2 * k_p * k[i, j, k_ijk-1] / (k_p + k[i, j, k_ijk-1])) * dx * dy / dz_b_k
        a_t[p] = k_ijk == nk ? 0.0 : (2 * k_p * k[i, j, k_ijk+1] / (k_p + k[i, j, k_ijk+1])) * dx * dy / dz_t_k

        # 対角項
        a_p[p] = a_w[p] + a_e[p] + a_s[p] + a_n[p] + a_b[p] + a_t[p] + a_p_0

        # 右辺項（隨伴問題特有）
        rhs = a_p_0 * lambda_initial[i, j, k_ijk]

        # 底面境界条件（温度観測データとの差）
        if k_ijk == 1  # 底面（k_ijk == 0 in Python が k_ijk == 1 in Julia）
            rhs += 2.0 * (T_cal[i, j] - Y_obs[i, j]) * dx * dy
        end

        b[p] = rhs
    end

    return a_w, a_e, a_s, a_n, a_b, a_t, a_p, b
end

"""
隨伴問題用sparse行列の構築
Python の assemble_A_Adjoint 相当
（直接問題と同じ構造）

引数:
- ni, nj, nk: 格子数
- a_w, a_e, a_s, a_n, a_b, a_t, a_p: 係数配列

戻り値:
- A_csr: CSR形式sparse行列
"""
function assemble_A_Adjoint(ni::Int, nj::Int, nk::Int,
                           a_w::Vector{Float64}, a_e::Vector{Float64}, a_s::Vector{Float64},
                           a_n::Vector{Float64}, a_b::Vector{Float64}, a_t::Vector{Float64}, a_p::Vector{Float64})

    # 直接問題と同じ行列構造を使用
    return assemble_A_DHCP(ni, nj, nk, a_w, a_e, a_s, a_n, a_b, a_t, a_p)
end

"""
隨伴問題の複数時間ステップソルバー
Python の multiple_time_step_solver_Adjoint 相当

引数:
- T_cal: 直接問題解の温度分布 (nt, ni, nj, nk)
- Y_obs: 観測温度分布 (nt, ni, nj)
- nt: 時間ステップ数
- rho: 密度
- cp_coeffs, k_coeffs: 熱物性値多項式係数
- dx, dy: 格子間隔
- dz, dz_b, dz_t: z方向格子パラメータ
- dt: 時間ステップ
- rtol: 共役勾配法の相対許容誤差
- maxiter: 最大反復数

戻り値:
- lambda_field: 隨伴変数分布 (nt, ni, nj, nk)
"""
function multiple_time_step_solver_Adjoint(T_cal::Array{Float64,4}, Y_obs::Array{Float64,3},
                                          nt::Int, rho::Float64, cp_coeffs::Vector{Float64}, k_coeffs::Vector{Float64},
                                          dx::Float64, dy::Float64, dz::Vector{Float64},
                                          dz_b::Vector{Float64}, dz_t::Vector{Float64}, dt::Float64,
                                          rtol::Float64, maxiter::Int)

    ni, nj, nk = size(T_cal)[2:4]
    lambda_field = Array{Float64}(undef, nt, ni, nj, nk)

    # 最終時刻の隨伴変数をゼロで初期化
    lambda_field[end, :, :, :] .= 0.0

    # 初期解ベクトル（列優先順序でflat化）
    x0 = reshape(lambda_field[end, :, :, :], :)

    # 時間を逆向きにループ（隨伴問題の特徴）
    for t in (nt-1):-1:1
        # 次の時間ステップの隨伴変数
        lambda_initial = lambda_field[t+1, :, :, :]

        # 現在時刻の温度から熱物性値を計算
        cp, k = thermal_properties_calculator(T_cal[t, :, :, :], cp_coeffs, k_coeffs)

        # 係数行列と右辺項を構築
        a_w, a_e, a_s, a_n, a_b, a_t, a_p, b = coeffs_and_rhs_building_Adjoint(
            lambda_initial, T_cal[t, :, :, 1], Y_obs[t, :, :], rho, cp, k, dx, dy, dz, dz_b, dz_t, dt
        )

        # sparse行列組み立て
        A_csr = assemble_A_Adjoint(ni, nj, nk, a_w, a_e, a_s, a_n, a_b, a_t, a_p)

        # 前処理行列（対角プリコンディショナ）
        diag_vec = diag(A_csr)
        inv_diag = [d != 0.0 ? 1.0/d : 0.0 for d in diag_vec]  # ゼロ除算回避

        # 共役勾配法で線形システムを解く
        x_result = cg(A_csr, b; Pl=Diagonal(inv_diag), reltol=rtol, maxiter=maxiter)

        # 結果を3D配列に再整形して保存
        lambda_field[t, :, :, :] = reshape(x_result, (ni, nj, nk))
        x0 = x_result  # 次の時間ステップのためのウォームスタート
    end

    return lambda_field
end

println("隨伴問題ソルバー（Adjoint）実装完了")