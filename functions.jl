function noImmunization!(du,u,p,t)
  λ, μ, avK, ks, pk = p
  ξ = sum(ks .* pk .* u)
  du[:] .= -μ * u + λ / avK * (1.0 .- u) .* ks * ξ
end

function getSolution(p)
    u0 = 1.0*ones(length(p[4]))
    prob = SteadyStateProblem(noImmunization!,u0,p = p)
    return(solve(prob))
end

function getρ(λ, μ, avK)
    return(λ/μ/avK)
end

function getz₁(sol, pk, ks)
    return(sum( sol.u .* pk .* ks))
end

function getϕ(ρ, pk, ks, z₁)
    return(ρ * sum( pk .* (ks ./ (1.0 .+ z₁*ρ*ks)).^2 ))
end

function getψ(ρ, pk, ks, z₁)
    return(sum( pk .* ks ./ ((1.0 .+ z₁*ρ*ks)).^2 ))
end

function getϵC(z₁, ϕ, ψ)
    return((2*ψ - (1-ϕ)*z₁)/(ψ - (1-ϕ)*z₁))
end

function getPrevalence(sol, pk)
    return(sum( sol.u .* pk ))
end

function SFDist(ks, γ)
    return( ks.^(-γ) / sum(ks.^(-γ)))
end

function getavK(ks, pk)
    return(sum( ks .* pk))
end

function getF(ϵ, ρ, z₁, ϕ, ψ, ks )
    return( -ϵ*ρ*z₁* ks ./ ( (1.0.+ρ*z₁*ks).*(1.0.+(1-ϵ)*ρ*z₁*ks) ) .* (1.0 .+ ρ*ψ/(1-ϕ)*ks) )
end

function getkStar(ϵ, ρ, z₁, ϕ, ψ)
    α = ρ*ψ/(1-ϕ)
    β = ρ*z₁
    a = β*(α*(2-ϵ)-β*(1-ϵ))
    b = 2*α
    c = 1
    if (ϵ < (2*ψ - (1-ϕ)*z₁)/(ψ - (1-ϕ)*z₁))
        return( (α + sqrt((α-β)*(α - β*(1-ϵ))) )/β/(-α*(2-ϵ)+β*(1-ϵ)))
    else
        return( NaN )
    end
end

function getCorrespondance(γ, kmax, μ, I)
    ks = [1:1:kmax;]
    pk = SFDist(ks, γ)
    avK = sum( ks .* pk)

    function noImmunizationOpt!(du,u,p,t)
      ξ = sum(ks .* pk .* u)
      du[:] .= -μ * u + p[1] / avK * (1.0 .- u) .* ks * ξ
    end

    function lossFunction(sol)
        return(abs(sum( sol.u .* pk ) - I))
    end

    u0 = 1.0*ones(length(ks))
    prob = SteadyStateProblem(noImmunizationOpt!,u0)

    cost_function = build_loss_objective(prob, Tsit5(),
        lossFunction)

    result = optimize(cost_function, 0.0, 10.0)
    return(result.minimizer)
end

function getCorrespondance(pk, μ, I)
    kmax = length(pk)
    ks = [1:1:kmax;]
    avK = sum( ks .* pk)

    function noImmunizationOpt!(du,u,p,t)
      ξ = sum(ks .* pk .* u)
      du[:] .= -μ * u + p[1] / avK * (1.0 .- u) .* ks * ξ
    end

    function lossFunction(sol)
        return(abs(sum( sol.u .* pk ) - I))
    end

    u0 = 1.0*ones(length(ks))
    prob = SteadyStateProblem(noImmunizationOpt!,u0)

    cost_function = build_loss_objective(prob, Tsit5(),
        lossFunction)

    result = optimize(cost_function, 0.0, 10.0)
    return(result.minimizer)
end

function getCorrespondance(pk, μ, I, ks)
    avK = sum( ks .* pk)

    function noImmunizationOpt!(du,u,p,t)
      ξ = sum(ks .* pk .* u)
      du[:] .= -μ * u + p[1] / avK * (1.0 .- u) .* ks * ξ
    end

    function lossFunction(sol)
        return(abs(sum( sol.u .* pk ) - I))
    end

    u0 = 1.0*ones(length(ks))
    prob = SteadyStateProblem(noImmunizationOpt!,u0)

    cost_function = build_loss_objective(prob, Tsit5(),
        lossFunction)

    result = optimize(cost_function, 0.0, 10.0)
    return(result.minimizer)
end

function getkStarFromScratch(γ, kmax, λ, μ, ϵ)
    ks = [1:1:kmax;]
    pk = SFDist(ks, γ)
    avK = sum( ks .* pk)

    p = [λ, μ, avK, ks, pk]
    sol = getSolution(p)

    z₁ = getz₁(sol, pk, ks)
    ρ = getρ(λ, μ, avK)
    ϕ = getϕ(ρ, pk, ks, z₁)
    ψ = getψ(ρ, pk, ks, z₁)

    kStar = getkStar(ϵ, ρ, z₁, ϕ, ψ)

    return(kStar)
end

function getFFromScratch(γ, kmax, λ, μ, ϵ)
    ks = [1:1:kmax;]
    pk = SFDist(ks, γ)
    avK = sum( ks .* pk)

    p = [λ, μ, avK, ks, pk]
    sol = getSolution(p)

    z₁ = getz₁(sol, pk, ks)
    ρ = getρ(λ, μ, avK)
    ϕ = getϕ(ρ, pk, ks, z₁)
    ψ = getψ(ρ, pk, ks, z₁)

    ks = [1:1:1000;]
    f = getF(ϵ, ρ, z₁, ϕ, ψ, ks )

    return(f)
end

function getϵCFromScratch(γ, kmax, λ, μ)
    ks = [1:1:kmax;]
    pk = SFDist(ks, γ)
    avK = sum( ks .* pk)

    p = [λ, μ, avK, ks, pk]
    sol = getSolution(p)

    z₁ = getz₁(sol, pk, ks)
    ρ = getρ(λ, μ, avK)
    ϕ = getϕ(ρ, pk, ks, z₁)
    ψ = getψ(ρ, pk, ks, z₁)

    return(getϵC(z₁, ϕ, ψ))
end

function getϵCFromScratch(pk, λ, μ)
    kmax = length(pk)
    ks = [1:1:kmax;]
    avK = sum( ks .* pk)

    p = [λ, μ, avK, ks, pk]
    sol = getSolution(p)

    z₁ = getz₁(sol, pk, ks)
    ρ = getρ(λ, μ, avK)
    ϕ = getϕ(ρ, pk, ks, z₁)
    ψ = getψ(ρ, pk, ks, z₁)

    return(getϵC(z₁, ϕ, ψ))
end


function getCriticalPrevalence(γ, kmax, μ, ϵ)
    ks = [1:1:kmax;]
    pk = SFDist(ks, γ)
    avK = sum( ks .* pk)

    function noImmunizationOpt!(du,u,p,t)
      ξ = sum(ks .* pk .* u)
      du[:] .= -μ * u + p[1] / avK * (1.0 .- u) .* ks * ξ
    end

    function lossFunction(sol)
        z₁ = getz₁(sol, pk, ks)
        ρ = getρ(sol.prob.p[1], μ, avK)
        ϕ = getϕ(sol.prob.p[1], pk, ks, z₁)
        ψ = getψ(ρ, pk, ks, z₁)
        return(abs(ψ*(1-ϕ)*(2-ϵ) - z₁*(1-ϵ)))
    end

    u0 = 1.0*ones(length(ks))
    prob = SteadyStateProblem(noImmunizationOpt!,u0)

    cost_function = build_loss_objective(prob, Tsit5(),
        lossFunction)

    result = optimize(cost_function, 0.0, 1000.0)
    return(result.minimizer)
end

function getAvF(fDist, pk, start)
    return sum( pk[start:end] .* fDist[start:end] ./ sum(pk[start:end]))
end

function paramsNegBin(mu, o)
    p = 1- 1.0/(1+mu/o)
    p = 1-p
    r = mu*(p)/(1-p)
    return(r,p)
end

function getICRegime(ϵth, pk, ks, avK, μ)

    function toOptimize(λ)
        return(abs(Ith - getϵCFromScratch(pk, λ, μ)))
    end

    res = optimize(toOptimize, 0.0, 10.0)
    res.minimizer
    p = [res.minimizer, μ, avK, ks, pk]
    sol = getSolution(p)
    Ireg = getPrevalence(sol, pk)

    return(Ireg)
end

function getIRRegime(ϵrth, pk, ks, avK, μ)

    function toOptimize(λ)
        m = 10000
        ϵs = range(0.0000001, 0.9999, length = m)
        p = [λ, μ, avK, ks, pk]
        sol = getSolution(p)
        z₁ = getz₁(sol, pk, ks)
        ρ = getρ(λ, μ, avK)
        ϕ = getϕ(ρ, pk, ks, z₁)
        ψ = getψ(ρ, pk, ks, z₁)

        avF = zeros(m)
        fksMax = zeros(m)

        for j in 1:m
            fksMax[j] = getF(ϵs[j], ρ, z₁, ϕ, ψ, maximum(ks))
            fDist = getF(ϵs[j], ρ, z₁, ϕ, ψ, ks)
            avF[j] = sum( pk[1:end] .* fDist[1:end] ./ sum(pk[1:end]))
        end
        ϵr = 0.0
        s = Int64.(sign.(fksMax .- avF))
        if (-1 in s)
            ϵr = ϵs[findfirst(s -> s == -1, s)]
        end

        return(abs(ϵrth - ϵr))
    end

    res = optimize(toOptimize, 0.0, 10.0)
    res.minimizer
    p = [res.minimizer, μ, avK, ks, pk]
    sol = getSolution(p)
    Ireg = getPrevalence(sol, pk)

    return(Ireg)
end

#---------------------------- Functions assortativity ---------------------
function assortativity!(du,u,p,t)
  kᵢ, pₖ, avK, gₖ, λ, μ, ϵ, α,  N  = p
  λᵢ = λ*kᵢ*α*sum(kᵢ .* pₖ .* ((1.0 .- gₖ) .* u[1:N] .+ gₖ .* u[(N+1):(2*N)]))/avK .+ (1-α)*λ*kᵢ .* ((1.0 .- gₖ) .* u[1:N] .+ gₖ .* u[(N+1):(2*N)])
  du[1:N] .= -μ*u[1:N] .+  (1.0 .- u[1:N]) .* λᵢ
  du[(N+1):(2*N)] .= -μ*u[(N+1):(2*N)] .+  (1-ϵ)*λᵢ .* (1.0 .- u[(N+1):(2*N)])
end

function getSolution_assort(p)
    u0 = ones(2*length(p[1]))
    prob = SteadyStateProblem(assortativity!,u0,p = p)
    return(solve(prob))
end

function getContactMatrix(pₖ, kᵢ, α)
    N_l = length(kᵢ)
    Cᵢⱼ = zeros(N_l, N_l)
    pProb_norm = pₖ .* kᵢ /(sum(pₖ .* kᵢ))
    for i in 1:N_l
        Cᵢⱼ[i,:] .= α*pProb_norm
        Cᵢⱼ[i,i] += (1-α)
    end
    return(Cᵢⱼ)
end

function getPrevalence_assort(sol, pₖ, gₖ, N)
    return(sum(((1.0 .- gₖ) .* sol.u[1:N] .+ gₖ .* sol.u[(N+1):(2*N)]) .* pₖ ))
end

function getCorrespondance_assort(pₖ, μ, I, kᵢ, α)
    N = length(kᵢ)
    avK = sum( kᵢ .* pₖ)
    gₖ = zeros(N)
    ϵ = 0.0

    function noImmunizationOpt_assort!(du,u,p,t)
        λᵢ = p[1]*kᵢ*α*sum(kᵢ .* pₖ .* ((1.0 .- gₖ) .* u[1:N] .+ gₖ .* u[(N+1):(2*N)]))/avK .+ (1-α)*p[1]*kᵢ .* ((1.0 .- gₖ) .* u[1:N] .+ gₖ .* u[(N+1):(2*N)])
        du[1:N] .= -μ*u[1:N] .+  (1.0 .- u[1:N]) .* λᵢ
        du[(N+1):(2*N)] .= -μ*u[(N+1):(2*N)] .+  (1-ϵ)*λᵢ .* (1.0 .- u[(N+1):(2*N)])
    end

    function lossFunction(sol)
        return(abs(getPrevalence_assort(sol, pₖ, gₖ, N) - I))
    end

    u0 = 1.0*ones(2*length(kᵢ))
    prob = SteadyStateProblem(noImmunizationOpt_assort!,u0)

    cost_function = build_loss_objective(prob, Tsit5(),
        lossFunction)
    result = optimize(cost_function, 0.0, 10.0)
    return(result.minimizer)
end

function getF_assort_detailed(ks, pk, avK, gk, λ, μ, ϵ, α,  N)
    N = length(ks)
    gk = zeros(N)
    λ_hat = λ/μ
    p = [ks, pk, avK, gk, λ, μ, ϵ, α,  N]
    sol_int = getSolution_assort(p)
    sol_int.u

    w = 1 - α

    z = sum( sol_int.u[1:N] .* pk .* ks)
    xₖ = sol_int.u[1:N]
    yₖ = (1.0 - ϵ) * xₖ ./ (1.0 .- ϵ*xₖ)

    d = 1.0 .+ λ_hat*ks .* (z/avK .+ w*(2*xₖ .- 1.0 .- z/avK))
    u = -λ_hat/avK*(1-w)*ks .* (1.0 .- xₖ)
    v = ks .* pk

    c = -sum( pk .* u ./ d) / (1.0 + sum( u .* v ./ d))
    Γ = 1.0/avK * sum( pk .* ks .* (1.0 .- xₖ) .* (1.0 .+ c*ks) ./ d)

    fDist_New = (yₖ .- xₖ) .* (1 .+ λ_hat*ks .* ((1.0 - w)*Γ .+ w * (1.0 .- xₖ) .* (1.0 .+ c .* ks) ./ d))
    f_dir_assort = (yₖ .- xₖ)
    f_indir_assort = (yₖ .- xₖ) .* (λ_hat*ks .* ((1.0 - w)*Γ .+ w * (1.0 .- xₖ) .* (1.0 .+ c .* ks) ./ d))


    return (-fDist_New, -f_indir_assort, -f_dir_assort)
end

function getF_assort(ks, pk, avK, gk, λ, μ, ϵ, α,  N)
    N = length(ks)
    gk = zeros(N)
    λ_hat = λ/μ
    p = [ks, pk, avK, gk, λ, μ, ϵ, α,  N]
    sol_int = getSolution_assort(p)
    sol_int.u

    w = 1 - α

    z = sum( sol_int.u[1:N] .* pk .* ks)

    xₖ = sol_int.u[1:N]
    yₖ = (1.0 - ϵ) * xₖ ./ (1.0 .- ϵ*xₖ)

    d = 1.0 .+ λ_hat*ks .* (z/avK .+ w*(2*xₖ .- 1.0 .- z/avK))
    u = -λ_hat/avK*(1-w)*ks .* (1.0 .- xₖ)
    v = ks .* pk

    c = -sum( pk .* u ./ d) / (1.0 + sum( u .* v ./ d))
    Γ = 1.0/avK * sum( pk .* ks .* (1.0 .- xₖ) .* (1.0 .+ c*ks) ./ d)

    fDist_New = (yₖ .- xₖ) .* (1 .+ λ_hat*ks .* ((1.0 - w)*Γ .+ w * (1.0 .- xₖ) .* (1.0 .+ c .* ks) ./ d))
    f_indir_assort = (yₖ .- xₖ)
    f_dir_assort = (yₖ .- xₖ) .* (λ_hat*ks .* ((1.0 - w)*Γ .+ w * (1.0 .- xₖ) .* (1.0 .+ c .* ks) ./ d))


    return (-fDist_New)
end

function getF_assort_sol(sol, ks, pk, avK, gk, λ, μ, ϵ, α,  N)
    N = length(ks)
    gk = zeros(N)
    λ_hat = λ/μ
    w = 1 - α

    z = sum( sol.u[1:N] .* pk .* ks)
    xₖ = sol.u[1:N]
    yₖ = (1.0 - ϵ) * xₖ ./ (1.0 .- ϵ*xₖ)

    d = 1.0 .+ λ_hat*ks .* (z/avK .+ w*(2*xₖ .- 1.0 .- z/avK))
    u = -λ_hat/avK*(1-w)*ks .* (1.0 .- xₖ)
    v = ks .* pk

    c = -sum( pk .* u ./ d) / (1.0 + sum( u .* v ./ d))
    Γ = 1.0/avK * sum( pk .* ks .* (1.0 .- xₖ) .* (1.0 .+ c*ks) ./ d)

    fDist_New = (yₖ .- xₖ) .* (1 .+ λ_hat*ks .* ((1.0 - w)*Γ .+ w * (1.0 .- xₖ) .* (1.0 .+ c .* ks) ./ d))
    f_indir_assort = (yₖ .- xₖ)
    f_dir_assort = (yₖ .- xₖ) .* (λ_hat*ks .* ((1.0 - w)*Γ .+ w * (1.0 .- xₖ) .* (1.0 .+ c .* ks) ./ d))


    return (-fDist_New)
end

function getIRRegime_assort(ϵrth, pk, ks, avK, μ, N, α)
    gk = zeros(N)
    function toOptimize(λ)
        p = [ks, pk, avK, gk, λ, μ, 0.0, α,  N]
        sol = getSolution_assort(p)

        m = 10000
        ϵs = range(0.0000001, 0.9999, length = m)

        avF = zeros(m)
        fksMax = zeros(m)
        fksStar = zeros(m)
        fksIndex = zeros(m)

        for j in 1:m
            fDist = getF_assort_sol(sol, ks, pk, avK, gk, λ, μ, ϵs[j], α,  N)
            fksMax[j] = fDist[end]
            avF[j] = sum(pk[1:end] .* fDist[1:end] ./ sum(pk[1:end]))
            fksStar[j] = maximum(fDist)
            fksIndex[j] = (fDist[N-1] > fDist[N])
        end
        ϵr = 0
        s = Int64.(sign.(fksMax .- avF))
        if (1 in s)
            ϵr = ϵs[findfirst(s -> s == 1, s)]
        end

        return(abs(ϵrth - ϵr))
    end

    res = optimize(toOptimize, 0.001, 10.0)
    p = [ks, pk, avK, gk, res.minimizer, μ, 0.0, α,  N]
    sol =  getSolution_assort(p)
    Ireg = getPrevalence_assort(sol, pk, gk, N)

    return(Ireg)
end

function getICRegime_assort(ϵcth, pk, ks, avK, μ, N, α)
    gk = zeros(N)
    function toOptimize(λ)
        p = [ks, pk, avK, gk, λ, μ, 0.0, α,  N]
        sol = getSolution_assort(p)

        m = 10000
        ϵs = range(0.0000001, 0.9999, length = m)

        avF = zeros(m)
        fksMax = zeros(m)
        fksStar = zeros(m)
        fksIndex = zeros(m)

        for j in 1:m
            fDist = getF_assort_sol(sol, ks, pk, avK, gk, λ, μ, ϵs[j], α,  N)
            fksMax[j] = fDist[end]
            avF[j] = sum(pk[1:end] .* fDist[1:end] ./ sum(pk[1:end]))
            fksStar[j] = maximum(fDist)
            fksIndex[j] = (fDist[N-1] > fDist[N])
        end
        ϵc = 0
        s = Int64.(sign.(fksMax .- fksStar))
        if (1 in fksIndex)
            ϵc = ϵs[findfirst(fksIndex -> fksIndex == 0, fksIndex)]
        end

        return(abs(ϵcth - ϵc))
    end

    res = optimize(toOptimize, 0.001, 10.0)
    p = [ks, pk, avK, gk, res.minimizer, μ, 0.0, α,  N]
    sol =  getSolution_assort(p)
    Ireg = getPrevalence_assort(sol, pk, gk, N)

    return(Ireg)
end
