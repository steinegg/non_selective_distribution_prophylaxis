using DifferentialEquations
using ParameterizedFunctions
using Statistics
using Plots
using LaTeXStrings
using Optim
using DiffEqParamEstim
using DelimitedFiles
using ColorSchemes
using Distributions
using CSV
using DataFrames
using Plots.PlotMeasures
using DoubleFloats

include("functions.jl")

#--------------------- Plot Panel 1 A/B ---------------
μ = 0.1
mu = 2.0
o = 0.05
coeff_var = sqrt(mu+1/o)
r,p = paramsNegBin(mu,o)
d = NegativeBinomial(r,p)

kmax = 1000
ks = [1:1:kmax;]
pk = map(x -> pdf(d, x), ks)
avK = getavK(ks,pk)

μ = 0.1
ϵ = 0.8
λ = 0.05
p = [λ, μ, avK, ks, pk]
sol = getSolution(p)

redL = λ/μ/avK

z₁ = getz₁(sol, pk, ks)
ρ = getρ(λ, μ, avK)
ϕ = getϕ(ρ, pk, ks, z₁)
ψ = getψ(ρ, pk, ks, z₁)
kStar = getkStar(ϵ, ρ, z₁, ϕ, ψ)

xk = z₁*ρ .* ks ./ (1.0 .+ z₁*ρ .* ks)
yk = (1-ϵ)*xk ./ (1 .- ϵ*xk)
Jkk = zeros(kmax,kmax)
for i in 1:kmax
    for j in 1:kmax
        Jkk[i,j] = ρ*ks[i]*(1-xk[i])*pk[j]*ks[j]*(yk[j]-xk[j])/(1+z₁*ρ*ks[i])/ (1-ϕ)
    end
end

firstTerm = zeros(kmax)
firstTerm .= yk - xk
secondTerm = zeros(kmax)
for i in 1:kmax
    secondTerm[i] = 1.0/pk[i] .* sum( pk .* Jkk[:,i] )
end

kStarP = 1/sqrt(1-ϵ)/ρ/z₁
xk = z₁*ρ * kStarP / (1.0 + z₁*ρ * kStarP)
yk = (1-ϵ)*xk / (1 - ϵ*xk)
valueMin = yk - xk

kPlot = 200
scatter(ks, -firstTerm,
    xscale = :log10,
    label = L"F_{dir}",
    xlim = (1,kPlot),
    markershape = :circle)
scatter!([kStarP], [-valueMin],
    label = :none,
    markercolor = :red,
    markershape = :hexagon,
    markerstrokecolor = :black,
    markerstrokewidth = 2,
    markersize = 8)

scatter!(-secondTerm,
    xscale = :log10,
    label = L"F_{indir}",
    xlim = (1,kPlot))


total = getF(ϵ, ρ, z₁, ϕ, ψ, ks)
minF = getF(ϵ, ρ, z₁, ϕ, ψ, kStar)
scatter!(-total,
    xscale = :log10,
    label = L"f(k)",
    xlim = (1,kPlot),
    xlabel = L"k")
scatter!([kStar], [-minF],
    label = :none,
    markercolor = :red,
    markershape = :hexagon,
    markerstrokecolor = :black,
    markerstrokewidth = 2,
    markersize = 8)
#savefig("Dropbox/Immunization/Plots/emergenceMin.pdf")

#writedlm("Dropbox/Immunization/Plots/data/panel1AB_kANDTerms.csv", [ks firstTerm secondTerm total])
#writedlm("Dropbox/Immunization/Plots/data/panel1AB_minima.csv", [kStarP valueMin kStar minF])

#------------------------------- Plot Panel 1 C --------------------------------
μ = 0.1
mu = 2.0
o = 0.05
r,p = paramsNegBin(mu,o)
d = NegativeBinomial(r,p)

kmax = 1000
ks = [1:1:kmax;]
pk = map(x -> pdf(d, x), ks)
avK = getavK(ks,pk)

μ = 0.1
ϵ = 0.8
kmax = 1000

l = 5
ϵs = [0.5, 0.7, 0.8, 0.85, 0.88]
l = length(ϵs)x
fs = zeros(kmax, l)
kStars = zeros(l)
vals = zeros(l)

λ = 0.025

p = [λ, μ, avK, ks, pk]
sol = getSolution(p)

z₁ = getz₁(sol, pk, ks)
ρ = getρ(λ, μ, avK)
ϕ = getϕ(ρ, pk, ks, z₁)
ψ = getψ(ρ, pk, ks, z₁)

for i in 1:l
    kStars[i] = getkStar(ϵs[i], ρ, z₁, ϕ, ψ)
    vals[i] = getF(ϵs[i], ρ, z₁, ϕ, ψ, kStars[i])
    fs[:,i] .= getF(ϵs[i], ρ, z₁, ϕ, ψ, ks)
end

N = 1000
ϵsAdd = range(0.1, 0.865, length = N)
kStarsAdd = zeros(N)
valsAdd = zeros(N)

for i in 1:N
    kStarsAdd[i] = getkStar(ϵsAdd[i], ρ, z₁, ϕ, ψ)
    valsAdd[i] = getF(ϵsAdd[i], ρ, z₁, ϕ, ψ, kStarsAdd[i])
end

stop = 500
lab = ["ϵ = "*string(round(i, sigdigits = 2)) for i in ϵs]
lab = reshape(lab, 1, length(lab))
plot(kStarsAdd, valsAdd,
    linestyle = :dash,
    color = :grey,
    label = L"k^*")
plot!(ks[1:500], fs[1:500,:],
    xscale = :log,
    marker = :dot,
    label = lab)
scatter!(kStars, vals,
    color = :grey,
    label = :none,
    marker = :dot,
    alpha = 0.5,
    markersize = 6,
    markerstrokewidth = 2)
#savefig("Dropbox/Immunization/Plots/fk.pdf")

#writedlm("Dropbox/Immunization/Plots/data/panel1C_fks.csv", fs[:,:])
#writedlm("Dropbox/Immunization/Plots/data/panel1C_ks.csv", ks[:,:])
#writedlm("Dropbox/Immunization/Plots/data/panel1C_kStarScatter.csv", [kStars vals])
#writedlm("Dropbox/Immunization/Plots/data/panel1C_kStarLine.csv", [kStarsAdd valsAdd])


#------------------------------- Plot Panel 2 A --------------------------------
mu = 2.0
o = 0.05
r,p = paramsNegBin(mu,o)
d = NegativeBinomial(r,p)

kmax = 1000
ks = [1.0:1.0:kmax;]
pk = map(x -> pdf(d, x), ks)
avK = getavK(ks,pk)

μ = 0.1
l = 100
kmax = 1000

ϵs = range(0.0,1.0, length = l)
ϵc = zeros(l)
res = zeros(l,l)
Is = range(0.01, 0.15, length = l)
Is = range(0.01, 0.8, length = l)

λs = zeros(l)
prevs = zeros(l)

Threads.@threads for i in 1:l
    λs[i] = getCorrespondance(pk, μ, Is[i])
end

#ks = [1:1:kmax;]

m = 10000
ϵsm = range(0.0000001, 0.9999, length = m)

ϵr = zeros(l)

Threads.@threads for i in 1:l
    p = [λs[i], μ, avK, ks, pk]
    sol = getSolution(p)

    prevs[i] = getPrevalence(sol, pk)
    z₁ = getz₁(sol, pk, ks)
    ρ = getρ(λs[i], μ, avK)
    ϕ = getϕ(ρ, pk, ks, z₁)
    ψ = getψ(ρ, pk, ks, z₁)

    ϵc[i] =  getϵC(z₁, ϕ, ψ)
    res[i,:] .= map(ϵ -> getkStar(ϵ, ρ, z₁, ϕ, ψ), ϵs)

    avF = zeros(m)
    fksMax = zeros(m)

    for j in 1:m
        fksMax[j] = ψ/z₁/(1-ϕ)*ϵsm[j]/(1-ϵsm[j])
        fDist = getF(ϵsm[j], ρ, z₁, ϕ, ψ, ks)
        avF[j] = -sum( pk[1:end] .* fDist[1:end] ./ sum(pk[1:end]))
    end

    s = Int64.(sign.(fksMax .- avF))
    if (1 in s)
        ϵr[i] = ϵsm[findfirst(s -> s == 1, s)]
    end
end


for i in 1:l
    for j in 1:l
        if res[i,j] < 0
            res[i,j] = NaN
        end
    end
end

for i in 1:l
    if ϵc[i] > 1.0
        ϵc[i] = -0.1
    end
end

heatmap(ϵs, Is, log10.(res),
    xlabel = L"\epsilon",
    ylabel = L"I",
    xlim = (minimum(ϵs), maximum(ϵs)),
    clim = (0,2))
plot!(ϵc, Is,
    color = :red,
    width = 2.0,
    label = L"\epsilon_c",
    legend = :bottomright)
plot!(ϵr, Is,
    color = :red,
    width = 2.0,
    label = L"\epsilon_c",
    legend = :bottomright)
#savefig("Dropbox/Immunization/Plots/heatmapKStar.pdf")

#writedlm("Dropbox/Immunization/Plots/data/panel2A_axis.csv", [ϵs, Is])
#writedlm("Dropbox/Immunization/Plots/data/panel2A_heatmap.csv", res)
#writedlm("Dropbox/Immunization/Plots/data/panel2A_critical.csv", [ϵc ϵr])

#--------------------- Plot Panel 2 B ---------------

mu = 2.0
l = 100
varL = range(3.0,20, length = l)
os = (varL.^2 ./ mu^2) .- mu

varPlot = sqrt.(os .+ 1.0/mu )
μ = 0.1
kmax = 1000
ks = [1:1:kmax;]

Is = [0.05, 0.1, 0.15, 0.2, 0.25]
m = length(Is)

λs = zeros(l,m)

Threads.@threads for i in 1:l
    for j in 1:m
        o = 1.0 / os[i]
        r,p = paramsNegBin(mu,o)
        d = NegativeBinomial(r,p)

        pk = map(x -> pdf(d, x), ks)
        avK = getavK(ks,pk)
        λs[i,j] = getCorrespondance(pk, μ, Is[j])
    end
end

ϵcs = zeros(l,m)

Threads.@threads for i in 1:l
    for j in 1:m
        o = 1.0/os[i]
        r,p = paramsNegBin(mu,o)
        d = NegativeBinomial(r,p)

        pk = map(x -> pdf(d, x), ks)
        avK = getavK(ks,pk)

        p = [λs[i,j], μ, avK, ks, pk]
        sol = getSolution(p)

        z₁ = getz₁(sol, pk, ks)
        ρ = getρ(λs[i,j], μ, avK)
        ϕ = getϕ(ρ, pk, ks, z₁)
        ψ = getψ(ρ, pk, ks, z₁)

        ϵcs[i,j] =  getϵC(z₁, ϕ, ψ)
        if ϵcs[i,j] > 1.0
            ϵcs[i,j] = -1
        end
    end
end

lab = ["I = "*string(round(i, sigdigits = 2)) for i in Is]
lab = reshape(lab, 1, length(lab))
cs = palette(:GnBu_6)[2:end]

scatter(varPlot, ϵcs,
    ylim = (0,1),
    label = lab,
    xlab = "Coefficient of Variation",
    ylab = L"\epsilon_c",
    linewidth = 2.0,
    palette = ColorSchemes.PuBu_6.colors[2:end],
    legend = :bottomright)

#writedlm("Dropbox/Immunization/Plots/data/panel2B_epsilons.csv", [ϵcs])
#writedlm("Dropbox/Immunization/Plots/data/panel2B_coeffvar.csv", [varPlot])
#writedlm("Dropbox/Immunization/Plots/data/panel2B_Is.csv", [Is])


#-------------------------- Analyze world data Panel 3 B ----------------
μ = 1.0
mu = 2.1
sd = 11.6
o = mu*mu/(sd*sd - mu)
r,p = paramsNegBin(mu,o)

d = NegativeBinomial(r,p)

kmax = 1000
ks = [1:1:kmax;]
pk = map(x -> pdf(d, x), ks)
avK = getavK(ks,pk)


d_vs = 0.76
d_art = 0.24

places = ["Cape Town", "Port Elizabeth", "Togo", "Senegal", "Ghana", "Ghambia",
    "Douala (Cameroon)", "Youndé (Cameroon)", "Kampala (Uganda)", "Malawi", "Mulanje (Malawi)", "Ougadougou",
    "Bobo Dioulasso (Burkina Faso)", "Kenya", "Lesotho", "Botswhana", "Namibia", "Swaziland",
    "Abidjan (Cote D'Ivoire)", "Senegal", "Lagos (Nigeria)", "Ibadan (Nigeria)", "Abuja (Nigeria)"]
prevs = [0.3, 0.51, 0.196, 0.215, 0.303, 0.098,
    0.255, 0.444, 0.137, 0.214, 0.245, 0.145,
    0.329, 0.267, 0.328, 0.1966, 0.124, 0.176,
    0.18, 0.218, 0.449, 0.28, 0.271]
art = [0.25, d_art, 0.062, 0.1, d_art, d_art,
    0.125, 0.125, 0.193, 0.08, 0.08, 0.25,
    0.557, 0.25, d_art, d_art, d_art, d_art,
    d_art, d_art, d_art, d_art, d_art]
vs = [d_vs, d_vs, d_vs, d_vs, d_vs, d_vs,
    d_vs, d_vs, d_vs, d_vs, d_vs, d_vs,
    d_vs, d_vs, d_vs, d_vs, d_vs, d_vs,
    d_vs, d_vs, d_vs, d_vs, d_vs]

data = DataFrame(places = places,
    prevs = prevs,
    art = art,
    vs = vs,
    sp = zeros(length(vs)))

data.effective_prevalence = data.prevs .* ( (1.0 .- data.art) .+ data.art.*(1.0 .- data.vs))

data = data[:, [:places, :effective_prevalence, :prevs, :art]]
rename!(data, :prevs => :prevalence)

places = ["Austria", "Croatia", "Denmark", "France", "Germany", "Greece", "Italy", "Netherlands",
    "Spain", "Sweden", "United Kingdom"]
prevs = [0.072, 0.033, 0.049, 0.12, 0.049, 0.065, 0.107, 0.06, 0.17, 0.04, 0.053]
sp = [0.861, 0.923, 0.976, 0.968, 0.933, 0.844, 0.924, 0.968, 0.886, 0.922, 0.935]


dataInter = DataFrame(places = places,
    prevs = prevs,
    sp = sp,
    art = zeros(length(prevs)),
    vs = zeros(length(prevs)))


dataInter.effective_prevalence = dataInter.prevs .* (1.0 .- dataInter.sp)
rename!(dataInter, :prevs => :prevalence)

dataInter = dataInter[:, [:places, :effective_prevalence, :prevalence, :art]]

data = outerjoin(data, dataInter, on = [:places, :effective_prevalence, :prevalence, :art])

d_vs = 0.5
d_art = 0.25

places = ["Buenos Aires", "Sao Paulo", "Brasilia", "Manaus", "Rio de Janeiro", "Santiago de Chile", "Bogota",
    "Cali", "Tijuana", "Panama", "Lima", "San Salvador", "Guatemala City", "Honduras"]
prevs = [0.173, 0.23, 0.058, 0.151, 0.152, 0.176, 0.163, 0.237, 0.202, 0.294, 0.205, 0.108, 0.119, 0.074]
art = [0.487, 0.41, 0.41, 0.41, 0.41, 0.485, 0.22, 0.22, 0.401, 0.333, 0.355, 0.252, 0.242, 0.315]
vs = [d_vs, d_vs, d_vs, d_vs, d_vs, d_vs, 0.14/0.22, 0.14/0.22, 0.025/0.401, d_vs, d_vs, d_vs, d_vs, d_vs]

dataInter = DataFrame(places = places,
    prevs = prevs,
    art = art,
    vs = vs,
    sp = length(vs))

dataInter.effective_prevalence = dataInter.prevs .* ( (1.0 .- dataInter.art) .+ dataInter.art.*(1.0 .- dataInter.vs))
rename!(dataInter, :prevs => :prevalence)
dataInter = dataInter[:, [:places, :effective_prevalence, :prevalence, :art]]

data = outerjoin(data, dataInter, on = [:places, :effective_prevalence, :prevalence, :art])

dataInter = CSV.read("Dropbox/Immunization/Data/final_data.csv", DataFrame)
rename!(dataInter, :ART_coverage => :art)
dataInter = dataInter[:, [:Country, :effective_prevalence, :prevalence, :art]]
rename!(dataInter, :Country => :places)
sort!(dataInter, [:effective_prevalence])

dataInter.origin = fill("UN", length(dataInter.places))
data.origin = fill("Studies", length(data.places))

data = data[.!(∈(dataInter.places).(data.places)),:]
data = outerjoin(data, dataInter, on = [:places, :effective_prevalence, :origin, :prevalence, :art])
data
sort!(data, [:effective_prevalence])

l = length(data.effective_prevalence)
ϵcs = zeros(l)
λs = zeros(l)

Threads.@threads for i in 1:l
    λs[i] = getCorrespondance(pk, μ, data.effective_prevalence[i])

    p = [λs[i], μ, avK, ks, pk]
    sol = getSolution(p)
    z₁ = getz₁(sol, pk, ks)
    ρ = getρ(λs[i], μ, avK)
    ϕ = getϕ(ρ, pk, ks, z₁)
    ψ = getψ(ρ, pk, ks, z₁)
    ϵcs[i] =  getϵC(z₁, ϕ, ψ)
end

for i in 1:l
    if ϵcs[i] < 0 || ϵcs[i] > 1
        ϵcs[i] = 0.0
    end
end

data.ϵc = ϵcs

ϵr = zeros(l)
m = 10000
ϵs = range(0.0000001, 0.9999, length = m)

Threads.@threads for i in 1:l
    p = [λs[i], μ, avK, ks, pk]
    sol = getSolution(p)

    z₁ = getz₁(sol, pk, ks)
    ρ = getρ(λs[i], μ, avK)
    ϕ = getϕ(ρ, pk, ks, z₁)
    ψ = getψ(ρ, pk, ks, z₁)

    avF = zeros(m)
    fksMax = zeros(m)

    for j in 1:m
        fksMax[j] = ψ/z₁/(1-ϕ)*ϵs[j]/(1-ϵs[j])
        fDist = getF(ϵs[j], ρ, z₁, ϕ, ψ, ks)
        avF[j] = -sum( pk[1:end] .* fDist[1:end] ./ sum(pk[1:end]))
    end

    s = Int64.(sign.(fksMax .- avF))
    if (1 in s)
        ϵr[i] = ϵs[findfirst(s -> s == 1, s)]
    end
end

data.ϵr = ϵr

p = scatter(data.places, data.ϵc,
    #ylim = (0.6,1),
    #ylabel = L"\epsilon_c",
    label = L"\epsilon_c",
    xrotation = 90,
    bottom_margin = 50mm,
    xticks = :all,
    legend = :bottomright,
    xtickfontsize=6)
scatter!(p,
    data.places, data.ϵr,
    label = L"\epsilon_r",
)
for i in 1:l
    plot!(p, [data.places[i], data.places[i]], [data.ϵr[i], data.ϵc[i]],
    label = :none,
    color = :black)
end
plot!(p, data.places, ones(l)*0.6,
    label = :none,
    color = :black)

scatter(data[data.art .> 0,:].art, data[data.art .> 0,:].ϵc,
    legend = :none,
    xlabel = "ART coverage",
    ylabel = L"\epsilon_c")
#savefig("Dropbox/Immunization/Plots/supp_scatter_ARTvsEps")


l = length(data.effective_prevalence)
ϵcs = zeros(l)
λs = zeros(l)

Threads.@threads for i in 1:l
    I = minimum([data.prevalence[i]*0.142, data.effective_prevalence[i]])
    λs[i] = getCorrespondance(pk, μ, I)

    p = [λs[i], μ, avK, ks, pk]
    sol = getSolution(p)
    z₁ = getz₁(sol, pk, ks)
    ρ = getρ(λs[i], μ, avK)
    ϕ = getϕ(ρ, pk, ks, z₁)
    ψ = getψ(ρ, pk, ks, z₁)
    ϵcs[i] =  getϵC(z₁, ϕ, ψ)
end

for i in 1:l
    if ϵcs[i] < 0 || ϵcs[i] > 1
        ϵcs[i] = 0.0
    end
end

data.ϵc_goal = ϵcs

ϵr = zeros(l)
m = 10000
ϵs = range(0.0000001, 0.9999, length = m)

Threads.@threads for i in 1:l
    p = [λs[i], μ, avK, ks, pk]
    sol = getSolution(p)

    z₁ = getz₁(sol, pk, ks)
    ρ = getρ(λs[i], μ, avK)
    ϕ = getϕ(ρ, pk, ks, z₁)
    ψ = getψ(ρ, pk, ks, z₁)

    avF = zeros(m)
    fksMax = zeros(m)

    for j in 1:m
        fksMax[j] = ψ/z₁/(1-ϕ)*ϵs[j]/(1-ϵs[j])
        fDist = getF(ϵs[j], ρ, z₁, ϕ, ψ, ks)
        avF[j] = -sum( pk[1:end] .* fDist[1:end] ./ sum(pk[1:end]))
    end

    s = Int64.(sign.(fksMax .- avF))
    if (1 in s)
        ϵr[i] = ϵs[findfirst(s -> s == 1, s)]
    end
end

data.ϵr_goal = ϵr

#show(data[data.places .== "Abidjan (Cote D'Ivoire)",:])

p = scatter(data.places, data.ϵc_goal,
    ylim = (0.0,1),
    #ylabel = L"\epsilon_c",
    label = L"\epsilon_c",
    xrotation = 90,
    bottom_margin = 50mm,
    xticks = :all,
    legend = :bottomright,
    xtickfontsize=6)
scatter!(p,
    data.places, data.ϵr_goal,
    label = L"\epsilon_r",
)
for i in 1:l
    plot!(p, [data.places[i], data.places[i]], [data.ϵr_goal[i], data.ϵc_goal[i]],
    label = :none,
    color = :black)
end
plot!(p, data.places, ones(l)*0.6,
    label = :none,
    color = :black)

#CSV.write("Dropbox/Immunization/Plots/data/mapAux.csv", data)

#--------------- Plot Panel 3 A-----------------------
l = 100
prevs = range(0.0000001, 0.15, length = l)

μ = 1.0
mu = 2.1
sd = 11.6
#sd = 1.0
o = mu*mu/(sd*sd - mu)
r,p = paramsNegBin(mu,o)
d = NegativeBinomial(r,p)
kmax = 1000
ks = [1.0:1.0:kmax;]
pk = map(x -> pdf(d, x), ks)

avK = getavK(ks, pk)
sqrt(sum(ks.*ks.*pk) - avK^2)

ϵcs = zeros(l)
ϵr = zeros(l)
λs = zeros(l)

m = 10000
ϵs = range(0.0000001, 0.9999, length = m)

plot(pk,
    xaxis = :log,
    yaxis = :log,
    ylabel = "PDF",
    legend = :none)
plot(pk,
    ylabel = "PDF",
    legend = :none)

Threads.@threads for i in 1:l
    λs[i] = getCorrespondance(pk, μ, prevs[i], ks)

    p = [λs[i], μ, avK, ks, pk]
    sol = getSolution(p)
    z₁ = getz₁(sol, pk, ks)
    ρ = getρ(λs[i], μ, avK)
    ϕ = getϕ(ρ, pk, ks, z₁)
    ψ = getψ(ρ, pk, ks, z₁)
    ϵcs[i] =  getϵC(z₁, ϕ, ψ)

    avF = zeros(m)
    fksMax = zeros(m)

    for j in 1:m
        fksMax[j] = ψ/z₁/(1-ϕ)*ϵs[j]/(1-ϵs[j])
        fDist = getF(ϵs[j], ρ, z₁, ϕ, ψ, ks)
        avF[j] = sum( -pk[1:end] .* fDist[1:end] ./ sum(pk[1:end]))
    end

    s = Int64.(sign.(fksMax .- avF))
    if (1 in s)
        ϵr[i] = ϵs[findfirst(s -> s == 1, s)]
    end
end

for i in 1:l
    if ϵcs[i] < 0 || ϵcs[i] > 1
        ϵcs[i] = 0.0
    end
end


plot(prevs, ϵcs, label = L"\epsilon_c",
    legend = :bottomright)
plot!(prevs, ϵr, label = L"\epsilon_r",
    xlabel = "Effective Prevalence")




#writedlm("Dropbox/Immunization/Plots/data/panel3A_prevs.csv", prevs)
#writedlm("Dropbox/Immunization/Plots/data/panel3A_epsilonc.csv", ϵcs)
#writedlm("Dropbox/Immunization/Plots/data/panel3A_epsilonr.csv", ϵr)


IC = getICRegime(0.6, pk, ks, avK, μ)
IR = getIRRegime(0.6, pk, ks, avK, μ)

#writedlm("Dropbox/Immunization/Plots/data/panel3A_thresholdValues.csv", [IC, IR])

IC = getICRegime(0.44, pk, ks, avK, μ)
IR = getIRRegime(0.44, pk, ks, avK, μ)

#writedlm("Dropbox/Immunization/Plots/data/panel3A_thresholdValuesLower.csv", [IC, IR])

IC = getICRegime(0.86, pk, ks, avK, μ)
IR = getIRRegime(0.86, pk, ks, avK, μ)

#writedlm("Dropbox/Immunization/Plots/data/panel3A_thresholdValuesUpper.csv", [IC, IR])

#------------------------ Supp Case Covid  ---------------------------------

μ =  1/(3.7+1.5+2.3)

mu = 13.4
kmax = 1000
ks = [1:1:kmax;]

m = 5
os = range(0.1, 10, length = m)

l = 100
Is = range(0.01, 0.8, length = l)
λs = zeros(l,m)
ϵcs = zeros(l,m)

Threads.@threads for i in 1:m
    for j in 1:l
        r,p = paramsNegBin(mu,1.0/os[i])
        d = NegativeBinomial(r,p)
        pk = map(x -> pdf(d, x), ks)
        avK = getavK(ks, pk)

        λs[j,i] = getCorrespondance(pk, μ, Is[j])

        p = [λs[j,i], μ, avK, ks, pk]
        sol = getSolution(p)
        z₁ = getz₁(sol, pk, ks)
        ρ = getρ(λs[j,i], μ, avK)
        ϕ = getϕ(ρ, pk, ks, z₁)
        ψ = getψ(ρ, pk, ks, z₁)
        ϵcs[j,i] =  getϵC(z₁, ϕ, ψ)
        if ϵcs[j,i] > 1 || ϵcs[j,i] < 0
            ϵcs[j,i] = 0
        end
    end
end

d = Poisson(mu)
pk = map(x -> pdf(d, x), ks)
pk = pk
avK = getavK(ks, pk)

ϵsAdd = zeros(l)

Threads.@threads for i in 1:l
    λ = getCorrespondance(pk, μ, Is[i])

    p = [λ, μ, avK, ks, pk]
    sol = getSolution(p)
    z₁ = getz₁(sol, pk, ks)
    ρ = getρ(λ, μ, avK)
    ϕ = getϕ(ρ, pk, ks, z₁)
    ψ = getψ(ρ, pk, ks, z₁)
    ϵsAdd[i] =  getϵC(z₁, ϕ, ψ)
end

for i in 1:l
    if ϵsAdd[i] > 1 || ϵsAdd[i] < 0
        ϵsAdd[i] = 0
    end
end

#[mu * sqrt(1.0/i + 1/mu) for i in os]

lab = [string(round(sqrt(i + 1/mu), sigdigits = 3)) for i in os]
lab = reshape(lab, 1, length(lab))

Is = Is.*100000

plot(Is, reverse(ϵcs, dims= 2),
    label = reshape([lab[end-i] for i in 0:(length(lab)-1)], 1, length(lab)),
    legend = :outerright,
    linewidth = 2.0,
    xlabel = "7-day incidence per 100,000 individuals",
    ylabel = L"\epsilon_c",
    legendtitle  = "CV",
    formatter = identity,
    palette = ColorSchemes.PuBu_7.colors[2:end])
plot!(
    Is, ϵsAdd,
    label = "Poisson",
    linewidth = 2.0,
    color = :black
    )
#savefig("Dropbox/Immunization/Plots/supp_COV.pdf")

#writedlm("Dropbox/Immunization/Plots/data/panelSupp1_Is.csv", Is)
#writedlm("Dropbox/Immunization/Plots/data/panelSupp1_eps.csv", ϵcs)
#writedlm("Dropbox/Immunization/Plots/data/panelSupp1_epsBin.csv", ϵsAdd)


#------------------------------- Supp Plot Panel 2 A Scale Free --------------------------------
kmax = 1000
ks = [1:1:kmax;]
pk = SFDist(ks, 1.5)
avK = getavK(ks,pk)

μ = 0.1
l = 25
kmax = 1000

ϵs = range(0.0,1.0, length = l)
ϵc = zeros(l)
res = zeros(l,l)
Is = range(0.01, 0.7, length = l)

λs = zeros(l)
prevs = zeros(l)

Threads.@threads for i in 1:l
    λs[i] = getCorrespondance(pk, μ, Is[i], ks)
end

m = 10000
ϵsm = range(0.0000001, 0.9999, length = m)

ϵr = zeros(l)

Threads.@threads for i in 1:l
    p = [λs[i], μ, avK, ks, pk]
    sol = getSolution(p)

    prevs[i] = getPrevalence(sol, pk)
    z₁ = getz₁(sol, pk, ks)
    ρ = getρ(λs[i], μ, avK)
    ϕ = getϕ(ρ, pk, ks, z₁)
    ψ = getψ(ρ, pk, ks, z₁)

    ϵc[i] =  getϵC(z₁, ϕ, ψ)
    res[i,:] .= map(ϵ -> getkStar(ϵ, ρ, z₁, ϕ, ψ), ϵs)

    avF = zeros(m)
    fksMax = zeros(m)

    for j in 1:m
        fksMax[j] = ψ/z₁/(1-ϕ)*ϵsm[j]/(1-ϵsm[j])
        fDist = getF(ϵsm[j], ρ, z₁, ϕ, ψ, ks)
        avF[j] = -sum( pk[1:end] .* fDist[1:end] ./ sum(pk[1:end]))
    end

    s = Int64.(sign.(fksMax .- avF))
    if (1 in s)
        ϵr[i] = ϵsm[findfirst(s -> s == 1, s)]
    end
end


for i in 1:l
    for j in 1:l
        if res[i,j] < 0
            res[i,j] = NaN
        end
    end
end

for i in 1:l
    if ϵc[i] > 1.0
        ϵc[i] = -0.1
    end
end

heatmap(ϵs, Is, log10.(res),
    xlabel = L"\epsilon",
    ylabel = L"I",
    xlim = (minimum(ϵs), maximum(ϵs)))
plot!(ϵc, Is,
    color = :red,
    width = 2.0,
    label = L"\epsilon_c",
    legend = :bottomright)
plot!(ϵr, Is,
    color = :white,
    width = 2.0,
    label = L"\epsilon_r",
    legend = :bottomright)
#savefig("Dropbox/Immunization/Plots/heatmapKStar.pdf")

#writedlm("Dropbox/Immunization/Plots/data/Supp_panel1C_axis.csv", [ϵs, Is])
#writedlm("Dropbox/Immunization/Plots/data/Supp_panel1C_heatmap.csv", res)
#writedlm("Dropbox/Immunization/Plots/data/Supp_panel1C_critical.csv", [ϵc ϵr])

#--------------------- Supp Plot Panel 2 B Scale Free ---------------

l = 50
γs = range(1.5, 3.0, length = l)
μ = 0.1
kmax = 1000
ks = [1:1:kmax;]

Is = [0.2, 0.3, 0.4, 0.5, 0.6]
m = length(Is)

λs = zeros(l,m)

Threads.@threads for i in 1:l
    for j in 1:m
        pk = SFDist(ks, γs[i])
        avK = getavK(ks,pk)
        λs[i,j] = getCorrespondance(pk, μ, Is[j])
    end
end

ϵcs = zeros(l,m)

Threads.@threads for i in 1:l
    for j in 1:m
        pk = SFDist(ks, γs[i])
        avK = getavK(ks,pk)

        p = [λs[i,j], μ, avK, ks, pk]
        sol = getSolution(p)

        z₁ = getz₁(sol, pk, ks)
        ρ = getρ(λs[i,j], μ, avK)
        ϕ = getϕ(ρ, pk, ks, z₁)
        ψ = getψ(ρ, pk, ks, z₁)

        ϵcs[i,j] =  getϵC(z₁, ϕ, ψ)
        if ϵcs[i,j] > 1.0
            ϵcs[i,j] = -1
        end
    end
end

lab = ["I = "*string(round(i, sigdigits = 2)) for i in Is]
lab = reshape(lab, 1, length(lab))
cs = palette(:GnBu_6)[2:end]

plot(γs, ϵcs,
    ylim = (0,1),
    label = lab,
    xlab = L"\gamma",
    ylab = L"\epsilon_c",
    linewidth = 2.0,
    palette = ColorSchemes.PuBu_6.colors[2:end],
    legend = :bottomright)

#writedlm("Dropbox/Immunization/Plots/data/supp_panel1D_epsilons.csv", [ϵcs])
#writedlm("Dropbox/Immunization/Plots/data/supp_panel1D_gammas.csv", [γs])
#writedlm("Dropbox/Immunization/Plots/data/supp_panel1D_Is.csv", [Is])

#--------------------- Supp Plot Panel 1 A/B Theory vs. Numerics ---------------
μ = 0.1
mu = 2.0
o = 0.05
coeff_var = sqrt(mu+1/o)
r,p = paramsNegBin(mu,o)
d = NegativeBinomial(r,p)

kmax = 500
ks = [1:1:kmax;]
pk = map(x -> pdf(d, x), ks)
avK = getavK(ks,pk)

μ = 0.1
ϵ = 0.8
λ = 0.05
p = [λ, μ, avK, ks, pk]
sol = getSolution(p)
getPrevalence(sol, pk)
N = length(ks)
gk = zeros(N)
α = 1.0
p = [ks, pk, avK, gk, λ, μ, ϵ, α,  N]
sol_assort = getSolution_assort(p)
prev0 = getPrevalence_assort(sol_assort, pk, gk, N)

minPk = minimum(pk)
intr = 1.0*minPk
fs_assort = zeros(N)
fs_indirect_assort = zeros(N)

Threads.@threads for i in 1:N
    gk = zeros(N)
    gk[i] = intr/pk[i]
    p = [ks, pk, avK, gk, λ, μ, ϵ, α,  N]
    sol_int = getSolution_assort(p)
    prev = getPrevalence_assort(sol_int, pk, gk, N)
    fs_assort[i] = (prev0 - prev)/intr
    fs_indirect_assort[i] = sum((sol_assort.u[1:N] - sol_int[1:N]) .* pk)/intr
end

redL = λ/μ/avK

z₁ = getz₁(sol, pk, ks)
ρ = getρ(λ, μ, avK)
ϕ = getϕ(ρ, pk, ks, z₁)
ψ = getψ(ρ, pk, ks, z₁)
kStar = getkStar(ϵ, ρ, z₁, ϕ, ψ)

xk = z₁*ρ .* ks ./ (1.0 .+ z₁*ρ .* ks)
yk = (1-ϵ)*xk ./ (1 .- ϵ*xk)
Jkk = zeros(kmax,kmax)
for i in 1:kmax
    for j in 1:kmax
        Jkk[i,j] = ρ*ks[i]*(1-xk[i])*pk[j]*ks[j]*(yk[j]-xk[j])/(1+z₁*ρ*ks[i])/ (1-ϕ)
    end
end

firstTerm = zeros(kmax)
firstTerm .= yk - xk
secondTerm = zeros(kmax)
for i in 1:kmax
    secondTerm[i] = 1.0/pk[i] .* sum( pk .* Jkk[:,i] )
end

kStarP = 1/sqrt(1-ϵ)/ρ/z₁
xk = z₁*ρ * kStarP / (1.0 + z₁*ρ * kStarP)
yk = (1-ϵ)*xk / (1 - ϵ*xk)
valueMin = yk - xk

total = firstTerm .+ secondTerm
minF = getF(ϵ, ρ, z₁, ϕ, ψ, kStar)

kPlot = 200
scatter(ks, sol_assort.u[1:500] - sol_assort.u[501:end],
    xscale = :log10,
    label = L"F_{dir}",
    xlim = (1,kPlot),
    markershape = :circle,
    color = :lightblue)
scatter!(ks, -firstTerm,
    markershape = :x,
    color = :black,
    label = :none)
scatter!(ks, fs_indirect_assort,
    label = L"F_{indir}",
    color = :lightgreen)
scatter!([kStarP], [-valueMin],
    label = :none,
    markercolor = :red,
    markershape = :hexagon,
    markerstrokecolor = :black,
    markerstrokewidth = 2,
    markersize = 8)
scatter!(fs_assort,
    xscale = :log10,
    label = L"f(k)",
    xlim = (1,kPlot),
    xlabel = L"k")
scatter!(-total,
    xscale = :log10,
    label = :none,
    xlim = (1,kPlot),
    markershape = :x,
    color = :black)
scatter!([kStar], [-minF],
    label = :none,
    markercolor = :red,
    markershape = :hexagon,
    markerstrokecolor = :black,
    markerstrokewidth = 2,
    markersize = 8)
scatter!(-secondTerm,
    xscale = :log10,
    xlim = (1,kPlot),
    markershape = :x,
    color = :black,
    label = "Theory")
savefig("Dropbox/Immunization/Plots/supp_TheoryVSNumerics_1.pdf")

#writedlm("Dropbox/Immunization/Plots/data/supp_panel1AB_theoryNumerics_ks.csv", [ks, sol_assort.u[1:500] - sol_assort.u[501:end], fs_indirect_assort, fs_assort])


#------------------------------- Supp Plot Panel 1 C Theory vs. numerics --------------------------------
μ = 0.1
mu = 2.0
o = 0.05
r,p = paramsNegBin(mu,o)
d = NegativeBinomial(r,p)

kmax = 500
ks = [1:1:kmax;]
pk = map(x -> pdf(d, x), ks)
avK = getavK(ks,pk)

μ = 0.1
ϵ = 0.8

l = 5
ϵs = [0.5, 0.7, 0.8, 0.85, 0.88]
l = length(ϵs)
fs = zeros(kmax, l)
kStars = zeros(l)
vals = zeros(l)

λ = 0.025

p = [λ, μ, avK, ks, pk]
sol = getSolution(p)
getPrevalence(sol, pk)

z₁ = getz₁(sol, pk, ks)
ρ = getρ(λ, μ, avK)
ϕ = getϕ(ρ, pk, ks, z₁)
ψ = getψ(ρ, pk, ks, z₁)

minPk = minimum(pk)
intr = 1.0*minPk
fs_assort = zeros(kmax, l)
N = length(ks)
gk = zeros(N)
α = 1.0
p = [ks, pk, avK, gk, λ, μ, ϵ, α, N]
sol_assort = getSolution_assort(p)
prev0 = getPrevalence_assort(sol_assort, pk, gk, N)

for i in 1:l
    print(i)
    print(" ")
    kStars[i] = getkStar(ϵs[i], ρ, z₁, ϕ, ψ)
    vals[i] = getF(ϵs[i], ρ, z₁, ϕ, ψ, kStars[i])
    fs[:,i] .= getF(ϵs[i], ρ, z₁, ϕ, ψ, ks)
    Threads.@threads for j in 1:N
        gk = zeros(N)
        gk[j] = intr/pk[j]
        p = [ks, pk, avK, gk, λ, μ, ϵs[i], α,  N]
        sol_int = getSolution_assort(p)
        prev = getPrevalence_assort(sol_int, pk, gk, N)
        fs_assort[j,i] = (prev0 - prev)/intr
    end
end


N = 500
ϵsAdd = range(0.1, 0.865, length = N)
kStarsAdd = zeros(N)
valsAdd = zeros(N)

for i in 1:N
    kStarsAdd[i] = getkStar(ϵsAdd[i], ρ, z₁, ϕ, ψ)
    valsAdd[i] = getF(ϵsAdd[i], ρ, z₁, ϕ, ψ, kStarsAdd[i])
end

stop = 500
lab = ["eps = "*string(round(i, sigdigits = 2)) for i in ϵs]
lab = reshape(lab, 1, length(lab))
plot(kStarsAdd, -valsAdd,
    linestyle = :dash,
    color = :grey,
    label = L"k^*")
plot!(ks, fs_assort,
    xscale = :log,
    marker = :dot,
    label = lab)
scatter!(kStars, -vals,
    color = :grey,
    label = :none,
    marker = :dot,
    alpha = 0.5,
    markersize = 6,
    markerstrokewidth = 2,
    xlabel = L"k")
plot!(ks, -fs,
    xscale = :log,
    marker = :x,
    color = :black,
    label = :none)
savefig("Dropbox/Immunization/Plots/supp_TheoryVSNumerics_2.pdf")

#writedlm("Dropbox/Immunization/Plots/data/supp_panel1C_theoryNumerics_fks.csv", fs[:,:])
#writedlm("Dropbox/Immunization/Plots/data/supp_panel1C_theoryNumerics_fks_numerics.csv", fs_assort[:,:])
#writedlm("Dropbox/Immunization/Plots/data/supp_panel1C_theoryNumerics_ks.csv", ks[:,:])
#writedlm("Dropbox/Immunization/Plots/data/supp_panel1C_theoryNumerics_kStarScatter.csv", [kStars vals])
#writedlm("Dropbox/Immunization/Plots/data/supp_panel1C_theoryNumerics_kStarLine.csv", [kStarsAdd valsAdd])


#------------------------------- Plot Panel 1 C - Assortativity --------------------------------
μ = 0.1
mu = 2.0
o = 0.05
r,p = paramsNegBin(mu,o)
d = NegativeBinomial(r,p)

kmax = 500
ks = [1:1:kmax;]
pk = map(x -> pdf(d, x), ks)
avK = getavK(ks,pk)

μ = 0.1
ϵ = 0.8

l = 5
ϵs = [0.5, 0.7, 0.8, 0.85, 0.88]
l = length(ϵs)
fs = zeros(kmax, l)
kStars = zeros(l)
vals = zeros(l)

λ = 0.025

p = [λ, μ, avK, ks, pk]
sol = getSolution(p)
prevMix = getPrevalence(sol, pk)

z₁ = getz₁(sol, pk, ks)
ρ = getρ(λ, μ, avK)
ϕ = getϕ(ρ, pk, ks, z₁)
ψ = getψ(ρ, pk, ks, z₁)

minPk = minimum(pk)
intr = 0.001*minPk
fs_assort = zeros(kmax, l)
N = length(ks)
gk = zeros(N)
α = 0.86
λAssort = getCorrespondance_assort(pk, μ, prevMix, ks, α)
p = [ks, pk, avK, gk, λAssort, μ, ϵ, α, N]
sol_assort = getSolution_assort(p)
prev0 = getPrevalence_assort(sol_assort, pk, gk, N)

for i in 1:l
    print(i)
    print(" ")
    kStars[i] = getkStar(ϵs[i], ρ, z₁, ϕ, ψ)
    vals[i] = getF(ϵs[i], ρ, z₁, ϕ, ψ, kStars[i])
    fs[:,i] .= getF(ϵs[i], ρ, z₁, ϕ, ψ, ks)
    Threads.@threads for j in 1:N
        gk = zeros(N)
        gk[j] = intr/pk[j]
        p = [ks, pk, avK, gk, λAssort, μ, ϵs[i], α,  N]
        sol_int = getSolution_assort(p)
        prev = getPrevalence_assort(sol_int, pk, gk, N)
        fs_assort[j,i] = (prev0 - prev)/intr
    end
end


N = 500
ϵsAdd = range(0.1, 0.865, length = N)
kStarsAdd = zeros(N)
valsAdd = zeros(N)

for i in 1:N
    kStarsAdd[i] = getkStar(ϵsAdd[i], ρ, z₁, ϕ, ψ)
    valsAdd[i] = getF(ϵsAdd[i], ρ, z₁, ϕ, ψ, kStarsAdd[i])
end

stop = 500
lab = ["eps = "*string(round(i, sigdigits = 2)) for i in ϵs]
lab = reshape(lab, 1, length(lab))
plot(kStarsAdd, -valsAdd,
    linestyle = :dash,
    color = :grey,
    label = L"k^*")
plot!(ks, fs_assort,
    xscale = :log,
    marker = :dot,
    label = lab)
scatter!(kStars, -vals,
    color = :grey,
    label = :none,
    marker = :dot,
    alpha = 0.5,
    markersize = 6,
    markerstrokewidth = 2,
    xlabel = L"k")
plot!(ks, -fs,
    xscale = :log,
    #marker = :x,
    color = :black,
    label = :none)
savefig("Dropbox/Immunization/Plots/supp_Assort_2_fixed.pdf")


#------------------ Supp - Plot Panel 1 A&B - Assortativity --------------------------
μ = 0.1
mu = 2.0
o = 0.05
coeff_var = sqrt(mu+1/o)
r,p = paramsNegBin(mu,o)
d = NegativeBinomial(r,p)

kmax = 1000
ks = [1:1:kmax;]
pk = map(x -> pdf(d, x), ks)
avK = getavK(ks,pk)

μ = 0.1
ϵ = 0.8
λ = 0.05

α = 1.0
N = length(ks)
gk = zeros(N)

p = [λ, μ, avK, ks, pk]
sol = getSolution(p)
prev = getPrevalence(sol, pk)

l = 4
αs = [1.0, 0.9, 0.8, 0.7]

λs = zeros(l)
Threads.@threads for i in 1:l
    λs[i] = getCorrespondance_assort(pk, μ, prev, ks, αs[i])
end

f_tot = zeros(N,l)
f_indir = zeros(N,l)
f_dir = zeros(N,l)

for i in 1:l
    a,b,c = getF_assort_detailed(ks, pk, avK, gk, λs[i], μ, ϵ, αs[i], N)
    print(length(a))
    f_tot[:,i] = a
    f_indir[:,i] = b
    f_dir[:,i] = c
end

lab = ["alpha = "*string(round(i, sigdigits = 2)) for i in αs]
lab = reshape(lab, 1, length(lab))

plot(ks, f_tot,
    label = lab,
    xscale = :log10,
    linecolor = ["red" "blue" "green" "black"],
    xlabel = L"k",
    ylabel = L"f(k)")
savefig("Dropbox/Immunization/Plots/fk_assort.pdf")

plot(f_indir,
    label = :none,
    xscale = :log10,
    line = ["red" "blue" "green" "black"],
    linestyle = :dash,
    xlabel = L"k")
plot!(f_dir, label = lab,
    line = ["red" "blue" "green" "black"])
plot!([10, 10], [0.2, 0.2],
    color = :black,
    label = L"f_{dir}")
plot!([10, 10], [0.2, 0.2],
    color = :black,
    linestyle = :dash,
    label = L"f_{indir}")
savefig("Dropbox/Immunization/Plots/fdir_indir_assort.pdf")

#writedlm("Dropbox/Immunization/Plots/data/f_assort_ks.csv", [ks])
#writedlm("Dropbox/Immunization/Plots/data/f_assort_f_tot.csv", [f_tot])
#writedlm("Dropbox/Immunization/Plots/data/f_assort_f_indir.csv", [f_indir])
#writedlm("Dropbox/Immunization/Plots/data/f_assort_f_dir.csv", [f_dir])

#-------------------------- Supp - Plot Panel 1c - Assortativity ---------------
μ = 0.1
mu = 2.0
o = 0.05
r,p = paramsNegBin(mu,o)
d = NegativeBinomial(r,p)

α = 0.86

kmax = 1000
ks = [1:1:kmax;]
pk = map(x -> pdf(d, x), ks)
avK = getavK(ks,pk)

μ = 0.1
ϵ = 0.8

l = 5
ϵs = [0.5, 0.7, 0.8, 0.85, 0.88]
l = length(ϵs)
fs = zeros(kmax, l)
kStars = zeros(l)
vals = zeros(l)

λ = 0.025

p = [λ, μ, avK, ks, pk]
sol = getSolution(p)
prevMix = getPrevalence(sol, pk)

z₁ = getz₁(sol, pk, ks)
ρ = getρ(λ, μ, avK)
ϕ = getϕ(ρ, pk, ks, z₁)
ψ = getψ(ρ, pk, ks, z₁)

fs_assort = zeros(kmax, l)
N = length(ks)
gk = zeros(N)

α = 0.86
λAssort = getCorrespondance_assort(pk, μ, prevMix, ks, α)
p = [ks, pk, avK, gk, λAssort, μ, ϵ, α, N]
sol_assort = getSolution_assort(p)
prev0 = getPrevalence_assort(sol_assort, pk, gk, N)

for i in 1:l
    print(i)
    print(" ")
    fs_assort[:,i] .= getF_assort(ks, pk, avK, gk, λAssort, μ, ϵs[i], α, N)
    kStars[i] = getkStar(ϵs[i], ρ, z₁, ϕ, ψ)
    vals[i] = getF(ϵs[i], ρ, z₁, ϕ, ψ, kStars[i])
    fs[:,i] .= getF(ϵs[i], ρ, z₁, ϕ, ψ, ks)
end

N = length(ks)
gk = zeros(N)
λ_hat = λAssort/μ
p = [ks, pk, avK, gk, λAssort, μ, ϵ, α,  N]
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

N = 1000
ϵsAdd = range(0.1, 0.896, length = N)
kStarsAdd = zeros(N)
valsAdd = zeros(N)

Nk = 100000
ksAdd = range(0.01, 5000, length = Nk)

for i in 1:l
    fk_assort_test = zeros(Nk)
    for j in 1:Nk
        a = λ_hat*w*ksAdd[j]
        b = 1 + λ_hat*(1-w)*ksAdd[j]*z/avK - λ_hat*w*ksAdd[j]
        c_add = -λ_hat*ksAdd[j]*(1-w)*z/avK

        xk = (-b + sqrt( b^2 - 4*a * c_add)) / (2*a)
        yk = (1.0 - ϵs[i]) * xk / (1.0 - ϵs[i]*xk)

        d = 1.0 + λ_hat*ksAdd[j] * (z/avK + w*(2*xk - 1.0 - z/avK))

        fk_assort_test[j] = (xk - yk) * (1 + λ_hat*ksAdd[j] * ((1.0 - w)*Γ + w * (1.0 - xk) * (1.0 + c * ksAdd[j]) / d))
    end
    maxInd = argmax(fk_assort_test)[1]
    kStars[i] = ksAdd[maxInd]
    vals[i] = fk_assort_test[maxInd]
end

Threads.@threads for i in 1:N
    fk_assort_test = zeros(Nk)
    for j in 1:Nk
        a = λ_hat*w*ksAdd[j]
        b = 1 + λ_hat*(1-w)*ksAdd[j]*z/avK - λ_hat*w*ksAdd[j]
        c_add = -λ_hat*ksAdd[j]*(1-w)*z/avK

        xk = (-b + sqrt( b^2 - 4*a * c_add)) / (2*a)
        yk = (1.0 - ϵsAdd[i]) * xk / (1.0 - ϵsAdd[i]*xk)

        d = 1.0 + λ_hat*ksAdd[j] * (z/avK + w*(2*xk - 1.0 - z/avK))

        fk_assort_test[j] = (xk - yk) * (1 + λ_hat*ksAdd[j] * ((1.0 - w)*Γ + w * (1.0 - xk) * (1.0 + c * ksAdd[j]) / d))
    end
    maxInd = argmax(fk_assort_test)[1]
    kStarsAdd[i] = ksAdd[maxInd]
    valsAdd[i] = fk_assort_test[maxInd]
end

lab = ["eps = "*string(round(i, sigdigits = 2)) for i in ϵs]
lab = reshape(lab, 1, length(lab))
plot(kStarsAdd, valsAdd,
    linestyle = :dash,
    color = :grey,
    label = L"k^*")
plot!(ks, fs_assort,
    xscale = :log,
    marker = :dot,
    label = lab)
scatter!(kStars, vals,
    color = :grey,
    label = :none,
    marker = :dot,
    alpha = 0.5,
    markersize = 6,
    markerstrokewidth = 2,
    xlabel = L"k",
    legend = :topleft)
plot!(ks, -fs,
    xscale = :log,
    #marker = :x,
    color = :black,
    label = :none,
    legend = :topleft)
savefig("Dropbox/Immunization/Plots/supp_Assort_2_fixed.pdf")

#writedlm("Dropbox/Immunization/Plots/data/supp_Assort_1c.csv", [fs_assort])
#writedlm("Dropbox/Immunization/Plots/data/supp_Assort_1c_add.csv", [kStarsAdd valsAdd])
#writedlm("Dropbox/Immunization/Plots/data/supp_Assort_1c_kStars.csv", [kStars, vals])


#--------------- Supp - Plot Panel 3 A&B - Assortativity -----------------------
l = 100
prevs = range(0.0000001, 0.15, length = l)

μ = 1.0
mu = 2.1
sd = 11.6
#sd = 1.0
o = mu*mu/(sd*sd - mu)
r,p = paramsNegBin(mu,o)
d = NegativeBinomial(r,p)
kmax = 1000
ks = [1.0:1.0:kmax;]
pk = map(x -> pdf(d, x), ks)

avK = getavK(ks, pk)

ϵcs = zeros(l)
ϵr = zeros(l)
λs = zeros(l)

m = 10000
ϵs = range(0.0000001, 0.9999, length = m)

α = 0.86
N = length(ks)
gk = zeros(N)

ϵr = zeros(l)
ϵc = zeros(l)

sols = [sol for i in 1:50]

Threads.@threads for i in 2:l
    λs[i] = getCorrespondance_assort(pk, μ, prevs[i], ks, α)

    avF = zeros(m)
    fksMax = zeros(m)
    fksStar = zeros(m)
    fksIndex = zeros(m)

    p = [ks, pk, avK, gk, λs[i], μ, 0.0, α,  N]
    sol = getSolution_assort(p)
    #sols[i] = sol

    for j in 1:m
        fDist = getF_assort_sol(sol, ks, pk, avK, gk, λs[i], μ, ϵs[j], α,  N)
        fksMax[j] = fDist[end]
        avF[j] = sum(pk[1:end] .* fDist[1:end] ./ sum(pk[1:end]))
        fksStar[j] = maximum(fDist)
        fksIndex[j] = (fDist[N-1] > fDist[N])
    end

    s = Int64.(sign.(fksMax .- avF))
    if (1 in s)
        ϵr[i] = ϵs[findfirst(s -> s == 1, s)]
    end

    s = Int64.(sign.(fksMax .- fksStar))
    if (1 in fksIndex)
        ϵc[i] = ϵs[findfirst(fksIndex -> fksIndex == 0, fksIndex)]
    end

end

prevs_or = readdlm("Dropbox/Immunization/Plots/data/panel3A_prevs.csv")
ϵc_or = readdlm("Dropbox/Immunization/Plots/data/panel3A_epsilonc.csv")
ϵr_or = readdlm("Dropbox/Immunization/Plots/data/panel3A_epsilonr.csv")


plot(prevs, ϵc,
    label = "epsilon_c Assort",
    xlabel = L"I",
    legend = :topleft,
    width = 2)
plot!(prevs, ϵr,
    label = "epsilon_r Assort",
    width = 2)
plot!(prevs_or, ϵc_or, label = "epsilon_c",
    width = 2)
plot!(prevs_or, ϵr_or, label = "epsilon_r",
    width = 2)

#writedlm("Dropbox/Immunization/Plots/data/supp_panel3A_assort.csv", [ϵc ϵr])

IRAssort_middle = getIRRegime_assort(0.6, pk, ks, avK, μ, N, α)
IRAssort_down = getIRRegime_assort(0.44, pk, ks, avK, μ, N, α)
IRAssort_up = getIRRegime_assort(0.86, pk, ks, avK, μ, N, α)

ICAssort_middle = getICRegime_assort(0.6, pk, ks, avK, μ, N, α)
ICAssort_down = getIRRegime_assort(0.44, pk, ks, avK, μ, N, α)
ICAssort_up = getIRRegime_assort(0.86, pk, ks, avK, μ, N, α)

#writedlm("Dropbox/Immunization/Plots/data/panel3A_thresholdValues_assort.csv", [ICAssort_middle, IRAssort_middle])

α = 0.72
IRAssort_middle = getIRRegime_assort(0.6, pk, ks, avK, μ, N, α)
ICAssort_middle = getICRegime_assort(0.6, pk, ks, avK, μ, N, α)

writedlm("Dropbox/Immunization/Plots/data/panel3A_thresholdValues_assort_2nd.csv", [ICAssort_middle, IRAssort_middle])


#------------------  Plot ϵc and ϵr for assortativity - numerically --------------------------
α = 0.86

μ = 0.1
mu = 2.1
sd = 11.6
o = mu*mu/(sd*sd - mu)
r,p = paramsNegBin(mu,o)

d = NegativeBinomial(r,p)

kmax = 500
ks = [1:1:kmax;]
pk = map(x -> pdf(d, x), ks)
avK = getavK(ks,pk)
N = length(ks)
intr = minimum(pk)

l = 10
Is = range(0.0000001, 0.12, length = l)
#Is = [0.06]

λs = zeros(l)
λsAssort = zeros(l)

Threads.@threads for i in 1:l
    λs[i] = getCorrespondance(pk, μ, Is[i])
end

Threads.@threads for i in 1:l
    λsAssort[i] = getCorrespondance_assort(pk, μ, Is[i], ks, α)
end

λs

p = [ks, pk, avK, zeros(N), λs[4], μ, ϵ, α,  N]

sol_int = getSolution_assort(p)
getPrevalence_assort(sol_int, pk, gk, N)

N = length(ks)
gk = zeros(N)
λAssort_1 = getCorrespondance_assort(pk, μ, Is[1], ks, α)
p = [ks, pk, avK, gk, λsAssort[4], μ, ϵ, α,  N]
sol_assort = getSolution_assort(p)
prev0 = getPrevalence_assort(sol_assort, pk, gk, N)
Is[4]
m = 20
ϵs = range(0.01, 0.99, length = m)

fInf = zeros(m, l)
fMax = zeros(m, l)
fRand = zeros(m, l)
epsilon_c = zeros(m,l)

prevs = zeros(l)

for i in 1:l
    p = [λs[i], μ, avK, ks, pk]
    sol = getSolution(p)

    prevs[i] = getPrevalence(sol, pk)
    z₁ = getz₁(sol, pk, ks)
    ρ = getρ(λs[i], μ, avK)
    ϕ = getϕ(ρ, pk, ks, z₁)
    ψ = getψ(ρ, pk, ks, z₁)

    for j in 1:m
        fDist = -1.0*getF(ϵs[j], ρ, z₁, ϕ, ψ, ks)
        fInf[j,i] = fDist[end]
        fMax[j,i] = maximum(fDist)
        fRand[j,i] = sum(fDist .* pk) / sum(pk)
        epsilon_c[j,i] = getϵC(z₁, ϕ, ψ)
    end
end

print(fInfAssort[1,:])

fInfAssort = zeros(m, l)
fMaxAssort = zeros(m, l)
fRandAssort = zeros(m, l)
epsilon_cAssort = zeros(m,l)

for i in 1:l
    gk = zeros(N)
    p = [ks, pk, avK, gk, λs[i], μ, 0.4, 1.0,  N]
    sol_int = getSolution_assort(p)
    prev = getPrevalence_assort(sol_int, pk, gk, N)
    print(prev - Is[i])
    print("\n")
end

for i in 1:l
    print("Prevalence:")
    print(" ")
    print(i)
    print("\n")
    for j in 1:m
        print(j)
        print(" ")
        fDist = zeros(N)
        Threads.@threads for k in 1:N
            gk = zeros(N)
            gk[k] = intr/pk[k]
            p = [ks, pk, avK, gk, λsAssort[i], μ, ϵs[j], α,  N]
            sol_int = getSolution_assort(p)
            prev = getPrevalence_assort(sol_int, pk, gk, N)
            fDist[k] = (Is[i] - prev)/intr
        end
        if findmax(fDist) != length(fDist)
            epsilon_cAssort[j,i] = 1.0
        end
        fInfAssort[j,i] = fDist[end]
        fMaxAssort[j,i] = maximum(fDist)
        fRandAssort[j,i] = sum(fDist .* pk) / sum(pk)
    end
    print("\n")
end

print(fInfAssort[1,:])

epsilon_r_ = zeros(l)
epsilon_c = zeros(l)

epsilon_r_Assort = zeros(l)
epsilon_c_Assort = zeros(l)

for i in 1:l
    bigger = 0
    j = 0
    while bigger == 0 && j < m
        if fInf[m-j,i] < fRand[m-j,i]
            bigger = 1
            epsilon_r[i] = ϵs[m-j]
        end
        j += 1
    end

    bigger = 0
    j = 0
    while bigger == 0 && j < m
        if fMax[m-j,i] != fInf[m-j,i]
            bigger = 1
            epsilon_c[i] = ϵs[m-j]
        end
        j += 1
    end
end

for i in 1:l
    bigger = 0
    j = 0
    while bigger == 0 && j < m
        if fInfAssort[m-j,i] < fRandAssort[m-j,i]
            bigger = 1
            epsilon_r_Assort[i] = ϵs[m-j]
        end
        j += 1
    end

    bigger = 0
    j = 0
    while bigger == 0 && j < m
        if fMaxAssort[m-j,i] != fInfAssort[m-j,i]
            bigger = 1
            epsilon_c_Assort[i] = ϵs[m-j]
        end
        j += 1
    end
end
epsilon_r

plot(Is, epsilon_r,
    label = L"\epsilon_r",
    color = "red")
plot!(Is, epsilon_c,
    label = L"\epsilon_c",
    xlabel = L"I",
    color = "blue")
plot!(Is, epsilon_r_Assort,
    label = :none,
    color = "red",
    linestyle = :dash)
plot!(Is, epsilon_c_Assort,
    label = :none,
    color = "blue",
    linestyle = :dash)
plot!([0.1, 0.1], [0.2,0.2],
    color = "black",
    label = "Random Mixing")
plot!([0.1, 0.1], [0.2,0.2],
        color = "black",
        label = "Assortative",
        linestyle = :dash)

#------------------  Plot f(k) random vs. assortative to verify the calculations --------------------------
μ = 0.1
mu = 2.0
o = 0.05
coeff_var = sqrt(mu+1/o)
r,p = paramsNegBin(mu,o)
d = NegativeBinomial(r,p)

kmax = 1000
ks = [1:1:kmax;]
pk = map(x -> pdf(d, x), ks)
avK = getavK(ks,pk)

μ = 0.1
ϵ = 0.8
λ = 0.05
p = [λ, μ, avK, ks, pk]
sol = getSolution(p)

redL = λ/μ/avK

z₁ = getz₁(sol, pk, ks)
ρ = getρ(λ, μ, avK)
ϕ = getϕ(ρ, pk, ks, z₁)
ψ = getψ(ρ, pk, ks, z₁)
kStar = getkStar(ϵ, ρ, z₁, ϕ, ψ)

xk = z₁*ρ .* ks ./ (1.0 .+ z₁*ρ .* ks)
yk = (1-ϵ)*xk ./ (1 .- ϵ*xk)
Jkk = zeros(kmax,kmax)
for i in 1:kmax
    for j in 1:kmax
        Jkk[i,j] = ρ*ks[i]*(1-xk[i])*pk[j]*ks[j]*(yk[j]-xk[j])/(1+z₁*ρ*ks[i])/ (1-ϕ)
    end
end

firstTerm = zeros(kmax)
firstTerm .= yk - xk
secondTerm = zeros(kmax)
for i in 1:kmax
    secondTerm[i] = 1.0/pk[i] .* sum( pk .* Jkk[:,i] )
end

kStarP = 1/sqrt(1-ϵ)/ρ/z₁
xk = z₁*ρ * kStarP / (1.0 + z₁*ρ * kStarP)
yk = (1-ϵ)*xk / (1 - ϵ*xk)
valueMin = yk - xk

kPlot = 200
scatter(ks, -firstTerm,
    xscale = :log10,
    label = L"F_{dir}",
    xlim = (1,kPlot),
    markershape = :circle)
scatter!([kStarP], [-valueMin],
    label = :none,
    markercolor = :red,
    markershape = :hexagon,
    markerstrokecolor = :black,
    markerstrokewidth = 2,
    markersize = 8)

scatter!(-secondTerm,
    xscale = :log10,
    label = L"F_{indir}",
    xlim = (1,kPlot))


total = getF(ϵ, ρ, z₁, ϕ, ψ, ks)
minF = getF(ϵ, ρ, z₁, ϕ, ψ, kStar)
scatter!(-total,
    xscale = :log10,
    label = L"f(k)",
    xlim = (1,kPlot),
    xlabel = L"k")
scatter!([kStar], [-minF],
    label = :none,
    markercolor = :red,
    markershape = :hexagon,
    markerstrokecolor = :black,
    markerstrokewidth = 2,
    markersize = 8)


α = 1.0
N = length(ks)
gk = zeros(N)

fDist_New, f_indir_assort, f_dir_assort = getF_assort_detailed(ks, pk, avK, gk, λ, μ, ϵ, α, N)

scatter(ks, f_indir_assort, label = "f_indir_assort")
scatter!(ks, -firstTerm,
    xscale = :log10,
    label = L"F_{dir}",
    xlim = (1,kPlot),
    markershape = :circle)
scatter!(ks, f_dir_assort, label = "f_dir_assort",
    xscale = :log10)
scatter!(-secondTerm,
    xscale = :log10,
    label = L"F_{indir}",
    xlim = (1,kPlot))

scatter(fDist_New)
scatter!(-total,
    xscale = :log10,
    label = L"f(k)",
    xlim = (1,kPlot),
    xlabel = L"k")

#---------------------  Plot Panel 1 A/B Assortativity - Numerically ---------------
α = 0.86

μ = 0.1
mu = 2.0
o = 0.05
coeff_var = sqrt(mu+1/o)
r,p = paramsNegBin(mu,o)
d = NegativeBinomial(r,p)

kmax = 500
ks = [1:1:kmax;]
pk = map(x -> pdf(d, x), ks)
avK = getavK(ks,pk)

μ = 0.1
ϵ = 0.8
λ = 0.05
p = [λ, μ, avK, ks, pk]
sol = getSolution(p)
prevMix = getPrevalence(sol, pk)

N = length(ks)
gk = zeros(N)
λAssort = getCorrespondance_assort(pk, μ, prevMix, ks, α)
p = [ks, pk, avK, gk, λAssort, μ, ϵ, α,  N]
sol_assort = getSolution_assort(p)
prev0 = getPrevalence_assort(sol_assort, pk, gk, N)

minPk = minimum(pk)
intr = 0.001*minPk
fs_assort = zeros(N)
fs_indirect_assort = zeros(N)

Threads.@threads for i in 1:N
    gk = zeros(N)
    gk[i] = intr/pk[i]
    p = [ks, pk, avK, gk, λAssort, μ, ϵ, α,  N]
    sol_int = getSolution_assort(p)
    prev = getPrevalence_assort(sol_int, pk, gk, N)
    fs_assort[i] = (prev0 - prev)/intr
    fs_indirect_assort[i] = sum((sol_assort.u[1:N] - sol_int[1:N]) .* pk)/intr
end

redL = λ/μ/avK

z₁ = getz₁(sol, pk, ks)
ρ = getρ(λ, μ, avK)
ϕ = getϕ(ρ, pk, ks, z₁)
ψ = getψ(ρ, pk, ks, z₁)
kStar = getkStar(ϵ, ρ, z₁, ϕ, ψ)

xk = z₁*ρ .* ks ./ (1.0 .+ z₁*ρ .* ks)
yk = (1-ϵ)*xk ./ (1 .- ϵ*xk)
Jkk = zeros(kmax,kmax)
for i in 1:kmax
    for j in 1:kmax
        Jkk[i,j] = ρ*ks[i]*(1-xk[i])*pk[j]*ks[j]*(yk[j]-xk[j])/(1+z₁*ρ*ks[i])/ (1-ϕ)
    end
end

firstTerm = zeros(kmax)
firstTerm .= yk - xk
secondTerm = zeros(kmax)
for i in 1:kmax
    secondTerm[i] = 1.0/pk[i] .* sum( pk .* Jkk[:,i] )
end

total = getF(ϵ, ρ, z₁, ϕ, ψ, ks)
minF = getF(ϵ, ρ, z₁, ϕ, ψ, kStar)

kStarP = 1/sqrt(1-ϵ)/ρ/z₁
xk = z₁*ρ * kStarP / (1.0 + z₁*ρ * kStarP)
yk = (1-ϵ)*xk / (1 - ϵ*xk)
valueMin = yk - xk
plot(fs_indirect_assort,
    xscale = :log10)
kPlot = 200
scatter(ks, sol_assort.u[1:500] - sol_assort.u[501:end],
    xscale = :log10,
    label = L"F_{dir}",
    xlim = (1,kPlot),
    markershape = :circle,
    color = :lightblue)
scatter!(ks, -firstTerm,
    markershape = :x,
    color = :black,
    label = :none)
scatter!(ks, fs_indirect_assort,
    label = L"F_{indir}",
    color = :lightgreen)
scatter!([kStarP], [-valueMin],
    label = :none,
    markercolor = :red,
    markershape = :hexagon,
    markerstrokecolor = :black,
    markerstrokewidth = 2,
    markersize = 8)
scatter!(fs_assort,
    xscale = :log10,
    label = L"f(k)",
    xlim = (1,kPlot),
    xlabel = L"k")
scatter!(-total,
    xscale = :log10,
    label = :none,
    xlim = (1,kPlot),
    markershape = :x,
    color = :black)
scatter!([kStar], [-minF],
    label = :none,
    markercolor = :red,
    markershape = :hexagon,
    markerstrokecolor = :black,
    markerstrokewidth = 2,
    markersize = 8)
scatter!(-secondTerm,
    xscale = :log10,
    xlim = (1,kPlot),
    markershape = :x,
    color = :black,
    label = "Theory")
savefig("Dropbox/Immunization/Plots/supp_Assort_1_fixed.pdf")

#savefig("Dropbox/Immunization/Plots/emergenceMin.pdf")

#writedlm("Dropbox/Immunization/Plots/data/panel1AB_kANDTerms.csv", [ks firstTerm secondTerm total])
#writedlm("Dropbox/Immunization/Plots/data/panel1AB_minima.csv", [kStarP valueMin kStar minF])
