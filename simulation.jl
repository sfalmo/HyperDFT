using Dates

Vext_sin(x; n::Int, A::Number, φ::Number, L::Number) = A * sin(2π * x * n / L + φ)

Vext_lin(x; x1::Number, x2::Number, E1::Number, E2::Number) = x > x1 && x < x2 ? E1 + (x - x1) * (E2 - E1) / (x2 - x1) : 0

Vext_wall(x; xw::Number, L::Number) = x < xw || x > L - xw ? Inf : 0

function generate_Vext(L::Number; num_sin=4, num_lin=rand(1:5), wall=true)
    Avar = 1.0
    sin_parameters = []
    for n in 1:num_sin
        push!(sin_parameters, (n = n, A = randn() * Avar, φ = rand() * 2π, L = L))
    end
    Evar = 1.0
    lin_parameters = []
    for _ in 1:num_lin
        push!(lin_parameters, (x1 = round(rand() * L, digits=2), x2 = round(rand() * L, digits=2), E1 = randn() * Evar, E2 = randn() * Evar))
    end
    xwmax = 1.0
    wall_params = (xw = round(rand() * xwmax, digits=2), L = L)
    function (x)
        result = 0.0
        for sin_params in sin_parameters
            result += Vext_sin(x; sin_params...)
        end
        for lin_params in lin_parameters
            result += Vext_lin(x; lin_params...)
        end
        if wall
            result += Vext_wall(x; wall_params...)
        end
        result
    end
end

struct System
    L::Float64
    μ::Float64
    β::Float64
    Vext::Function
    ϕ::Function
    particles::Vector{Float64}
    System(L::Number, μ::Number, T::Number, Vext::Function, ϕ::Function) = new(L, μ, 1 / T, Vext, ϕ, [])
end

function largest_cluster(system::System; cutoff=1.2)
    N = length(system.particles)
    coordination_of_particle::Vector{Int} = zeros(N)
    cluster_of_particle::Vector{Int} = 1:N
    particles_in_cluster::Vector{Vector{Int}} = [[i] for i in 1:N]
    for (i, xi) in enumerate(system.particles)
        for (j, xj) in enumerate(system.particles)
            if j <= i || dist(xi, xj, system.L) > cutoff
                continue
            end
            coordination_of_particle[i] += 1
            coordination_of_particle[j] += 1
            new_cluster, old_cluster = minmax(cluster_of_particle[i], cluster_of_particle[j])
            if new_cluster == old_cluster
                continue
            end
            for k in particles_in_cluster[old_cluster]
                cluster_of_particle[k] = new_cluster
            end
            push!(particles_in_cluster[new_cluster], particles_in_cluster[old_cluster]...)
            empty!(particles_in_cluster[old_cluster])
        end
    end
    coordination_count::Dict{Int,Int} = Dict(c => 0 for c in 0:N)
    for coordination in coordination_of_particle
        coordination_count[coordination] += 1
    end
    cluster_sizes::Vector{Int} = []
    for cluster in particles_in_cluster
        cluster_size = length(cluster)
        if cluster_size > 0
            push!(cluster_sizes, cluster_size)
        end
    end
    maximum(cluster_sizes; init=0)
end

mutable struct Histograms
    Â::Dict{String,Function}
    scalar::Dict{String,Float64}
    bins::Int
    dx::Float64
    onebody::Dict{String,Vector{Float64}}
    count::Int
    function Histograms(system::System; dx=0.01, Â=Dict("N" => system -> length(system.particles), "cluster" => largest_cluster))
        bins = Int(system.L / dx)
        push!(Â, "1" => _ -> 1)
        scalar = Dict(key => 0 for key in keys(Â))
        onebody = Dict(key => zeros(bins) for key in keys(Â))
        new(Â, scalar, bins, dx, onebody, 0)
    end
end

bin(histograms::Histograms, system::System, x::Number) = ceil(Int, x / system.L * histograms.bins)

function sample(histograms::Histograms, system::System)
    for (key, Â) in histograms.Â
        q = Â(system)
        histograms.scalar[key] += q
        for x in system.particles
            histograms.onebody[key][bin(histograms, system, x)] += q
        end
    end
    histograms.count += 1
end

struct Results
    scalar::Dict{String,Float64}
    xs::Vector{Float64}
    onebody::Dict{String,Vector{Float64}}
    function Results(histograms::Histograms, system::System)
        scalar_normed = copy(histograms.scalar)
        for q in keys(scalar_normed)
            scalar_normed[q] /= histograms.count
        end
        dx = histograms.dx
        xs = collect(dx/2:dx:system.L-dx/2)
        onebody_normed = copy(histograms.onebody)
        for q in keys(onebody_normed)
            onebody_normed[q] ./= histograms.count * dx
        end
        for q in keys(scalar_normed) ∩ keys(onebody_normed)
            onebody_normed["χ_"*q] = onebody_normed[q] - scalar_normed[q] * onebody_normed["1"]
        end
        onebody_normed["ρ"] = onebody_normed["1"]
        new(scalar_normed, xs, onebody_normed)
    end
end

function pbc!(system::System, i)
    system.particles[i] -= floor(system.particles[i] / system.L) * system.L
end

function dist(xi, xj, L)
    result = xj - xi
    result -= round(result / L) * L
    abs(result)
end

function add_particle!(system::System, x)
    push!(system.particles, x)
end

function remove_particle!(system::System, i)
    deleteat!(system.particles, i)
end

function calc_energy(system::System, i)
    xi = system.particles[i]
    E = system.Vext(xi)
    for xj in system.particles
        if xi == xj
            continue
        end
        E += system.ϕ(dist(xi, xj, system.L))
        if isinf(E)
            break
        end
    end
    E
end

function trial_move(system::System; Δxmax=0.1)
    if isempty(system.particles)
        return
    end
    i = rand(1:length(system.particles))
    xbefore = system.particles[i]
    Ebefore = calc_energy(system, i)
    system.particles[i] += Δxmax * (2 * rand() - 1)
    pbc!(system, i)
    Eafter = calc_energy(system, i)
    if rand() > exp(-system.β * (Eafter - Ebefore))
        system.particles[i] = xbefore
    end
end

function trial_insert(system::System)
    add_particle!(system, rand() * system.L)
    i = length(system.particles)
    ΔE = calc_energy(system, i)
    if rand() > system.L / length(system.particles) * exp(system.β * (system.μ - ΔE))
        remove_particle!(system, i)
    end
end

function trial_delete(system::System)
    if isempty(system.particles)
        return
    end
    i = rand(1:length(system.particles))
    ΔE = calc_energy(system, i)
    if rand() < length(system.particles) / system.L * exp(system.β * (ΔE - system.μ))
        remove_particle!(system, i)
    end
end

function sweep(system::System; transitions=10, insert_delete_probability=0.2)
    for _ in 1:transitions
        if rand() < insert_delete_probability
            rand() < 0.5 ? trial_insert(system) : trial_delete(system)
        else
            trial_move(system)
        end
    end
end

function simulate(L::Number, μ::Number, T::Number, Vext::Function, ϕ::Function; equilibration_time=Dates.Second(1), production_time=Dates.Second(2), sweep_transitions=10, Â=Dict("N" => system -> length(system.particles), "cluster" => largest_cluster))
    system = System(L, μ, T, Vext, ϕ)
    histograms = Histograms(system; Â)
    equilibration_start = now()
    while now() - equilibration_start < equilibration_time
        sweep(system; transitions=sweep_transitions)
    end
    production_start = now()
    while now() - production_start < production_time
        sweep(system; transitions=sweep_transitions)
        sample(histograms, system)
    end
    Results(histograms, system)
end
