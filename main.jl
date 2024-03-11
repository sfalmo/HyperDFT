include("simulation.jl")
include("dft.jl")
include("neural.jl")

using BSON, CUDA, Dates, Flux, JLD2, Plots, Printf
plotly()


function run_sims(datadir="data_$(now())"; L=10.0, μlim=(-5.0, 5.0), T=1.0, ϕ=r -> r < 1.0 ? Inf : 0, equilibration_time=Dates.Second(30), production_time=Dates.Second(600), num_sim=512)
    mkdir(datadir)
    Threads.@threads for sim_id in 1:num_sim
        println("Starting simulation $(sim_id)")
        μ = μlim[1] + rand() * (μlim[2] - μlim[1])
        Vext_generated = generate_Vext(L)
        results = simulate(L, μ, T, Vext_generated, ϕ; equilibration_time, production_time)
        μloc = μ .- Vext_generated.(results.xs)
        jldsave("$(datadir)/$(sim_id).jld2"; L, T, μloc, results)
        println("Simulation $(sim_id) done")
    end
end

function calc_profiles(dir, c1; L=10.0, dx=0.01)
    xs = dx/2:dx:L
    c2 = get_c2_autodiff(c1, xs)
    ρ_profiles = Vector{Vector{Float64}}()
    c_N_profiles = Vector{Vector{Float64}}()
    c_cluster_profiles = Vector{Vector{Float64}}()
    for sim in readdir(dir, join=true)
        println("Processing $(sim)")
        if !endswith(sim, ".jld2")
            continue
        end
        data = load(sim)
        results = data["results"]
        xs, ρ, χ_N, χ_cluster = results.xs, results.onebody["ρ"], results.onebody["χ_N"], results.onebody["χ_cluster"]
        c2_ρ = c2(ρ)
        c_N = χ_N ./ ρ .- c2_ρ * χ_N * dx
        c_cluster = χ_cluster ./ ρ .- c2_ρ * χ_cluster * dx
        push!(ρ_profiles, ρ)
        push!(c_N_profiles, c_N)
        push!(c_cluster_profiles, c_cluster)
    end
    Dict("ρ" => ρ_profiles, "c_N" => c_N_profiles, "c_cluster" => c_cluster_profiles)
end

function filter_profiles!(profiles; c_N_tol=0.05, ρ_min=0.0001)
    for (ρ, c_N, c_cluster) in zip(profiles["ρ"], profiles["c_N"], profiles["c_cluster"])
        c_cluster[abs.(c_N .- 1) .> c_N_tol] .= NaN
        c_cluster[ρ .< ρ_min] .= NaN
    end
end

function train(profiles; cA="c_cluster", window_width=4.99, dx=0.01, every=1, epochs=300)
    ρ_windows, c_cluster_values = generate_inout(profiles["ρ"], profiles[cA]; window_width, dx, every)
    println("Input-output data size: ", size(ρ_windows), size(c_cluster_values))
    ρ_windows, c_cluster_values = (ρ_windows, c_cluster_values) |> gpu

    model_cluster = Chain(
        Dense(size(ρ_windows)[1] => 512, softplus),
        Dense(512 => 512, softplus),
        Dense(512 => 512, softplus),
        Dense(512 => 1)
    ) |> gpu
    display(model_cluster)

    opt = Flux.setup(Adam(), model_cluster)
    loader = Flux.DataLoader((ρ_windows, c_cluster_values), batchsize=256, shuffle=true)
    loss(m, x, y) = Flux.mse(m(x), y)
    metric(m, x, y) = Flux.mae(m(x), y)
    get_learning_rate(epoch; initial=0.0001, rate=0.015, wait=3) = epoch < wait ? initial : initial * (1 - rate)^(epoch - wait)

    for epoch in 1:epochs
        learning_rate = get_learning_rate(epoch)
        Flux.adjust!(opt, learning_rate)
        @printf "Epoch: %3i (learning_rate: %.2e)..." epoch learning_rate; flush(stdout)
        Flux.train!(loss, model_cluster, loader, opt)
        @printf " loss: %.5f, metric: %.5f\n" loss(model_cluster, ρ_windows, c_cluster_values) metric(model_cluster, ρ_windows, c_cluster_values); flush(stdout)
    end

    model_cluster
end


function main(particles="HR", L=10; do_sims=true, do_profiles=true, do_train=true)
    ϕ = Dict(
        "HR" => r -> r < 1 ? Inf : 0,
        "SW1.2" => function (r)
            if r < 1
                return Inf
            end
            if r < 1.2
                return -1
            end
            return 0
        end
    )

    datadir = "data_$(particles)_L$(L)"

    if do_sims
        run_sims(datadir; L, ϕ=ϕ[particles])
    end

    if do_profiles
        BSON.@load "model_$(particles).bson" model
        c1 = get_c1_neural(model)

        profiles = calc_profiles(datadir, c1; L)
        jldsave("$(datadir).jld2"; ρ=profiles["ρ"], c_N=profiles["c_N"], c_cluster=profiles["c_cluster"])
    else
        profiles = load("$(datadir).jld2")
    end

    if do_train
        filter_profiles!(profiles)
        model_cluster = train(profiles; window_width=L/2 - 0.01)
        model_cluster = model_cluster |> cpu
        BSON.@save "model_cluster_$(particles)_L$(L).bson" model_cluster
    end
end


main("HR", 10; do_sims=false, do_profiles=false, do_train=true)
