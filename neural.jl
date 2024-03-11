import CUDA, Flux.Zygote

function generate_windows(ρ; window_bins)
    ρ_windows = Zygote.Buffer(zeros(Float32, window_bins, length(ρ)))  # We use a Zygote Buffer here to keep autodifferentiability
    pad = window_bins ÷ 2 - 1
    ρpad = vcat(ρ[end-pad:end], ρ, ρ[1:1+pad])
    for i in 1:length(ρ)
        ρ_windows[:,i] = ρpad[i:i+window_bins-1]
    end
    copy(ρ_windows)  # copy needed due to Zygote.Buffer
end

function generate_inout(ρ_profiles, cA_profiles; window_width, dx, every=1)
    window_bins = 2 * round(Int, window_width / dx) + 1
    ρ_windows_all = Vector{Vector{Float32}}()
    cA_values_all = Vector{Float32}()
    for (ρ, cA) in zip(ρ_profiles, cA_profiles)
        ρ_windows = generate_windows(ρ; window_bins)
        for i in 1:every:length(cA)
            if !isfinite(cA[i])
                continue
            end
            push!(ρ_windows_all, ρ_windows[:, i])
            push!(cA_values_all, cA[i])
            push!(ρ_windows_all, reverse(ρ_windows[:, i]))
            push!(cA_values_all, cA[i])
        end
    end
    ρ_windows_all = reduce(hcat, ρ_windows_all)
    println(Base.format_bytes(Base.summarysize(ρ_windows_all)))
    ρ_windows_all, cA_values_all'
end

function get_c1_neural(model)
    window_bins = length(model.layers[1].weight[1, :])  # Get the number of input bins from the shape of the first layer
    model = model |> gpu
    function (ρ)
        ρ_windows = generate_windows(ρ; window_bins) |> gpu  # The helper function generate_windows is defined in neural.jl
        model(ρ_windows) |> cpu |> vec  # Evaluate the model, make sure the result gets back to the CPU, and transpose it to a vector
    end
end

function get_c2_autodiff(c1, xs)
    dx = xs[2] - xs[1]
    function (ρ)
        Flux.jacobian(c1, ρ)[1] / dx
    end
end

function funcintegral(c1, xs, ρ; num_a=50)
    dx = xs[2] - xs[1]
    da = 1 / num_a
    as = da/2:da:1
    aintegral = zero(ρ)
    for a in as
        aintegral .+= c1(a .* ρ)
    end
    sum(ρ .* aintegral) * dx * da
end
