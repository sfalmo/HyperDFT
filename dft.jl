function minimize(L::Number, μ::Number, T::Number, Vext::Function, c1; α::Number=0.03, maxiter::Int=10000, dx::Number=0.01, floattype::Type=Float32, tol::Number=max(eps(floattype(1e3)), 1e-8))
    L, μ, T = floattype.((L, μ, T))  # Technical detail: we will use Float32 in the machine learning part as neural networks usually operate on single precision floats
    xs = collect(floattype, dx/2:dx:L)  # Construct the numerical grid
    Vext = Vext.(xs)  # Evaluate the external potential on the grid
    infiniteVext = isinf.(Vext)  # Check where Vext is infinite to set ρ = 0 there
    ρ, ρEL = zero(xs), zero(xs)  # Preallocate the density profile and an intermediate buffer for iteration
    fill!(ρ, 0.5)  # Start with a bulk density of 0.5
    i = 0
    while true
        ρEL .= exp.((μ .- Vext) ./ T .+ c1(ρ))  # Evaluate the RHS of the Euler-Lagrange equation
        ρ .= (1 - α) .* ρ .+ α .* ρEL  # Do a Picard iteration step to update ρ
        ρ[infiniteVext] .= 0  # Set ρ to 0 where Vext = ∞
        clamp!(ρ, 0, Inf)  # Make sure that ρ does not become negative
        Δρmax = maximum(abs.(ρ - ρEL)[.!infiniteVext])  # Calculate the remaining discrepancy to check convergence
        i += 1
        if Δρmax < tol
            println("Converged (step: $(i), ‖Δρ‖ = $(Δρmax) < $(tol) = tolerance)")
            break  # The remaining discrepancy is below the tolerance: break out of the loop and return the result
        end
        if !isfinite(Δρmax) || i >= maxiter
            println("Did not converge (step: $(i) of $(maxiter), ‖Δρ‖: $(Δρmax), tolerance: $(tol))")
            return nothing  # The iteration did not converge, there is no valid result
        end
    end
    xs, ρ
end