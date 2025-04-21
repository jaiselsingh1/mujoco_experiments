using MuJoCo
using .Threads
using LinearAlgebra


model = load_model("../models/cartpole.xml")
data = init_data(model)

function get_state(data)
    return vcat(copy(data.qpos), copy(data.qvel))
end

function running_cost(data)
end

function terminal_cost(data)
end

# K is the number of samples to generate (the number of control sequences)
# T is the number of time steps
function mppi(model, data, K, T; Σ, Φ, λ, q)
    nu = model.nu # number of control inputs
    U = zeros(nu, T)
    S = zeros(K) # S is the costs
    ϵ = [rand(nu, T) for _ = 1:K] # generating noise for T time steps for each of the K samples
    x_0 = get_state(data)

    for k = 1:K
        local_data = copy(data)
        x = x_0
        for t = 1:T
            u_t = U[:, t] .+ (ϵ[k][:, t])
            local_data.ctrl .= u_t
            step!(model, local_data)
            S[k] += running_cost(local_data) + (λ .* (U[:, t]' .* inv(Σ) .* ϵ[k][:, t]))
        end

        S[k] += terminal_cost(local_data)
    end

    β = minimum(S)
    weights = exp.((-1.0 / λ) * (S .- β))
    η = sum(weights)
    weights ./= η

    for t = 1:T
        U[:, t] += sum(weights[k]ϵ[k][:, t] for k = 1:K)
    end

    for t = 2:T
        U[:, t-1] .= U[:, t]
    end

    U[:, T] .= zeros(nu)  # this is to re-initialize the last control step
    return U
end


function mppi_controller!(model, data)

end
