using MuJoCo
using .Threads
using LinearAlgebra

function get_state(data)
    return vcat(copy(data.qpos), copy(data.qvel))
end

function running_cost(data)
end

# K is the number of samples to generate (the number of control sequences)
# T is the number of time steps
function mppi_rollouts(model, data, K, T; Σ, Φ, λ, q)
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
           step!(model, data)
           S[k] += running_cost(local_data) + (λ .* (U[:, t]' .* inv(Σ) .* ϵ[k][:, t])
        end

    end



end
