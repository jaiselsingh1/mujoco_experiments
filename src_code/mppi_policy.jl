using MuJoCo
using .Threads
using LinearAlgebra


model = load_model("../models/cartpole.xml")
data = init_data(model)

function get_state(data)
    return vcat(copy(data.qpos), copy(data.qvel))
end

function running_cost(data, ctrl)
    x = data.qpos[1]
    θ = data.qpos[2]
    x_dot = data.qvel[1]
    θ_dot = data.qvel[2]

    pos_cost = 1.0 * x^2
    theta_cost = 20.0 * (cos(θ) - 1.0)^2
    vel_cost = 0.1 * x_dot^2
    thetadot_cost = 0.1 * (θ_dot)^2
    ctrl_cost = ctrl[1]^2

    cost = pos_cost + theta_cost + vel_cost + thetadot_cost + ctrl_cost
    return cost
end

function terminal_cost(data, ctrl)
    return 10.0 * running_cost(data, ctrl)
end

# K is the number of samples to generate (the number of control sequences)
# T is the number of time steps
function mppi(model, data; K=30, T=100, Σ=1.0, Φ=0.0, λ=1.0, q=0.0)
    nu = model.nu # number of control inputs
    U = zeros(nu, T)
    S = zeros(K) # S is the costs
    ϵ = [rand(nu, T) for _ = 1:K] # generating noise for T time steps for each of the K samples
    x_0 = get_state(data)
    for k = 1:K
        local_data = init_data(model)
        local_data.qpos .= data.qpos
        local_data.qvel .= data.qvel
        x = x_0
        for t = 1:T
            u_t = U[:, t] .+ ϵ[k][:, t]  # Generate noisy control for this timestep
            local_data.ctrl .= u_t       # Apply only the current control
            step!(model, local_data)
            S[k] += running_cost(local_data, local_data.ctrl) + λ * dot(U[:, t], ϵ[k][:, t])

        end
        S[k] += terminal_cost(local_data, local_data.ctrl)
    end
    β = minimum(S)
    weights = exp.((-1.0 / λ) * (S .- β))
    η = sum(weights)
    weights ./= η
    for t = 1:T
        U[:, t] .+= sum(weights[k] * ϵ[k][:, t] for k = 1:K)
    end
    # Shift controls forward
    for t = 2:T
        U[:, t-1] .= U[:, t]
    end
    U[:, T] .= zeros(nu)  # this is to re-initialize the last control step
    return U
end


function mppi_controller!(model, data)
    data.ctrl .= mppi(model, data)[:, 1]
    nothing
end

mj_resetData(model, data)
init_visualiser()
visualise!(model, data; controller=mppi_controller!)
