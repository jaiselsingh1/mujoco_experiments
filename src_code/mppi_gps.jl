using MuJoCo
using LinearAlgebra
using .Threads
using JLD2
using Base

struct HopperModel
    model::Model
    data::Data
    ν::Int
    state_dim::Int
end

function hopper_model(model_path="../models/hopper.xml")
    model = load_model(model_path)
    data = init_data(model)
    ν = model.nu
    state_dim = length(get_physics_state(model, data))

    return HopperModel(model, data, ν, state_dim)
end

struct Cartpole
    model::Model
    data::Data
    ν::Int
    state_dim::Int
end

function cartpole_model(model_path="../models/cartpole.xml")
    model = load_model(model_path)
    data = init_data(model)
    ν = model.nu
    state_dim = length(get_physics_state(model, data))

    return Cartpole(model, data, ν, state_dim)
end

struct MPPIPlanner # doesn't need to be mutable
    U::Matrix{Float64}
    T::Int # number of steps
end

function running_cost_hp(data)
    fwd_vel = data.qvel[1]
    ctrl = data.ctrl

    alive_bonus = -1.0
    velocity_reward = -100.0 * fwd_vel
    control_cost = 0.001 * sum(ctrl .^ 2)

    cost = alive_bonus + velocity_reward + control_cost
    return cost
end

function terminal_cost_hp(data)
    h = data.qpos[2]
    return (h < 0.9) ? 500 : 0.0
end

function running_cost_cartpole(d)
    ctrl = d.ctrl
    x = d.qpos[1]
    θ = d.qpos[2]
    x_dot = d.qvel[1]
    θ_dot = d.qvel[2]

    pos_cost = 1.0 * x^2
    theta_cost = 20.0 * (cos(θ) - 1.0)^2
    vel_cost = 0.1 * x_dot^2
    thetadot_cost = 0.1 * (θ_dot)^2
    ctrl_cost = ctrl[1]^2

    return (pos_cost + theta_cost + vel_cost + thetadot_cost + ctrl_cost)

end

function terminal_cost_cartpole(d)
    return 10.0 * running_cost_cartpole(d)
end

struct Trajectory
    states::Matrix{Float64}
    controls::Matrix{Float64}
end

# K -> number of rollouts to do
# T -> number of steps per rollout
function mppi_traj!(env, planner::MPPIPlanner; K=100, T=500, Σ=1.0, Φ=0.0, λ=1.0, q=0.0) # planning
    ν = env.ν
    U = planner.U
    T = planner.T

    model = env.model
    data = env.data

    # for saving trajectories
    state_dim = env.state_dim
    all_states = [zeros(state_dim, T+1) for _ in 1:K]
    all_controls = [zeros(ν, T) for _ in 1:K]

    ϵ = [randn(ν, T) for _ in 1:K] # noise samples
    S = zeros(K) # costs

    local_datas = [init_data(model) for _ in 1:Threads.nthreads()]
    @threads for k in 1:K
        local_d = local_datas[threadid()]
        local_d.qpos .= data.qpos
        local_d.qvel .= data.qvel

        all_states[k][:, 1] .= get_physics_state(model, local_d)

        for t in 1:T
            noisy_control = clamp.(U[:, t] + ϵ[k][:, t], -1.0, 1.0)
            local_d.ctrl .= noisy_control
            step!(model, local_d)

            all_controls[k][:, t] .= noisy_control
            all_states[k][:, t+1] .= get_physics_state(model, local_d)

            step_cost = running_cost_cartpole(local_d)
            S[k] += step_cost + (λ * inv(Σ) * U[:, t]' * ϵ[k][:, t])
        end
        S[k] += terminal_cost_cartpole(local_d)

    end

    β = minimum(S)
    weights = exp.((-1.0 / λ) * (S .- β))
    η = sum(weights)
    weights ./= η

    for t in 1:T
        planner.U[:, t] += sum(weights[k] * ϵ[k][:, t] for k = 1:K)
    end

    best_k = argmin(S)
    return Trajectory(all_states[best_k], all_controls[best_k])
end

function generate_trajectories()
end

function save_trajectories(trajectories, filename)
    JLD2.@save(filename, trajectories)
    println("trajectories saved to $filename")
end

function load_trajectories(filename)
    JLD2.@load(filename, trajectories)
    println("trajectories loaded from $filename")
    return trajectories
end

env = cartpole_model()
init_visualiser()
T = 100
planner = MPPIPlanner(0.2 .* randn(env.ν, T), T) # try to warm start the planner/controls
all_trajectories = Trajectory[]

function mppi_controller!(m, d)
    # plan
    best_traj = mppi_traj!(env, planner)
    push!(all_trajectories, best_traj)

    # act
    d.ctrl .= planner.U[:, 1]
    # step!(m, d) apparently the visualiser does the actual stepping when it is called

    # shift
    planner.U[:, 1:end-1] .= planner.U[:, 2:end]
    planner.U[:, end] .= 0.0
end

reset!(env.model, env.data)
visualise!(env.model, env.data, controller=mppi_controller!)
save_trajectories(all_trajectories, "trajectories.JLD2")
