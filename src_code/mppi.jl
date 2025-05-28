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

struct Trajectory
    states::Matrix{Float64}  # (state_dim, T+1)
    controls::Matrix{Float64}  # (ν, T)
    cost::Float64
    weight::Float64
end

struct MPPIPlanner # doesn't need to be mutable
    U::Matrix{Float64}
    T::Int # number of steps
end

function hopper_model(model_path="../models/hopper.xml")
    model = load_model(model_path)
    data = init_data(model)
    ν = model.nu
    state_dim = length(get_physics_state(model, data))

    return HopperModel(model, data, ν, state_dim)
end

function running_cost_hp(data)
    fwd_vel = data.qvel[1]
    ctrl = data.ctrl

    alive_bonus = -1.0
    velocity_reward = -10.0 * fwd_vel
    control_cost = 0.001 * sum(ctrl .^ 2)

    cost = alive_bonus + velocity_reward + control_cost
    return cost
end

function terminal_cost_hp(data)
    return 0.0
end

# K -> number of rollouts to do
# T -> number of steps per rollout
function mppi_traj!(env::HopperModel, planner::MPPIPlanner; K=100, T=500, Σ=0.5, Φ=0.0, λ=1.0, q=0.0) # planning
    ν = env.ν
    U = planner.U
    T = planner.T

    model = env.model
    data = env.data

    ϵ = [randn(ν, T) for _ in 1:K] # noise samples
    S = zeros(K) # costs
    w = zeros(K) # weights


    local_datas = [init_data(model) for _ in 1:Threads.nthreads()]
    @threads for k in 1:K
        local_d = local_datas[threadid()]
        local_d.qpos .= data.qpos
        local_d.qvel .= data.qvel

        for t in 1:T
            noisy_control = clamp.(U[:, t] + ϵ[k][:, t], -1.0, 1.0)
            local_d.ctrl .= noisy_control
            step!(model, local_d)

            step_cost = running_cost_hp(local_d)
            S[k] += step_cost + (λ * inv(Σ) * U[:, t]' * ϵ[k][:, t])
        end
        S[k] += terminal_cost_hp(local_d)
    end

    β = minimum(S)
    weights = exp.((-1.0 / λ) * (S .- β))
    η = sum(weights)
    weights ./= η

    for t in 1:T
        planner.U[:, t] .+= sum(weights[k] * ϵ[k][:, t] for k = 1:K)
    end

end

function mppi_step!(planner::MPPIPlanner, env::HopperModel)
    # plan
    mppi_traj!(env, planner)

    # act
    env.data.ctrl .= planner.U[:, 1]
    step!(env.model, env.data)

    # shift
    planner.U[:, 1:end-1] .= planner.U[:, 2:end]
    planner.U[:, end] .= 0.0
end


env = hopper_model()
init_visualiser()
T = 100
planner = MPPIPlanner(zeros(env.ν, T), T)
reset!(env.model, env.data)

function mppi_controller!(m::Model, d::Data)
    mppi_step!(planner, env)
end

visualise!(env.model, env.data, controller=mppi_controller!)
