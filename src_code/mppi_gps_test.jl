using MuJoCo: mjData
# clone one step policy and then action chunk policy -> see if they do something different
# one step and jacobian policy
using JLD2
using MuJoCo
using .Threads
using LinearAlgebra
using Base

struct HopperModel
    model::Model
    data::Data
    ν::Int # number of controls
    state_dim::Int # size of the state
end

struct cartpole
    model::Model
    data::Data
    ν::Int
    state_dim::Int
end

struct Trajectory
    states::Matrix{Float64}
    controls::Matrix{Float64}
    cost::Float64
    weight::Float64
end

function get_state(data)
    return vcat(copy(data.qpos), copy(data.qvel))
end

function cartpole_model(model_path="../models/cartpole.xml")
    model = load_model(model_path)
    data = init_data(model)

    ν = model.nu
    state_dim = length(get_physics_state(model, data))

    return cartpole(model, data, ν, state_dim)
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
    velocity_reward = -1.0 * fwd_vel
    control_cost = 0.001 * sum(ctrl .^ 2)

    cost = alive_bonus + velocity_reward + control_cost
    return cost
end


function terminal_cost_hp(data; min_height=0.7, max_height=2.0)
    height = data.qpos[2]

    if height > max_height || height < min_height
        return 100.0
    end

    return 0.0
    # return 10.0 * running_cost_hp(data, ctrl)
end

function mppi_traj(env::HopperModel; K=100, T=500, Σ=0.5, Φ=0.0, λ=1.0, q=0.0)
    ν = env.ν # number of control inputs
    state_dim = env.state_dim
    model = env.model
    data = env.data

    U = zeros(ν, T)
    S = zeros(K) # S is the costs
    ϵ = [randn(ν, T) for _ in 1:K] # noise for the controls

    all_states = [zeros(state_dim, T + 1) for _ in 1:K]  # the last control input at T would result in a state at T+1
    all_controls = [zeros(ν, T) for _ in 1:K]
    all_costs = zeros(K)

    local_datas = [init_data(model) for _ = 1:Threads.nthreads()]

    Threads.@threads for k in 1:K
        t_id = Threads.threadid()
        local_data = local_datas[t_id]
        local_data.qpos .= data.qpos
        local_data.qvel .= data.qvel

        all_states[:][k] .= get_physics_state(model, local_data)
        #all_states[:][k] .= get_state(local_data)

        for t in 1:T
            noisy_control = clamp.(U[:, t] + ϵ[k][:, t], -1.0, 1.0)
            local_data.ctrl .= noisy_control
            # all_controls[k][:, t] = noisy_control
            step!(model, local_data)

            all_states[k][:, t+1] = get_physics_state(model, local_data)
            step_cost = running_cost_hp(local_data)
            S[k] += step_cost + (λ * inv(Σ) * U[:, t]' * ϵ[k][:, t])
        end
        S[k] += terminal_cost_hp(local_data)
        all_costs[k] = S[k]
    end

    # MPPI weightage calculation
    β = minimum(S)
    weights = exp.((-1.0 / λ) * (S .- β))
    η = sum(weights)
    weights ./= η

    # update nominal control
    for t = 1:T
        U[:, t] .+= sum(weights[k] * ϵ[k][:, t] for k = 1:K)
    end
    data.ctrl .= U[:, 1]
    step!(model, data)

    # shift controls forward
    for t = 2:T
        U[:, t-1] .= U[:, t] # setting the previous control to be the current
    end
    U[:, T] .= zeros(ν) # re-initialize the last control step

    trajectories = [Trajectory(all_states[k], all_controls[k], all_costs[k], weights[k]) for k = 1:K]

    sort!(trajectories, by=traj -> traj.cost) # this is an in-line function call that uses an anonymous function

    return U, trajectories
end

function save_trajectories(trajectories, filename)
    save_object = Dict(
        "num_trajectories" => length(trajectories),
        "trajectories" => trajectories
    )
    JLD2.save(filename, save_object)
    println("Saved $(length(trajectories)) trajectories to $filename")
end

function load_trajectories(filename)
    data = JLD2.load(filename)
    println("Loaded $(data["num_trajectories"]) trajectories from $filename")
    return data["trajectories"]
end

function generate_trajectories(env::HopperModel; num_batches=5, top_k=5, save=false, save_path="trajectories.jld2")
    all_trajectories = Trajectory[]

    for batch in 1:num_batches
        U, trajectories = mppi_traj(env)

        for k in 1:min(top_k, length(trajectories)) #lowest-cost trajectories
            push!(all_trajectories, trajectories[k])
        end
    end

    if save
        save_trajectories(all_trajectories, save_path)
    end

    return all_trajectories
end

env = hopper_model()
trajectories = generate_trajectories(env; num_batches=5, top_k=5)

model = env.model
data = env.data
init_visualiser()
traj_states = getfield.(trajectories, :states)
visualise!(model, data; trajectories=traj_states)


#=
function visualise_trajectories(env::HopperModel, trajectories::Vector{Trajectory})
    model = env.model
    data = init_data(model)
    init_visualiser()
    traj_states = Matrix{Float64}[]
    for trajectory in trajectories
        push!(traj_states, trajectory.states)
    end

    Base.invokelatest(visualise!, model, data; trajectories=traj_states)
    # visualise!(model, data; trajectories=traj_states)
end
=#

#=
function visualize_trajectories_sequential(env::HopperModel, trajectories::Vector{Trajectory};
    fps=60, pause_between=1.0, show_info=true)

    init_visualiser()
    for (idx, trajectory) in enumerate(trajectories)
        if show_info
            println("\n=== Trajectory $idx / $(length(trajectories)) ===")
            println("Cost: $(trajectory.cost)")
            println("Weight: $(trajectory.weight)")
        end

        model = env.model
        data = init_data(model)

        # Set initial state
        data.qpos .= trajectory.states[1:model.nq, 1]
        data.qvel .= trajectory.states[model.nq+1:end, 1]

        T = size(trajectory.controls, 2)

        step_ref = Ref(1)

        function controller!(m, d)
            if step_ref[] <= T
                d.ctrl .= trajectory.controls[:, step_ref[]]
                step_ref[] += 1
            end
        end

        trimmed_states = trajectory.states[:, 1:end-1]

        Base.invokelatest(
            visualise!,
            model,
            data;
            controller=controller!,
            trajectories=trimmed_states
        )

        if idx < length(trajectories)
            println("Pausing for $pause_between seconds...")
            sleep(pause_between)
        end
    end
end

function visualize_trajectory(env::HopperModel, trajectory::Trajectory; fps=60)
    init_visualiser()

    model = env.model
    data = init_data(model)
    reset!(model, data)

    # put the sim in the trajectory’s initial state
    data.qpos .= trajectory.states[1:model.nq, 1]
    data.qvel .= trajectory.states[model.nq+1:end, 1]

    trimmed_states = trajectory.states[:, 1:end-1]
    T = size(trimmed_states, 2)
    step_ref = Ref(1)

    function ctrl!(m, d)
        if step_ref[] ≤ T
            d.ctrl .= trajectory.controls[:, step_ref[]]
            step_ref[] += 1
        end
    end

    Base.invokelatest(
        visualise!,               # function object
        model,                    # positional
        data;                     # positional
        controller=ctrl!,     # keywords
        trajectories=trimmed_states
    )
end
 =#

# visualize_trajectories_sequential(env, trajectories)
# visualize_trajectory(env, trajectories[1])  # visualize best trajectory
