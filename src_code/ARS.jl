using MuJoCo
using UnicodePlots
using LinearAlgebra
using .Threads
using Statistics

model = load_model("../models/hopper.xml")
data = init_data(model)
init_qpos = copy(data.qpos)
init_qvel = copy(data.qvel)


function create_policy(model, data)
    num_act = model.nu
    num_obs = 2 * model.nq   # can have this be 2 * state_vector
    policy = zeros(num_act, num_obs)
    return policy
end

function get_state(data)
    return vcat(copy(data.qpos), copy(data.qvel))
end

function hop_reward(data)
    reward = 0.0

    fwd_velocity = data.qvel[1]
    reward += fwd_velocity

    return reward
end

function stand_reward(data)
    reward = 0.0

    upright_bonus = 1.0
    height = data.qpos[2]
    t_height = 1.0
    if height > t_height
        reward += upright_bonus
        reward += data.qvel[1] #data.qvel[1]*4
    else
        reward -= abs(t_height - height)
    end

    return reward
end


function rollout(model, data, policy; H=1000)
    data.qpos .= copy(init_qpos)
    data.qvel .= copy(init_qvel)

    total_reward = 0.0
    for h = 1:H

        observation = get_state(data) # can be tweaked later on to be something else
        action = policy * observation

        data.ctrl .= action

        step!(model, data)
        total_reward += stand_reward(data)
    end

    return total_reward
end


# α is step size
# N is number of directions
# ν is std of the noise
# H is Horizon
function BRS(model, data; α=0.01, ν=0.02, N=500, H=1000, num_episodes=100) # basic random search
    policy = create_policy(model, data) # initialize policy

    ep_rewards = Float64[] # to help track the learning

    for episode = 1:num_episodes
        deltas = [ν .* randn(size(policy)) for _ = 1:N]

        # the two policies
        π_plus = [policy + deltas[i] for i = 1:N]
        π_minus = [policy - deltas[i] for i = 1:N]

        #storing the 2 rewards
        R_plus = Vector{Float64}(undef, N)
        R_minus = Vector{Float64}(undef, N)
        thread_datas = [init_data(model) for _ in 1:Threads.nthreads()]

        Threads.@threads for k = 1:N  # can make this part multi-threaded
            t_id = Threads.threadid()
            local_data = thread_datas[t_id]

            R_plus[k] = rollout(model, local_data, π_plus[k])
            R_minus[k] = rollout(model, local_data, π_minus[k])
        end

        # update
        update = zeros(size(policy))
        for k = 1:N
            update .+= (R_plus[k] - R_minus[k]) .* deltas[k]
        end

        policy .+= (α / N) .* update

        current_reward = rollout(model, data, policy, H=H)
        push!(ep_rewards, current_reward)

        if episode % 10 == 0
            println("Episode $episode, Reward: $current_reward")
        end
    end

    display(heatmap(policy))
    display(lineplot(ep_rewards))

    return policy
end

# BRS_policy = BRS(model, data)

function BRS_controller!(model, data)
    observation = get_state(data)
    data.ctrl .= clamp.(BRS_policy * observation, -1.0, 1.0)
    nothing
end

function ARS_V1(model, data; α=0.01, ν=0.02, N=500, H=1000, num_episodes=100) #scaling BRS by the standard deviation of the collected rewards
    policy = create_policy(model, data)
    ep_rewards = Float64[]

    for episode = 1:num_episodes
        deltas = [ν .* randn(size(policy)) for _ = 1:N]

        π_plus = [policy + deltas[k] for k = 1:N]
        π_minus = [policy - deltas[k] for k = 1:N]

        R_plus = Vector{Float64}(undef, N)
        R_minus = Vector{Float64}(undef, N)
        max_R = Vector{Float64}(undef, N)
        #max_R = Vector{Tuple{Float64,Float64,Float64,Vector{Float64}}}()

        thread_datas = [init_data(model) for _ in 1:Threads.nthreads()]

        Threads.@threads for k = 1:N
            t_id = Threads.threadid()
            local_data = thread_datas[t_id]

            R_plus[k] = rollout(model, local_data, π_plus[k])
            R_minus[k] = rollout(model, local_data, π_minus[k])

            # elite selection in evolutionary strategies
            #max_reward = max(R_plus[k], R_minus[k])
            max_R[k] = (max(R_plus[k], R_minus[k]))
        end

        σ = std(vcat(R_plus, R_minus)) # standard deviation of the rewards from the rollouts

        sorted_indices = sortperm(max_R)
        sorted_R_plus = R_plus[sorted_indices]
        sorted_R_minus = R_minus[sorted_indices]
        sorted_deltas = deltas[sorted_indices]


        update = zeros(size(policy))
        for k = 1:N
            update .+= (sorted_R_plus[k] - sorted_R_minus[k]) .* sorted_deltas[k]
        end

        policy .+= (α / (σ * N)) .* update


        #=
        sorted_rewards = sort(max_R, by=x->x[1], rev=true)
        sorted_R_plus = [x[2] for x in max_R]
        sorted_R_minus = [x[3] for x in max_R]
        sorted_deltas = [x[4] for x in max_R]
        rewards_list = collect(zip(R_plus, R_minus, deltas))
        sorted_tuples = sort(rewards_list, by=x -> (x[1], x[2], rev=true))  # the by x is an anon function
        =#

        current_reward = rollout(model, data, policy, H=H)
        push!(ep_rewards, current_reward)

        if episode % 10 == 0
            println("Episode $episode, Reward: $current_reward")
        end
    end

    display(heatmap(policy))
    display(lineplot(ep_rewards))

    return policy
end

ARS_policy = ARS_V1(model, data)

function ARS_controller!(model, data)
    observation = get_state(data)
    data.ctrl .= clamp.(ARS_policy * observation, -1.0, 1.0)
    nothing
end


function ARS_V2() # "linear maps of states normalized by a mean and standard deviation computed online"

end







mj_resetData(model, data)
init_visualiser()
visualise!(model, data, controller=ARS_controller!)
