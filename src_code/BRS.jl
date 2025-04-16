using MuJoCo 
using UnicodePlots 
using LinearAlgebra
using .Threads

model = load_model("../models/hopper.xml")
data = init_data(model)
init_qpos = copy(data.qpos)
init_qvel = copy(data.qvel)


function create_policy(model, data)
    num_act = model.nu 
    num_obs = 2*model.nq   # can have this be 2 * state_vector 
    policy = zeros(num_act, num_obs)
    return policy 
end 

function get_state(data)
    return vcat(copy(data.qpos), copy(data.qvel))
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
        reward -= abs(t_height-height)
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
function BRS(model, data; α=0.01, ν=0.03, N=100, H=1000, num_episodes=100) # basic random search 
    policy = create_policy(model, data) # initialize policy 

    ep_rewards = Float64[] # to help track the learning 

    for episode = 1:num_episodes
        deltas = [ν .* randn(size(policy)) for _ = 1:N]
        
        # the two policies
        π_plus = [policy + deltas[i] for i = 1:N]
        π_minus = [policy - deltas[i] for i = 1:N]

        #storing the 2 rewards 
        R_plus = Float64[]
        R_minus = Float64[]

        Threads.@threads for k = 1:N  # can make this part multi-threaded 
            local_data1 = init_data(model)
            local_data2 = init_data(model)

            push!(R_plus, rollout(model, local_data1, π_plus[k]))
            push!(R_minus, rollout(model, local_data2, π_minus[k]))
        end 

        # update 
        update = zeros(size(policy))
        for k = 1:N 
            update .+= (R_plus[k] - R_minus[k]) .* deltas[k]
        end 

        policy .+= (α/N) .* update 
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

BRS_policy = BRS(model, data)

function BRS_controller!(model, data)
    observation = get_state(data)
    data.ctrl .= clamp.(BRS_policy * observation, -1.0, 1.0)
    nothing 
end 


mj_resetData(model, data)
init_visualiser()
visualise!(model, data, controller=BRS_controller!)



