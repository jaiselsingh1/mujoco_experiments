using MuJoCo 
using UnicodePlots 
using LinearAlgebra

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
    data.qpos .= init_qpos 
    data.qvel .= init_qvel 

    for h = 1:H 
        total_reward = 0.0 
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
function BRS(model, data; α=0.01, ν=0.03, N=10, H=1000, num_episodes=100) # basic random search 
    policy = create_policy(model, data) # initialize policy 

    ep_rewards = Float64[] # to help track the learning 

    for episode = 1:num_episodes
        deltas = [(ν * randn(size(policy)) for _ = 1:N)]
        
        # the two policies
        π_plus = [policy + delta for delta in deltas]
        π_minus = [policy - delta for delta in deltas]

        #storing the 2 rewards 
        R_plus = Float64[]
        R_minus = Float64[]

        for k = 1:N 
            push!(R_plus, rollout(model, data, π_plus[k]))
            push!(R_minus, rollout(model, data, π_minus[k]))
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
            println("Iteration $iter, Reward: $current_reward")
        end
    end 

    display(heatmap(policy))
    display(lineplot(ep_rewards))
    
    return policy
end 

function BRS_controller!(model, data)
    BRS_policy = BRS(model, data)
    observation = get_state(data)
    data.ctrl .= BRS_policy * observation
end 



mj_resetData(model, data)
init_visualiser()
visualise!(model,data, controller = BRS_controller!)



