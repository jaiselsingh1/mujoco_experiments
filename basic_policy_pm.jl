using MuJoCo 
#using Flux 
using LinearAlgebra 
using Statistics
using UnicodePlots

model = load_model("pointmass.xml")
data = init_data(model)

mj_resetData(model, data)
init_qpos = copy(data.qpos)
init_qvel = copy(data.qvel)

num_observations = 2*model.nq # number of observable states 
num_actions = model.nu # number of actuators 

base_policy = 0.0 * randn(num_actions, num_observations) 
global best_reward = -Inf
global best_policy = copy(base_policy)
global best_total_reward = -Inf  # best total trajectory reward 

num_trajectories = 2*length(base_policy) # proportional to the number of parameters in the policy 
num_episodes = 500 # total training episodes 
max_steps = 500 # maximum steps per trajectory 
noise_scale = 0.05 # for policy updates
learning_rate = 0.2

ep_rewards = Float64[]
for episode in 1:num_episodes
    global base_policy, best_policy, best_reward, best_total_reward 
    policies = []
    rewards = Float64[] 
    episode_best_reward = -Inf

    for i in 1:num_trajectories
        # get one or more trajectories and their reward -> add noise every time to the policy 
        #mj_resetData(model, data)
        data.qpos .= init_qpos
        data.qvel .= init_qvel

        # random placement of the pointmass for exploration 
        data.qpos[1] = 0.2 * (rand() - 0.5) 
        data.qpos[2] = 0.2 * (rand() - 0.5)


        policy = base_policy .+ randn(size(base_policy)) * noise_scale
        push!(policies, policy)

        total_reward = 0.0

        for step in 1:max_steps 
            # get reward/ observations (multiply with policy to get actions -> apply actions to sample_model_and_data)
            # step forward in time (get trajectory/ get reward)
            observation = vcat(data.qpos, data.qvel) 
            if size(observation, 2) != 1
                observation = reshape(observation, :, 1) # make it into a column vector 
            end 

            action = policy * observation #simple linear policy

            data.ctrl .= clamp.(action, -1.0, 1.0)
        
            mj_step(model, data)

            x_pos = data.qpos[1]
            y_pos = data.qpos[2]

            # let the origin be the target 
            dist_target = sqrt(x_pos^2 + y_pos^2)

            # higher reward for closer to target 
            position_reward = -dist_target #exp(-10.0 * dist_target)
            # println("position reward $position_reward")

            # small penalty for high velocity (for smooth movement)
            velocity_penalty = 0.01 * (data.qvel[1]^2 + data.qvel[2]^2)
            # println("velocity penalty $velocity_penalty")

            # Small penalty for large control inputs (for energy efficiency)
            control_penalty = 0.005 * sum(abs.(data.ctrl))
            # println("control penalty $control_penalty")

            # combined reward
            #println("$position_reward $velocity_penalty $control_penalty")
            step_reward = position_reward #- velocity_penalty - control_penalty
            
            total_reward += step_reward 

        end 

        if total_reward > best_total_reward
            best_total_reward = total_reward 
            best_policy = copy(policy)
            println("New best policy found Reward: $best_total_reward")
        end 

        if total_reward > episode_best_reward
            episode_best_reward = total_reward 
        end 

        push!(rewards, episode_best_reward)

    end 

    normalized_rewards = (rewards .- mean(rewards)) ./ (std(rewards) + 1e-8)
    push!(ep_rewards, mean(rewards))

    gradient = zeros(size(base_policy))
    for i in 1:num_trajectories
        noise = policies[i] - base_policy 
        gradient .+= noise .* normalized_rewards[i]
    end 
    
    display(heatmap(base_policy))
    display(gradient)
    display(lineplot(ep_rewards))
    base_policy .+= learning_rate * gradient/num_trajectories 

    if episode % 5 == 0 || episode == 1
        println("Episode $episode | Avg Reward: $(mean(rewards)) | Best Episode: $episode_best_reward | All-time Best: $best_total_reward")
    end
end 

function trained_policy_controller!(m::Model, d::Data)
    state = vcat(d.qpos, d.qvel)
    d.ctrl .= clamp.(base_policy * state, -1.0, 1.0)
    nothing
end


mj_resetData(model, data)
data.qpos[1] = 0.2  # random initial poisitions for test 
data.qpos[2] = 0.2 
init_visualiser()
visualise!(model, data, controller = trained_policy_controller!)
