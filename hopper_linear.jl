using MuJoCo
using UnicodePlots
using Statistics
using LinearAlgebra
using Threads 
Threads.set_num_threads(4)

model = load_model("hopper.xml")
data = init_data(model)
# model.nq = 7
# model.nu = 4
mj_resetData(model, data)
init_qpos = copy(data.qpos) # to use within the loop to reset the data
init_qvel = copy(data.qvel)
num_observations = 2*model.nq
num_actions = model.nu
noise_scale = 0.1
learning_rate = 0.2
base_policy = 0.0 * randn(num_actions, num_observations)
global best_policy = copy(base_policy)
global best_reward = -Inf
num_trajectories = 2*length(base_policy)
num_episodes = 1500
max_steps = 500
ep_rewards = Float64[]


function perturb_state!(data, init_qpos, init_qvel, pertubation_scale = 0.1)
    data.qpos .= copy(init_qpos)
    data.qvel .= copy(init_qvel)
    joint_angles = data.qpos[4:end]
    data.qpos[4:end] .+= pertubation_scale * randn(size(joint_angles))
    data.qvel .+= pertubation_scale * randn(size(data.qvel))
    nothing 
end 

function stand_reward(data)
    reward = 0.0 
    upright_bonus = 1.0 
    height = data.qpos[2]
    t_height = 0.0 
    if height > t_height
        reward += upright_bonus
        reward += abs(data.qvel[1]-2)  #data.qvel[1]*4
    else
        reward -= abs(t_height-height)
    end
    reward -= 1e-3 * norm(data.ctrl)^2

    return reward 
end 

function hop_reward(data)
    reward = 0.0
    
    #= 
    upright_bonus = 1.0 
    height = data.qpos[2]
    t_height = 0 

    if height > t_height 
        reward += upright_bonus
    else 
        reward -= abs(height-t_height)
    end 
    =#
    reward += data.qvel[1]    
    return reward 
end 

for episode in 1:num_episodes
    global best_policy, best_reward
    policies = Vector{typeof(base_policy)}(undef, num_trajectories) # pre allocates the memory (vector is just a 1d Array)
    rewards = zeros(Float64, num_trajectories)
    episode_best_reward = -Inf

    thread_datas = [init_data(model) for _ in 1:Threads.nthreads()]

    Threads.@threads for traj in 1:num_trajectories

        t_id = Threads.threadid()
        local_data = thread.datas[t_id]

        local_data.qpos .= copy(init_qpos)
        local_data.qvel .= copy(init_qvel)
        perturb_state!(local_data, init_qpos, init_qvel, 0.1)


        policy = base_policy .+ randn(size(base_policy)).*noise_scale
        policies[traj] = policy 
        total_reward = 0.0
        
        for step in 1:max_steps
            observation = vcat(local_data.qpos, local_data.qvel)
            observation[3] = sin(observation[3])
            action = policy * observation
            local_data.ctrl .= clamp.(action, -1.0, 1.0)

            step!(model, local_data)
            total_reward += hop_reward(local_data)
            
            #= 
            # COM based reward 
            torso_position = data.xpos[:, 1]
            foot_position = data.xpos[:, 3]

            torso_x = torso_position[1]
            foot_x = foot_position[1]
            balance_reward = -0.2 * abs(torso_x - foot_x)

            energy_penalty = -0.001 * (data.qpos[4]^2 + data.qpos[5]^2 + data.qpos[6]^2 + data.qpos[7]^2)  # penalize any angle deviations for joints 
            target_height = 0.0 
            height_reward = -0.1 * abs(data.qpos[2] - target_height) # data.qpos[2] is rootz 
            velocity_reward = -0.1 * abs(data.qvel[1])
            =# 
        end
        
        if total_reward > best_reward
            best_reward = total_reward
            best_policy = copy(policy)
            println("New best policy found! Reward: $best_reward")
        end
        
        if total_reward > episode_best_reward
            episode_best_reward = total_reward
        end
        
        rewards[traj] = total_reward 
    end

    episode_best_idx = argmax(rewards)
    episode_best_reward = rewards[episode_best_idx]
    
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
        println("Episode $episode | Avg Reward: $(mean(rewards)) | Best Episode: $episode_best_reward | All-time Best: $best_reward")
    end
end

function trained_policy_controller!(m::Model, d::Data)
    state = vcat(d.qpos, d.qvel)
    d.ctrl .= clamp.(best_policy * state, -1.0, 1.0)
    nothing
end

mj_resetData(model, data)
init_visualiser()
visualise!(model, data, controller = trained_policy_controller!)

