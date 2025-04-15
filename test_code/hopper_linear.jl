using MuJoCo
using UnicodePlots
using Statistics
using LinearAlgebra
using .Threads 
using Optim 



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
learning_rate = 0.3

base_policy = 0.0 * randn(num_actions, num_observations + 1) # the +1 symbolizes having an added bias term 

global best_reward = -Inf
 # global best_policy = copy(base_policy)
 # global best_bias = copy(base_bias)

num_trajectories = 2*length(base_policy)
num_episodes = 2000
max_steps = 500
ep_rewards = Float64[]

# clip the max/min velocities (-4.0 and 4.0 based on emperical testing)
min_vel = -4.0 
max_vel = 4.0 

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
    t_height = 1.0 
    if height > t_height
        reward += upright_bonus
        reward += data.qvel[1] #data.qvel[1]*4
    else
        reward -= abs(t_height-height)
    end
    # reward -= 1e-3 * norm(data.ctrl)^2  # standing tends to get substantially better when the control cost is taken away 

    return reward 
end 

function hop_reward(data)
    reward = 0.0
    
    fwd_velocity = data.qvel[1]
    reward += fwd_velocity
    
    upright_bonus = 1.0 
    t_height = 0.0
    height = data.qpos[2]
    if height > t_height
        reward += upright_bonus
        #reward += 2*abs(height)
    else 
        reward -= abs(height-t_height)
    end 

    return reward 

end 

thread_datas = [init_data(model) for _ in 1:Threads.nthreads()]
policies = Vector{typeof(base_policy)}(undef, num_trajectories) # pre allocates the memory (vector is just a 1d Array)
rewards = zeros(Float64, num_trajectories)

for episode in 1:num_episodes
    global best_reward
    episode_best_reward = -Inf


    Threads.@threads for traj in 1:num_trajectories
        t_id = Threads.threadid()
        local_data = thread_datas[t_id]

        local_data.qpos .= copy(init_qpos)
        local_data.qvel .= copy(init_qvel)
        perturb_state!(local_data, init_qpos, init_qvel, 0.1)

        # bias = base_bias .+ rand(size(base_bias)).*noise_scale
        policy = base_policy .+ randn(size(base_policy)) .* noise_scale 
        policies[traj] = policy 
        total_reward = 0.0
        
        for step in 1:max_steps
            observation = vcat(local_data.qpos, local_data.qvel)
            observation[3] = sin(observation[3]) # trying to make the non-linearities more easily understood 
            observation = vcat(observation, 1.0)
            
            # clip the velocity (got rid of this because you should let the dynamics actually roll out)
            # local_data.qvel .= clamp.(local_data.qvel, min_vel, max_vel)

            action = policy * observation
            local_data.ctrl .= clamp.(action, -1.0, 1.0)

            step!(model, local_data)
            total_reward += stand_reward(local_data)
            
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
        
        rewards[traj] = total_reward 
    end

    episode_best_idx = argmax(rewards)
    episode_best_reward = rewards[episode_best_idx]

    if episode_best_reward > best_reward 
        best_reward = episode_best_reward 
        # best_policy = copy(policies[episode_best_idx])
        # best_bias = copy(biases[episode_best_idx])
        println("new best policy found! reward = $best_reward")
    end 
    
    normalized_rewards = (rewards .- mean(rewards)) ./ (std(rewards) + 1e-8)
    push!(ep_rewards, mean(rewards))
    
    gradient = zeros(size(base_policy))
    for i in 1:num_trajectories
        noise = policies[i] - base_policy
        gradient .+= noise .* normalized_rewards[i]
    end

    #= 
    this approach is silly because when you are doing gradients, you assume that one thing is being held constant for the other gradient to be valid
    bias_gradient = zeros(size(base_bias))
    for i in 1:num_trajectories
        bias_noise = biases[i] - base_bias
        bias_gradient .+= bias_noise .* normalized_rewards[i]
    end 
    =#
    
    display(heatmap(base_policy))
    display(gradient)
    display(lineplot(ep_rewards))
    
    base_policy .+= learning_rate * gradient # /num_trajectories
    # base_bias .+= learning_rate * bias_gradient/num_trajectories
    
    if episode % 100 == 0 
        println("base_policy $base_policy and learning rate = $learning_rate and gradeint = $gradient")
    end 


    if episode % 5 == 0 || episode == 1
        println("Episode $episode | Avg Reward: $(mean(rewards)) | Best Episode: $episode_best_reward | All-time Best: $best_reward")
    end
end

function trained_policy_controller!(m::Model, d::Data)
    state = vcat(d.qpos, d.qvel)
    state[3] = sin(state[3])
    state = vcat(state, 1.0)
    # augmented_state = vcat(state, 1.0)
    d.ctrl .= clamp.(base_policy * state, -1.0, 1.0)
    nothing
end

mj_resetData(model, data)
init_visualiser()
visualise!(model, data, controller = trained_policy_controller!)

