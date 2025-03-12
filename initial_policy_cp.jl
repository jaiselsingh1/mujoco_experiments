using MuJoCo 
# using Flux 
using LinearAlgebra 
using Statistics
using UnicodePlots

model = load_model("cartpole.xml")
data = init_data(model)

init_qpos = copy(data.qpos)
init_qvel = copy(data.qvel)

num_observations = 2*model.nq # number of observable states 
num_actions = model.nu # number of actuators 

function feature_dim(state_dim, order)
    return 1 + state_dim * 2 * order # there are 2 features per state per order 
end 

global fourier_order = 3 
feature_size = feature_dim(num_observations, fourier_order)


function fourier_features(state, order)
    features = [1.0] # constant that's always needed 
    normalized_state = state ./ [1.0, 1.0, 1.0, 1.0]
    for i in 1:order
        for j in 1:length(state)
            push!(features, cos(π * i * normalized_state[j]))
            push!(features, sin(π * i * normalized_state[j]))
        end 
    end 
    return features 
end 


base_policy = 0.0 * randn(num_actions, feature_size) 
global best_reward = -Inf
global best_policy = copy(base_policy)
global best_total_reward = -Inf  # best total trajectory reward 

num_trajectories = 2*length(base_policy)
num_episodes = 100 # total training episodes 
max_steps = 1000 # maximum steps per trajectory 
noise_scale = 0.05 # for policy updates
learning_rate = 0.3

# outermost is training loop -> policy parameters change per steps
# outer for loop -> sampling policies 
# inner for loop ->  using policies to get trajectories

# Training loop is outermost where the policy params change through steps. 
# Outer loop is sampling policies. Inter loop is using policy to get trajectories


ep_rewards = Float64[]
for episode in 1:num_episodes
    global base_policy, best_policy, best_reward, best_total_reward, fourier_order
    policies = []
    rewards = Float64[] 
    episode_best_reward = -Inf

    for i in 1:num_trajectories
        # get one or more trajectories and their reward -> add noise every time to the policy 
        data.qpos .= init_qpos  # better to use this than the mj_reset() which is slow 
        data.qvel .= init_qvel 

        policy = base_policy .+ randn(size(base_policy)) * noise_scale
        push!(policies, policy)

        total_reward = 0.0

        for step in 1:max_steps 
            # get reward/ observations (multiply with policy to get actions -> apply actions to sample_model_and_data)
            # step forward in time (get trajectory/ get reward)
            state = vcat(data.qpos, data.qvel)
            features = fourier_features(state, fourier_order) # 3rd order to start 
            
            action = policy * features # fourier feature network to capture any non-linearities 

            data.ctrl .= clamp.(action, -1.0, 1.0)
        
            mj_step(model, data)

            pole_angle = data.qpos[2]
            cart_pos = data.qpos[1] 
            
            angle_reward = cos(pole_angle) # straight angle is better hence reward for the theta to be close to 0 
            pos_penalty = 0.1 * abs(cart_pos)  # Small penalty for distance from center
            
            angle_vel_penalty = 0.05 * abs(data.qvel[2])  # Penalize fast angle changes
            pos_vel_penalty = 0.05 * abs(data.qvel[1])    # Penalize fast cart movement
            
            # Combined reward
            step_reward = angle_reward - pos_penalty - angle_vel_penalty - pos_vel_penalty
            
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
    

    #combine -> weight all the policies with the rewards generated to approximate a gradient direction and take a step 
    # training loop 
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
    feature_vector = fourier_features(state, fourier_order)
    d.ctrl .= clamp.(best_policy * feature_vector, -1.0, 1.0)
    nothing
end

mj_resetData(model, data)
init_visualiser()
visualise!(model, data, controller = trained_policy_controller!)







