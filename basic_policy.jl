using MuJoCo 
using Flux 
using LinearAlgebra 

model = load_model("carpole.xml")
data = init_data(model)
num_observations = model.qpos + model.qvel # number of observable states 
num_actions = model.nu # number of actuators 

base_policy = mj_zeros(num_observations, num_actions)


num_trajectories = 10 
num_episodes = 100 # total training episodes 
max_steps = 1000 # maximum steps per trajectory 
noise_scale = 0.05 # for policy updates


# outermost is training loop -> policy parameters change per steps
# outer for loop -> sampling policies 
# inner for loop ->  using policies to get trajectories

# Training loop is outermost where the policy params change through steps. 
# Outer loop is sampling policies. Inter loop is using policy to get trajectories

for episode in 1:num_episodes
    policies = []
    rewards = [] 
    trajectories = []

    for i in 1:num_trajectories
        # get one or more trajectories and their reward -> add noise every time to the policy 
        mj_resetData(model, data)

        policy = base_policy .+ randn(size(base_policy)) * noise_scale
        push!(policies, policy)

        total_reward = 0.0

        for step in 1:max_steps 
            # get reward/ observations (multiply with policy to get actions -> apply actions to sample_model_and_data)
            # step forward in time (get trajectory/ get reward)
            observations = vcat(model.qpos, model.qvel) 
            action = policy * observation #simple linear policy

            data.ctrl .= clamp(action, -1.0, 1.0)
        
            mj_step(model, data)

            angle_reward = cos(data.qpos[2]) # cos(pole angle)
            pos_penalty = 0.1 * abs(data.qpos[1]) #keep the cart centered
            reward = angle_reward - pos_penalty
            total_reward += reward 

            push!(trajectories, data)
        end 
    end 
    
    push!(rewards, total_reward)
    #combine -> weight all the policies with the rewards generated to approximate a gradient direction and take a step 
    # training loop 
end 



