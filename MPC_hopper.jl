using MuJoCo 
using Statistics 
using LinearAlgebra 
using UnicodePlots 
using .Threads 

# Load model first
model = load_model("hopper.xml")
data = init_data(model)

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
    # reward -= 1e-3 * norm(data.ctrl)^2  # standing tends to get substantially better when the control cost is taken away 

    return reward 
end 


sim_datas = [init_data(model) for _ in 1:Threads.nthreads()]
thread_rewards = fill(-Inf, Threads.nthreads())
thread_indices = fill(0, Threads.nthreads())  # Use fill instead of Vector{Int}(undef, ...)

function mpc_controller!(model, data)
    H = 100 # horizon 
    N = 100 # number of candidates 
    actions = Vector{Matrix{Float64}}(undef, N)

    # Reset thread-local results
    fill!(thread_rewards, -Inf)
    fill!(thread_indices, 0)

    Threads.@threads for i in 1:N
        actions[i] = clamp.(randn(model.nu, H), -1.0, 1.0)
    end 

    Threads.@threads for i in 1:N # number of candidates 
        t_id = Threads.threadid()
        sim_data = sim_datas[t_id]
        action = actions[i]

        sim_data.qpos .= data.qpos 
        sim_data.qvel .= data.qvel

        rewards = 0.0 

        for t in 1:H  
            sim_data.ctrl .= action[:, t]  # Use the action for current timestep
            step!(model, sim_data)
            rewards += stand_reward(sim_data)
        end 

        if rewards > thread_rewards[t_id]
            thread_rewards[t_id] = rewards 
            thread_indices[t_id] = i
        end 
    end 

    best_reward = -Inf 
    best_idx = 0 

    for i in 1:Threads.nthreads()
        if thread_rewards[i] > best_reward
            best_reward = thread_rewards[i]
            best_idx = thread_indices[i]
        end 
    end 

    if best_idx > 0 
        data.ctrl .= actions[best_idx][:, 1]
    else 
        data.ctrl .= zeros(model.nu)
    end 

    nothing 
end 

# Reset data before visualization
mj_resetData(model, data)
init_visualiser()
visualise!(model, data, controller = mpc_controller!)

#=
function mpc_controller!(model, data)
    state = vcat(data.qpos, data.qvel)
    
    H = 100 # Horizon 
    num_candidates = 100 
    action_sequences = Vector{Matrix{Float64}}(undef, num_candidates)

    # Generate random action sequences (pre-allocate then fill)
    Threads.@threads for i in 1:num_candidates
        sequence = randn(model.nu, H)
        sequence = clamp.(sequence, -1.0, 1.0)
        action_sequences[i] = sequence # Assign to pre-allocated slot
    end 
     
    # Atomic variables for thread safety
    best_reward = Atomic{Float64}(-Inf)
    best_idx = Atomic{Int}(0)

    sim_datas = [init_data(model) for _ in 1:Threads.nthreads()]
    
    Threads.@threads for i in 1:num_candidates
        t_id = Threads.threadid()
        sim_data = sim_datas[t_id]
        actions = action_sequences[i]

        sim_data.qpos .= data.qpos 
        sim_data.qvel .= data.qvel

        total_reward = 0.0 

        for t in 1:H 
            sim_data.ctrl .= actions[:, t] 
            step!(model, sim_data)
            reward = stand_reward(sim_data)
            total_reward += reward
        end 

        current_best = best_reward[]
        if total_reward > current_best
            atomic_cas!(best_reward, current_best, total_reward)
            if best_reward[] == total_reward
                atomic_cas!(best_idx, best_idx[], i)
            end
        end
    end 

    if best_idx[] > 0
        data.ctrl .= action_sequences[best_idx[]][:, 1]
    else
        data.ctrl .= zeros(model.nu)
    end
    
    nothing
end
=#

