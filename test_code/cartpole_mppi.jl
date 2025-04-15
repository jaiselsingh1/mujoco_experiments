using MuJoCo 

model = load_model("cartpole.xml")
data = init_data(model)

global const K = 30  #number of trajectories 
global const T = 100 #number of time steps 

# cost function parameters 
const Σ = 1.0
const ϕ = 0
const q = 0  
const λ = 1.0

nx = 2 * model.nq # number of states
nu = model.nu # number of controls 

function terminal_cost(d, ctrl)
    return 10.0 * running_cost(d, ctrl)
end 

function running_cost(d, ctrl)
    x = d.qpos[1]
    θ = d.qpos[2]
    x_dot = d.qvel[1]
    θ_dot = d.qvel[2]

    pos_cost = 1.0 * x^2
    theta_cost = 20.0 * (cos(θ) - 1.0)^2
    vel_cost = 0.1 * x_dot^2
    thetadot_cost = 0.1 * (θ_dot)^2
    ctrl_cost = ctrl[1]^2
    
    q = pos_cost + theta_cost + vel_cost + thetadot_cost + ctrl_cost
    return q 
end 


global U_t = zeros(nu, T) # global controls 

function mppi(m, d)
    # init_state = vcat(data.qpos, data.qvel)
    ϵ = [randn(nu, T) for k in 1:K]  # sample noise up-front 
    S = zeros(K)  # costs

    for k in 1:K
        d_copy = init_data(m)
        d_copy.qpos .= d.qpos 
        d_copy.qvel .= d.qvel
        # weights = zeros(size(ϵ), K)

        for t in 1:T 
            d_copy.ctrl .= U_t[:, t] + ϵ[k][:, t]
            step!(m, d_copy)
            S[k] += running_cost(d_copy, d_copy.ctrl) + λ * U_t[:, t]' * ϵ[k][:, t] 
        end

        S[k] += terminal_cost(d_copy, d_copy.ctrl)
    end 

        β = minimum(S)
        weights = exp.(-1/λ * (S .- β))
        η = sum(weights) # eta is just a weight summation 
        weights ./= η

        # update the control sequence with the appropriate noise 
        for t in 1:T
            weighted_noise = zeros(nu)
            for k in 1:K
                weighted_noise .+= weights[k] * ϵ[k][:, t]
            end
            U_t[:, t] .+= weighted_noise
        end 

    for t in 1:T-1
        U_t[:, t] .= U_t[:, t+1] # shift over the controls to the next sequence 
    end 

    U_t[:, T] .= zeros(nu) # re initialize controls   
end 


function mppi_controller!(m, d)
    mppi(m, d)
    d.ctrl .= U_t[:, 1]
end 

init_visualiser()
visualise!(model, data, controller = mppi_controller!)
