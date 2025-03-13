using MuJoCo 


model = load_model("cartpole.xml")
data = init_data(model)

global const K = 30  #number of trajectories 
global const T = 100 #number of time steps 

# cost function parameters 
const Σ = 0.1
const ϕ = 0
const q = 0  
const λ = 0.1 

nx = 2 * model.nq # number of states
nu = model.nu # number of controls 

function terminal_cost(d::data)
    return 10.0 * running_cost(d)
end 

function running_cost(d::data)
    q = zeros(K)
    x = d.qpos[1]
    θ = d.qpos[2]
    x_dot = d.qvel[1]
    θ_dot = d.qvel[2]
    q =  10*x^2 + (500*(cos(θ)+ 1)^2) + x_dot^2 + 15*(θ_dot^2) #direct cost function that is described in the paper
    return q 
end 


global U_t = zeros(nu, T) # global controls 

function mppi(m::data, d::data)
    # init_state = vcat(data.qpos, data.qvel)
    ϵ = [randn(nu, T) for k in 1:K]  # sample noise up-front 

    for k in 1:K
        d_copy = init_data(m)
        d_copy.qpos .= d.qpos 
        d_copy.qvel .= d.qvel

        S = zeros(K)  # costs
        # weights = zeros(size(ϵ), K)

        for t in 1:T 
            d.ctrl .= U_t[:, t] + ϵ[k][:, t]
            mj_step!(m, d_copy)
            S[k] += running_cost(d_copy) + λ * U_t[:, t]' * ϵ[k][:, t] 
        end

        S[k] += terminal_cost(d_copy)
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


function mppi_controller!(m::model, d::data)
    mppi(m, d)
    d.ctrl .= U_t[:, 1]
end 

init_visualizer()
visualise!(model, data, controller = mppi_controller!)
