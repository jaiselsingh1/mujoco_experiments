using MuJoCo
using LinearAlgebra
using MatrixEquations


model = load_model("cartpole.xml")
data = init_data(model)


function lqr(m::Model, d::Data, x_current)
    # number of states and control inputs
    nx = 2*model.nq
    nu = model.nu
    # finite-difference parameters
    c = 1e-6
    centred = true

    # use the current state in order to update the data of the model
    d.qpos .= x_current[1 : m.nq]
    d.qvel .= x_current[m.nq+1 : end]

    # compute the matrices (A, B)
    A = mj_zeros(nx,nx)
    B = mj_zeros(nx,nu)
    mjd_transitionFD(model, data, c, centred, A, B, nothing, nothing)

    Q = diagm([1.0, 10.0, 1.0, 5.0]) # Weights for the state vector
    R = diagm([1.0]) 

    S = zeros(nx, nu) # cross term penalty costs
    _, _, K, _ = ared(A,B,R,Q,S)
    
    return K, A, B, Q, R 
end 


H = 50 # H is the horizon
function MPC_lqr!(m::Model, d::Data)
    nx = 2 * model.nq
    nu = model.nu
    
    #predicted
    pred_states = zeros(nx, H+1)
    U_pred = zeros(nu, H) # initial controls are just zeros
    
    x0 = vcat(d.qpos, d.qvel)
    pred_states[:, 1] = x0
    
    m_sim = load_model("cartpole.xml")
    d_sim = init_data(m_sim)
    
    # copy in the state from the actual sim
    d_sim.qpos .= d.qpos
    d_sim.qvel .= d.qvel
    
    # Define target state (upright pendulum)
    x_target = zeros(nx)
    x_target[2] = π
    
    # run a prediction scheme
    for t in 1:H
        # Calculate K matrix at each timestep with current state
        current_state = pred_states[:, t]
        
        # Update simulation state
        d_sim.qpos .= current_state[1:m_sim.nq]
        d_sim.qvel .= current_state[m_sim.nq+1:end]
        
        # Compute LQR matrices
        A = zeros(nx, nx)
        B = zeros(nx, nu)
        c = 1e-6
        centred = true
        mjd_transitionFD(m_sim, d_sim, c, centred, A, B, nothing, nothing)
        
        # LQR cost matrices
        Q = diagm([1.0, 10.0, 1.0, 5.0]) # Weights for the state vector
        R = diagm([1.0])
        S = zeros(nx, nu) # cross term penalty costs
        
        # Calculate K matrix
        _, _, K, _ = ared(A, B, R, Q, S)
        
        # Calculate control input for current state
        U_pred[:, t] = -K * (current_state - x_target)
        
        # Apply constraints
        U_pred[:, t] = clamp.(U_pred[:, t], -1.0, 1.0) # constraints based on the xml file
        
        # Apply control to simulation and step forward
        d_sim.ctrl .= U_pred[:, t]
        step!(m_sim, d_sim)
        
        # Update predicted state for next timestep
        pred_states[:, t+1] = vcat(d_sim.qpos, d_sim.qvel)
    end
    
    # Apply first control input to actual model
    d.ctrl .= U_pred[:, 1]
    nothing
end

# do a rollout for all of the horizon and get the K matrix for the entire but only apply the first instance and then reset and go on from there 

mj_resetData(model, data)
init_visualiser()
visualise!(model, data, controller = MPC_lqr!)



