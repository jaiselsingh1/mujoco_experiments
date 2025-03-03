using MuJoCo 

model, data = MuJoCo.sample_model_and_data()
println("original joint positions", data.qpos)

function random_controller!(m::Model, d::Data)
    nu = m.nu  
    d.ctrl .= 2*rand(nu) .- 1  # the nu is the number of control inputs and then the . means in place 
    return nothing 
end 

for t in 1:100
    random_controller!(model, data)
    step!(model, data)  # step "steps" the model, data in time and modifies the data in place (! means in place)
end 

println("new joint positions", data.qpos)

mj_resetData(model, data)

# the @ means that something is a macro which means that it runs at compile time instead of run time 

init_visualiser()
visualise!(model, data, controller = random_controller!)

# the forward!() evaluates the ode at the current step 
# the step!() function actually integrates the ode over time 
