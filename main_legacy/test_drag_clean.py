import numpy as np
from cop_dragforce_v3_ant import BoxDropSimulator, PHYSICS_PARAMETERS

# Mock calculate_all_metrics to avoid printing progress
import cop_dragforce_v3_ant
def mock_calc(self, dynamics_func):
    n = len(self.t)
    self.com['h'] = self.states[0]
    self.com['v'] = self.states[1]
    self.com['a'] = np.zeros(n)
    self.com['f'] = np.zeros(n)
    self.aero_data['squeeze'] = np.zeros(n)
    self.aero_data['drag'] = np.zeros(n)
    self.contact_data['f_contact'] = np.zeros(n)
    for i in range(n):
        y = self.states[:, i]
        h, v, phi, theta, psi = y[0:5]
        omegas = y[5:8]
        f_sq, f_dr, tau, f_total_contact = dynamics_func(h, v, [phi, theta, psi], omegas)
        self.aero_data['drag'][i] = f_dr
        self.aero_data['squeeze'][i] = f_sq
        self.contact_data['f_contact'][i] = f_total_contact
        self.com['f'][i] = f_total_contact + f_sq

cop_dragforce_v3_ant.BoxDropResult.calculate_all_metrics = mock_calc

def check_drag(scale):
    params = PHYSICS_PARAMETERS.copy()
    params['scale_drag'] = scale
    params['scale_squeeze'] = 0.0
    
    sim = BoxDropSimulator(sim_params=params)
    initial_corners = [1.0, 1.0, 1.0, 1.0]
    simulation_time = 0.4
    solver_max_step = 0.001
    
    result = sim.run(
        h_corners=initial_corners, 
        t_sim=simulation_time, 
        max_step=solver_max_step
    )
    
    max_drag = np.max(np.abs(result.aero_data['drag']))
    final_v = result.states[1, -1]
    final_h = result.states[0, -1]
    
    print(f"Scale Drag: {scale}")
    print(f"Max Drag Force: {max_drag:.6f} N")
    print(f"Final Velocity: {final_v:.6f} m/s")
    print(f"Final Height: {final_h:.6f} m")
    return max_drag, final_v, final_h

print("Testing Drag Force Implementation...")
check_drag(0.0)
print("-" * 30)
check_drag(1.0)
print("-" * 30)
