from mmg_class import mmg_class
import numpy as np
import matplotlib.pyplot as plt

def test_physics():
    mmg = mmg_class(u0=0, nP0=0, delta0=0)
    mmg.reset()
    
    # Set constant RPS
    nP_target = 20.0
    mmg.u[0] = nP_target
    
    dt = 0.1
    steps = 1000
    
    print(f"Starting physics test with nP={nP_target}...")
    
    for i in range(steps):
        mmg.timestep(dt)
        # Keep applying control (timestep updates state but not control input variable if we don't change it, 
        # but mmg.u is used in x_dot_func. 
        # Wait, mmg.timestep calls x_dot_func which uses self.u. 
        # So we just need to set self.u once if we want it constant.)
        
        if i % 100 == 0:
            print(f"Step {i}: x={mmg.x[0]:.2f}, u={mmg.x[3]:.2f}")
            
    print(f"Final x: {mmg.x[0]:.2f}")
    print(f"Final u: {mmg.x[3]:.2f}")
    
    if mmg.x[0] > 1.0:
        print("Physics test PASSED: Ship moved.")
    else:
        print("Physics test FAILED: Ship did not move.")

if __name__ == "__main__":
    test_physics()
