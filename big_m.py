import numpy as np
import pandas as pd

def print_tableau(tableau, columns, basic_vars, iteration):
    print(f"\n--- Iteration {iteration} ---")
    df = pd.DataFrame(tableau, columns=columns)
    df.insert(0, 'Basic_Var', basic_vars)
    print(df.to_string(index=False))

def big_m_method():
    M = 10000 
    
    columns = ['x1', 'x2', 's1', 's2', 'a1', 'RHS']
    
    C = np.array([3, 5, 0, 0, -M, 0], dtype=float)
    
    tableau = np.array([
        [1, 0, 1, 0, 0, 4],   
        [0, 2, 0, 1, 0, 12],  
        [3, 2, 0, 0, 1, 18]   
    ], dtype=float)
    
    basic_indices = [2, 3, 4] 
    basic_vars = ['s1', 's2', 'a1']
    
    iteration = 0
    while True:
        C_B = C[basic_indices]
        Z = np.dot(C_B, tableau)
        C_minus_Z = C - Z
        
        C_minus_Z_vars = C_minus_Z[:-1] 
        
        print_tableau(tableau, columns, basic_vars, iteration)
        print("C_j - Z_j:", np.round(C_minus_Z_vars, 2))
        print(f"Current Z value: {np.round(Z[-1], 2)}")
        
        if np.all(C_minus_Z_vars <= 1e-5):
            print("\nOptimal solution reached!")
            break
            
        entering_idx = np.argmax(C_minus_Z_vars)
        entering_var = columns[entering_idx]
        
        pivot_col = tableau[:, entering_idx]
        rhs = tableau[:, -1]
        
        ratios = []
        for i in range(len(rhs)):
            if pivot_col[i] > 1e-5:
                ratios.append(rhs[i] / pivot_col[i])
            else:
                ratios.append(float('inf'))
                
        if all(r == float('inf') for r in ratios):
            print("Problem is unbounded!")
            break
            
        leaving_idx = np.argmin(ratios)
        leaving_var = basic_vars[leaving_idx]
        
        print(f"Entering: {entering_var}, Leaving: {leaving_var}")
        
        basic_indices[leaving_idx] = entering_idx
        basic_vars[leaving_idx] = entering_var
        
        pivot_element = tableau[leaving_idx, entering_idx]
        tableau[leaving_idx, :] = tableau[leaving_idx, :] / pivot_element
        
        for i in range(tableau.shape[0]):
            if i != leaving_idx:
                multiplier = tableau[i, entering_idx]
                tableau[i, :] = tableau[i, :] - multiplier * tableau[leaving_idx, :]
                
        iteration += 1

    print("\n--- Final Optimal Values ---")
    for var, val in zip(basic_vars, tableau[:, -1]):
        if var in ['x1', 'x2']:
            print(f"{var} = {round(val, 2)}")
    print(f"Optimal Z = {round(Z[-1], 2)}")

if __name__ == "__main__":
    big_m_method()