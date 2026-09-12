import numpy as np
import pandas as pd
import copy

costs = np.array([
    [8, 5, 6],
    [15, 10, 12],
    [3, 9, 10]
])
supply = np.array([120, 80, 80])
demand = np.array([150, 70, 60])

def print_grid(matrix, title):
    print(f"\n--- {title} ---")
    df = pd.DataFrame(matrix, columns=['W1', 'W2', 'W3'], index=['F1', 'F2', 'F3'])
    print(df)

def vam(costs, supply, demand):
    print("\nStarting VAM to find Initial Basic Feasible Solution...")
    s = supply.copy()
    d = demand.copy()
    allocations = np.zeros_like(costs)
    
    iteration = 1
    while s.sum() > 0 and d.sum() > 0:
        row_penalties = []
        col_penalties = []
        
        for i in range(costs.shape[0]):
            if s[i] > 0:
                valid_costs = [costs[i][j] for j in range(costs.shape[1]) if d[j] > 0]
                if len(valid_costs) >= 2:
                    sorted_costs = sorted(valid_costs)
                    row_penalties.append(sorted_costs[1] - sorted_costs[0])
                elif len(valid_costs) == 1:
                    row_penalties.append(valid_costs[0])
                else:
                    row_penalties.append(-1)
            else:
                row_penalties.append(-1)
                
        for j in range(costs.shape[1]):
            if d[j] > 0:
                valid_costs = [costs[i][j] for i in range(costs.shape[0]) if s[i] > 0]
                if len(valid_costs) >= 2:
                    sorted_costs = sorted(valid_costs)
                    col_penalties.append(sorted_costs[1] - sorted_costs[0])
                elif len(valid_costs) == 1:
                    col_penalties.append(valid_costs[0])
                else:
                    col_penalties.append(-1)
            else:
                col_penalties.append(-1)
                
        max_row_pen = max(row_penalties)
        max_col_pen = max(col_penalties)
        
        if max_row_pen >= max_col_pen:
            row_idx = row_penalties.index(max_row_pen)
            valid_col_indices = [j for j in range(costs.shape[1]) if d[j] > 0]
            col_idx = min(valid_col_indices, key=lambda j: costs[row_idx][j])
        else:
            col_idx = col_penalties.index(max_col_pen)
            valid_row_indices = [i for i in range(costs.shape[0]) if s[i] > 0]
            row_idx = min(valid_row_indices, key=lambda i: costs[i][col_idx])
            
        qty = min(s[row_idx], d[col_idx])
        allocations[row_idx][col_idx] = qty
        s[row_idx] -= qty
        d[col_idx] -= qty
        
        print(f"Step {iteration}: Allocated {qty} at (F{row_idx+1}, W{col_idx+1})")
        iteration += 1
        
    print_grid(allocations, "Initial Basic Feasible Solution (VAM)")
    initial_cost = np.sum(allocations * costs)
    print(f"Initial Transportation Cost: ${initial_cost}")
    return allocations

def get_uv(allocations, costs):
    u = [None] * costs.shape[0]
    v = [None] * costs.shape[1]
    u[0] = 0 
    
    basic_cells = [(i, j) for i in range(costs.shape[0]) for j in range(costs.shape[1]) if allocations[i][j] > 0]
    
    while None in u or None in v:
        for (i, j) in basic_cells:
            if u[i] is not None and v[j] is None:
                v[j] = costs[i][j] - u[i]
            elif v[j] is not None and u[i] is None:
                u[i] = costs[i][j] - v[j]
    return u, v

def get_penalties(allocations, costs, u, v):
    penalties = np.zeros_like(costs)
    for i in range(costs.shape[0]):
        for j in range(costs.shape[1]):
            if allocations[i][j] == 0:
                penalties[i][j] = u[i] + v[j] - costs[i][j]
    return penalties

def find_loop(start_cell, allocations):
    def dfs(current_cell, path, is_vertical):
        if len(path) > 3 and current_cell == start_cell:
            return path
            
        for next_cell in [(r, c) for r in range(allocations.shape[0]) for c in range(allocations.shape[1])]:
            if next_cell == current_cell or (next_cell != start_cell and allocations[next_cell[0]][next_cell[1]] == 0):
                continue
                
            if is_vertical and next_cell[1] == current_cell[1]:
                if next_cell not in path or next_cell == start_cell:
                    res = dfs(next_cell, path + [next_cell], not is_vertical)
                    if res: return res
                    
            elif not is_vertical and next_cell[0] == current_cell[0]:
                if next_cell not in path or next_cell == start_cell:
                    res = dfs(next_cell, path + [next_cell], not is_vertical)
                    if res: return res
        return None

    return dfs(start_cell, [start_cell], False)

def modi(allocations, costs):
    print("\nStarting MODI Optimization...")
    iteration = 1
    
    while True:
        print(f"\n--- MODI Iteration {iteration} ---")
        u, v = get_uv(allocations, costs)
        print(f"u values: {u}")
        print(f"v values: {v}")
        
        penalties = get_penalties(allocations, costs, u, v)
        print_grid(penalties, "Penalty Matrix (u_i + v_j - c_ij)")
        
        if np.all(penalties <= 0):
            print("\nOptimal solution verified! All penalties <= 0.")
            break
            
        max_pen_idx = np.unravel_index(np.argmax(penalties), penalties.shape)
        print(f"Most positive penalty at cell F{max_pen_idx[0]+1}, W{max_pen_idx[1]+1}. Shifting allocation.")
        
        loop = find_loop(max_pen_idx, allocations)
        loop = loop[:-1] 
        
        minus_cells = loop[1::2]
        theta = min([allocations[r][c] for r, c in minus_cells])
        
        print(f"Closed loop found: {loop}")
        print(f"Shift amount (theta) = {theta}")
        
        for k, (r, c) in enumerate(loop):
            if k % 2 == 0:
                allocations[r][c] += theta 
            else:
                allocations[r][c] -= theta 
                
        iteration += 1

    print_grid(allocations, "Final Optimal Allocation Matrix")
    final_cost = np.sum(allocations * costs)
    print(f"\nMinimum Total Transportation Cost: ${final_cost}")

if __name__ == "__main__":
    initial_alloc = vam(costs, supply, demand)
    modi(initial_alloc, costs)