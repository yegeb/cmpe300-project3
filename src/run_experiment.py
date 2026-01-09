import subprocess
import os
import random
import time
import matplotlib.pyplot as plt
import pandas as pd

# =============================================================================
# Configuration
# =============================================================================
GEN_GRAPH_CMD = "./dependencies/gen_graph"
ZK_VERIFIER_CMD = "./dependencies/zk_verifier"
HONEST_PROVER_SCRIPT = "src/honest_prover.py"
OUTPUT_DIR = "outputs"
TRIALS_PER_GRAPH = 100                          # Number of verification trials to estimate acceptance rate
LARGE_GRAPH_SIZES = range(200, 951, 37)         # Graph sizes to test
SAMPLE_COUNT = 15000                            # Large sample size to overcome coupon collector limit
GREEDY_ITERATIONS = 100                         # High iterations for finding global optima
PENALTY_MULTIPLIER = 10000                      # Heavy penalty for violating hot edges


# =============================================================================
# Graph Handling
# =============================================================================

def generate_graph(n: int, output_file: str) -> None:
    """
    Generates a graph of size n using gen_graph executable.
    """
    cmd = [GEN_GRAPH_CMD, "--size", str(n)]
    with open(output_file, "w") as f:
        subprocess.run(cmd, stdout=f, check=True)

def parse_graph(graph_file: str) -> tuple[int, list[tuple[int, int]], dict[int, list[int]]]:
    """
    Parses the graph file to get vertex count, edges list, and adjacency map.
    Returns: (n, edges_list, adjacency_dict)
    """
    with open(graph_file, "r") as f:
        content = f.read()
    
    tokens = content.split()
    if not tokens:
        raise ValueError("Graph file is empty")
        
    n = int(tokens[0])
    m = int(tokens[1])
    
    adj = {i: [] for i in range(n)}
    edges = []
    
    # Edges start from token index 2. Each edge is 2 tokens.
    idx = 2
    for _ in range(m):
        u = int(tokens[idx])
        v = int(tokens[idx+1])
        adj[u].append(v)
        adj[v].append(u)
        edges.append((u, v))
        idx += 2
        
    return n, edges, adj

# =============================================================================
# Probing & Verification
# =============================================================================

def probe_verifier(graph_file: str, samples: int = 1000) -> dict[tuple[int, int], int]:
    """
    Runs the verifier ONCE for 'samples' rounds to learn the distribution of 
    challenged edges efficiently ('Fast Probing' strategy).
    
    Sends dummy answers 'ANSWER 0 1' to keep the verifier going. This allows
    harvesting thousands of challenges in a single process.
    """
    edge_counts = {}
    
    cmd = [ZK_VERIFIER_CMD, "--graph", graph_file, "--rounds", str(samples)]
    
    try:
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        
        while True:
            line = p.stdout.readline()
            if not line:
                break
            
            line = line.strip()
            if line.startswith("CHALLENGE"):
                parts = line.split()
                if len(parts) >= 3:
                    u, v = int(parts[1]), int(parts[2])
                    
                    # Normalize edge to (min, max)
                    if u > v: u, v = v, u
                    edge = (u, v)
                    
                    edge_counts[edge] = edge_counts.get(edge, 0) + 1
                    
                    # Send dummy valid answer. The verifier only checks inequality of current pair.
                    # It does not check consistency with previous rounds or commitments.
                    try:
                        p.stdin.write("ANSWER 0 1\n")
                        p.stdin.flush()
                    except BrokenPipeError:
                        break
            elif line.startswith("ACCEPT") or line.startswith("REJECT"):
                break
                
        p.terminate()
        try:
            p.wait(timeout=1)
        except subprocess.TimeoutExpired:
            p.kill()
            
    except Exception as e:
        print(f"  Probe failed: {e}")
        
    return edge_counts

def run_verification(graph_file: str, coloring_file: str, rounds: int = 50) -> bool:
    """
    Runs the honest prover with the target verifier for a fixed number of rounds.
    Returns True if ACCEPT, False if REJECT.
    """
    verifier_cmd_str = f"{ZK_VERIFIER_CMD} --graph {graph_file} --rounds {rounds}"
    cmd = [
        "python3", HONEST_PROVER_SCRIPT,
        "--verifier", verifier_cmd_str,
        "--graph", graph_file,
        "--coloring", coloring_file
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if "ACCEPT" in result.stdout:
            return True
        elif result.returncode != 0:
            print(f"Verifier failed with return code {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
        else:
            # Rejection without crash
            out = result.stdout.strip()
            
        return False
    except Exception as e:
        print(f"Exception during verification: {e}")
        return False

# =============================================================================
# Coloring Algorithms 
# =============================================================================

def optimize_coloring(
    n: int, 
    adj: dict[int, list[int]], 
    initial_colors: list[int], 
    edge_weights: dict[tuple[int, int], int], 
    max_steps: int = 1000
) -> list[int]:
    """
    Refines a coloring using Min-Conflicts local search.
    The cost function is WEIGHTED by 'edge_weights' (frequency of verifier checks).
    """
    colors = list(initial_colors)
    nodes = list(range(n))
    
    for _ in range(max_steps):
        improved = False
        random.shuffle(nodes)
        
        for u in nodes:
            current_cost = 0
            
            # Identify colored neighbors and calculate current conflict cost
            neighbors_with_weights = []
            for v in adj[u]:
                edge = (min(u, v), max(u, v))
                w = edge_weights.get(edge, 1) # Default weight 1
                neighbors_with_weights.append((v, w))
                
                if colors[v] == colors[u] and colors[v] != -1:
                    current_cost += w
            
            # If there is a conflict, try to reduce it
            if current_cost > 0:
                # Calculate cost for each candidate color (0, 1, 2)
                neighbor_colors = [(colors[v], w) for v, w in neighbors_with_weights if colors[v] != -1]
                costs = {0: 0, 1: 0, 2: 0}
                for c_n, w in neighbor_colors:
                    costs[c_n] += w
                
                # Pick color with min cost
                min_c = min(costs, key=costs.get)
                min_cost_val = costs[min_c]
                
                if min_cost_val < current_cost:
                    colors[u] = min_c
                    improved = True
        
        if not improved:
            break
            
    return colors

def greedy_coloring(
    n: int, 
    adj: dict[int, list[int]], 
    edge_weights: dict[tuple[int, int], int], 
    iterations: int = 20,
    hot_degrees: dict[int, int] = None
) -> list[int]:
    """
    Runs weighted greedy coloring multiple times, followed by Local Search Refinement.
    Returns the best coloring found.
    """
    best_colors = None
    min_bad_cost = float('inf')
    
    # Define visit orders for greedy pass
    orders = []
    
    # 1. Hot Node Priority (Critical for exploiting verifier bias)
    if hot_degrees:
        orders.append(sorted(range(n), key=lambda x: hot_degrees.get(x, 0), reverse=True))
        
    orders.append(list(range(n))) 
    orders.append(sorted(range(n), key=lambda x: len(adj[x]), reverse=True)) 
    
    # Random orders for remaining iterations
    # If we added hot_degrees order, we have 3 deterministic orders, else 2.
    deterministic_count = len(orders)
    if iterations > deterministic_count:
        for _ in range(iterations - deterministic_count):
            nodes = list(range(n))
            random.shuffle(nodes)
            orders.append(nodes)
    
    # Pre-calculate list of all edges for fast evaluation
    all_edges = []
    for u in adj:
        for v in adj[u]:
            if u < v:
                all_edges.append((u, v))
    
    for nodes in orders:
        # 1. Greedy Pass (Weighted)
        colors = [-1] * n
        for u in nodes:
            costs = {0: 0, 1: 0, 2: 0}
            for v in adj[u]:
                if colors[v] != -1:
                    edge = (min(u, v), max(u, v))
                    w = edge_weights.get(edge, 1)
                    costs[colors[v]] += w
            
            best_c = min(costs, key=costs.get)
            colors[u] = best_c
            
        # 2. Local Search Refinement
        # Scale steps with N to ensure large graphs get enough optimization time
        colors = optimize_coloring(n, adj, colors, edge_weights, max_steps=5*n) 
        
        # 3. Evaluate Total Cost (Weighted)
        cost = 0
        for u, v in all_edges:
            if colors[u] == colors[v]:
                edge = (u, v)
                cost += edge_weights.get(edge, 1)
                
        if cost < min_bad_cost:
            min_bad_cost = cost
            best_colors = colors
            if min_bad_cost == 0:
                break
            
    return best_colors

# =============================================================================
# Utilities
# =============================================================================

def count_bad_edges(colors: list[int], edges: list[tuple[int, int]]) -> int:
    """Counts strict number of monochromatic edges (unweighted)."""
    bad = 0
    for u, v in edges:
        if colors[u] == colors[v]:
            bad += 1
    return bad

def count_hot_collisions(colors: list[int], edge_weights: dict[tuple[int, int], int]) -> int:
    """Counts collisions on 'hot' edges (weight > 1)."""
    collisions = 0
    for edge, w in edge_weights.items():
        if w > 1:
            u, v = edge
            if colors[u] == colors[v]:
                collisions += 1
    return collisions

# =============================================================================
# Main Execution
# =============================================================================

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    results = []
    
    graph_file = os.path.join(OUTPUT_DIR, "G.txt")
    coloring_file = os.path.join(OUTPUT_DIR, "coloring.txt")
    
    print(f"Starting experiments with sizes: {list(LARGE_GRAPH_SIZES)}")
    print(f"Configuration: Samples={SAMPLE_COUNT}, Iterations={GREEDY_ITERATIONS}")
    
    for n in LARGE_GRAPH_SIZES:
        # Use unique filenames for each size to avoid race conditions if multiple scripts run
        graph_file = os.path.join(OUTPUT_DIR, f"G_{n}.txt")
        coloring_file = os.path.join(OUTPUT_DIR, f"coloring_{n}.txt")

        start_time = time.time()
        
        # 1. Generate Graph
        try:
            generate_graph(n, graph_file)
        except Exception as e:
            print(f"Failed to generate graph for N={n}: {e}")
            continue
            
        # 2. Parse Graph
        n_actual, edges, adj = parse_graph(graph_file)
        m = len(edges)
        
        # 3. Probe Verifier for Bias
        t0 = time.time()
        print(f"  Probing verifier for N={n_actual} ({SAMPLE_COUNT} samples)...")
        hot_edges = probe_verifier(graph_file, samples=SAMPLE_COUNT)
        probe_time = time.time() - t0
        
        # Build weighted constraint map
        edge_weights = {}
        hot_degrees = {} # Count total frequency of checks per node
        max_freq = 0
        hot_edge_count = len(hot_edges)
        
        for e, count in hot_edges.items():
            # Heavy penalty for frequent edges. 
            edge_weights[e] = 1 + (count**2 * 100) 
            if count > max_freq: max_freq = count
            
            # Update node hotness
            u, v = e
            hot_degrees[u] = hot_degrees.get(u, 0) + count
            hot_degrees[v] = hot_degrees.get(v, 0) + count
            
        print(f"  -> Found {hot_edge_count} hot edges (Max freq: {max_freq}).")
        
        # Print Top 5 Hot Nodes to verify user hypothesis
        top_hot_nodes = sorted(hot_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"  -> Top 5 Hot Nodes: {top_hot_nodes}")
        
        # 4. Coloring Optimization
        t1 = time.time()
        best_colors = None
        min_hot_collisions = float('inf')
        
        # Retry logic for robustness
        for attempt in range(5):
            colors = greedy_coloring(n_actual, adj, edge_weights, iterations=GREEDY_ITERATIONS, hot_degrees=hot_degrees)
            
            hot_collisions = count_hot_collisions(colors, edge_weights)
            
            if best_colors is None or hot_collisions < min_hot_collisions:
                best_colors = colors
                min_hot_collisions = hot_collisions
            
            if hot_collisions == 0:
                print(f"  -> Satisfied hot edges on attempt {attempt+1}.")
                break
        coloring_time = time.time() - t1
        
        colors = best_colors
        final_hot_collisions = count_hot_collisions(colors, edge_weights)
        bad_edges_count = count_bad_edges(colors, edges)
         
        # Save coloring for verification
        with open(coloring_file, "w") as f:
            for c in colors:
                f.write(f"{c}\n")
        
        # 5. Run Verification Trials
        t2 = time.time()
        success_count = 0
        
        # Run verification trials sequentially
        for _ in range(TRIALS_PER_GRAPH):
            if run_verification(graph_file, coloring_file, rounds=50):
                success_count += 1
                
        verification_time = time.time() - t2
        
        acceptance_rate = success_count / TRIALS_PER_GRAPH
        bad_ratio = (bad_edges_count / m) if m > 0 else 0
        total_time = time.time() - start_time
        
        print(f"  -> Bad: {bad_edges_count} ({bad_ratio:.2%}), Hot Violations: {final_hot_collisions}, Accept Rate: {acceptance_rate:.2f}")
        print(f"     [Time] Total: {total_time:.2f}s (Probe: {probe_time:.2f}s, Coloring: {coloring_time:.2f}s, Verify: {verification_time:.2f}s)")
        
        results.append({
            "Graph Size": n_actual,
            "Edges": m,
            "Bad Edges": bad_edges_count,
            "Bad Edge Ratio": bad_ratio,
            "Hot Violations": final_hot_collisions,
            "Acceptance Rate": acceptance_rate,
            "Time Total (s)": round(total_time, 2),
            "Time Probe (s)": round(probe_time, 2),
            "Time Color (s)": round(coloring_time, 2),
            "Time Verify (s)": round(verification_time, 2)
        })
        
    # 6. Save Data
    df = pd.DataFrame(results)
    csv_path = os.path.join(OUTPUT_DIR, "experiment_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nData saved to {csv_path}")
    
    # 7. Plotting
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(df["Graph Size"], df["Acceptance Rate"], marker='o', linestyle='-', color='b')
        plt.title("Graph Size vs. Acceptance Rate (Optimized Attack)")
        plt.xlabel("Graph Size (N)")
        plt.ylabel("Acceptance Rate")
        plt.grid(True)
        plt.ylim(-0.05, 1.05)
        
        plot_path = os.path.join(OUTPUT_DIR, "experiment_results.png")
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")
    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == "__main__":
    main()
