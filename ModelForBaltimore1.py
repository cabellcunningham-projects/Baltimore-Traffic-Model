import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# 1. Load dataset
df_edges = pd.read_csv('edges_all.csv', low_memory=False)

# 2. Clean and convert data from OpenStreetMap columns
# Convert length (meters -> miles)
df_edges['length_miles'] = df_edges['length'] * 0.000621371

# Extract numeric speed limits from 'maxspeed' (e.g., '35 mph' -> 35.0), default to 35 mph if missing
df_edges['speed_limit_mph'] = (
    df_edges['maxspeed'].astype(str)
    .str.extract(r'(\d+)')
    .astype(float)
    .fillna(35.0)
)

# Free-Flow Travel Time (FFTT) in minutes
df_edges['fftt'] = (df_edges['length_miles'] / df_edges['speed_limit_mph']) * 60.0

# Extract lane count and compute capacity (default: 1 lane = 1800 veh/hr)
df_edges['lanes_num'] = (
    df_edges['lanes'].astype(str)
    .str.extract(r'(\d+)')
    .astype(float)
    .fillna(1.0)
)
df_edges['capacity'] = df_edges['lanes_num'] * 1800.0

# Identify tunnels and the Key Bridge
df_edges['is_tunnel'] = df_edges['tunnel'].fillna('').astype(str).str.lower().apply(
    lambda x: 1 if x in ['yes', 'building_passage', '1', 'true'] else 0
)

# Tag Key Bridge edges based on street name or ref (I-695 bridge over Patapsco)
name_match = df_edges['name'].fillna('').astype(str).str.contains('Francis Scott Key|Key Bridge', case=False)
ref_bridge_match = (
    df_edges['bridge'].fillna('').astype(str).str.lower().isin(['yes', '1', 'true']) &
    df_edges['ref'].fillna('').astype(str).str.contains('695', case=False)
)
df_edges['is_key_bridge'] = (name_match | ref_bridge_match).astype(int)


# 3. Congestion and Network Building Functions
def bpr_cost(fftt, volume, capacity, alpha=0.15, beta=4.0):
    return fftt * (1.0 + alpha * (volume / capacity) ** beta)

def build_graph(df):
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(
            row['u'],  # Source node
            row['v'],  # Target node
            length=row['length_miles'],
            fftt=row['fftt'],
            capacity=row['capacity'],
            current_cost=row['fftt'],
            volume=0.0,
            is_tunnel=row['is_tunnel'],
            is_key_bridge=row['is_key_bridge']
        )
    return G

def run_msa_traffic_assignment(G, od_matrix, iterations=5):
    # Group OD demand by origin to calculate single-source shortest paths efficiently
    origin_demands = {}
    for (origin, destination), demand in od_matrix.items():
        origin_demands.setdefault(origin, []).append((destination, demand))

    for u, v in G.edges():
        G[u][v]['volume'] = 0.0
        G[u][v]['current_cost'] = G[u][v]['fftt']

    for k in range(1, iterations + 1):
        aux_volumes = {edge: 0.0 for edge in G.edges()}

        for origin, dest_list in origin_demands.items():
            if not G.has_node(origin):
                continue
            
            # Compute shortest paths from 'origin' to all reachable nodes in ONE operation
            paths = nx.single_source_dijkstra_path(G, origin, weight='current_cost')

            for destination, demand in dest_list:
                if destination in paths:
                    path = paths[destination]
                    for i in range(len(path) - 1):
                        aux_volumes[(path[i], path[i+1])] += demand

        gamma = 1.0 / k
        for u, v in G.edges():
            edge = (u, v)
            G[u][v]['volume'] = (1 - gamma) * G[u][v]['volume'] + gamma * aux_volumes[edge]
            G[u][v]['current_cost'] = bpr_cost(
                G[u][v]['fftt'],
                G[u][v]['volume'],
                G[u][v]['capacity']
            )
    return G


# 4. Define Realistic Origin-Destination Demand
# Get all node IDs directly from the edge dataset
all_nodes = list(set(df_edges['u']).union(set(df_edges['v'])))

# Select origins and destinations across the network
od_demand = {}
sample_origins = all_nodes[:5]
sample_destinations = all_nodes[-5:]

# Generate vehicles flowing across regional origins and destinations
for origin in sample_origins:
    for dest in sample_destinations:
        if origin != dest:
            od_demand[(origin, dest)] = 500  # 500 trips per OD pair

# 5. Execute Scenarios
print("Building baseline graph...")
G_base = build_graph(df_edges)
print("Running baseline traffic assignment...")
G_base = run_msa_traffic_assignment(G_base, od_demand)

print("Building disrupted graph...")
G_disrupted = build_graph(df_edges)

# Remove Key Bridge edges
key_bridge_edges = [(u, v) for u, v, d in G_disrupted.edges(data=True) if d['is_key_bridge'] == 1]
if key_bridge_edges:
    G_disrupted.remove_edges_from(key_bridge_edges)
    print(f"Removed {len(key_bridge_edges)} Key Bridge edge(s).")
else:
    print("Notice: No explicit Key Bridge edges found by name/ref; running general disruption.")

# Apply freight tunnel penalties permanently to free-flow travel time
for u, v, d in G_disrupted.edges(data=True):
    if d['is_tunnel'] == 1:
        d['fftt'] *= 100.0
        d['current_cost'] = d['fftt']

print("Running disrupted traffic assignment...")
G_disrupted = run_msa_traffic_assignment(G_disrupted, od_demand)


# 6. Extract and Display Metrics
def calculate_network_metrics(G):
    total_vht = sum(d['volume'] * (d['current_cost'] / 60.0) for _, _, d in G.edges(data=True))
    total_vmt = sum(d['volume'] * d['length'] for _, _, d in G.edges(data=True))

    bottlenecks = []
    for u, v, d in G.edges(data=True):
        vc_ratio = d['volume'] / d['capacity'] if d['capacity'] > 0 else 0
        if vc_ratio > 0.8:
            bottlenecks.append((u, v, vc_ratio, d['current_cost']))
    return total_vht, total_vmt, pd.DataFrame(bottlenecks, columns=['source', 'target', 'vc_ratio', 'delay_min'])

vht_base, vmt_base, _ = calculate_network_metrics(G_base)
vht_dis, vmt_dis, bottlenecks_dis = calculate_network_metrics(G_disrupted)

print("\n--- RESULTS ---")
print(f"Baseline VHT: {vht_base:,.2f} hours | Disrupted VHT: {vht_dis:,.2f} hours")
if vht_base > 0:
    print(f"Total Daily Delay Increase: {vht_dis - vht_base:,.2f} hours ({((vht_dis - vht_base)/vht_base)*100:.1f}%)")

# Compare edge flow differences and save output
results = []
for u, v, d in G_base.edges(data=True):
    vol_base = d['volume']
    vol_dis = G_disrupted[u][v]['volume'] if G_disrupted.has_edge(u, v) else 0
    results.append({
        'source': u, 'target': v,
        'vol_base': vol_base, 'vol_dis': vol_dis,
        'vol_change': vol_dis - vol_base
    })

df_results = pd.DataFrame(results)
df_results.to_csv('part1_traffic_impact_results.csv', index=False)
print("Saved traffic simulation results to 'part1_traffic_impact_results.csv'.")
