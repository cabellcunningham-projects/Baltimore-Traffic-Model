import pandas as pd
import matplotlib.pyplot as plt

# Load results
df_results = pd.read_csv('part1_traffic_impact_results.csv')
df_edges = pd.read_csv('edges_all.csv', low_memory=False)

# Merging road names
df_plot = pd.merge(
    df_results,
    df_edges[['u', 'v', 'name', 'ref']],
    left_on=['source', 'target'],
    right_on=['u', 'v'],
    how='left'
)

# Clean road labels
df_plot['road_label'] = df_plot['ref'].fillna(df_plot['name'])
df_plot['road_label'] = df_plot['road_label'].fillna('Local Connector')

# Group by unique road name so segments don't stack on top of each other
df_grouped = df_plot.groupby('road_label', as_index=False)['vol_change'].max()

# Top 10 Unique Diverted Corridors
top_increases = df_grouped[df_grouped['vol_change'] > 0].sort_values(by='vol_change', ascending=False).head(10)

# Create horizontal bar chart
plt.figure(figsize=(12, 6))
bars = plt.barh(top_increases['road_label'], top_increases['vol_change'], color='#d95f02', edgecolor='black')

plt.xlabel('Peak Net Vehicle Increase (Vehicles/hr)', fontsize=11)
plt.ylabel('Roadway Corridor', fontsize=11)
plt.title('Baltimore Traffic Re-Routing: Top 10 Diverted Corridors', fontsize=13, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)


for bar in bars:
    width = bar.get_width()
    plt.text(width + 15, bar.get_y() + bar.get_height()/2, f'+{int(width):,}', va='center', fontsize=10)

plt.tight_layout()
plt.show()
