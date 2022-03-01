#!/usr/bin/env python
# coding: utf-8

# # Loading libraries and defining helper functions
import numpy as np
import pandas as pd
import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.stats import linregress

plt.ion()

def write_query(dict_cols_str, dict_cols_num):
	# Writes a query containing all conditions in the dictionaries dict_cols_str and dict_cols_num
	# 
	# Input:
	# dict_cols_str: dictionary containing string conditions
	# dict_cols_num: dictionary containing numerical conditions
	# 
	# Output:
	# q: string containing the query to be applied to the DataFrame

	# Part of the query related to string conditions
	# The value of each dictionary key is a list.
	# This way we can filter multiple conditions for the same field. For example:
	# select rows where the columns 'country' is equal to 'Austria' and 'Germany'
	q = ''
	for (k,v) in dict_cols_str.items():
		q += '('
		for i in v:
			q += f"`{k}` == '{i}' or "
		q = q[:-4]
		q += ") and "

	# Part of the query related to numerical conditions
	# Because numerical conditions are not constrained to 'equal' (as the string conditions above),
	# we need to additionally specify the kind of condition ('<', '<=', '==', '>', '>=')
	for (k,v) in dict_cols_num.items():
		q += f"(`{k}` {v[1]} {v[0]}) and "

	q = q[:-5]
	return q

def drop_nan_columns(data):
	# Drop columns where all values are nan
	for c in data.columns:
		if data[c].isnull().values.all():
			print(f'Column {c} is empty')
			data.drop(columns=[c], inplace=True)
	return data



# Read data
sep = ','
# sep = ';'
decimal = '.'
# decimal = '.'

path = '.'
# filename = 'DatabaseCoffeeColombia_2000_19.csv'
filename = 'data/coffee_2014-2018.csv'

data = pd.read_csv(f'{path}/{filename}', sep=sep, index_col=None, decimal=decimal)
data = drop_nan_columns(data)
data.fillna(0, inplace=True)

data.loc[data['Exporter'] == 'United States', 'Exporter'] = 'United States of America'
data.loc[data['Importer'] == 'United States', 'Importer'] = 'United States of America'
data.loc[data['Exporter'] == "Cote d'Ivoire", 'Exporter'] = 'Ivory Coast'
data.loc[data['Importer'] == "Cote d'Ivoire", 'Importer'] = 'Ivory Coast'
data.loc[data['Exporter'] == "Czech Republic", 'Exporter'] = 'Czechia'
data.loc[data['Importer'] == "Czech Republic", 'Importer'] = 'Czechia'
data.loc[data['Exporter'] == "China, Hong Kong SAR", 'Exporter'] = 'Hong Kong S.A.R.'
data.loc[data['Importer'] == "China, Hong Kong SAR", 'Importer'] = 'Hong Kong S.A.R.'
data.loc[data['Exporter'] == "China, Macao SAR", 'Exporter'] = 'Macao S.A.R'
data.loc[data['Importer'] == "China, Macao SAR", 'Importer'] = 'Macao S.A.R'
data.loc[data['Exporter'] == "Korea, Republic", 'Exporter'] = 'South Korea'
data.loc[data['Importer'] == "Korea, Republic", 'Importer'] = 'South Korea'
data.loc[data['Exporter'] == "Russian Federation", 'Exporter'] = 'Russia'
data.loc[data['Importer'] == "Russian Federation", 'Importer'] = 'Russia'

data = data.loc[data['Exporter'] != 'Other Asia, nes',:]
data = data.loc[data['Importer'] != 'Other Asia, nes',:]
data = data.loc[data['Exporter'] != 'Curacao',:]
data = data.loc[data['Importer'] != 'Curacao',:]
data = data.loc[data['Exporter'] != 'Areas, nes',:]
data = data.loc[data['Importer'] != 'Areas, nes',:]

# Remove not useful columns
data.drop(columns=['Exporter M.49', 'Exporter ISO3', 'Importer M.49', 'Importer ISO3', 'HS 1996', 'Quarantine code (value)', 'Quarantine code (weight)'], inplace=True)

# Remove lines where a country is exporting to itself
data.drop(data[data["Exporter"] == data["Importer"]].index, inplace=True)

# # Filtering data

# Define filter to apply to the DataFrame
# 
# Dictionary containing string conditions
dict_cols_str = {}
# dict_cols_str['Resource'] = ['Coffee, not roasted, not decaffeinated']
# dict_cols_str['Exporter'] = ['Colombia']

# 
# Dictionary containing numerical conditions
dict_cols_num = {}
dict_cols_num['Year'] = [2018, '==']
dict_cols_num['Weight (1000kg)'] = [1000, '>=']

# Write the query
q = write_query(dict_cols_str, dict_cols_num)

# Create a new DataFrame with filtered data
data_filt = data.query(q)

# Limit number of trades (sort descending by weight)
maxrows = 500
data_filt = data_filt.sort_values('Weight (1000kg)', ascending=False).head(maxrows)

# Select only biggest exporters and importers
nexp = 10
nodes_exp = data_filt.groupby("Exporter").aggregate(np.sum).sort_values("Weight (1000kg)", ascending=False).index[:nexp].tolist()
nimp = 10
nodes_imp = data_filt.groupby("Importer").aggregate(np.sum).sort_values("Weight (1000kg)", ascending=False).index[:nimp].tolist()
excl = False
if excl is True:
	# Select only pairs both in biggest exporters AND biggest importers
	data_filt = data_filt.loc[data_filt["Exporter"].isin(nodes_exp) & data_filt['Importer'].isin(nodes_imp), :]
else:
	# Select pairs where either the exporter OR the importer is among the biggest
	data_filt = data_filt.loc[data_filt["Exporter"].isin(nodes_exp) | data_filt['Importer'].isin(nodes_imp), :]

# Aggregate (sum) all kinds of coffees (roasted, not roasted, caffeinated, etc)
data_filt = data_filt.groupby(['Exporter', "Importer"], as_index=False).aggregate(np.sum)

# Remove empty columns
data_filt = drop_nan_columns(data_filt)

# Export to file
data_filt.to_csv('data_filt.csv')


# # Defining the graph
# We now have a dataset with the relationship exporter -> importer, where we applied some conditions to unencumber the graph (type of coffee, year, minimum weight, max number of lines).
# 
# Because of that relationship, the natural choice is a directed graph, which networkx provides.

# Create empty graph
G = nx.DiGraph()




# First we have to define the graph nodes. As we have Importer and Exporter columns in our dataset, it's easy. We just need to ensure that no duplicate nodes exist (a country that imports and exports).
imp = data_filt['Importer'].unique()
exp = data_filt['Exporter'].unique()
nodes = imp
nodes = np.append(nodes, exp)
nodes = np.unique(nodes)


# And the edges between the nodes are the flows from exporters to importers. Here we'll assume that if the column Weight is not zero, there is a an edge between that pair of (exporter, importer).
# 
# We also define an additional column, representing the relative weight. This might be useful to plot each edge in a different color or thickness.

# In[7]:


# Edges are the flows from each Exporter to each Importer
pd_flow = pd.DataFrame(columns=['Exporter', 'Importer', 'Weight', 'Relative weight', 'coord_x', 'coord_y'])
pd_flow['Exporter'] = data_filt['Exporter']
pd_flow['Importer'] = data_filt['Importer']
pd_flow['Weight'] = data_filt['Weight (1000kg)']
pd_flow['Relative weight'] = 100*data_filt['Weight (1000kg)']/data_filt['Weight (1000kg)'].sum()
cap_rel_weight = 10
pd_flow.loc[pd_flow['Relative weight'] > cap_rel_weight, 'Relative weight'] = cap_rel_weight


G.add_weighted_edges_from(pd_flow[['Exporter', 'Importer', 'Weight']].values)



# Until now our graph is an abstract concept. We (and networkx) have no idea of the position of each node. 
# 
# The simplest approach (at least for testing purposes) is to let networkx find the positions.
pos = nx.spring_layout(G)




# And now we can view the graph.
fig, ax = plt.subplots(figsize=(16, 12))
nx.draw_networkx_nodes(G, pos, node_color="b", alpha=0.5, node_size=1)
nx.draw_networkx_labels(G, pos, font_size=8)
nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color="m")



# Cool, we have a graph! But it's horrendous.
# 
# Fortunately, as our nodes are countries, we can use theirs coordinates. To do this, we need some dataset of each country. NaturalEarth has some really good datasets.
path_gpd = 'data/ne_50m_admin_0_countries.zip'
gdf = gpd.read_file(path_gpd)
# gdf = gdf.to_crs(epsg=3395) # mercator projection
gdf = gdf.to_crs('+proj=robin') # robinson projection 
gdf = gdf[(gdf['SOVEREIGNT'] != "Antarctica")]


# Let's recreate the graph just to make sure nothing is being inserted twice.
G = nx.DiGraph()




# Because now we have two datasets from two different sources (coffee data and countries data), we need first to guarantee some consistency between them. Let's select only those countries for which we have info in both datasets. 
# 
# And for those countries, let's find the centroid and use is as the position.

# Check if all nodes are in the geopandas dataset
coords = {}
q = ''
# for n in G.nodes():
for n in nodes:
	if n not in gdf['ADMIN'].values:
		print(f'Node {n} not in the geopandas dataset')
		q += f"(`Importer` != '{n}') and (`Exporter` != '{n}') and "
	else:
		G.add_node(n)
		country_polygon = gdf.loc[gdf['ADMIN'] == n, 'geometry']
		coords[n] = (country_polygon.centroid.x.values[0], country_polygon.centroid.y.values[0])
q = q[:-5]
if q != '':
	pd_flow.query(q, inplace=True)


# Now we add the edges, their labels and position.
# Add edges to the graph
G.add_weighted_edges_from(pd_flow[['Exporter', 'Importer', 'Weight']].values)

# Find the position of each node (there are other algorithms, check documentation)
pos = coords

# Let's manually change the position of Canada and US
# If using the mercator projection (epsg 3395), uncomment both lines below
# pos["Canada"] = (-1.25e7, 8.5e6)
# pos['United States of America'] = (-1.1e7, 4.8e6)

# If using the robinson projection, uncomment both lines below
pos["Canada"] = (-8.6e6, 6.2e6)
pos['United States of America'] = (-8.8e6, 4.2e6)


# Let's use the relative weight as the label of each edge
label_edges = {(row['Exporter'],row['Importer']): f'{row["Weight"]:.1f}' for ind, row in pd_flow.iterrows()}


# # Visualization

# Drawing
fig, ax = plt.subplots(figsize=(16, 8))

# Draw world map
gdf.plot(edgecolor='tab:blue', facecolor='w', ax=ax)

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_color="b", alpha=0.5, node_size=1)

# Draw node labels (countries names)
# nx.draw_networkx_labels(G, pos, font_size=8)

# Draw edges
#nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color="m", width=0.25*flow_weight_rel) (Calcular o peso)

# nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color="m")
nx.draw_networkx_edges(G, pos, edge_color='tab:red', width=pd_flow['Relative weight'].values)

# Strech axes and remove them
fig.tight_layout()
ax.axis('off')

# Export figure
fig.savefig('graph_coffee.png', dpi=200)

adj = nx.adjacency_matrix(G)


# For example, we can check, for each country, from how many other countries it imports coffee.
print('Imports from how many countries')
for i, n in enumerate(G.nodes):
	n_imp = adj.getcol(i).count_nonzero()
	print(f'{n}: {n_imp}')
print('')


# And the opposite: to how many countries each country exports coffee.
print('Exports to how many countries')
for i, n in enumerate(G.nodes):
	n_exp = adj.getrow(i).count_nonzero()
	print(f'{n}: {n_exp}')
print('')


# And finally we can try to visualize the adjacency matrix.
fact_size = max(len(G.nodes)/40, 1)
fig, ax = plt.subplots(figsize=(8*fact_size, 8*fact_size))
ax.spy(adj)

ax.xaxis.set_label_text('Importer')
ax.xaxis.set_ticks(range(0,adj.shape[0]))
ax.xaxis.set_ticklabels(G.nodes)
ax.xaxis.set_tick_params(rotation=90)
ax.xaxis.set_label_position('top')

ax.yaxis.set_label_text('Exporter')
ax.yaxis.set_ticks(range(0,adj.shape[0]))
ax.yaxis.set_ticklabels(G.nodes)
ax.grid()

fig.tight_layout()
fig.savefig('spy_coffee.png', dpi=200)



df_properties = pd.DataFrame(index=G.nodes())
df_properties['degree_centrality'] = nx.degree_centrality(G).values()
df_properties['betweenness_centrality'] = nx.betweenness_centrality(G).values()
df_properties['closeness_centrality'] = nx.closeness_centrality(G).values()
df_properties['pagerank'] = nx.pagerank(G).values()

# In edges and out edges
for n in G.nodes:
	df_properties.loc[n, 'in edges degree'] = G.in_degree(n)
	df_properties.loc[n, 'out edges degree'] = G.out_degree(n)
	export_total = sum(G.get_edge_data(n, n_out)["weight"] for n_out in G[n])
	df_properties.loc[n, 'total export'] = export_total
df_properties['total export relative'] = df_properties['total export']/df_properties['total export'].sum()

df_properties.to_csv('properties_graph_coffee.csv')


for (e1, e2) in G.edges():
	if (e2, e1) in G.edges():
		exp_e1_to_e2 = G.get_edge_data(e1, e2)["weight"]
		exp_e2_to_e1 = G.get_edge_data(e2, e1)["weight"]
		ratio = max(exp_e1_to_e2, exp_e2_to_e1)/min(exp_e1_to_e2, exp_e2_to_e1)
		print(f'({e1}, {e2}) is a loop. {e1} exports {exp_e1_to_e2} to {e2}. {e2} exports {exp_e2_to_e1} to {e1}. The ratio is {ratio:.1f}')

# Find adjacency matrix with weight for each pair (exporter,importer)
S = pd.DataFrame()
for n1 in G.nodes():
	for n2 in G.nodes():
		if n2 in G[n1]:
			S.loc[n1,n2] = G.get_edge_data(n1, n2)["weight"]
		else:
			S.loc[n1,n2] = 0
S.to_csv('matrix_S_full.csv')


# Export list of importers for the biggest exporters
big5 = data_filt.groupby("Exporter").aggregate(np.sum).sort_values("Weight (1000kg)", ascending=False).index[:5].tolist()
for n in big5:
	df = data_filt.loc[data_filt['Exporter'] == n, :].sort_values("Weight (1000kg)", ascending=False)

	df.to_csv(f'biggest_importers_{n}.csv', sep=';', decimal=',')