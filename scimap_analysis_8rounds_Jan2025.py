python -m cellpose --dir /scratch/zqn7td/FA1_FA2_FA3_4i/imageanalysis/analysis4/FA1-2_morecrop.ome.tiff --pretrained_model cyto3 --chan 0 --chan2 0 --diameter 7 --save_tif

python -m mcquant.cli --masks /Volumes/FallahiLab/Maize-Data/Leica-Thunder/Kimberly_Nguyen/20240624_HetMel/imageanalysis/analysis4/FA1-2_morecrop_hoechst_cp_masks.tif --image /Volumes/FallahiLab/Maize-Data/Leica-Thunder/Kimberly_Nguyen/20240624_HetMel/imageanalysis/analysis4/FA1-2_morecrop.ome.tiff --output /scratch/zqn7td/FA1_FA2_FA3_4i/imageanalysis/analysis4 --channel_names /Volumes/FallahiLab/Maize-Data/Leica-Thunder/Kimberly_Nguyen/20240624_HetMel/imageanalysis/analysis2/markers.csv

python


conda create --name scimapKN python=3.10
conda activate scimapKN
pip install git+https://github.com/kimnguyen72/scimap.git
conda update git+https://github.com/kimnguyen72/scimap.git
pip install git+https://github.com/kimnguyen72/scimap.git
################SETUP##########
conda activate scimap
python

import os

QT_QPA_PLATFORM='offscreen'

os.chdir('/Volumes/FallahiLab/Maize-Data/Leica-Thunder/Kimberly_Nguyen/20240624_HetMel/FA3-2/2025_analysis')
os.chdir('/standard/vol328/FallahiLab/Maize-Data/Leica-Thunder/Kimberly_Nguyen/20240624_HetMel/FA3-2/2025_analysis')
os.getcwd()

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import anndata as ad

import scimap as sm

import scanpy as sp
import pandas as pd
import napari as napari


import matplotlib
import matplotlib.pyplot as plt

import itertools 

try:
    from . import scimapKN     # "myapp" case
except:
    import scimapKN

######Phenotyping single cell quantification data#################

feature_table_path=["/Volumes/FallahiLab/Maize-Data/Leica-Thunder/Kimberly_Nguyen/20240624_HetMel/FA3-2/2025_analysis/quantification-master/2025_8rounds_FullStack_BgSub_Z0_Hoechst_cp_masks.csv"]
feature_table_path=["/standard/vol328/FallahiLab/Maize-Data/Leica-Thunder/Kimberly_Nguyen/20240624_HetMel/FA3-2/2025_analysis/quantification-master/2025_8rounds_FullStack_BgSub_Z0_Hoechst_cp_masks.csv"]

adata = sm.pp.mcmicro_to_scimap(feature_table_path)



sm.pp.log1p(adata, layer='log', verbose=True)
#contents of expression matrix
adata


adata.X
adata.var
adata

#################        Write and Read           ######
adata.write('8rounds_naparigater_excluded-low-counts_spatial_v4.h5ad')
adata = ad.read_h5ad('/standard/vol328/FallahiLab/Maize-Data/Leica-Thunder/Kimberly_Nguyen/20240624_HetMel/FA3-2/2025_analysis/8rounds_naparigater_excluded-low-counts_spatial_v4.h5ad')
adata = ad.read_h5ad('/Volumes/FallahiLab/Maize-Data/Leica-Thunder/Kimberly_Nguyen/20240624_HetMel/FA3-2/2025_analysis/8rounds_naparigater_excluded-low-counts_spatial_v4.h5ad')

adata = ad.read_h5ad('/Volumes/FallahiLab/Maize-Data/Leica-Thunder/Kimberly_Nguyen/20240624_HetMel/FA3-2/2025_analysis/8rounds_naparigater_excluded-low-counts_spatial_v4.h5ad')

adata = ad.read_h5ad('/standard/vol328/FallahiLab/Maize-Data/Leica-Thunder/Kimberly_Nguyen/20240624_HetMel/FA3-2/2025_analysis/8rounds_phenotyped_combinations.h5ad')


adata = sm.tl.spatial_interaction(adata, x_coordinate='X_centroid', y_coordinate='Y_centroid',
                            method='knn', knn=15, permutation=1000, pval_method='zscore',
                            label='interaction_knn_zscore')

subsetMarkers1 = ['cFOS', 'FRA1', 'FRA2', 'cJUN', 'JUNB', 'JUND']
sm.pl.heatmap(adata, groupBy='phenotype', standardScale='column', figsize=(8,5), subsetMarkers=subsetMarkers1, subsetGroups = all_combinations, showPrevalence=True, vmin=-1, vmax=1, clusterColumns=True, clusterRows=True)  
sm.pl.heatmap(adata, groupBy='phenotype', standardScale='column', figsize=(8,5), subsetMarkers=subsetMarkers1, subsetGroups = combinations_above_n500, showPrevalence=True, vmin=-1, vmax=1, clusterColumns=True, clusterRows=True)  

#manual gating using napari viewer


sm.pl.napariGater (image_path, adata)
adata = sm.pp.rescale (adata, gate=adata.uns['gates'])


viewer = napari.Viewer()
image_path = 'quantification-master/2025_8rounds_FullStack_BgSub_Z0.ome.tif'
marker_of_interest = 'MITF'
marker_of_interest = 'SOX10'
marker_of_interest = 'NGFR'
marker_of_interest = 'cFOS'
marker_of_interest = 'FRA1'
marker_of_interest = 'FRA2'
marker_of_interest = 'cJUN'
marker_of_interest = 'JUNB'
marker_of_interest = 'JUND'
marker_of_interest = 'FAP'
marker_of_interest = 'CD45'
marker_of_interest = 'SOX9'
marker_of_interest = 'CD3D'
marker_of_interest = 'FOXP3'
marker_of_interest = 'CD4'
marker_of_interest = 'CD31'
marker_of_interest = 'PRAME'
sm.pl.gate_finder(image_path, adata, marker_of_interest, from_gate = 0, to_gate = 7, increment = 0.1, point_size=6)

#read in manual gates and scale
marker_gates = pd.read_csv('/standard/vol328/FallahiLab/Maize-Data/Leica-Thunder/Kimberly_Nguyen/20240624_HetMel/FA3-2/2025_analysis/marker_gates_v3.csv',
    header = [0], index_col =[0])

adata.uns['gates'] = marker_gates
adata = sm.pp.rescale(adata, gate=adata.uns['gates'])

#read in phentyping workflow csv
phenotype = pd.read_csv('phenotyping_workflow_8rounds_v4_combinations_exclude_n500.csv')

#execute phenotyping
phenotype.style.format(na_rep='')
adata= sm.tl.phenotype_cells (adata, phenotype=phenotype, label="phenotype") 
adata.obs['phenotype'].value_counts()


sm.pl.spatial_scatterPlot(adata, colorBy = ['phenotype'],figsize=(9,6), s=0.3, fontsize=5)

adata.write('8rounds_phenotyped_combinations.h5ad')

adata = ad.read('8rounds_phenotyped_named.h5ad')
sm.pl.markerCorrelation(adata)

#Spatial functions
adata = sm.tl.spatial_distance (adata, phenotype='phenotype')
adata = sm.tl.spatial_distance(adata, x_coordinate='X_centroid', y_coordinate='Y_centroid',
                         phenotype='phenotype', label='spatial_distance')
sm.pl.spatial_distance(adata, method='heatmap', phenotype='phenotype', imageid='sample_id')

sm.pl.spatial_distance(adata, method='numeric', distance_from='MITF+ SOX10+', phenotype='phenotype', log = True, plot_type='boxen')
sm.pl.spatial_distance(adata, phenotype='phenotype',figsize=(20,18),
                       fileName='FA3-2_spatialdistance.png',
                       saveDir='/standard/vol328/FallahiLab/Maize-Data/Leica-Thunder/Kimberly_Nguyen/20240624_HetMel/FA3-2/2025_analysis/')

sm.pl.spatial_distance(adata, phenotype='phenotype',figsize=(6,18),
                       fileName='FA3-2_spatialdistance_colcluster.pdf',
                       saveDir='/home/zqn7td/figures',
                       subset_col = 'phenotype',
                       subset_value = ['Fibroblast', 'Endothelial Cell', 'Immune Cell','Cytotoxic T Cell', 'Helper T Cell','T-Regulatory Cell','Other Immune Cell'],
                       heatmap_standard_scale = 1,
                       heatmap_col_cluster=True)
                       
sm.pl.spatial_distance (adata, method='numeric',distance_from='SOX10+ NGFR+ SOX9+', log=True, #subset_col = 'phenotype',
                       #subset_value = ['Fibroblast', 'Endothelial Cell', 'Immune Cell','Cytotoxic T Cell', 'Helper T Cell','T-Regulatory Cell','Other Immune Cell'],
                       height=10, aspect=3/4, fileName='FA3-2_spatialdistance_numeric_distancefromSOX10NGFRSOX9_log_2.png',
                       saveDir='/home/zqn7td/figures',)

sm.pl.spatial_distance (adata, method='numeric',distance_from='SOX10+ NGFR+', log=True, subset_col = 'phenotype',
                       #subset_value = ['Fibroblast'],#, 'Endothelial Cell', 'Immune Cell','Cytotoxic T Cell', 'Helper T Cell','T-Regulatory Cell','Other Immune Cell'],
                       height=10, aspect=3/4, )

###Spatial interactions
sm.tl.spatial_interaction(adata, x_coordinate='X_centroid', y_coordinate='Y_centroid', z_coordinate=None, phenotype='phenotype', method='radius', radius=30, knn=10, permutation=1000, imageid='imageid', subset=None, pval_method='zscore', verbose=True, label='spatial_interaction')
sm.tl.spatial_interaction(adata, x_coordinate='X_centroid', y_coordinate='Y_centroid', z_coordinate=None, phenotype='phenotype', method='radius', radius=15, knn=1, permutation=1000, imageid='imageid', subset=None, pval_method='zscore', verbose=True, label='spatial_interaction_15')

sm.pl.spatial_interaction(adata, summarize_plot=True, spatial_interaction ='spatial_interaction_15',p_val=0.01, cmap='coolwarm', nonsig_color='lightgrey',
                    binary_view=True, row_cluster=True, col_cluster=True, subset_phenotype = ['Fibroblast', 'Endothelial Cell', 'Immune Cell','Cytotoxic T Cell', 'Helper T Cell','T-Regulatory Cell','Other Immune Cell'],
                    subset_neighbour_phenotype =['MITF+ SOX10+', 'MITF+ NGFR+', 'MITF+ SOX9+', 'MITF+ PRAME+', 'SOX10+ NGFR+', 'SOX10+ SOX9+', 'SOX10+ PRAME+', 'NGFR+ SOX9+', 'MITF+ SOX10+ NGFR+', 'MITF+ SOX10+ SOX9+', 'MITF+ SOX10+ PRAME+', 'SOX10+ NGFR+ SOX9+',  'MITF+ SOX10+ NGFR+ SOX9+', 'MITF+ SOX10+ NGFR+ PRAME+', 'MITF+ SOX10+ SOX9+ PRAME+', 'MITF+', 'SOX10+', 'NGFR+', 'SOX9+', 'PRAME+', 'MITF+ SOX10+ NGFR+ SOX9+ PRAME+'],
                    fileName='FA3-2_spatialinteraction.png',
                    saveDir='/home/zqn7td/figures',)

sm.pl.spatialInteractionNetwork(adata, #subsetPhenotype=['Fibroblast', 'Endothelial Cell', 'Immune Cell','Cytotoxic T Cell', 'Helper T Cell','T-Regulatory Cell','Other Immune Cell'], 
                                fileName='TME_interaction.png', saveDir='/home/zqn7td/figures', figsize=(80,80), fontSize=20)
###Spatial clustering
adata = sm.tl.spatial_expression(adata, x_coordinate='X_centroid', y_coordinate='Y_centroid',
                           method='radius', radius=15, 
                           label='expression_radius_15')
adata = sm.tl.spatial_cluster(adata, df_name='expression_radius_30', method='kmeans', k=10, label='cluster_kmeans')
adata = sm.tl.spatial_cluster(adata, df_name='expression_radius_15', method='kmeans', k=10, label='cluster_kmeans_expression_radius_15')

adata = sm.tl.spatial_count(adata, x_coordinate='X_centroid', y_coordinate='Y_centroid',
                      phenotype='phenotype', method='radius', radius=15,
                      label='spatial_count_neighborhood_radius15')
adata = sm.tl.spatial_cluster(adata, df_name='spatial_count_neighborhood_radius15', method='kmeans', k=10, label='cluster_neighborhood_radius15')

cluster_plots(adata, group_by, subsample=100000, palette='viridis', use_raw=False, size=None, output_dir=None)
sm.pl.cluster_plots(adata, group_by='leiden', palette='plasma', subsample=50000)
adata = sm.tl.spatial_cluster(adata, df_name='spatial_count', method='kmeans', k=10, label='cluster_kmeans')


sm.pl.cluster_plots(adata, group_by='cluster_neighborhood_radius30', subsample=None, use_raw=False, output_dir='/home/zqn7td/figures')
####spatial pscore

adata = sm.tl.spatial_pscore(adata, proximity=['Neural crest-like', 'Immune Cell'],
                       method='radius', radius=30, label='proximity_score_all')
adata.obs['proximity_score_all']
adata.uns['proximity_score_all']
sm.pl.spatial_pscore(adata, label='proximity_score_all', plot_score='both', color='skyblue',
                     fileName='FA3-2_spatialpscore.png',
                    saveDir='/home/zqn7td/figures')


sm.pl.densityPlot2D(adata, markerA="cJUN")
############font

SMALL_SIZE = 20
MEDIUM_SIZE = 24
BIGGER_SIZE = 28

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]

sm.pl.heatmap(adata, groupBy='phenotype', standardScale='column', figsize=(5,3), showPrevalence=True, vmin=-1, vmax=1)

###umap font

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]

#############


for i,r in adata.obs.iterrows():
    if r['phenotype']=="likely-MITF+":
        adata.obs.loc[i,'phenotype'] = "MITF+"
for i,r in adata.obs.iterrows():
    if r['phenotype']=="likely-SOX10+":
        adata.obs.loc[i,'phenotype'] = "SOX10+"
for i,r in adata.obs.iterrows():
    if r['phenotype']=="likely-Immune":
        adata.obs.loc[i,'phenotype'] = "Immune"
for i,r in adata.obs.iterrows():
    if r['phenotype']=="likely-EC":
        adata.obs.loc[i,'phenotype'] = "EC"
for i,r in adata.obs.iterrows():
    if r['phenotype']=="likely-MITF+ SOX10+ NGFR+ ":
        adata.obs.loc[i,'phenotype'] = "MITF+ SOX10+ NGFR+ "
for i,r in adata.obs.iterrows():
    if r['phenotype']=="likely-Fibroblast":
        adata.obs.loc[i,'phenotype'] = "Fibroblast"
for i,r in adata.obs.iterrows():
    if r['phenotype']=="likely-SOX10+ NGFR+":
        adata.obs.loc[i,'phenotype'] = "SOX10+ NGFR+ "
for i,r in adata.obs.iterrows():
    if r['phenotype']=="likely-MITF+ SOX10+":
        adata.obs.loc[i,'phenotype'] = "MITF+ SOX10+"
for i,r in adata.obs.iterrows():
    if r['phenotype']=="likely-SOX10+ NGFR+":
        adata.obs.loc[i,'phenotype'] = "Unknown"
for i,r in adata.obs.iterrows():
    if r['phenotype']=="likely-MITF+":
        adata.obs.loc[i,'phenotype'] = "Unknown"
for i,r in adata.obs.iterrows():
    if r['phenotype']=="likely-Endothelial Cell":
        adata.obs.loc[i,'phenotype'] = "Endothelial Cell"
for i,r in adata.obs.iterrows():
    if r['phenotype']=="likely-Neural crest-like":
        adata.obs.loc[i,'phenotype'] = "Unknown"
for i,r in adata.obs.iterrows():
    if r['phenotype']=="likely-Melanocytic":
        adata.obs.loc[i,'phenotype'] = "Unknown"

for i,r in adata.obs.iterrows():
    if r['phenotype']=="likely-Transitory":
        adata.obs.loc[i,'phenotype'] = "Transitory"

for i,r in adata.obs.iterrows():
    if r['phenotype']=="Tumor cell":
        adata.obs.loc[i,'phenotype'] = "Unknown"


###pixel intensity distribution
sm.pl.distPlot(adata, layer='log', markers=['cFOS','cJUN','FRA1','FRA2','JUNB','JUND'], ncols=3, fontsize=6, figsize=(5,2))

sm.pl.distPlot(adata, layer=None, vline = 'auto', markers=['MITF', 'NGFR', 'cFOS', 'SOX10', 'cJUN', 'JUND', 'FRA1', 'FRA2',
       'FAP', 'JUNB', 'SOX9', 'CD45', 'CD31', 'CD4', 'CD3D', 'FOXP3',
       'PRAME'], ncols=4, fontsize=6, figsize=(5,5))


######          UMAP            #######

#perform umap analysis on data
adata = sm.tl.umap(adata)

#color plot
palettte = {
    "Melanocytic":"#e6194B",
    "Transitory": "#4363d8",
    "Neural crest-like":"#3cb44b",
    "Fibroblast":"#ffe119",
    "Immune Cell":"#42d4f4",
    "Endothelial Cell":"#9A6324",
    "Unknown":"#a9a9a9"
}


####plot UMAP 

sm.pl.umap(adata, color =["phenotype"],figsize=[8,5],tight_layout=True), #palette = palettte)


#############HEATMAPS##################

######Heatmap setup for all combinations of differentiation state markers


diff_markers = ['MITF','SOX10','NGFR','SOX9','PRAME'] 
all_combinations = []

pairwise_combinations = list(itertools.combinations(diff_markers, 2))
print("Pairwise combinations:")
for combo in pairwise_combinations:
    m = " ".join([diff_markers + "+" for diff_markers in combo])
    print(m)
    all_combinations.append(m)

threeway_combinations = list(itertools.combinations(diff_markers, 3))
print("Three-way combinations:")
for combo in threeway_combinations:
    m = " ".join([diff_markers + "+" for diff_markers in combo])
    print(m)
    all_combinations.append(m)

fourway_combinations = list(itertools.combinations(diff_markers, 4))
print("Four-way combinations:")
for combo in fourway_combinations:
    m = " ".join([diff_markers + "+" for diff_markers in combo])
    print(m)
    all_combinations.append(m)


one_combinations = list(itertools.combinations(diff_markers,1))
for combo in one_combinations:
    m = " ".join([diff_markers + "+" for diff_markers in combo])
    print(m)
    all_combinations.append(m)

all_combinations.append("MITF+ SOX10+ NGFR+ SOX9+ PRAME+")

all_combinations = ['MITF+ SOX10+', 'MITF+ NGFR+', 'MITF+ SOX9+', 'MITF+ PRAME+', 'SOX10+ NGFR+', 'SOX10+ SOX9+', 'SOX10+ PRAME+', 'NGFR+ SOX9+', 'NGFR+ PRAME+', 'SOX9+ PRAME+', 'MITF+ SOX10+ NGFR+', 'MITF+ SOX10+ SOX9+', 'MITF+ SOX10+ PRAME+', 'MITF+ NGFR+ SOX9+', 'MITF+ NGFR+ PRAME+', 'MITF+ SOX9+ PRAME+', 'SOX10+ NGFR+ SOX9+', 'SOX10+ NGFR+ PRAME+', 'SOX10+ SOX9+ PRAME+', 'NGFR+ SOX9+ PRAME+', 'MITF+ SOX10+ NGFR+ SOX9+', 'MITF+ SOX10+ NGFR+ PRAME+', 'MITF+ SOX10+ SOX9+ PRAME+', 'MITF+ NGFR+ SOX9+ PRAME+', 'SOX10+ NGFR+ SOX9+ PRAME+', 'MITF+', 'SOX10+', 'NGFR+', 'SOX9+', 'PRAME+', 'MITF+ SOX10+ NGFR+ SOX9+ PRAME+']
combinations_above_n500 = ['MITF+ SOX10+', 'MITF+ NGFR+', 'MITF+ SOX9+', 'MITF+ PRAME+', 'SOX10+ NGFR+', 'SOX10+ SOX9+', 'SOX10+ PRAME+', 'NGFR+ SOX9+', 'MITF+ SOX10+ NGFR+', 'MITF+ SOX10+ SOX9+', 'MITF+ SOX10+ PRAME+', 'SOX10+ NGFR+ SOX9+',  'MITF+ SOX10+ NGFR+ SOX9+', 'MITF+ SOX10+ NGFR+ PRAME+', 'MITF+ SOX10+ SOX9+ PRAME+', 'MITF+', 'SOX10+', 'NGFR+', 'SOX9+', 'PRAME+', 'MITF+ SOX10+ NGFR+ SOX9+ PRAME+']

sm.pl.heatmap(adata, groupBy='phenotype', standardScale='column', figsize=(8,5), 
            subsetMarkers=subsetMarkers1, subsetGroups = combinations_above_n500, 
            showPrevalence=True, vmin=-1, vmax=1, clusterColumns=True, clusterRows=True,
            saveDir='/standard/vol328/FallahiLab/Maize-Data/Leica-Thunder/Kimberly_Nguyen/20240624_HetMel/FA3-2/2025_analysis',
            fileName='FA3-2_heatmap_combinations_above_n500_6-AP-1.pdf')  


#Named Diff State / AP-1 heat map
subsetMarkers1 = ['cFOS', 'FRA1', 'FRA2', 'cJUN', 'JUNB', 'JUND']
subsetGroups1= ["Melanocytic","Transitory","Neural crest-like"]
sm.pl.heatmap(adata, groupBy='phenotype', standardScale='column', figsize=(8,5), subsetMarkers=subsetMarkers1, subsetGroups = subsetGroups1, showPrevalence=True, vmin=-1, vmax=1, clusterColumns=False, clusterRows=False,  
              orderColumn=["cFOS","FRA1","FRA2","cJUN","JUNB","JUND"], 
              orderRow=["Melanocytic","Transitory","Neural crest-like"])

#new
subsetMarkers1 = ['PRAME','MITF','SOX10','NGFR','SOX9','cFOS', 'FRA1', 'FRA2', 'cJUN', 'JUNB', 'JUND']
subsetGroups1 = ["MITF+ SOX10+","MITF+ SOX10+ SOX9+","MITF+ SOX10+ NGFR+","SOX10+ NGFR+","SOX10+ NGFR+ SOX9+","SOX9+","MITF+ SOX9+","SOX9+ SOX10+","NGFR+ SOX9+"]


##all combinations
subsetMarkers1 = ['cFOS', 'FRA1', 'FRA2', 'cJUN', 'JUNB', 'JUND']
sm.pl.heatmap(adata, groupBy='phenotype', standardScale='column', figsize=(8,5), subsetMarkers=subsetMarkers1, subsetGroups = all_combinations, showPrevalence=True, vmin=-1, vmax=1, clusterColumns=False, clusterRows=False)  
#ordered rows

subsetMarkers1 = ["PRAME","MITF","SOX10","NGFR","SOX9",'cFOS', 'FRA1', 'FRA2', 'cJUN', 'JUNB', 'JUND']
sm.pl.heatmap(adata, groupBy='phenotype', standardScale='column', figsize=(8,5), subsetMarkers=subsetMarkers1, subsetGroups = subsetGroups1, showPrevalence=True, vmin=-1, vmax=1, clusterColumns=False, clusterRows=False,  
              orderColumn=["PRAME","MITF","SOX10","NGFR","SOX9","cFOS","FRA1","FRA2","cJUN","JUNB","JUND"], 
              orderRow=["MITF+ SOX10+","MITF+ SOX10+ SOX9+","MITF+ SOX10+ NGFR+","SOX10+ NGFR+","SOX10+ NGFR+ SOX9+","SOX9+","MITF+ SOX9+","SOX9+ SOX10+","NGFR+ SOX9+"])


subsetGroups1= ["MITF+ SOX10+","MITF+ SOX10+ SOX9+","MITF+ SOX10+ NGFR+","SOX10+ NGFR+","SOX10+ NGFR+ SOX9+","SOX9+","MITF+ SOX9+","SOX9+ SOX10+","NGFR+ SOX9+"]
sm.pl.heatmap(adata, groupBy='phenotype', standardScale='column', figsize=(8,5), subsetMarkers=subsetMarkers1, subsetGroups = subsetGroups1, showPrevalence=True, vmin=-1, vmax=1, clusterColumns=False, clusterRows=False,  
              orderColumn=["PRAME","MITF","SOX10","NGFR","SOX9","cFOS","FRA1","FRA2","cJUN","JUNB","JUND"], 
              orderRow=["MITF+ SOX10+","MITF+ SOX10+ SOX9+","MITF+ SOX10+ NGFR+","SOX10+ NGFR+","SOX10+ NGFR+ SOX9+","SOX9+","MITF+ SOX9+","SOX9+ SOX10+","NGFR+ SOX9+"])

subsetMarkers1 = ["PRAME",'cFOS', 'FRA1', 'FRA2', 'cJUN', 'JUNB', 'JUND']
sm.pl.heatmap(adata, groupBy='phenotype', standardScale='column', figsize=(8,5), subsetMarkers=subsetMarkers1, subsetGroups = subsetGroups1, showPrevalence=True, vmin=-1, vmax=1, clusterColumns=False, clusterRows=False,  
              orderColumn=["PRAME","cFOS","FRA1","FRA2","cJUN","JUNB","JUND"], 
              orderRow=["MITF+ SOX10+","MITF+ SOX10+ SOX9+","MITF+ SOX10+ NGFR+","SOX10+ NGFR+","SOX10+ NGFR+ SOX9+","SOX9+","MITF+ SOX9+","SOX9+ SOX10+","NGFR+ SOX9+"])

sm.pl.heatmap(adata, groupBy='phenotype', standardScale='column', figsize=(8,5), subsetMarkers=subsetMarkers1, subsetGroups = subsetGroups1, showPrevalence=True, vmin=-1, vmax=1)

#AP-1 markers only
sm.pl.heatmap(adata, groupBy='phenotype', standardScale='column', figsize=(8,5), subsetMarkers=subsetMarkers1, subsetGroups = subsetGroups1, showPrevalence=True, vmin=-1, vmax=1, clusterColumns=False, clusterRows=False,  
              orderColumn=["PRAME","cFOS","FRA1","FRA2","cJUN","JUNB","JUND"], 
              orderRow=["MITF+ SOX10+","MITF+ SOX10+ SOX9+","MITF+ SOX10+ NGFR+","SOX10+ NGFR+","SOX10+ NGFR+ SOX9+","SOX9+","MITF+ SOX9+","SOX9+ SOX10+","NGFR+ SOX9+"])


sm.pl.spatial_scatterPlot(adata, colorBy = ['phenotype'],figsize=(9,6), s=0.3, fontsize=5,fileName='ScatterPlot_phenotype.png',customColors = palettte,saveDir="/Volumes/FallahiLab/Maize-Data/Leica-Thunder/Kimberly_Nguyen/20240624_HetMel/imageanalysis/figures/FA2-2",dpi=300)
sm.pl.spatial_scatterPlot(adata, colorBy = ['phenotype'],figsize=(9,6), s=0.3, fontsize=5,customColors = palettte)

sm.pl.pie(adata, phenotype='phenotype', title=["Nodule 2 Phenotype Composition"],legend = True)

##############FIGURES####################

##font

SMALL_SIZE = 28
MEDIUM_SIZE = 32
BIGGER_SIZE = 36

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]

###phenotype heatmap

#6 AP-1
subsetMarkers1 = ['cFOS', 'FRA1', 'FRA2', 'cJUN', 'JUNB', 'JUND']
all_combinations = ['MITF+ SOX10+', 'MITF+ NGFR+', 'MITF+ SOX9+', 'MITF+ PRAME+', 'SOX10+ NGFR+', 'SOX10+ SOX9+', 'SOX10+ PRAME+', 'NGFR+ SOX9+', 'NGFR+ PRAME+', 'SOX9+ PRAME+', 'MITF+ SOX10+ NGFR+', 'MITF+ SOX10+ SOX9+', 'MITF+ SOX10+ PRAME+', 'MITF+ NGFR+ SOX9+', 'MITF+ NGFR+ PRAME+', 'MITF+ SOX9+ PRAME+', 'SOX10+ NGFR+ SOX9+', 'SOX10+ NGFR+ PRAME+', 'SOX10+ SOX9+ PRAME+', 'NGFR+ SOX9+ PRAME+', 'MITF+ SOX10+ NGFR+ SOX9+', 'MITF+ SOX10+ NGFR+ PRAME+', 'MITF+ SOX10+ SOX9+ PRAME+', 'MITF+ NGFR+ SOX9+ PRAME+', 'SOX10+ NGFR+ SOX9+ PRAME+', 'MITF+', 'SOX10+', 'NGFR+', 'SOX9+', 'PRAME+', 'MITF+ SOX10+ NGFR+ SOX9+ PRAME+']
combinations_above_n500 = ['MITF+ SOX10+', 'MITF+ NGFR+', 'MITF+ SOX9+', 'MITF+ PRAME+', 'SOX10+ NGFR+', 'SOX10+ SOX9+', 'SOX10+ PRAME+', 'NGFR+ SOX9+', 'MITF+ SOX10+ NGFR+', 'MITF+ SOX10+ SOX9+', 'MITF+ SOX10+ PRAME+', 'SOX10+ NGFR+ SOX9+',  'MITF+ SOX10+ NGFR+ SOX9+', 'MITF+ SOX10+ NGFR+ PRAME+', 'MITF+ SOX10+ SOX9+ PRAME+', 'MITF+', 'SOX10+', 'NGFR+', 'SOX9+', 'PRAME+', 'MITF+ SOX10+ NGFR+ SOX9+ PRAME+']

heatmap=sm.pl.heatmap(adata, groupBy='phenotype', standardScale='column', figsize=(20,18), 
            subsetMarkers=subsetMarkers1, subsetGroups = combinations_above_n500, 
            showPrevalence=True, vmin=-1, vmax=1, clusterColumns=True, clusterRows=True)
            #saveDir='/standard/vol328/FallahiLab/Maize-Data/Leica-Thunder/Kimberly_Nguyen/20240624_HetMel/FA3-2/2025_analysis/figures',
            #fileName='FA3-2_heatmap_combinations_above_n500_6-AP-1.pdf')  

# Generate the heatmap plot
sm.pl.heatmap(adata, groupBy='phenotype', standardScale='column', figsize=(20,18), 
              subsetMarkers=subsetMarkers1, subsetGroups=combinations_above_n500, 
              showPrevalence=True, vmin=-1, vmax=1, clusterColumns=True, clusterRows=True)

# Access the current plot
ax = plt.gca()

# Extract the y-axis labels from the plot
x_axis_labels = ax.get_xticklabels()
phenotype_order = [label.get_text() for label in x_axis_labels]

# Create a DataFrame with the phenotype order
phenotype_order_df = pd.DataFrame(phenotype_order, columns=['Phenotype'])

# Print the DataFrame
print(phenotype_order_df)


#Exclude FRA1
subsetMarkers1 = ['cFOS', 'FRA2', 'cJUN', 'JUNB', 'JUND']
sm.pl.heatmap(adata, groupBy='phenotype', standardScale='column', figsize=(20,18), 
            subsetMarkers=subsetMarkers1, subsetGroups = combinations_above_n500, 
            showPrevalence=True, vmin=-1, vmax=1, clusterColumns=True, clusterRows=True,
            saveDir='/standard/vol328/FallahiLab/Maize-Data/Leica-Thunder/Kimberly_Nguyen/20240624_HetMel/FA3-2/2025_analysis/figures',
            fileName='FA3-2_heatmap_combinations_above_n500_5-AP-1.pdf')  






##########################################

log1p(adata, layer='log', verbose=True)
sp.pp.scale(adata_zscored)

feature_table_path=["/Users/zqn7td/Desktop/run_mcmicro/feature_extraction/nodule_channel13_8_cell_nodule2_hoechst_cp_masks.csv"]
adata = sm.pp.mcmicro_to_scimap(feature_table_path,drop_markers=['Hoechst_round1','AF488_CD31','AF647_FAP','AF488_CD45'])

subsetMarkers = ['cFOS', 'FRA1', 'FRA2', 'cJUN', 'JUNB', 'JUND']
subsetGroups = ['Melanocytic','Transitory','Neural crest-like']
sm.pl.heatmap(adata, groupBy='phenotype', layer = None,  standardScale='column', subsetMarkers=subsetMarkers, subsetGroups = subsetGroups, figsize=(10,6), showPrevalence=True, vmin=-1, vmax=1, clusterColumns=False, clusterRows=False,
              orderColumn=['cFOS', 'FRA1', 'FRA2', 'cJUN', 'JUNB', 'JUND'],
              orderRow=["Melanocytic","Transitory","Neural crest-like"])

adata.var['FOS/JUN'] = adata.var.index['cFOS']/adata.var.index['cJUN']

subsetMarkers = ['MITF','SOX10','NGFR', 'FAP']
sm.pl.heatmap(adata, groupBy='phenotype', standardScale=None, subsetMarkers=subsetMarkers, figsize=(10,6), showPrevalence=True, vmin=0, vmax=1, clusterColumns=False, clusterRows=False,
              orderColumn=['MITF','SOX10','NGFR', 'FAP'],
              orderRow=["Melanocytic","Transitory","Neural crest-like", "Fibroblast", "Unknown"])


phenotype = pd.read_csv('/Users/zqn7td/Desktop/run_mcmicro/phenotyping_workflow_6_AP1only.csv')
phenotype.style.format(na_rep='')
adata = sm.tl.phenotype_cells (adata, phenotype=phenotype, label="phenotype") 
adata.obs['phenotype'].value_counts()

ax1 = df.plot.scatter (x='MITF',y='SOX10',c='cFOS')

adata.write('/Volumes/FallahiLab/Maize-Data/Leica-Thunder/Kimberly_Nguyen/20240624_HetMel/imageanalysis/analysis3/analysis3_FA2-2_6_noAP1.h5ad')
adata = ad.read_h5ad('/Users/zqn7td/Desktop/run_mcmicro/pilotanalysis_6_AP1only.h5ad')

df= ad.AnnData.to_df(adata)

#remove AF647_
df.rename(columns=lambda x: x.split('_')[1], inplace=True)



df_subsetSOX10 = df.loc[df['SOX10'] >= 0.1]

df_subsetSOX102_sampled = df_subsetSOX102.sample(25000)

ax1 = df_subsetSOX10_sampled.plot.scatter(x='MITF',y='FOSoverJUN',c='SOX10',figsize=(7.5,6))
plt.ylim(0, 1)
plt.xlim(0,1)

df_subsetSOX102_sampled['FOS/JUN'] = df['cFOS']/df['cJUN']
df_subsetSOX102_sampled['NGFR/MITF'] = df['NGFR']/df['MITF']

df_no_outlier = df_subsetSOX10_sampled.loc[df_subsetSOX10_sampled['FOSoverJUN'] <= 5]
ax1 = df_no_outlier.plot.scatter(x='NGFR',y='FOSoverJUN',c='SOX10'),figsize=(7.5,6)
plt.tight_layout()
ax1 = df_no_outlier.plot.scatter(x='MITF',y='FOSoverJUN',c='NGFR',figsize=(7.5,6))

df_subsetSOX102_sampled = df_subsetSOX102_sampled.rename(columns={'FOS/JUN': 'FOSoverJUN', 'NGFR/MITF': 'NGFRoverMITF'})

ax1 = df_no_outlier.plot.scatter(x='cJUN',y='cFOS',c='MITF',figsize=(7.5,6),ylim=(0,0.8),xlim=(0,1))

ax1 = df_subsetSOX10_sampled.plot.scatter(x='MITF',y='FOSoverJUN',figsize=(7.5,6),ylim=(0,0.8),xlim=(0,0.8))

ax1 = df_subsetSOX10_sampled.plot.scatter(x='JUND',y='cFOS',c='SOX10') #,figsize=(7.5,6),xlim=(0,0.8))

ax1 = df_subsetSOX10_sampled.plot.scatter(x='MITF',y='SOX10',c='NGFR',figsize=(7.5,6))

ax1 = df_subsetSOX10_sampled.plot.scatter(x='MITF',y='NGFR',c='cJUN',figsize=(7.5,6))

plt.show()


plt.savefig("/Users/zqn7td/Downloads/fosoverjun_mitf_ngfr.pdf",dpi=300)

df_raw_sampled_subset = df_raw_sampled.loc[(df_raw_sampled['CD45'] <= 0.5) & (df_raw_sampled['FAP'] <=0.5) & (df_raw_sampled['CD31']<=0.5 ) ]
df_nophenotype = df_raw_sampled_subset 
ax2 = df_nophenotype.plot.scatter(x='cFOS',y='JUND',c='cFOS')


df_no_outlier['FOSoverJUN'].corr(df_no_outlier['MITF'])

df_no_outlier.corr()


import numpy as np
rho= df_nophenotype.corr()
pval = df_no_outlier.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(*rho.shape)
p = pval.applymap(lambda x: ''.join(['*' for t in [.05, .01, .001] if x<=t]))
import stats
rho= df_no_outlier.corr()
pval = df_no_outlier.corr(method=lambda x, y: stats.pearsonr(x, y)[1]) - np.eye(*rho.shape)
p = pval.applymap(lambda x: ''.join(['*' for t in [.05, .01, .001] if x<=t]))

# %%
 
# Define the markers and their corresponding values
markers = ['MITF', 'SOX10', 'PRAME' 'NGFR', 'SOX9']
marker_values = {marker: i + 1 for i, marker in enumerate(markers)}

# Initialize a 2151 matrix with zeros
matrix = np.zeros((21, 5), dtype=int)

# Populate the matrix based on the presence of markers in each combination
for col, combination in enumerate(combinations_above_n500):
    for marker, value in marker_values.items():
        if marker in combination:
            matrix[value - 1, col] = value

print(matrix)

############
markers = ['MITF', 'SOX10', 'PRAME', 'NGFR', 'SOX9']

# Create a new list to store the updated combinations
combinations_with_negatives = []

# Iterate through each combination and add missing markers with a negative sign
for combination in combinations_above_n500:
    positive_markers = combination.split()
    negative_markers = [f"{marker}-" for marker in markers if marker not in combination]
    updated_combination = ' '.join(positive_markers + negative_markers)
    combinations_with_negatives.append(updated_combination)

# Print the updated combinations
for combo in combinations_with_negatives:
    print(combo)