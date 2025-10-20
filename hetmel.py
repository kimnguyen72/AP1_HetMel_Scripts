import anndata as ad
import scimap as sm
import copy
import os
import importlib
import re

import scanpy as sp
import pandas as pd
import numpy as np
import napari as napari
import math
import seaborn as sns
import itertools
import umap as um

from scipy import stats
from scipy.stats import zscore
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import colors
from matplotlib.colors import LogNorm
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
from matplotlib.colors import Normalize

#####################################################################

def replace_phenotype(adata, old_value, new_value):
    for i,r in adata.obs.iterrows():
        if r['phenotype']==old_value:
            adata.obs.loc[i,'phenotype'] = new_value
            
    return adata

#####################################################################

def categorize_phenotype(phenotype):
    TME_phenotypes= ['Immune cell', 'Fibroblast', 'Endothelial cell','Cytotoxic T cell', 'Keratinocyte',
                     'Helper T cell','T-Regulatory cell','B cell','Macrophage','Pro-tumor macrophage','Inflammatory macrophage']
    if any(marker in phenotype for marker in ['MITF', 'SOX10', 'PRAME']):
        return 'Tumor'
    elif any(cell_type in phenotype for cell_type in TME_phenotypes):
        return 'TME'
    else:
        return 'Other'
    
#####################################################################

def threshold_phenotype_by_prevalence(adata, threshold= 0.01, verbose=True,return_adata=False):
    counts = adata.obs['phenotype'].value_counts()
    count_df = counts.to_frame()  
    total_count = count_df['count'].sum()
    subset_count_df = count_df[count_df['count'] > total_count * threshold]
    phenotypes_above_prevalence_threshold = subset_count_df.index.tolist()
    phenotypes_above_prevalence_threshold 
    if verbose == True:
        print(subset_count_df)
        print("total count:" + str(total_count))
    if return_adata == True:
        bdata = adata[adata.obs['phenotype'].isin(phenotypes_above_prevalence_threshold)]
        return bdata
    else:
        return phenotypes_above_prevalence_threshold
    
#####################################################################

def assign_clusters_from_dendogram(adata, phenotype_cluster_dict, label='cluster'):
    adata.obs[label] = adata.obs['phenotype'].map(phenotype_cluster_dict)
    adata.obs[label] = adata.obs[label].astype('category')
    for i in phenotype_cluster_dict:
        print(f"Phenotype: {i}, Cluster: {phenotype_cluster_dict[i]}")
    print(adata.obs[label].value_counts())
    return adata

#####################################################################

def distPlot(
    adata,
    layer=None,
    markers=None,
    subset=None,
    imageid='imageid',
    vline=None,
    vlinewidth=None,
    plotGrid=True,
    ncols=None,
    color=None,
    xticks=None,
    figsize=(5, 5),
    fontsize=None,
    dpi=200,
    saveDir=None,
    fileName='scimapDistPlot.png',
    scale_x1=None,
    scale_x2=None,
):
    
    # subset data if neede
    if subset is not None:
        if isinstance(subset, str):
            subset = [subset]
        if layer == 'raw':
            bdata = adata.copy()
            bdata.X = adata.raw.X
            bdata = bdata[bdata.obs[imageid].isin(subset)]
        else:
            bdata = adata.copy()
            bdata = bdata[bdata.obs[imageid].isin(subset)]
    else:
        bdata = adata.copy()

    # isolate the data
    if layer is None:
        data = pd.DataFrame(bdata.X, index=bdata.obs.index, columns=bdata.var.index)
    elif layer == 'raw':
        data = pd.DataFrame(bdata.raw.X, index=bdata.obs.index, columns=bdata.var.index)
    else:
        data = pd.DataFrame(
            bdata.layers[layer], index=bdata.obs.index, columns=bdata.var.index
        )

    # keep only columns that are required
    if markers is not None:
        if isinstance(markers, str):
            markers = [markers]
        # subset the list
        data = data[markers]

    # auto identify rows and columns in the grid plot
    def calculate_grid_dimensions(num_items, num_columns=None):
        """
        Calculates the number of rows and columns for a square grid
        based on the number of items.
        """
        if num_columns is None:
            num_rows_columns = int(math.ceil(math.sqrt(num_items)))
            return num_rows_columns, num_rows_columns
        else:
            num_rows = int(math.ceil(num_items / num_columns))
            return num_rows, num_columns

    if plotGrid is False:
        # Create a figure and axis object
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        # Loop through each column in the DataFrame and plot a KDE with the
        # user-defined color or the default color (grey)
        if color is None:
            for column in data.columns:
                data[column].plot.kde(ax=ax, label=column)
        else:
            for column in data.columns:
                c = color.get(column, 'grey')
                data[column].plot.kde(ax=ax, label=column, color=c)
        ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=fontsize)
        ax.tick_params(axis='both', which='major', width=1, labelsize=fontsize)
        plt.tight_layout()
        if xticks is not None:
            ax.set_xticks(xticks)
            ax.set_xticklabels([str(x) for x in xticks])

        if vline == 'auto':
            ax.axvline((data[column].max() + data[column].min()) / 2, color='black')
        elif vline is None:
            pass
        else:
            ax.axvline(vline, color='black')
            
        if scale_x:
            ax.set_xlim(data.min().min(), data.max().max())

        # save figure
        if outputDir is not None:
            plt.savefig(pathlib.Path(outputDir) / outputFileName)

    else:
        # calculate the number of rows and columns
        num_rows, num_cols = calculate_grid_dimensions(
            len(data.columns), num_columns=ncols
        )

        # set colors
        if color is None:
            # Define a color cycle of 10 colors
            color_cycle = itertools.cycle(
                plt.rcParams['axes.prop_cycle'].by_key()['color']
            )
            # Assign a different color to each column
            color = {col: next(color_cycle) for col in data.columns}

        # Set the size of the figure
        fig, axes = plt.subplots(
            nrows=num_rows, ncols=num_cols, figsize=figsize, dpi=dpi
        )
        axes = np.atleast_2d(axes)
        # Set the spacing between subplots
        # fig.subplots_adjust(bottom=0.1, hspace=0.1)

        # Loop through each column in the DataFrame and plot a KDE with the
        # user-defined color or the default color (grey) in the corresponding subplot
        for i, column in enumerate(data.columns):
            c = color.get(column, 'grey')
            row_idx = i // num_cols
            col_idx = i % num_cols
            data[column].plot.kde(ax=axes[row_idx, col_idx], label=column, color=c)
            axes[row_idx, col_idx].set_title(column)
            axes[row_idx, col_idx].tick_params(
                axis='both', which='major', width=1, labelsize=fontsize
            )
            axes[row_idx, col_idx].set_ylabel('')

            if vline == 'auto':
                axes[row_idx, col_idx].axvline(
                    (data[column].max() + data[column].min()) / 2, color='gray', dashes=[2, 2], linewidth=1
                )
                if vlinewidth is not None:
                    axes[row_idx, col_idx].axvline(
                        (data[column].max() + data[column].min()) / 2,color='gray', dashes=[2, 2], linewidth=vlinewidth)
            elif vline is None:
                pass
            else:
                axes[row_idx, col_idx].axvline(vline, color='black')

            if xticks is not None:
                axes[row_idx, col_idx].set_xticks(xticks)
                axes[row_idx, col_idx].set_xticklabels([str(x) for x in xticks])
                
            if scale_x1 is not None and scale_x2 is not None:
                axes[row_idx, col_idx].set_xlim(scale_x1, scale_x2)
            else:    
                axes[row_idx, col_idx].set_xlim(data[column].min(), data[column].max())
                

        # Remove any empty subplots
        num_plots = len(data.columns)
        for i in range(num_plots, num_rows * num_cols):
            row_idx = i // num_cols
            col_idx = i % num_cols
            fig.delaxes(axes[row_idx, col_idx])

        # Set font size for tick labels on both axes
        plt.tick_params(axis='both', labelsize=fontsize)
        plt.tight_layout()

        # Save the figure to a file
        if saveDir:
            if not os.path.exists(saveDir):
                os.makedirs(saveDir)
            full_path = os.path.join(saveDir, fileName)
            plt.savefig(full_path, dpi=300)
            plt.close()
            print(f"Saved heatmap to {full_path}")
        else:
            plt.show()
            

#####################################################################

def heatmap(
    adata,
    groupBy,
    layer=None,
    subsetMarkers=None,
    subsetGroups=None,
    clusterRows=True,
    clusterColumns=True,
    standardScale=None,
    orderRow=None,
    orderColumn=None,
    showPrevalence=False,
    cmap='vlag',
    figsize=None,
    saveDir=None,
    fileName=None,
    verbose=True,
    scale_title=None,
    **kwargs,
):

    # load adata
    if isinstance(adata, str):
        adata = ad.read_h5ad(adata)

    # check if the location is provided if the user wishes to save the image
    if (saveDir is None and fileName is not None) or (
        saveDir is not None and fileName is None
    ):
        raise ValueError(
            "Both 'saveDir' and 'fileName' must be provided together or not at all."
        )

    # subset data if user requests
    subsetadata = None  # intialize subsetted data
    if subsetGroups:
        subsetGroups = (
            [subsetGroups] if isinstance(subsetGroups, str) else subsetGroups
        )  # convert to list
        subsetadata = adata[adata.obs[groupBy].isin(subsetGroups)]
        # also identify the categories to be plotted
        categories = subsetadata.obs[groupBy].values
    else:
        # also identify the categories to be plotted
        categories = adata.obs[groupBy].values

    # subset the markers if user requests
    if subsetMarkers:
        subsetMarkers = (
            [subsetMarkers] if isinstance(subsetMarkers, str) else subsetMarkers
        )  # convert to list
        if subsetadata:
            # isolate the data
            if layer == 'raw':
                data = subsetadata[:, subsetMarkers].raw.X
            elif layer is None:
                data = subsetadata[:, subsetMarkers].X
            else:
                data = subsetadata[:, subsetMarkers].layers[layer]
        else:
            # isolate the data
            if layer == 'raw':
                data = adata[:, subsetMarkers].raw.X
            elif layer is None:
                data = adata[:, subsetMarkers].X
            else:
                data = adata[:, subsetMarkers].layers[layer]
    else:
        # take the whole data if the user does not subset anything
        if layer == 'raw':
            data = adata.raw.X
        elif layer is None:
            data = adata.X
        else:
            data = adata.layers[layer]

    # intialize the markers to be plotted
    if subsetMarkers is None:
        subsetMarkers = adata.var.index.tolist()

    # The actual plotting function
    def plot_category_heatmap_vectorized(
        data,
        marker_names,
        categories,
        clusterRows,
        clusterColumns,
        standardScale,
        orderRow,
        orderColumn,
        showPrevalence,
        cmap,
        figsize,
        saveDir,
        fileName,
        **kwargs,
    ):
        # Validate clustering and ordering options
        if (clusterRows or clusterColumns) and (
            orderRow is not None or orderColumn is not None
        ):
            raise ValueError(
                "Cannot use clustering and manual ordering together. Please choose one or the other."
            )

        if standardScale not in [None, 'row', 'column']:
            raise ValueError("standardScale must be 'row', 'column', or None.")

        # Convert marker_names to list if it's a pandas Index
        # if isinstance(marker_names, pd.Index):
        #    marker_names = marker_names.tolist()

        # Data preprocessing
        sorted_indices = np.argsort(categories)
        data = data[sorted_indices, :]
        categories = categories[sorted_indices]
        unique_categories, category_counts = np.unique(categories, return_counts=True)

        # Compute mean values for each category
        mean_data = np.array(
            [
                np.mean(data[categories == category, :], axis=0)
                for category in unique_categories
            ]
        )

        # Apply standard scaling if specified
        if standardScale == 'row':
            scaler = StandardScaler()
            mean_data = scaler.fit_transform(mean_data)
        elif standardScale == 'column':
            scaler = StandardScaler()
            mean_data = scaler.fit_transform(mean_data.T).T
            
    
       
        # Apply manual ordering if specified
        if orderRow:
            # Ensure orderRow is a list
            if isinstance(orderRow, pd.Index):
                orderRow = orderRow.tolist()
            row_order = [unique_categories.tolist().index(r) for r in orderRow]
            mean_data = mean_data[row_order, :]
            unique_categories = [unique_categories[i] for i in row_order]
            category_counts = [category_counts[i] for i in row_order]

        if orderColumn:
            # Ensure orderColumn is a list0000
            if isinstance(orderColumn, pd.Index):
                orderColumn = orderColumn.tolist()
            col_order = [marker_names.index(c) for c in orderColumn]
            mean_data = mean_data[:, col_order]
            marker_names = [marker_names[i] for i in col_order]

            # Clustering
        if clusterRows:
            # Perform hierarchical clustering
            row_linkage = linkage(pdist(mean_data), method='average')
            # Reorder data according to the clustering
            row_order = dendrogram(row_linkage, no_plot=True)['leaves']
            mean_data = mean_data[row_order, :]
            unique_categories = unique_categories[row_order]
            category_counts = category_counts[row_order]

        if clusterColumns:
            # Perform hierarchical clustering
            col_linkage = linkage(pdist(mean_data.T), method='average')
            # Reorder data according to the clustering
            col_order = dendrogram(col_linkage, no_plot=True)['leaves']
            mean_data = mean_data[:, col_order]
            marker_names = [marker_names[i] for i in col_order]

        # Plotting
        # Dynamic figsize calculation
        if figsize is None:
            base_size = 0.5  # Base size for each cell in inches
            figsize_width = max(10, len(marker_names) * base_size)
            figsize_height = max(8, len(unique_categories) * base_size)
            figsize = (figsize_width, figsize_height)

        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)



        # Heatmap
        # Extract vmin and vmax from kwargs if present, else default to min and max of mean_data
        vmin = kwargs.pop('vmin', np.min(mean_data))
        vmax = kwargs.pop('vmax', np.max(mean_data))

        # Create the Normalize instance with vmin and vmax
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

        c = ax.imshow(mean_data, aspect='auto', cmap=cmap, norm=norm, **kwargs)

        # Prevalence text
        if showPrevalence:
            # Calculate text offset from the last column of the heatmap
            text_offset = (
                mean_data.shape[1] * 0.001
            )  # Small offset from the right edge of the heatmap
            total_cells = sum(category_counts)

            for index, count in enumerate(category_counts):
                percentage = (count / total_cells) * 100
                # Position text immediately to the right of the heatmap
                ax.text(
                    mean_data.shape[1] + text_offset,
                    index,
                    f"n={count} ({percentage:.1f}%)",
                    va='center',
                    ha='left',
                )

        # Setting the tick labels
        ax.set_xticks(np.arange(mean_data.shape[1]))
        ax.set_xticklabels(marker_names, rotation=90, ha="right")
        ax.set_yticks(np.arange(mean_data.shape[0]))
        ax.set_yticklabels(unique_categories)

        # Move the colorbar to the top left corner
        # cbar_ax = fig.add_axes([0.125, 0.92, 0.2, 0.02])  # x, y, width, height
        cbar_ax = ax.inset_axes([-0.5, -1.5, 4, 0.5], transform=ax.transData)
        cbar = plt.colorbar(c, cax=cbar_ax, orientation='horizontal')
        cbar_ax.xaxis.set_ticks_position('top')
        cbar_ax.xaxis.set_label_position('top')
        if scale_title:
            cbar.set_label(scale_title)
        else:
            cbar.set_label('Mean expression in group')

        ax.set_xlabel('Markers')
        ax.set_ylabel('Categories')

        # plt.tight_layout(rect=[0, 0, 0.9, 0.9]) # Adjust the layout

        # Saving the figure if saveDir and fileName are provided
        if saveDir:
            if not os.path.exists(saveDir):
                os.makedirs(saveDir)
            full_path = os.path.join(saveDir, fileName)
            plt.savefig(full_path, dpi=300)
            plt.close(fig)
            print(f"Saved heatmap to {full_path}")
        else:
            plt.show()

    # call the plotting function
    plot_category_heatmap_vectorized(
        data=data,
        marker_names=subsetMarkers,
        categories=categories,
        clusterRows=clusterRows,
        clusterColumns=clusterColumns,
        standardScale=standardScale,
        orderRow=orderRow,
        orderColumn=orderColumn,
        showPrevalence=showPrevalence,
        cmap=cmap,
        figsize=figsize,
        saveDir=saveDir,
        fileName=fileName,
        **kwargs,
    )
    
#####################################################################

def plot_umap(
    adata,
    color=None,
    layer=None,
    subset=None,
    standardScale=False,
    use_raw=False,
    log=False,
    label='umap',
    cmap='vlag',
    palette=None,
    createLegend=True,
    alpha=0.8,
    figsize=(5, 5),
    s=None,
    ncols=None,
    balance_colors=False,
    tight_layout=False,
    return_data=False,
    saveDir=None,
    fileName='umap.pdf',
    **kwargs,
):

    # check if umap tool has been run
    try:
        adata.obsm[label]
    except KeyError:
        raise KeyError("Please run `sm.tl.umap(adata)` first")



    # identify the coordinates
    umap_coordinates = pd.DataFrame(
        adata.obsm[label], index=adata.obs.index, columns=['umap-1', 'umap-2']
    )
    umap_coordinates['phenotype'] = adata.obs['phenotype']
    
    
    # other data that the user requests
    if color is not None:
        if isinstance(color, str):
            color = [color]
        # identify if all elemets of color are available
        if (
            set(color).issubset(list(adata.var.index) + list(adata.obs.columns))
            is False
        ):
            raise ValueError(
                "Element passed to `color` is not found in adata, please check!"
            )

        # organise the data
        if any(item in color for item in list(adata.obs.columns)):
            adataobs = adata.obs.loc[:, adata.obs.columns.isin(color)]
            adataobs = adataobs.apply(lambda x: x.astype('category'))

        else:
            adataobs = None

        if any(item in color for item in list(adata.var.index)):
            # find the index of the marker
            marker_index = np.where(np.isin(list(adata.var.index), color))[0]
            if layer is not None:
                adatavar = adata.layers[layer][:, np.r_[marker_index]]
            elif use_raw is True:
                adatavar = adata.raw.X[:, np.r_[marker_index]]
            else:
                adatavar = adata.X[:, np.r_[marker_index]]
            adatavar = pd.DataFrame(
                adatavar,
                index=adata.obs.index,
                columns=list(adata.var.index[marker_index]),
            )
        else:
            adatavar = None

        if standardScale is True:
            if adatavar is not None:
                adatavar = adatavar.apply(lambda x: (x - x.mean()) / x.std(), axis=0)


        # combine all color data
        if adataobs is not None and adatavar is not None:
            color_data = pd.concat([adataobs, adatavar], axis=1)
        elif adataobs is not None and adatavar is None:
            color_data = adataobs
        elif adataobs is None and adatavar is not None:
            color_data = adatavar
    else:
        color_data = None

    # combine color data with umap coordinates
    if color_data is not None:
        final_data = pd.concat([umap_coordinates, color_data], axis=1)
    else:
        final_data = umap_coordinates

    # Balance colors if requested
    if balance_colors and color is not None:
        for col in color:
            if col in final_data.columns:
                grouped = final_data.groupby(col)
                min_size = grouped.size().min()
                final_data = grouped.apply(lambda x: x.sample(n=min_size, random_state=0))
                final_data.index = final_data.index.droplevel(0)  # Reset multi-index
    
    # create some reasonable defaults
    # estimate number of columns in subpolt
    nplots = len(final_data.columns) - 2  # total number of plots
    if ncols is None:
        if nplots >= 4:
            subplot = [math.ceil(nplots / 4), 4]
        elif nplots == 0:
            subplot = [1, 1]
        else:
            subplot = [math.ceil(nplots / nplots), nplots]
    else:
        subplot = [math.ceil(nplots / ncols), ncols]

    if nplots == 0:
        n_plots_to_remove = 0
    else:
        n_plots_to_remove = (
            np.prod(subplot) - nplots
        )  # figure if we have to remove any subplots

    # size of points
    if s is None:
        if nplots == 0:
            s = 100000 / adata.shape[0]
        else:
            s = (100000 / adata.shape[0]) / nplots

    # if there are categorical data then assign colors to them
    if final_data.select_dtypes(exclude=["number", "bool_", "object_"]).shape[1] > 0:
        # find all categories in the dataframe
        cat_data = final_data.select_dtypes(exclude=["number", "bool_", "object_"])
        # find all categories
        all_cat = []
        for i in cat_data.columns:
            all_cat.append(list(cat_data[i].cat.categories))

        # generate colormapping for all categories
        less_9 = [colors.rgb2hex(x) for x in sns.color_palette('Set1')]
        nineto20 = [colors.rgb2hex(x) for x in sns.color_palette('tab20')]
        greater20 = [
            colors.rgb2hex(x)
            for x in sns.color_palette('gist_ncar', max([len(i) for i in all_cat]))
        ]

        all_cat_colormap = dict()
        for i in range(len(all_cat)):
            if len(all_cat[i]) <= 9:
                dict1 = dict(zip(all_cat[i], less_9[: len(all_cat[i])]))
            elif len(all_cat[i]) > 9 and len(all_cat[i]) <= 20:
                dict1 = dict(zip(all_cat[i], nineto20[: len(all_cat[i])]))
            else:
                dict1 = dict(zip(all_cat[i], greater20[: len(all_cat[i])]))
            all_cat_colormap.update(dict1)

        # if user has passed in custom colours update the colors
        if palette is not None:
            all_cat_colormap.update(palette)
    else:
        all_cat_colormap = None

    
    #subset data if needed
    if subset is not None:
        if isinstance(subset, str):
            subset = [subset]
        final_data = final_data[final_data['phenotype'].isin(subset)]


    # plot
    fig, ax = plt.subplots(subplot[0], subplot[1], figsize=figsize)
    plt.rcdefaults()
    # plt.rcParams['axes.facecolor'] = 'white'

    # remove unwanted axes
    # fig.delaxes(ax[-1])
    if n_plots_to_remove > 0:
        for i in range(n_plots_to_remove):
            fig.delaxes(ax[-1][(len(ax[-1]) - 1) - i : (len(ax[-1])) - i][0])

    # to make sure the ax is always 2x2
    if any(i > 1 for i in subplot):
        if any(i == 1 for i in subplot):
            ax = ax.reshape(subplot[0], subplot[1])



    if nplots == 0:
        ax.scatter(
            x=final_data['umap-1'],
            y=final_data['umap-2'],
            s=s,
            cmap=cmap,
            alpha=alpha,
            **kwargs,
        )
        plt.xlabel("UMAP-1")
        plt.ylabel("UMAP-2")
        plt.tick_params(right=False, top=False, left=False, bottom=False)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        if tight_layout is True:
            plt.tight_layout()

    elif all(i == 1 for i in subplot):
        column_to_plot = [
                e for e in list(final_data.columns) if e not in ('umap-1', 'umap-2')
            ][0]
        '''
        if ['phenotype'] not in color:
            column_to_plot = [
                e for e in list(final_data.columns) if e not in ('umap-1', 'umap-2','phenotype')
            ][0]
        elif ['phenotype'] in color:
            column_to_plot = [
                e for e in list(final_data.columns) if e not in ('umap-1', 'umap-2')
            ][0]
        '''
        if all_cat_colormap is None:
            im = ax.scatter(
                x=final_data['umap-1'],
                y=final_data['umap-2'],
                s=s,
                c=final_data[column_to_plot],
                cmap=cmap,
                alpha=alpha,
                **kwargs,
            )
            plt.colorbar(im, ax=ax)
            
        else:
            ax.scatter(
                x=final_data['umap-1'],
                y=final_data['umap-2'],
                s=s,
                c=final_data[column_to_plot].map(all_cat_colormap),
                cmap=cmap,
                alpha=alpha,
                **kwargs,
            )
            # create legend
            if createLegend==True:
                patchList = []
                for key in list(final_data[column_to_plot].unique()):
                    data_key = mpatches.Patch(color=all_cat_colormap[key], label=key)
                    patchList.append(data_key)
                    ax.legend(
                        handles=patchList,
                        bbox_to_anchor=(1.05, 1),
                        loc='upper left',
                        borderaxespad=0.0,
                )

        plt.xlabel("UMAP-1")
        plt.ylabel("UMAP-2")
        plt.title(column_to_plot)
        plt.tick_params(right=False, top=False, left=False, bottom=False)
        ax.set(xticklabels=([]))
        ax.set(yticklabels=([]))
        if tight_layout is True:
            plt.tight_layout()

    else:
        column_to_plot = [
            e for e in list(final_data.columns) if e not in ('umap-1', 'umap-2')
        ]
        k = 0
        for i, j in itertools.product(range(subplot[0]), range(subplot[1])):

            if final_data[column_to_plot[k]].dtype == 'category':
                ax[i, j].scatter(
                    x=final_data['umap-1'],
                    y=final_data['umap-2'],
                    s=s,
                    c=final_data[column_to_plot[k]].map(all_cat_colormap),
                    cmap=cmap,
                    alpha=alpha,
                    **kwargs,
                )
                # create legend
                
                if createLegend==True:
                    patchList = []
                    for key in list(final_data[column_to_plot[k]].unique()):
                        data_key = mpatches.Patch(color=all_cat_colormap[key], label=key)
                        patchList.append(data_key)
                        ax[i, j].legend(
                            handles=patchList,
                            bbox_to_anchor=(1.05, 1),
                            loc='upper left',
                            borderaxespad=0.0,
                        )
            else:
                norm = colors.TwoSlopeNorm(vmin=final_data[column_to_plot[k]].min(), vcenter=0, vmax=final_data[column_to_plot[k]].max())
                im = ax[i, j].scatter(
                    x=final_data['umap-1'],
                    y=final_data['umap-2'],
                    s=s,
                    c=final_data[column_to_plot[k]],
                    cmap=cmap,
                    norm=norm,
                    alpha=alpha,
                    **kwargs,
                )
                plt.colorbar(im, ax=ax[i, j])

            ax[i, j].tick_params(right=False, top=False, left=False, bottom=False)
            ax[i, j].set_xticklabels([])
            ax[i, j].set_yticklabels([])
            ax[i, j].set_xlabel("UMAP-1")
            ax[i, j].set_ylabel("UMAP-2")
            ax[i, j].set_title(column_to_plot[k])
            if tight_layout is True:
                plt.tight_layout()
            k = k + 1  # iterator

    # if save figure is requested
    if saveDir:
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)
        full_path = os.path.join(saveDir, fileName)
        plt.savefig(full_path, dpi=300)
        plt.close(fig)
        print(f"Saved heatmap to {full_path}")
    else:
        plt.show()

    # return data if needed
    if return_data is True:
        return final_data
    
    
#####################################################################
    
def spatial_scatterPlot(
    adata,
    colorBy,
    subset_values=None,
    subset_layer=None,
    topLayer=None,
    x_coordinate='X_centroid',
    y_coordinate='Y_centroid',
    imageid='imageid',
    layer=None,
    subset=None,
    s=None,
    ncols=None,
    alpha=1,
    dpi=200,
    fontsize=None,
    plotLegend=True,
    cmap='RdBu_r',
    catCmap='tab20',
    vmin=None,
    vmax=None,
    customColors=None,
    figsize=(5, 5),
    invert_yaxis=True,
    saveDir=None,
    fileName='scimapScatterPlot.png',
    **kwargs,
):
    """
    Parameters:
        adata (anndata.AnnData):  
            Pass the `adata` loaded into memory or a path to the `adata`
            file (.h5ad).

        colorBy (str):  
                The column name that will be used for color-coding the points. This can be
                either markers (data stored in `adata.var`) or observations (data stored in `adata.obs`).

        topLayer (list, optional):  
                A list of categories that should be plotted on the top layer. These categories
                must be present in the `colorBy` data. Helps to highlight cell types or cluster that is of interest.

        x_coordinate (str, optional):
            The column name in `spatial feature table` that records the
            X coordinates for each cell.

        y_coordinate (str, optional):
            The column name in `single-cell spatial table` that records the
            Y coordinates for each cell.

        imageid (str, optional):
            The column name in `spatial feature table` that contains the image ID
            for each cell.

        layer (str or None, optional):
            The layer in `adata.layers` that contains the expression data to use.
            If `None`, `adata.X` is used. use `raw` to use the data stored in `adata.raw.X`.

        subset (list or None, optional):
            `imageid` of a single or multiple images to be subsetted for plotting purposes.

        s (float, optional):
            The size of the markers.

        ncols (int, optional):
            The number of columns in the final plot when multiple variables are plotted.

        alpha (float, optional):
            The alpha value of the points (controls opacity).

        dpi (int, optional):
            The DPI of the figure.

        fontsize (int, optional):
            The size of the fonts in plot.

        plotLegend (bool, optional):
            Whether to include a legend.

        cmap (str, optional):
            The colormap to use for continuous data.

        catCmap (str, optional):
            The colormap to use for categorical data.

        vmin (float or None, optional):
            The minimum value of the color scale.

        vmax (float or None, optional):
            The maximum value of the color scale.

        customColors (dict or None, optional):
            A dictionary mapping color categories to colors.

        figsize (tuple, optional):
            The size of the figure. Default is (5, 5).

        invert_yaxis (bool, optional):
            Invert the Y-axis of the plot.

        saveDir (str or None, optional):
            The directory to save the output plot. If None, the plot will not be saved.

        fileName (str, optional):
            The name of the output file. Use desired file format as
            suffix (e.g. `.png` or `.pdf`). Default is 'scimapScatterPlot.png'.

        **kwargs:
            Additional keyword arguments to be passed to the matplotlib scatter function.


    Returns:
        Plot (image):
            If `saveDir` is provided the plot will saved within the
            provided saveDir.

    Example:
            ```python

            customColors = { 'Unknown' : '#e5e5e5',
                            'CD8+ T' : '#ffd166',
                            'Non T CD4+ cells' : '#06d6a0',
                            'CD4+ T' : '#118ab2',
                            'ECAD+' : '#ef476f',
                            'Immune' : '#073b4c',
                            'KI67+ ECAD+' : '#000000'
                }

            sm.pl.spatial_scatterPlot (adata=core6,
                             colorBy = ['ECAD', 'phenotype_gator'],
                             subset = 'unmicst-6_cellMask',
                             figsize=(4,4),
                             s=0.5,
                             plotLegend=True,
                             fontsize=3,
                             dpi=300,
                             vmin=0,
                             vmax=1,
                             customColors=customColors,
                             fileName='scimapScatterPlot.svg',
                             saveDir='/Users/aj/Downloads')


            ```

    """

    # Load the andata object
    if isinstance(adata, str):
        adata = ad.read_h5ad(adata)
    else:
        adata = adata.copy()

    # subset data if neede
    if subset is not None:
        if isinstance(subset, str):
            subset = [subset]
        if layer == 'raw':
            bdata = adata.copy()
            bdata.X = adata.raw.X
            bdata = bdata[bdata.obs[imageid].isin(subset)]
        else:
            bdata = adata.copy()
            bdata = bdata[bdata.obs[imageid].isin(subset)]
    else:
        bdata = adata.copy()

      # Subset data by phenotypes if needed
    if subset_layer is not None and subset_values is not None:
        if isinstance(subset_values, str):
            subset_values = [subset_values]
        if subset_layer not in bdata.obs.columns:
            raise ValueError(f"Layer '{subset_layer}' not found in the data.")
        if not all(value in bdata.obs[subset_layer].unique() for value in subset_values):
            raise ValueError(
                f"One or more subset values '{subset_values}' not found in layer '{subset_layer}'."
            )
        # Filter the data based on the specified layer and values
        bdata = bdata[bdata.obs[subset_layer].isin(subset_values)]
        
    # isolate the data
    if layer is None:
        data = pd.DataFrame(bdata.X, index=bdata.obs.index, columns=bdata.var.index)
    elif layer == 'raw':
        data = pd.DataFrame(bdata.raw.X, index=bdata.obs.index, columns=bdata.var.index)
    else:
        data = pd.DataFrame(
            bdata.layers[layer], index=bdata.obs.index, columns=bdata.var.index
        )

    # isolate the meta data
    meta = bdata.obs

    # toplayer logic
    if isinstance(topLayer, str):
        topLayer = [topLayer]

    # identify the things to color
    if isinstance(colorBy, str):
        colorBy = [colorBy]
    # extract columns from data and meta
    data_cols = [col for col in data.columns if col in colorBy]
    meta_cols = [col for col in meta.columns if col in colorBy]
    # combine extracted columns from data and meta
    colorColumns = pd.concat([data[data_cols], meta[meta_cols]], axis=1)

    # identify the x and y coordinates
    x = meta[x_coordinate]
    y = meta[y_coordinate]

    # auto identify rows and columns in the grid plot
    def calculate_grid_dimensions(num_items, num_columns=None):
        """
        Calculates the number of rows and columns for a square grid
        based on the number of items.
        """
        if num_columns is None:
            num_rows_columns = int(math.ceil(math.sqrt(num_items)))
            return num_rows_columns, num_rows_columns
        else:
            num_rows = int(math.ceil(num_items / num_columns))
            return num_rows, num_columns

    # calculate the number of rows and columns
    nrows, ncols = calculate_grid_dimensions(
        len(colorColumns.columns), num_columns=ncols
    )

    # resolve figsize
    # figsize = (figsize[0]*ncols, figsize[1]*nrows)

    # Estimate point size
    if s is None:
        s = (10000 / bdata.shape[0]) / len(colorColumns.columns)

    # Define the categorical colormap (optional)
    cmap_cat = plt.get_cmap(catCmap)

    # FIIGURE
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=dpi)

    # Flatten the axs array for easier indexing
    if nrows == 1 and ncols == 1:
        axs = [axs]  # wrap single subplot in a list
    else:
        axs = axs.flatten()

    # Loop over the columns of the DataFrame
    for i, col in enumerate(colorColumns):
        # Select the current axis
        ax = axs[i]

        # invert y-axis
        if invert_yaxis is True:
            ax.invert_yaxis()

        # Scatter plot for continuous data
        if colorColumns[col].dtype.kind in 'iufc':
            scatter = ax.scatter(
                x=x,
                y=y,
                c=colorColumns[col],
                cmap=cmap,
                s=s,
                vmin=vmin,
                vmax=vmax,
                linewidths=0,
                alpha=alpha,
                **kwargs,
            )
            if plotLegend is True:
                cbar = plt.colorbar(scatter, ax=ax, pad=0)
                cbar.ax.tick_params(labelsize=fontsize)

        # Scatter plot for categorical data
        else:
            # Get the unique categories in the column
            categories = colorColumns[col].unique()

            # Map the categories to colors using either the custom colors or the categorical colormap
            if customColors:
                colors = {
                    cat: customColors[cat] for cat in categories if cat in customColors
                }
            else:
                colors = {cat: cmap_cat(i) for i, cat in enumerate(categories)}

            # Ensure topLayer categories are plotted last
            categories_to_plot_last = (
                [cat for cat in topLayer if cat in categories] if topLayer else []
            )
            categories_to_plot_first = [
                cat for cat in categories if cat not in categories_to_plot_last
            ]

            # Plot non-topLayer categories first
            for cat in categories_to_plot_first:
                cat_mask = colorColumns[col] == cat
                ax.scatter(
                    x=x[cat_mask],
                    y=y[cat_mask],
                    c=[colors.get(cat, cmap_cat(np.where(categories == cat)[0][0]))],
                    s=s,
                    linewidths=0,
                    alpha=alpha,
                    **kwargs,
                )

            # Then plot topLayer categories
            for cat in categories_to_plot_last:
                cat_mask = colorColumns[col] == cat
                ax.scatter(
                    x=x[cat_mask],
                    y=y[cat_mask],
                    c=[colors.get(cat, cmap_cat(np.where(categories == cat)[0][0]))],
                    s=s,
                    linewidths=0,
                    alpha=alpha,
                    **kwargs,
                )

            if plotLegend is True:
                # Adjust legend to include all categories
                handles = [
                    mpatches.Patch(
                        color=colors.get(
                            cat, cmap_cat(np.where(categories == cat)[0][0])
                        ),
                        label=cat,
                    )
                    for cat in categories
                ]
                ax.legend(
                    handles=handles,
                    bbox_to_anchor=(1.0, 1.0),
                    loc='upper left',
                    bbox_transform=ax.transAxes,
                    fontsize=fontsize,
                )

        ax.set_title(col)  # fontsize=fontsize
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])

    # Remove any empty subplots
    num_plots = len(colorColumns.columns)
    for i in range(num_plots, nrows * ncols):
        ax = axs[i]
        fig.delaxes(ax)

    # Adjust the layout of the subplots grid
    plt.tick_params(axis='both', labelsize=fontsize)
    plt.tight_layout()

    # save figure
    if saveDir:
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)
        full_path = os.path.join(saveDir, fileName)
        plt.savefig(full_path, dpi=dpi)
        plt.close()
        print(f"Saved plot to {full_path}")
    else:
        plt.show()
        
#####################################################################
def plot_colored_grid(categories, ax, colors=['white','black', 'black','black','black','black','black'], 
                    bounds=[0, 1 , 2, 3,4,5,6,7],
                    grid=True, columnlabels=False, frame=True,
                    verbose=False,
                    
                    ):
    # create discrete colormap
    from matplotlib import colors as mplt
    cmap = mplt.ListedColormap(colors)
    norm = mplt.BoundaryNorm(bounds, cmap.N)
    
    data = []
    for category in categories:
        row = [
            1 if "MITF+" in category else 0,
            2 if "SOX10+" in category else 0,
            3 if "PRAME+" in category else 0,
            4 if "SOX9+" in category else 0,
            5 if "NGFR+" in category else 0,
            6 if "AXL+" in category else 0
        ]
        data.append(row)
    data = np.array(data)
    if verbose:
        print(data)
        print(categories)
    # enable or disable frame
    ax.imshow(data, cmap=cmap,  aspect='auto', norm=norm, extent=[0, data.shape[1], 0, data.shape[0]])
    
    # remove tick labels
    ax.tick_params(bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False) 
    
    
    if not frame:
        ax.axis('off')
        
    # show grid
    if grid:
        ax.set_xticks(np.arange(0, data.shape[1] + 1, 1), minor=False)
        ax.set_yticks(np.arange(0, data.shape[0] + 1, 1), minor=False)
        ax.grid(which="major", color="grey", linestyle='-', linewidth=1)
        ax.tick_params(which="major", size=0, labelbottom=False, labelleft=False)
        
    
    # column labels
    if columnlabels:
        for i, label in enumerate(columnlabels):
            
            ax.text(i + 0.5, data.shape[0] + 0.3, label, ha='center', va='bottom', rotation=90, fontsize=10)
            
      
###################################################################



#####################################################################

def heatmapnew(
    adata,
    groupBy,
    layer=None,
    subsetMarkers=None,
    subsetGroups=None,
    clusterRows=True,
    clusterColumns=True,
    standardScale=None,
    orderRow=None,
    orderColumn=None,
    showPrevalence=False,
    cmap='vlag',
    figsize=None,
    saveDir=None,
    fileName=None,
    verbose=True,
    scale_title=None,
    dend_threshold=None,
    dend_axis = False,
    y_lab='Categories',
    row_dendrogram=None,
    phenotype_matrix=None,
    phenotype_labels=None,
    clustering_method='average',
    return_row_linkage=False,
    x_tick_rotation=90,
    **kwargs,
):

    # load adata
    if isinstance(adata, str):
        adata = ad.read_h5ad(adata)

    # check if the location is provided if the user wishes to save the image
    if (saveDir is None and fileName is not None) or (
        saveDir is not None and fileName is None
    ):
        raise ValueError(
            "Both 'saveDir' and 'fileName' must be provided together or not at all."
        )

    # subset data if user requests
    subsetadata = None  # intialize subsetted data
    if subsetGroups:
        subsetGroups = (
            [subsetGroups] if isinstance(subsetGroups, str) else subsetGroups
        )  # convert to list
        subsetadata = adata[adata.obs[groupBy].isin(subsetGroups)]
        # also identify the categories to be plotted
        categories = subsetadata.obs[groupBy].values
    else:
        # also identify the categories to be plotted
        categories = adata.obs[groupBy].values

    # subset the markers if user requests
    if subsetMarkers:
        subsetMarkers = (
            [subsetMarkers] if isinstance(subsetMarkers, str) else subsetMarkers
        )  # convert to list
        if subsetadata:
            # isolate the data
            if layer == 'raw':
                data = subsetadata[:, subsetMarkers].raw.X
            elif layer is None:
                data = subsetadata[:, subsetMarkers].X
            else:
                data = subsetadata[:, subsetMarkers].layers[layer]
        else:
            # isolate the data
            if layer == 'raw':
                data = adata[:, subsetMarkers].raw.X
            elif layer is None:
                data = adata[:, subsetMarkers].X
            else:
                data = adata[:, subsetMarkers].layers[layer]
    else:
        # take the whole data if the user does not subset anything
        if layer == 'raw':
            data = adata.raw.X
        elif layer is None:
            data = adata.X
        else:
            data = adata.layers[layer]

    # intialize the markers to be plotted
    if subsetMarkers is None:
        subsetMarkers = adata.var.index.tolist()

    # The actual plotting function
    def plot_category_heatmap_vectorized(
        data,
        marker_names,
        categories,
        clusterRows,
        clusterColumns,
        standardScale,
        orderRow,
        orderColumn,
        showPrevalence,
        cmap,
        figsize,
        dend_threshold,
        clustering_method,
        row_dendrogram,
        x_tick_rotation,
        phenotype_matrix,
        dend_axis,
        y_lab,
        phenotype_labels,
        saveDir,
        fileName,
        **kwargs,
    ):
        # Validate clustering and ordering options
        if (clusterRows) and (orderRow is not None):
            raise ValueError(
                "Cannot use clustering and manual ordering together. Please choose one or the other."
            )
        
        if (clusterColumns) and (orderColumn is not None):
            raise ValueError(
                "Cannot use clustering and manual ordering together. Please choose one or the other."
            )

        if standardScale not in [None, 'row', 'column']:
            raise ValueError("standardScale must be 'row', 'column', or None.")

        # Convert marker_names to list if it's a pandas Index
        # if isinstance(marker_names, pd.Index):
        #    marker_names = marker_names.tolist()

        # Data preprocessing
        sorted_indices = np.argsort(categories)
        data = data[sorted_indices, :]
        categories = categories[sorted_indices]
        unique_categories, category_counts = np.unique(categories, return_counts=True)

        # Compute mean values for each category
        mean_data = np.array(
            [
                np.mean(data[categories == category, :], axis=0)
                for category in unique_categories
            ]
        )

        # Apply standard scaling if specified
        if standardScale == 'row':
            scaler = StandardScaler()
            mean_data = scaler.fit_transform(mean_data)
        elif standardScale == 'column':
            scaler = StandardScaler()
            mean_data = scaler.fit_transform(mean_data.T).T
            
    
       
        # Apply manual ordering if specified
        if orderRow:
            # Ensure orderRow is a list
            if isinstance(orderRow, pd.Index):
                orderRow = orderRow.tolist()
            row_order = [unique_categories.tolist().index(r) for r in orderRow]
            mean_data = mean_data[row_order, :]
            unique_categories_ordered = [unique_categories[i] for i in row_order]
            category_counts = [category_counts[i] for i in row_order]

        if orderColumn:
            # Ensure orderColumn is a list0000
            if isinstance(orderColumn, pd.Index):
                orderColumn = orderColumn.tolist()
            col_order = [marker_names.index(c) for c in orderColumn]
            mean_data = mean_data[:, col_order]
            marker_names = [marker_names[i] for i in col_order]

            # Clustering
        row_linkage = None    
        if clusterRows:
            # Perform hierarchical clustering
            row_linkage = linkage(pdist(mean_data), method=clustering_method)
            
        #Row dendogram
        
        if clusterColumns:
            # Perform hierarchical clustering
            col_linkage = linkage(pdist(mean_data.T), method = clustering_method)
            
            
        # Plotting
        # Dynamic figsize calculation
        if figsize is None:
            base_size = 0.5  # Base size for each cell in inches
            figsize_width = max(10, len(marker_names) * base_size)
            figsize_height = max(8, len(unique_categories) * base_size)
            figsize = (figsize_width, figsize_height)
        
        # Create a grid layout for the heatmap and dendrograms
        fig = plt.figure(figsize=figsize)
        spec = plt.GridSpec(nrows=2, ncols=3, 
                            width_ratios=[1,1,3],
                            height_ratios=[1, 5],
                            wspace=0.05,
                            figure=fig)



       
         
        

        # Column dendrogram
        '''
        if clusterColumns:
            ax_col_dendro = fig.add_subplot(spec[0, 2])
            col_dendro = dendrogram(col_linkage, ax=ax_col_dendro)
            col_order = col_dendro['leaves']
            mean_data = mean_data[:, col_order]
            marker_names = [marker_names[i] for i in col_order]
            ax_col_dendro.axis('off')
'''
        if clusterColumns:
            # Reorder data according to the clustering
            col_order = dendrogram(col_linkage, no_plot=True)['leaves']
            mean_data = mean_data[:, col_order]
            marker_names = [marker_names[i] for i in col_order]

        
        
        
        
        
        #Row dendogram
        if clusterRows:
            if row_dendrogram:
                ax_row_dendro = fig.add_subplot(spec[1, 0])
                row_dendro = dendrogram(row_linkage, orientation='left', ax=ax_row_dendro, color_threshold=dend_threshold)
                row_order = row_dendro['leaves']
                
                mean_data = mean_data[row_order, :]
                unique_categories_ordered = unique_categories[row_order]
                category_counts = category_counts[row_order]
                if dend_axis:
                    ax_row_dendro.axis('on')
                    max_distance = np.max(row_linkage[:, 2])  # Get the maximum distance from the linkage matrix
                    num_ticks = 5  # Number of ticks you want
                    tick_positions = np.linspace(0, max_distance, num_ticks)  # Generate tick positions
                    ax_row_dendro.set_xticks(tick_positions)  # Set x-tick positions
                    ax_row_dendro.set_xticklabels([int(tick) for tick in tick_positions], fontsize=8)  # Set integer x-tick labels
                else:
                    ax_row_dendro.axis('off')
            else:
                row_order = dendrogram(row_linkage, no_plot=True)['leaves']
                mean_data = mean_data[row_order, :]
                unique_categories_ordered = unique_categories[row_order]
                category_counts = category_counts[row_order]
       
        #Phenotype matrix
        if phenotype_matrix:
            ax_colored_grid = fig.add_subplot(spec[1, 1])
            plot_colored_grid(
                categories=unique_categories_ordered,
                ax= ax_colored_grid,
                grid=True,
                columnlabels=phenotype_labels,
                frame=True,
                verbose=verbose,
                )
            ax_colored_grid.axis('on')
        
        
        # Heatmap
        ax = fig.add_subplot(spec[1,2])
        vmin = kwargs.pop('vmin', np.min(mean_data))
        vmax = kwargs.pop('vmax', np.max(mean_data))
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        #norm = colors.TwoSlopeNorm(vmin=vmin, vmax=vmax)
        c = ax.imshow(mean_data, aspect='auto', cmap=cmap, norm=norm, **kwargs)
        #c = ax.imshow(mean_data, aspect='auto', cmap=cmap, **kwargs)
        # Prevalence text
        if showPrevalence:
            # Calculate text offset from the last column of the heatmap
            text_offset = (
                mean_data.shape[1] * 0.001
            )  # Small offset from the right edge of the heatmap
            total_cells = sum(category_counts)

            for index, count in enumerate(category_counts):
                percentage = (count / total_cells) * 100
                # Position text immediately to the right of the heatmap
                ax.text(
                    mean_data.shape[1] + text_offset,
                    index,
                    f"n={count} ({percentage:.1f}%)",
                    va='center',
                    ha='left',
                )
        
        # Setting the tick labels
        ax.set_xticks(np.arange(mean_data.shape[1]))
        ax.tick_params(axis='x', pad=0)
        if x_tick_rotation > 0:
            ax.set_xticklabels(marker_names, rotation=x_tick_rotation, ha="right",va='top',)
            
        elif x_tick_rotation == 90:
            ax.set_xticklabels(marker_names, rotation=x_tick_rotation, ha="center",va='top',)
        else:
             ax.set_xticklabels(marker_names, rotation=x_tick_rotation, ha="center",va='center',)
        ax.set_yticks(np.arange(mean_data.shape[0]))
        if phenotype_matrix:
            ax.set_yticks([])
        else:
            ax.set_yticklabels(unique_categories_ordered)
        
        #ax.set_yticklabels(unique_categories)
        
        # Move the colorbar to the top left corner
        # cbar_ax = fig.add_axes([0.125, 0.92, 0.2, 0.02])  # x, y, width, height
        cbar_ax = ax.inset_axes([1.5, -1.5, 4, 0.5], transform=ax.transData)
        cbar = plt.colorbar(c, cax=cbar_ax, orientation='horizontal')
        cbar_ax.xaxis.set_ticks_position('top')
        cbar_ax.xaxis.set_label_position('top')
        if scale_title:
            cbar.set_label(scale_title)
        else:
            cbar.set_label('Mean expression in group')

        ax.set_xlabel('Markers',labelpad=15)
        if phenotype_matrix is None:
            ax.set_ylabel(y_lab)
        
        plt.tight_layout()

        # Saving the figure if saveDir and fileName are provided
        if saveDir:
            if not os.path.exists(saveDir):
                os.makedirs(saveDir)
            full_path = os.path.join(saveDir, fileName)
            plt.savefig(full_path, bbox_inches='tight', pad_inches=0.1,dpi=300)
            plt.close(fig)
            print(f"Saved heatmap to {full_path}")
        
        else:
            plt.show()

        if return_row_linkage:
            cluster_labels = fcluster(row_linkage, t=dend_threshold, criterion='distance')
            
            reversed_categories = unique_categories_ordered[::-1]
            
            ordered_df = pd.DataFrame({
                'key': row_order,
                'phenotype': [reversed_categories[i] for i, index in enumerate(row_order)]
            })
            cluster_labels_final =[]
            for i in ordered_df['key']:    
                cluster_labels_final.append(cluster_labels[i])
                #ordered_df.at[i, 'cluster'] = cluster_labels[i]
            
            ordered_df['cluster'] = cluster_labels_final
            ordered_df['cluster'] = ordered_df['cluster'].astype('category')
            phenotype_cluster_mapping= dict(zip(ordered_df['phenotype'], ordered_df['cluster']))
            #phenotype_cluster_mapping = ordered_df
            
            return phenotype_cluster_mapping
        
            
        
    # call the plotting function
    phenotype_cluster_mapping = plot_category_heatmap_vectorized(
        data=data,
        marker_names=subsetMarkers,
        categories=categories,
        clusterRows=clusterRows,
        clusterColumns=clusterColumns,
        standardScale=standardScale,
        orderRow=orderRow,
        orderColumn=orderColumn,
        showPrevalence=showPrevalence,
        cmap=cmap,
    
        dend_threshold=dend_threshold,
        clustering_method=clustering_method,
        phenotype_labels=phenotype_labels,
        dend_axis=dend_axis,
        y_lab=y_lab,
        row_dendrogram=row_dendrogram,
        phenotype_matrix=phenotype_matrix,
        x_tick_rotation=x_tick_rotation,
        figsize=figsize,
        saveDir=saveDir,
        fileName=fileName,
        **kwargs,
    )
    
    if return_row_linkage:
        return phenotype_cluster_mapping
    


############################################################################

def densityPlot2D(
    adata,
    markerA,
    markerB=None,
    markerC=None,
    layer=None,
    subset=None,
    imageid='imageid',
    ncols=None,
    cmap='jet',
    figsize=(3, 3),
    hline='auto',
    vline='auto',
    fontsize=None,
    dpi=100,
    xticks=None,
    yticks=None,
    saveDir=None,
    fileName='densityPlot2D.pdf',
):
    """
    Parameters:
        adata (anndata.AnnData):
            Annotated data matrix containing single-cell gene expression data.

        markerA (str):
            The name of the first marker whose expression will be plotted.

        markerB (list, optional):
            The name of the second marker or a list of second markers whose expression will be plotted.
            If not provided, a 2D density plot of `markerA` against all markers in the dataset will be plotted.

        layer (str or list of str, optional):
            The layer in adata.layers that contains the expression data to use.
            If None, adata.X is used. use `raw` to use the data stored in `adata.raw.X`

        subset (list, optional):
            `imageid` of a single or multiple images to be subsetted for plotting purposes.

        imageid (str, optional):
            Column name of the column containing the image id. Use in conjunction with `subset`.

        ncols (int, optional):
            The number of columns in the grid of density plots.

        cmap (str, optional):
            The name of the colormap to use. Defaults to 'jet'.

        figsize (tuple, optional):
            The size of the figure in inches.

        hline (float or 'auto', optional):
            The y-coordinate of the horizontal line to plot. If set to `None`, a horizontal line is not plotted.
            Use 'auto' to draw a vline at the center point.

        vline (float or 'auto', optional):
            The x-coordinate of the vertical line to plot. If set to `None`, a vertical line is not plotted.
            Use 'auto' to draw a vline at the center point.

        fontsize (int, optional):
            The size of the font of the axis labels.

        dpi (int, optional):
            The DPI of the figure. Use this to control the point size. Lower the dpi, larger the point size.

        xticks (list of float, optional):
            Custom x-axis tick values.

        yticks (list of float, optional):
            Custom y-axis tick values.

        saveDir (str, optional):
            The directory to save the output plot.

        fileName (str, optional):
            The name of the output file. Use desired file format as suffix (e.g. `.png` or `.pdf`).

    Returns:
        Plot (image):
            If `outputDir` is not provided, the plot is displayed on the screen.
            Otherwise, the plot is saved in the provided `outputDir` directory.

    Example:
        ```python

        # create a 2D density plot of the expression of 'CD3D' against 'CD8A' in the dataset 'adata'
        sm.pl.densityPlot2D(adata, markerA='CD3D', markerB='CD8A')

        # create a 2D density plot of the expression of 'CD3D' against all markers in the dataset 'adata'
        sm.pl.densityPlot2D(adata, markerA='CD3D')
        ```

    """
    # testing
    # import anndata as ad
    # adata = ad.read(r"C:\Users\aj\Dropbox (Partners HealthCare)\nirmal lab\softwares\scimap\scimap\tests\_data\example_data.h5ad")
    # adata = ad.read('/Users/aj/Dropbox (Partners HealthCare)/nirmal lab/softwares/scimap/scimap/tests/_data/example_data.h5ad')
    # markerA ='CD3E'; layers=None; markerB='CD163'; plotGrid=True; ncols=None; color=None; figsize=(10, 10); fontsize=None; subset=None; imageid='imageid'; xticks=None; dpi=200; outputDir=None;
    # hline = 'auto'; vline = 'auto'
    # outputFileName='densityPlot2D.png'
    # color = {'markerA': '#000000', 'markerB': '#FF0000'}
    # outputDir = r"C:\Users\aj\Downloads"

    # densityPlot2D (adata, markerA='CD3D', markerB=['CD2', 'CD10', 'CD163'], dpi=50, outputDir=r"C:\Users\aj\Downloads")

    # set color
    # cp = copy.copy(cm.get_cmap(cmap))
    # cp.set_under(alpha=0)

    cp = copy.copy(plt.colormaps[cmap])
    cp.set_under(alpha=0)

    # Subset data if needed
    if subset is not None:
        if isinstance(subset, str):
            subset = [subset]
        if layer == 'raw':
            bdata = adata.copy()
            bdata.X = adata.raw.X
            bdata = bdata[bdata.obs[imageid].isin(subset)]
        else:
            bdata = adata.copy()
            bdata = bdata[bdata.obs[imageid].isin(subset)]
    else:
        bdata = adata.copy()

    # Isolate the data
    if layer is None:
        data = pd.DataFrame(bdata.X, index=bdata.obs.index, columns=bdata.var.index)
    elif layer == 'raw':
        data = pd.DataFrame(bdata.raw.X, index=bdata.obs.index, columns=bdata.var.index)
    else:
        data = pd.DataFrame(bdata.layers[layer], index=bdata.obs.index, columns=bdata.var.index)

    # Keep only columns that are required
    x = data[markerA]

    if markerB is None:
        y = data.drop(markerA, axis=1)
    elif markerC is not None:
        if (isinstance(markerC, str) and isinstance(markerB, str)):
            markerC = [markerC]
            markerB = [markerB]
        
        c = data[markerC]
        y = data[markerB]# Convert to 1D array
        df = pd.concat([x,y,c],axis=1)
    else:
        if isinstance(markerB, str):
            markerB = [markerB]
        y = data[markerB]

    # Auto identify rows and columns in the grid plot
    def calculate_grid_dimensions(num_items, num_columns=None):
        """
        Calculates the number of rows and columns for a square grid
        based on the number of items.
        """
        if num_columns is None:
            num_rows_columns = int(math.ceil(math.sqrt(num_items)))
            return num_rows_columns, num_rows_columns
        else:
            num_rows = int(math.ceil(num_items / num_columns))
            return num_rows, num_columns

    # Calculate the number of rows and columns
    num_rows, num_cols = calculate_grid_dimensions(len(y.columns), num_columns=ncols)

    fig, axs = plt.subplots(
        nrows=num_rows,
        ncols=num_cols,
        figsize=(num_cols * figsize[0], num_rows * figsize[0]),
        subplot_kw={'projection': 'scatter_density'},
    )
    if num_rows == 1 and num_cols == 1:
        axs = [axs]  # wrap single subplot in a list
    else:
        axs = axs.flatten()

    for i, col in enumerate(y.columns):
        ax = axs[i]

        # Scatter plot color by markerC if provided, else default by x
        if markerC is not None:
            
             
            scatter = df.plot.scatter(
                x, y,c, norm=LogNorm(vmin=0.5, vmax=x.size),
            )
            fig.colorbar(scatter, ax=ax, label=markerC)  # add colorbar
        else:
            ax.scatter_density(
                x, y[col], dpi=dpi, cmap=cp, norm=LogNorm(vmin=0.5, vmax=x.size)
            )

        ax.set_xlabel(markerA, size=fontsize)
        ax.set_ylabel(col, size=fontsize)

        if hline == 'auto':
            ax.axhline((y[col].max() + y[col].min()) / 2, color='grey')
        elif hline is None:
            pass
        else:
            ax.axhline(hline, color='grey')

        if vline == 'auto':
            ax.axvline((x.max() + x.min()) / 2, color='grey')
        elif vline is None:
            pass
        else:
            ax.axvline(vline, color='grey')

        # Control x and y ticks
        if xticks is not None:
            ax.set_xticks(xticks)
            ax.set_xticklabels([str(x) for x in xticks])

        if yticks is not None:
            ax.set_yticks(yticks)
            ax.set_yticklabels([str(x) for x in yticks])

    # Remove any empty subplots
    num_plots = len(y.columns)
    for i in range(num_plots, num_rows * num_cols):
        ax = axs[i]
        fig.delaxes(ax)

    plt.tick_params(axis='both', labelsize=fontsize)
    plt.tight_layout()

    # Save the figure to a file
    if saveDir:
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)
        full_path = os.path.join(saveDir, fileName)
        plt.savefig(full_path, dpi=300)
        plt.close(fig)
        print(f"Saved heatmap to {full_path}")
    else:
        plt.show()

##########################################################################

def run_umap(
    adata, 
    use_layer=None, 
    use_raw=False, 
    log=False,
    n_neighbors=15, 
    n_components=2, 
    metric='euclidean',
    min_dist=0.1, 
    random_state=0, 
    balance_key=None,
    label='umap',
    subset_columns=None,  # Optional: Subset specific columns
    **kwargs
):
    """
    Parameters:
        adata (anndata.AnnData):  
            Annotated data matrix or path to an AnnData object, containing spatial gene expression data.

        use_layer (str, optional):  
            Specifies a layer in `adata.layers` for UMAP. Defaults to using `adata.X`.

        use_raw (bool, optional):  
            Whether to use `adata.raw.X` for the analysis.

        log (bool, optional):  
            Applies natural log transformation to the data if `True`.

        n_neighbors (int, optional):  
            Number of neighboring points used in manifold approximation.

        n_components (int, optional):  
            Dimensionality of the target embedding space.

        metric (str, optional):  
            Metric used to compute distances in high-dimensional space.

        min_dist (float, optional):  
            Effective minimum distance between embedded points.

        random_state (int, optional):  
            Seed used by the random number generator for reproducibility.

        balance_key (str, optional):  
            Key in `adata.obs` to balance sample sizes across groups.

        label (str, optional):  
            Key for storing UMAP results in `adata.obsm`.

        subset_columns (list, optional):  
            List of column names to subset the data for UMAP. Defaults to using all columns.

    Returns:
        adata (anndata.AnnData):  
            The input `adata` object, updated with UMAP embedding results in `adata.obsm[label]`.
    """
    # Balance sample sizes if requested
    if balance_key is not None:   
        if balance_key not in adata.obs:
            raise ValueError(f"Balance key '{balance_key}' not found in `adata.obs`.")
        sample_sizes = adata.obs[balance_key].value_counts()
        min_size = sample_sizes.min()
        balanced_indices = adata.obs.groupby(balance_key).apply(
            lambda x: x.sample(n=min_size, random_state=random_state)
        ).index.get_level_values(1)
        adata = adata[balanced_indices]

    # Load data
    if use_layer is not None:
        data = adata.layers[use_layer]
    elif use_raw is True:
        data = adata.raw.X
    else:
        data = adata.X

    # Subset columns if specified
    if subset_columns is not None:
        if not set(subset_columns).issubset(adata.var.index):
            raise ValueError("Some subset_columns are not found in adata.var.index.")
        column_indices = [adata.var.index.get_loc(col) for col in subset_columns]
        data = data[:, column_indices]

    # Log-transform the data if requested
    if log:
        data = np.log1p(data)

    # Run UMAP
    embedding = um.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        metric=metric,
        min_dist=min_dist,
        random_state=random_state
    ).fit_transform(data)

    # Store the embedding in adata
    adata.obsm[label] = embedding
    return adata

#################################################q##############################
def plot_umap_new(
    adata,
    color=None,
    layer=None,
    subset=None,
    standardScale=False,
    use_raw=False,
    log=False,
    label='umap',
    cmap='vlag',
    palette=None,
    createLegend=True,
    alpha=0.8,
    figsize=(5, 5),
    s=None,
    ncols=1,  # Default to 1 column for separate rows
    balance_colors=False,
    colorbar_range=None,
    tight_layout=True,
    return_data=False,
    saveDir=None,
    fileName='umap.pdf',
    show_layer=None,  # New parameter to toggle phenotype plot
    **kwargs,
):

    # Check if UMAP tool has been run
    try:
        adata.obsm[label]
    except KeyError:
        raise KeyError("Please run `sm.tl.umap(adata)` first")

    # Identify the coordinates
    umap_coordinates = pd.DataFrame(
        adata.obsm[label], index=adata.obs.index, columns=['umap-1', 'umap-2']
    )
    if show_layer is not None:
        umap_coordinates[show_layer] = adata.obs[show_layer]

    # Handle additional color data
    if color is not None:
        if isinstance(color, str):
            color = [color]
        if not set(color).issubset(list(adata.var.index) + list(adata.obs.columns)):
            raise ValueError(
                "Element passed to `color` is not found in adata, please check!"
            )

        adataobs = adata.obs[color] if any(c in adata.obs.columns for c in color) else None
        adatavar = (
            pd.DataFrame(
                adata.layers[layer][:, adata.var.index.isin(color)]
                if layer
                else adata.X[:, adata.var.index.isin(color)],
                index=adata.obs.index,
                columns=adata.var.index[adata.var.index.isin(color)],
            )
            if any(c in adata.var.index for c in color)
            else None
        )

        if standardScale and adatavar is not None:
            adatavar = adatavar.apply(lambda x: (x - x.mean()) / x.std(), axis=0)

        # Combine color data
        if adataobs is not None and adatavar is not None:
            color_data = pd.concat([adataobs, adatavar], axis=1)
        elif adataobs is not None:
            color_data = adataobs
        elif adatavar is not None:
            color_data = adatavar
    else:
        color_data = None

    # Combine color data with UMAP coordinates
    final_data = pd.concat([umap_coordinates, color_data], axis=1) if color_data is not None else umap_coordinates

    # Balance colors if requested
    if balance_colors and color is not None:
        for col in color:
            if col in final_data.columns:
                grouped = final_data.groupby(col)
                min_size = grouped.size().min()
                final_data = grouped.apply(lambda x: x.sample(n=min_size, random_state=0))
                final_data.index = final_data.index.droplevel(0)

    # Subset data if needed
    if subset is not None:
        subset = [subset] if isinstance(subset, str) else subset
        final_data = final_data[final_data['phenotype'].isin(subset)]

    # Determine the number of plots
    nplots = len(final_data.columns) - 2  # Exclude 'umap-1' and 'umap-2'
    if show_layer:
        nplots += 1  # Include phenotype plot if enabled
    nrows = math.ceil(nplots / ncols)

    # Size of points
    s = s or (100000 / adata.shape[0]) / nplots

    # Generate colormap for categorical data
    if final_data.select_dtypes(exclude=["number", "bool_", "object_"]).shape[1] > 0:
        cat_data = final_data.select_dtypes(exclude=["number", "bool_", "object_"])
        all_cat = [list(cat_data[col].cat.categories) for col in cat_data.columns]
        all_cat_colormap = {
            cat: colors.rgb2hex(c)
            for cats in all_cat
            for cat, c in zip(cats, sns.color_palette("tab20", len(cats)))
        }
        if palette:
            all_cat_colormap.update(palette)
    else:
        all_cat_colormap = None

    # Create subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize[0] * ncols, figsize[1] * nrows))
    axes = axes.flatten() if nrows * ncols > 1 else [axes]

    # Plot specific layer in the first subplot if enabled
    if show_layer:
        axes[0].scatter(
            x=final_data['umap-1'],
            y=final_data['umap-2'],
            s=s,
            c=final_data[show_layer].map(all_cat_colormap),
            cmap=cmap,
            alpha=alpha,
            **kwargs,
        )
        axes[0].set_title(str(show_layer))
        axes[0].set_xlabel("UMAP-1")
        axes[0].set_ylabel("UMAP-2")
        axes[0].tick_params(right=False, top=False, left=False, bottom=False)
        axes[0].set_xticks([])
        axes[0].set_yticks([])

        # Add legend for show_layer with padding
        if createLegend and all_cat_colormap:
            handles = [
                mpatches.Patch(color=all_cat_colormap[cat], label=cat)
                for cat in final_data[show_layer].cat.categories
            ]
            axes[0].legend(
                handles=handles,
                loc='upper left',
                bbox_to_anchor=(1.05, 1),  # Add padding to the right
                fontsize=8,
            )

    # Plot other color data in subsequent subplots
    if show_layer is not None:
        column_to_plot = [col for col in final_data.columns if col not in ('umap-1', 'umap-2', show_layer)]
    else: 
        column_to_plot = [col for col in final_data.columns if col not in ('umap-1', 'umap-2')]
    start_index = 1 if show_layer else 0
    for i, col in enumerate(column_to_plot, start=start_index):
        if final_data[col].dtype.name == 'category':
            axes[i].scatter(
                x=final_data['umap-1'],
                y=final_data['umap-2'],
                s=s,
                c=final_data[col].map(all_cat_colormap),
                cmap=cmap,
                alpha=alpha,
                **kwargs,
            )
            if createLegend:
                handles = [
                    mpatches.Patch(color=all_cat_colormap[cat], label=cat)
                    for cat in final_data[col].cat.categories
                ]
                axes[i].legend(
                    handles=handles,
                    loc='upper left',
                    bbox_to_anchor=(1.05, 1),  # Add padding to the right
                    fontsize=8,
                )
        else:
            if colorbar_range is not None:
                vmin, vmax = colorbar_range
                norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
                
            else:
                vmin = final_data[col].min()
                vmax = final_data[col].max()
                abs_max = max(abs(vmin), abs(vmax))  # Match the higher absolute value
                norm = colors.TwoSlopeNorm(
                    vmin=-abs_max,
                    vcenter=0,
                    vmax=abs_max,
                )
            im = axes[i].scatter(
                x=final_data['umap-1'],
                y=final_data['umap-2'],
                s=s,
                c=final_data[col],
                cmap=cmap,
                norm=norm,
                alpha=alpha,
                **kwargs,
            )
            plt.colorbar(im, ax=axes[i], orientation='vertical', pad=0.01)

        axes[i].set_title(col)
        axes[i].set_xlabel("UMAP-1")
        axes[i].set_ylabel("UMAP-2")
        axes[i].tick_params(right=False, top=False, left=False, bottom=False)
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    # Remove unused subplots
    for ax in axes[nplots:]:
        fig.delaxes(ax)

    # Adjust layout
    if tight_layout:
        plt.tight_layout()

    # Save or show the plot
    if saveDir:
        os.makedirs(saveDir, exist_ok=True)
        plt.savefig(os.path.join(saveDir, fileName), dpi=300)
        plt.close(fig)
        print(f"Saved UMAP plot to {os.path.join(saveDir, fileName)}")
    else:
        plt.show()

    # Return data if needed
    if return_data:
        return final_data
    
#################################################################


def plot_scatter(
    adata, x_marker, y_marker, color_marker, layer='tumor_zscore', figsize=(8, 6),
    sample_fraction=0.1, x_divisor=None, y_divisor=None, zscore_points=False, zscore_threshold=None,
    saveDir=None, fileName=None,
    title_prefix='',
):
    """
    Plots a scatter plot for tumor adata with options for x and y ratios, z-scoring, and outlier removal.
    
    Parameters:
        adata: AnnData object
            The annotated data matrix.
        x_marker: str
            Marker for the x-axis (or numerator if x_divisor is provided).
        y_marker: str
            Marker for the y-axis (or numerator if y_divisor is provided).
        color_marker: str
            Marker for the color scale.
        layer: str
            The layer in adata to extract the data from.
        sample_fraction: float
            Fraction of the data to sample (e.g., 0.1 for 10%).
        x_divisor: str or None
            Marker for the denominator of the x-axis ratio. If None, x_marker is used as-is.
        y_divisor: str or None
            Marker for the denominator of the y-axis ratio. If None, y_marker is used as-is.
        zscore_points: bool
            If True, z-scores the x and y values before plotting.
        zscore_threshold: float or None
            If provided, removes points where the z-score of x or y exceeds this threshold.
    """
    # Randomly sample the data
    sampled_indices = np.random.choice(adata.n_obs, size=int(adata.n_obs * sample_fraction), replace=False)
    sampled_adata = adata[sampled_indices]
    
    # Extract data for the specified markers
    x = sampled_adata[:, x_marker].layers[layer].flatten()
    y = sampled_adata[:, y_marker].layers[layer].flatten()
    color = sampled_adata[:, color_marker].layers[layer].flatten()
    
    # Compute the ratio for the x-axis if x_divisor is provided
    if x_divisor is not None:
        divisor_x = sampled_adata[:, x_divisor].layers[layer].flatten()
        x = x / (divisor_x + 1e-8)  # Add a small value to avoid division by zero
    
    # Compute the ratio for the y-axis if y_divisor is provided
    if y_divisor is not None:
        divisor_y = sampled_adata[:, y_divisor].layers[layer].flatten()
        y = y / (divisor_y + 1e-8)  # Add a small value to avoid division by zero
    
    # Z-score the points if requested
    if zscore_points:
        x = zscore(x)
        y = zscore(y)
    
    # Remove outliers based on z-score threshold
    if zscore_threshold is not None:
        x_zscore = zscore(x)
        y_zscore = zscore(y)
        valid_indices = (np.abs(x_zscore) < zscore_threshold) & (np.abs(y_zscore) < zscore_threshold)
        x = x[valid_indices]
        y = y[valid_indices]
        color = color[valid_indices]
    
      # Center the color bar on zero
    vmin = min(color.min(), -color.max())
    vmax = max(color.max(), -color.min())
    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    
    # Create the scatter plot
    plt.figure(figsize= figsize)
    scatter = plt.scatter(x, y, c=color, cmap='viridis', norm=norm, alpha=0.8)
    plt.colorbar(scatter, label=color_marker)
    
    # Center the x and y axes on 0
     #Determine the maximum absolute value for both axes
    max_abs_value = max(abs(x).max(), abs(y).max())
    plt.xlim(-max_abs_value, max_abs_value)
    plt.ylim(-max_abs_value, max_abs_value)
    
    # Add labels and title
    x_label = f'{x_marker}/{x_divisor}' if x_divisor else x_marker
    y_label = f'{y_marker}/{y_divisor}' if y_divisor else y_marker
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'{title_prefix} {x_label} vs {y_label} (Color: {color_marker})')
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)  # Add horizontal line at y=0
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)  # Add vertical line at x=0
    plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
    
        # Determine the filename if not provided
    if fileName is None:
        if title_prefix:
            fileName = f"{title_prefix}_{x_label}vs{y_label}vs{color_marker}_sample{sample_fraction}.png"
        else:
            fileName = f"{x_label}vs{y_label}vs{color_marker}_sample{sample_fraction}.png"
    else:
        fileName = f"{fileName}.png" if not fileName.endswith('.png') else fileName
    
    if saveDir:
        os.makedirs(saveDir, exist_ok=True)
        plt.savefig(os.path.join(saveDir, fileName), dpi=300)
        plt.close()
        print(f"Saved scatterplot to {os.path.join(saveDir, fileName)}")
    else:
        plt.show()

################################################

def spatial_scatterPlot_new(
    adata,
    colorBy,
    subset_values=None,
    subset_layer=None,
    topLayer=None,
    x_coordinate='X_centroid',
    y_coordinate='Y_centroid',
    imageid='imageid',
    layer=None,
    subset=None,
    s=None,
    ncols=None,
    alpha=1,
    dpi=200,
    fontsize=None,
    plotLegend=True,
    cmap='RdBu_r',
    catCmap='tab20',
    vmin=None,
    vmax=None,
    customColors=None,
    figsize=(5, 5),
    invert_yaxis=True,
    saveDir=None,
    fileName='scimapScatterPlot.png',
    **kwargs,
):
    """
    Updated to handle multiple subset_values with grid subplots.
    """
    # Load the andata object
    if isinstance(adata, str):
        adata = ad.read_h5ad(adata)
    else:
        adata = adata.copy()

    # Subset data if needed
    if subset is not None:
        if isinstance(subset, str):
            subset = [subset]
        if layer == 'raw':
            bdata = adata.copy()
            bdata.X = adata.raw.X
            bdata = bdata[bdata.obs[imageid].isin(subset)]
        else:
            bdata = adata.copy()
            bdata = bdata[bdata.obs[imageid].isin(subset)]
    else:
        bdata = adata.copy()

    # Subset data by phenotypes if needed
    if subset_layer is not None and subset_values is not None:
        if isinstance(subset_values, str):
            subset_values = [subset_values]
        if subset_layer not in bdata.obs.columns:
            raise ValueError(f"Layer '{subset_layer}' not found in the data.")
        if not all(value in bdata.obs[subset_layer].unique() for value in subset_values):
            raise ValueError(
                f"One or more subset values '{subset_values}' not found in layer '{subset_layer}'."
            )

        # If multiple subset_values, create a grid of subplots
        if len(subset_values) > 1:
            ncols = ncols or 3  # Default number of columns
            nrows = (len(subset_values) + ncols - 1) // ncols  # Calculate rows
            fig, axes = plt.subplots(
                nrows=nrows, ncols=ncols, figsize=(figsize[0] * ncols, figsize[1] * nrows), dpi=dpi
            )
            axes = axes.flatten()  # Flatten axes for easy indexing

            for i, value in enumerate(subset_values):
                ax = axes[i]
                subset_data = bdata[bdata.obs[subset_layer] == value]
                x = subset_data.obs[x_coordinate]
                y = subset_data.obs[y_coordinate]
                color = subset_data.obs[colorBy[0]] if colorBy[0] in subset_data.obs else subset_data[:, colorBy[0]].X.flatten()

                scatter = ax.scatter(
                    x=x,
                    y=y,
                    c=color,
                    cmap=cmap,
                    s=s or (10000 / subset_data.shape[0]),
                    vmin=vmin,
                    vmax=vmax,
                    alpha=alpha,
                    **kwargs,
                )
                ax.set_title(f"{subset_layer}: {value}", fontsize=fontsize)
                ax.set_xticks([])
                ax.set_yticks([])
                if invert_yaxis:
                    ax.invert_yaxis()
                if plotLegend:
                    plt.colorbar(scatter, ax=ax, pad=0.01)

            # Remove unused subplots
            for j in range(len(subset_values), len(axes)):
                fig.delaxes(axes[j])

            plt.tight_layout()
            if saveDir:
                if not os.path.exists(saveDir):
                    os.makedirs(saveDir)
                full_path = os.path.join(saveDir, fileName)
                plt.savefig(full_path, dpi=dpi)
                plt.close(fig)
                print(f"Saved plot to {full_path}")
            else:
                plt.show()
            return

        # If only one subset_value, filter the data
        bdata = bdata[bdata.obs[subset_layer].isin(subset_values)]

    # Isolate the data
    if layer is None:
        data = pd.DataFrame(bdata.X, index=bdata.obs.index, columns=bdata.var.index)
    elif layer == 'raw':
        data = pd.DataFrame(bdata.raw.X, index=bdata.obs.index, columns=bdata.var.index)
    else:
        data = pd.DataFrame(
            bdata.layers[layer], index=bdata.obs.index, columns=bdata.var.index
        )

    # Isolate the meta data
    meta = bdata.obs

    # Identify the x and y coordinates
    x = meta[x_coordinate]
    y = meta[y_coordinate]

    # Plot for a single subset or no subset
    plt.figure(figsize=figsize, dpi=dpi)
    scatter = plt.scatter(
        x=x,
        y=y,
        c=meta[colorBy[0]] if colorBy[0] in meta else data[colorBy[0]],
        cmap=cmap,
        s=s or (10000 / bdata.shape[0]),
        vmin=vmin,
        vmax=vmax,
        alpha=alpha,
        **kwargs,
    )
    if invert_yaxis:
        plt.gca().invert_yaxis()
    if plotLegend:
        plt.colorbar(scatter, pad=0.01)
    plt.title(colorBy[0], fontsize=fontsize)
    plt.xticks([])
    plt.yticks([])
    if saveDir:
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)
        full_path = os.path.join(saveDir, fileName)
        plt.savefig(full_path, dpi=dpi)
        plt.close()
        print(f"Saved plot to {full_path}")
    else:
        plt.show()
        
        
####################################################################


def distPlot3(
    adata,
    layer=None,
    markers=None,
    subset=None,
    imageid='imageid',
    vline=None,
    vlinewidth=None,
    plotGrid=True,
    ncols=None,
    color=None,
    xticks=None,
    figsize=(5, 5),
    fontsize=None,
    dpi=200,
    saveDir=None,
    fileName='scimapDistPlot.png',
    scale_x1=None,
    scale_x2=None,
    multiple_cases=False,
    y_label='Case',         # <-- add this
    x_label='Marker',                 # <-- add this
    plot_title='Pixel Distribution'
):
    
    # subset data if neede
    if subset is not None:
        if isinstance(subset, str):
            subset = [subset]
        if layer == 'raw':
            bdata = adata.copy()
            bdata.X = adata.raw.X
            bdata = bdata[bdata.obs[imageid].isin(subset)]
        else:
            bdata = adata.copy()
            bdata = bdata[bdata.obs[imageid].isin(subset)]
    else:
        bdata = adata.copy()

    # isolate the data
    if layer is None:
        data = pd.DataFrame(bdata.X, index=bdata.obs.index, columns=bdata.var.index)
    elif layer == 'raw':
        data = pd.DataFrame(bdata.raw.X, index=bdata.obs.index, columns=bdata.var.index)
    else:
        data = pd.DataFrame(
            bdata.layers[layer], index=bdata.obs.index, columns=bdata.var.index
        )

    # keep only columns that are required
    if markers is not None:
        if isinstance(markers, str):
            markers = [markers]
        # subset the list
        data = data[markers]
        
    if multiple_cases:
        # Get unique cases
        cases = bdata.obs[imageid].unique()
        n_cases = len(cases)
        n_markers = len(data.columns)
        fig, axes = plt.subplots(
            nrows=n_cases, ncols=n_markers,
            figsize=(n_markers * 3, n_cases * 2.5),
            dpi=dpi, sharex='col', sharey='row'
        )
        if n_cases == 1:
            axes = np.expand_dims(axes, 0)
        if n_markers == 1:
            axes = np.expand_dims(axes, 1)
        for i, case in enumerate(cases):
            case_mask = bdata.obs[imageid] == case
            for j, marker in enumerate(data.columns):
                ax = axes[i, j]
                sns.kdeplot(
                    data.loc[case_mask, marker],
                    ax=ax,
                    fill=True,
                    color=color.get(marker, None) if color else None,
                    linewidth=1.5,
                )
                # Calculate percentiles and mean/median
                values = data.loc[case_mask, marker].dropna()
                #mean = values.mean()
                #median = values.median()
                p1 = np.percentile(values, 1)
                p99 = np.percentile(values, 99)

                # Plot vertical lines with different colors
                vlines = [
                    #(mean, 'red', '-', 'Mean'),
                    #(median, 'blue', '--', 'Median'),
                    (p1, 'green', '--', '1st Percentile'),
                    (p99, 'purple', '--', '99th Percentile'),
                ]
                for v, c, ls, lbl in vlines:
                    ax.axvline(v, color=c, linestyle=ls, linewidth=1, label=lbl, alpha=0.5)

                

                if vline is not None:
                    ax.axvline(vline, color='black', alpha = 0.7, linewidth='1.5')
                if scale_x1 is not None and scale_x2 is not None:
                    ax.set_xlim(scale_x1, scale_x2)
                ax.set_yticks([])
                if i == n_cases - 1:
                    ax.set_xlabel(marker, fontsize=fontsize)
                else:
                    ax.set_xlabel("")
                if j == 0:
                    ax.set_ylabel(str(case), rotation = 0, ha = 'right', fontsize=fontsize)
                else:
                    ax.set_ylabel("")
                if xticks is not None:
                    ax.set_xticks(xticks)
                    
                ax.tick_params(axis='x', labelsize=fontsize-4)
                    
        handles, labels = axes[0,0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc='center left', bbox_to_anchor=(1.01, 0.5),
            fontsize=20, frameon=False        
        )       
        # Add y-label, x-label, and title to the figure
        fig.text(0.04, 0.5, y_label, va='center', rotation='vertical', fontsize=fontsize or 16)
        fig.text(0.5, 0.04, x_label, ha='center', fontsize=fontsize or 16)
        fig.suptitle(plot_title, fontsize=fontsize or 20, y=1.02)
        plt.tight_layout(rect=[0.05, 0.05, 1, 0.97])
        if saveDir:
            if not os.path.exists(saveDir):
                os.makedirs(saveDir)
            full_path = os.path.join(saveDir, fileName)
            plt.savefig(full_path, dpi=300)
            plt.close()
            print(f"Saved pixel distribution grid to {full_path}")
        else:
            plt.show()
        return


    # auto identify rows and columns in the grid plot
    def calculate_grid_dimensions(num_items, num_columns=None):
        """
        Calculates the number of rows and columns for a square grid
        based on the number of items.
        """
        if num_columns is None:
            num_rows_columns = int(math.ceil(math.sqrt(num_items)))
            return num_rows_columns, num_rows_columns
        else:
            num_rows = int(math.ceil(num_items / num_columns))
            return num_rows, num_columns

    if plotGrid is False:
        # Create a figure and axis object
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        # Loop through each column in the DataFrame and plot a KDE with the
        # user-defined color or the default color (grey)
        if color is None:
            for column in data.columns:
                data[column].plot.kde(ax=ax, label=column)
        else:
            for column in data.columns:
                c = color.get(column, 'grey')
                data[column].plot.kde(ax=ax, label=column, color=c)
        ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=fontsize)
        ax.tick_params(axis='both', which='major', width=1, labelsize=fontsize)
        plt.tight_layout()
        if xticks is not None:
            ax.set_xticks(xticks)
            ax.set_xticklabels([str(x) for x in xticks])

        if vline == 'auto':
            ax.axvline((data[column].max() + data[column].min()) / 2, color='black')
        elif vline is None:
            pass
        else:
            ax.axvline(vline, color='black',linestyle='--')
            
        if scale_x:
            ax.set_xlim(data.min().min(), data.max().max())

        # save figure
        if outputDir is not None:
            plt.savefig(pathlib.Path(outputDir) / outputFileName)

    else:
        # calculate the number of rows and columns
        num_rows, num_cols = calculate_grid_dimensions(
            len(data.columns), num_columns=ncols
        )

        # set colors
        if color is None:
            # Define a color cycle of 10 colors
            color_cycle = itertools.cycle(
                plt.rcParams['axes.prop_cycle'].by_key()['color']
            )
            # Assign a different color to each column
            color = {col: next(color_cycle) for col in data.columns}

        # Set the size of the figure
        fig, axes = plt.subplots(
            nrows=num_rows, ncols=num_cols, figsize=figsize, dpi=dpi
        )
        axes = np.atleast_2d(axes)
        # Set the spacing between subplots
        # fig.subplots_adjust(bottom=0.1, hspace=0.1)

        # Loop through each column in the DataFrame and plot a KDE with the
        # user-defined color or the default color (grey) in the corresponding subplot
        for i, column in enumerate(data.columns):
            c = color.get(column, 'grey')
            row_idx = i // num_cols
            col_idx = i % num_cols
            data[column].plot.kde(ax=axes[row_idx, col_idx], label=column, color=c)
            axes[row_idx, col_idx].set_title(column)
            axes[row_idx, col_idx].tick_params(
                axis='both', which='major', width=1, labelsize=fontsize
            )
            axes[row_idx, col_idx].set_ylabel('')

            if vline == 'auto':
                axes[row_idx, col_idx].axvline(
                    (data[column].max() + data[column].min()) / 2, color='gray', dashes=[2, 2], linewidth=1
                )
                if vlinewidth is not None:
                    axes[row_idx, col_idx].axvline(
                        (data[column].max() + data[column].min()) / 2,color='gray', dashes=[2, 2], linewidth=vlinewidth)
            elif vline is None:
                pass
            else:
                axes[row_idx, col_idx].axvline(vline, color='black')

            if xticks is not None:
                axes[row_idx, col_idx].set_xticks(xticks)
                axes[row_idx, col_idx].set_xticklabels([str(x) for x in xticks])
                
            if scale_x1 is not None and scale_x2 is not None:
                axes[row_idx, col_idx].set_xlim(scale_x1, scale_x2)
            else:    
                axes[row_idx, col_idx].set_xlim(data[column].min(), data[column].max())
                

        # Remove any empty subplots
        num_plots = len(data.columns)
        for i in range(num_plots, num_rows * num_cols):
            row_idx = i // num_cols
            col_idx = i % num_cols
            fig.delaxes(axes[row_idx, col_idx])

        # Set font size for tick labels on both axes
        plt.tick_params(axis='both', labelsize=fontsize)
        plt.tight_layout()

        # Save the figure to a file
        if saveDir:
            if not os.path.exists(saveDir):
                os.makedirs(saveDir)
            full_path = os.path.join(saveDir, fileName)
            plt.savefig(full_path, dpi=300)
            plt.close()
            print(f"Saved heatmap to {full_path}")
        else:
            plt.show()
            
#####################################################################

def normalize_to_gates(adata):
    gates = adata.uns['gates']
    # Select the 2nd column by position, not by name
    value_cols = [col for col in gates.columns if re.search("Case", col, re.IGNORECASE)]
    if not value_cols:
        raise ValueError("No column containing 'Case' found in gates DataFrame.")
    gates_series = gates[value_cols[0]]
    gates_dict = gates_series.to_dict()
    var_names = list(adata.var_names)
    gate_vector = np.array([gates_dict.get(marker, 0) for marker in var_names])
    
    # Verbose output: print the gate value for each marker
    print("Subtracting the following gate values from each marker:")
    for marker, gate_val in zip(var_names, gate_vector):
        print(f"  {marker}: {gate_val}")
    
    adata.layers["log_background_normalized"] = (adata.layers["log1p_raw"] - gate_vector)
    print("Background normalization complete. Layer 'log_background_normalized' added.")