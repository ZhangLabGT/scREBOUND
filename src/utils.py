import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd


# ColorMap
# Define the colormap from grey to white to purple
colors = [(0.7, 0.7, 0.7), (1, 1, 1), (1, 0, 0)]  # Grey to White to Purple
cmap_name = 'grey_to_white_to_purple'
cmap_expr = LinearSegmentedColormap.from_list(cmap_name, colors)
green_red = LinearSegmentedColormap.from_list('green_to_red', [(0, 0.7, 0), (1, 1, 1), (1, 0, 0), (0.5, 0, 0.5)], N=500)
SUPER_MAGMA = LinearSegmentedColormap.from_list('super_magma', colors=['#e0e0e0', '#dedede', '#fff68f', '#ffec8b', '#ffc125', '#ee7600', '#ee5c42', '#cd3278', '#c71585', '#68228b'], N=500)


def plot_embeds(embed, annos, figsize = (20,10), axis_label = "Latent", label_inplace = False, legend = True, **kwargs):
    """\
    Description:
    ----------------
        Plot latent space, 
    Parameters
        embed:
            the cell embedding, of the shape (ncells, nembeds)
        annos:
            the data frame of cluster annotations of the cells in `embed'
        save:
            file name for the figure
        figsize:
            figure size

    """
    _kwargs = {
        "s": 10,
        "alpha": 0.5,
        "markerscale": 1,
        "text_size": "large",
        "ncols": None,
        "colormap": None,
        "vmax": None,
    }
    _kwargs.update(kwargs)

    if _kwargs["ncols"] is None:
        ncols = annos.shape[1]
    else:
        ncols = _kwargs["ncols"]
    nrows = int(np.ceil(annos.shape[1]/ncols))

    fig = plt.figure(figsize = (figsize[0] * ncols, figsize[1] * nrows), dpi = 300, constrained_layout=True)
    axs = fig.subplots(nrows = nrows, ncols = ncols)
    for idx, anno_name in enumerate(annos.columns):

        if (nrows == 1) & (ncols == 1): 
            ax = axs
        elif (nrows == 1) | (ncols == 1):
            ax = axs[idx]
        else:
            ax = axs[idx//_kwargs["ncols"], idx%_kwargs["ncols"]]

        if isinstance(annos[anno_name].dtype, pd.CategoricalDtype):
            # categorical values
            unique_anno = annos[anno_name].cat.categories
            anno = np.array(annos[anno_name].values)
            
            if _kwargs["colormap"] is None:
                colormap = plt.cm.get_cmap("tab20", len(unique_anno))
            else:
                colormap = _kwargs["colormap"]

            texts = []
            for i, cluster_type in enumerate(unique_anno):
                # print(np.where(anno == cluster_type)[0])
                embed_clust = embed[np.where(anno == cluster_type)[0],:]
                ax.scatter(embed_clust[:,0], embed_clust[:,1], color = colormap(i), label = cluster_type, s = _kwargs["s"], alpha = _kwargs["alpha"])
                # text on plot
                if label_inplace:
                    texts.append(ax.text(np.median(embed_clust[:,0]), np.median(embed_clust[:,1]), color = "black", s = unique_anno[i], fontsize = _kwargs["text_size"], weight = 'semibold', in_layout = True))

            if legend:
                leg = ax.legend(loc='upper left', prop={'size': 15}, frameon = False, ncol = (len(unique_anno) // 10) + 1, bbox_to_anchor=(1.04, 1), markerscale = _kwargs["markerscale"])
                for lh in leg.legend_handles: 
                    lh.set_alpha(1)
        else:
            # continuous values
            if _kwargs["colormap"] is None:
                colormap = SUPER_MAGMA
            else:
                colormap = _kwargs["colormap"]
            anno = np.array(annos[anno_name].values)
            p = ax.scatter(embed[:,0], embed[:,1], c = anno, cmap = colormap, s = _kwargs["s"], alpha = _kwargs["alpha"])

            cbar = fig.colorbar(p, fraction=0.046, pad=0.04, ax = ax)
            if _kwargs["vmax"] is not None:
                p.set_clim(0, _kwargs["vmax"][idx])
                cbar.update_normal(p)
            cbar.ax.tick_params(labelsize = 20)

        ax.tick_params(axis = "both", which = "major", labelsize = 15)

        ax.set_xlabel(axis_label + " 1", fontsize = 19)
        ax.set_ylabel(axis_label + " 2", fontsize = 19)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_title(anno_name, fontsize = 20)
        # adjust position
        # if label_inplace:
        #     adjust_text(texts, only_move={'points':'xy', 'texts':'xy'})

    return fig


def plot_by_batch(x_rep, annos = None, batches = None, figsize = (20,10), axis_label = "Latent", label_inplace = False, legend = True, **kwargs):
    """\
    Description:
    -------------
        Plot the cell embedding (x_rep) by batches

    Parameters:
    -------------
        x_rep:
            cell representation (ncells, ndims)
        annos:
            cell annotation (ncells,)
        batches:
            cell batches (n_samples,)
        save
            file name for the figure
        figsize
            figure size
    """
    _kwargs = {
        "s": 10,
        "alpha": 0.7,
        "markerscale": 1,
        "text_size": "large",
        "colormap": None,
        "ncols":1
    }
    _kwargs.update(kwargs)

    unique_batch = np.unique(batches)
    unique_cluster = np.unique(annos) 

    nrows = int(np.ceil(len(unique_batch)/_kwargs["ncols"]))
    ncols = _kwargs["ncols"]    
    fig = plt.figure(figsize = (figsize[0] * ncols, figsize[1] * nrows), dpi = 300, constrained_layout=True)

    axs = fig.subplots(nrows = nrows, ncols = ncols)

    # load colormap for annos
    if _kwargs["colormap"] is None:
        colormap = plt.cm.get_cmap("tab20b", len(unique_cluster))
    else:
        colormap = _kwargs["colormap"]

    # loop through unique batches
    for idx, batch in enumerate(unique_batch):
        x_batch = x_rep[batches == batch]
        annos_batch = annos[batches == batch]
        texts = []
        for j, cluster_type in enumerate(unique_cluster):
            index = np.where(annos_batch == cluster_type)[0]
            if len(index) > 0:
                if (nrows == 1) & (ncols == 1): 
                    ax = axs
                elif (nrows == 1) | (ncols == 1):
                    ax = axs[idx]
                else:
                    ax = axs[idx//_kwargs["ncols"], idx%_kwargs["ncols"]]
                ax.scatter(x_batch[index,0], x_batch[index,1], color = colormap(j), label = cluster_type, s = _kwargs["s"], alpha = _kwargs["alpha"])
                # text on plot
                if label_inplace:
                    # if exist cells
                    if x_batch[index,0].shape[0] > 0:
                        texts.append(ax.text(np.median(x_batch[index,0]), np.median(x_batch[index,1]), color = "black", s = cluster_type, fontsize = _kwargs["text_size"], weight = 'semibold', in_layout = True))
        
        if legend:
            leg = ax.legend(loc='upper left', prop={'size': 15}, frameon = False, ncol = (len(unique_cluster) // 15) + 1, bbox_to_anchor=(1.04, 1), markerscale = _kwargs["markerscale"])
            for lh in leg.legend_handles: 
                lh.set_alpha(1)

            
        ax.set_title(batch, fontsize = 25)

        ax.tick_params(axis = "both", which = "major", labelsize = 15)

        ax.set_xlabel(axis_label + " 1", fontsize = 19)
        ax.set_ylabel(axis_label + " 2", fontsize = 19)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)  

        ax.set_xlim(np.min(x_batch[:,0]), np.max(x_batch[:,0]))
        ax.set_ylim(np.min(x_batch[:,1]), np.max(x_batch[:,1]))
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

        # if label_inplace:
        #     adjust_text(texts, only_move={'points':'xy', 'texts':'xy'})        
        plt.tight_layout()
    return fig


def plot_volcano(df, x = "logFC", y = "adj.P.Val", x_cutoffs = [-np.inf, 2], y_cutoff = 0.01, gene_name = True, ylim = None, xlim = None, figsize = (10, 7)):
    from adjustText import adjust_text
    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot()
    if ylim is None:
        ylim = np.inf
    if xlim is None:
        xlim = np.inf
    else:
        df.loc[df[x] > xlim, x] = xlim
        df.loc[df[x] < -xlim, x] = -xlim

    ax.scatter(x = df[x], y = df[y].apply(lambda x:-np.log10(max(x, ylim))), s = 1, color = "gray")#, label = "Not significant")
    
    # highlight down- or up- regulated genes
    down = df[(df[x] <= x_cutoffs[0]) & (df[y] <= y_cutoff)]
    up = df[(df[x] >= x_cutoffs[1]) & (df[y] <= y_cutoff)]
    ax.scatter(x = down[x], y = down[y].apply(lambda x:-np.log10(max(x, ylim))), s = 3, label = "Down-regulated", color = "blue")
    ax.scatter(x = up[x], y = up[y].apply(lambda x:-np.log10(max(x, ylim))), s = 3, label = "Up-regulated", color = "red")

    # add legends
    ax.set_xlabel("logFC", fontsize = 15)
    ax.set_ylabel("-logPVal", fontsize = 15)
    ax.set_xlim([-np.max(np.abs(df[x].values)) - 0.5, np.max(np.abs(df[x].values)) + 0.5])
    # ax.set_ylim(-3, 50)
    if len(down) > 0:
        ax.axvline(x_cutoffs[0], color = "grey", linestyle = "--")
    if len(up) > 0:
        ax.axvline(x_cutoffs[1], color = "grey", linestyle = "--")
    ax.axhline(-np.log10(y_cutoff), color = "grey", linestyle = "--")
    leg = ax.legend(loc='upper left', prop={'size': 15}, frameon = False, bbox_to_anchor=(1.04, 1), markerscale = 3)
    for lh in leg.legend_handles: 
        lh.set_alpha(1)

    # add gene names
    if gene_name:
        texts = []
        for i,r in down.iterrows():
            texts.append(plt.text(x = r[x], y = -np.log10(max(r[y], ylim)), s = i, fontsize = 7))

        for i,r in up.iterrows():
            texts.append(plt.text(x = r[x], y = -np.log10(max(r[y], ylim)), s = i, fontsize = 7))
        # # optional, adjust text
        adjust_text(texts)#,arrowprops=dict(arrowstyle="-", color='black', lw=0.5))
    
    return fig, ax