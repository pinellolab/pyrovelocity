import anndata
import numpy as np
from scipy.sparse import issparse


def get_clone_trajectory(
    adata,
    average_start_point=True,
    global_traj=True,
    times=[2, 4, 6],
    clone_num=None,
):
    if not average_start_point:
        adata.obsm["clone_vector_emb"] = np.zeros((adata.shape[0], 2))

    adatas = []
    clones = []
    centroids = []
    cen_clones = []
    print(adata.shape)
    adata.obs["clones"] = 0
    if "noWell" in adata.obs.columns:
        for w in adata.obs.Well.unique():
            adata_w = adata[adata.obs.Well == w]
            clone_adata_w = clone_adata[clone_adata.obs.Well == w]
            for j in range(clone_adata_w.shape[1]):
                adata_w.obs["clonei"] = 0
                # belongs to same clone
                adata_w.obs.loc[
                    clone_adata_w[:, j].X.toarray()[:, 0] >= 1, "clonei"
                ] = 1

                if not average_start_point:
                    for i in np.where(
                        (adata_w.obs.time == 2) & (adata_w.obs.clonei == 1)
                    )[0]:
                        next_time = np.where(
                            (adata_w.obs.time == 4) & (adata_w.obs.clonei == 1)
                        )[0]
                        adata_w.obsm["velocity_umap"][i] = (
                            adata_w.obsm["X_umap"][next_time].mean(axis=0)
                            - adata_w.obsm["X_umap"][i]
                        )
                    for i in np.where(
                        (adata_w.obs.time == 4) & (adata_w.obs.clonei == 1)
                    )[0]:
                        next_time = np.where(
                            (adata_w.obs.time == 6) & (adata_w.obs.clonei == 1)
                        )[0]
                        adata_w.obsm["velocity_umap"][i] = (
                            adata_w.obsm["X_umap"][next_time].mean(axis=0)
                            - adata_w.obsm["X_umap"][i]
                        )
                else:
                    time2 = np.where(
                        (adata_w.obs.time == 2) & (adata_w.obs.clonei == 1)
                    )[0]
                    time4 = np.where(
                        (adata_w.obs.time == 4) & (adata_w.obs.clonei == 1)
                    )[0]
                    time6 = np.where(
                        (adata_w.obs.time == 6) & (adata_w.obs.clonei == 1)
                    )[0]
                    if (
                        time2.shape[0] == 0
                        and time4.shape[0] == 0
                        and time6.shape[0] == 0
                    ):
                        continue
                    if (
                        time2.shape[0] > 0
                        and time4.shape[0] == 0
                        and time6.shape[0] > 0
                    ):
                        continue
                    adata_new = anndata.AnnData(
                        np.vstack(
                            [
                                adata_w[time2].X.toarray().mean(axis=0),
                                adata_w[time4].X.toarray().mean(axis=0),
                                adata_w[time6].X.toarray().mean(axis=0),
                            ]
                        ),
                        layers={
                            "spliced": np.vstack(
                                [
                                    adata_w[time2]
                                    .layers["spliced"]
                                    .toarray()
                                    .mean(axis=0),
                                    adata_w[time4]
                                    .layers["spliced"]
                                    .toarray()
                                    .mean(axis=0),
                                    adata_w[time6]
                                    .layers["spliced"]
                                    .toarray()
                                    .mean(axis=0),
                                ]
                            ),
                            "unspliced": np.vstack(
                                [
                                    adata_w[time2]
                                    .layers["unspliced"]
                                    .toarray()
                                    .mean(axis=0),
                                    adata_w[time4]
                                    .layers["unspliced"]
                                    .toarray()
                                    .mean(axis=0),
                                    adata_w[time6]
                                    .layers["unspliced"]
                                    .toarray()
                                    .mean(axis=0),
                                ]
                            ),
                        },
                        var=adata_w.var,
                    )

                    adata_new.obs.loc[:, "time"] = [2, 4, 6]
                    adata_new.obs.loc[:, "Cell type annotation"] = "Centroid"
                    print(adata_w[time6].obs.clonetype.unique())
                    print(adata_w[time6].obs)

                    adata_new.obs.loc[:, "clonetype"] = (
                        adata_w[time6].obs.clonetype.unique()
                    )  # use cell fate from last time point
                    adata_new.obs.loc[:, "clones"] = int(j)
                    if "Well" in adata_w[time6].obs.columns:
                        adata_new.obs.loc[:, "Well"] = adata_w[
                            time6
                        ].obs.Well.unique()

                    adata_new.obsm["X_umap"] = np.vstack(
                        [
                            adata_w[time2].obsm["X_umap"].mean(axis=0),
                            adata_w[time4].obsm["X_umap"].mean(axis=0),
                            adata_w[time6].obsm["X_umap"].mean(axis=0),
                        ]
                    )
                    adata_new.obsm["velocity_umap"] = np.vstack(
                        [
                            adata_w.obsm["X_umap"][time4].mean(axis=0)
                            - adata_w.obsm["X_umap"][time2].mean(axis=0),
                            adata_w.obsm["X_umap"][time6].mean(axis=0)
                            - adata_w.obsm["X_umap"][time4].mean(axis=0),
                            np.zeros(2),
                        ]
                    )
                    centroids.append(adata_new)
                    clone_new = anndata.AnnData(
                        np.vstack(
                            [
                                clone_adata_w[time2].X.toarray().mean(axis=0),
                                clone_adata_w[time4].X.toarray().mean(axis=0),
                                clone_adata_w[time6].X.toarray().mean(axis=0),
                            ]
                        ),
                        obs=adata_new.obs,
                    )
                    clone_new.var_names = clone_adata.var_names
                    clone_new.var = clone_adata.var
                    # print(clone_new.shape)
                    cen_clones.append(clone_new)

            adata_new = adata_w.concatenate(
                centroids[0].concatenate(centroids[1:]), join="outer"
            )
            clone_new = clone_adata_w.concatenate(
                cen_clones[0].concatenate(cen_clones[1:]), join="outer"
            )
            adatas.append(adata_new)
            clones.append(clone_new)
        return adatas[0].concatenate(adatas[1]), clones[0].concatenate(
            clones[1]
        )
    else:
        if clone_num is None:
            clone_num = adata.obsm["X_clone"].shape[1]
        for j in range(clone_num):
            print(j)
            adata.obs["clonei"] = 0
            # print('----------aa------')
            if issparse(adata.obsm["X_clone"]):
                adata.obs.loc[
                    adata.obsm["X_clone"].toarray()[:, j] >= 1, "clonei"
                ] = 1
            else:
                adata.obs.loc[adata.obsm["X_clone"][:, j] >= 1, "clonei"] = 1
            # print('----------bb------')

            if not average_start_point:
                for i in np.where(
                    (adata.obs.time == 2) & (adata.obs.clonei == 1)
                )[0]:
                    next_time = np.where(
                        (adata.obs.time == 4) & (adata.obs.clonei == 1)
                    )[0]
                    adata.obsm["velocity_umap"][i] = (
                        adata.obsm["X_umap"][next_time].mean(axis=0)
                        - adata.obsm["X_umap"][i]
                    )
                for i in np.where(
                    (adata.obs.time == 4) & (adata.obs.clonei == 1)
                )[0]:
                    next_time = np.where(
                        (adata.obs.time == 6) & (adata.obs.clonei == 1)
                    )[0]
                    adata.obsm["velocity_umap"][i] = (
                        adata.obsm["X_umap"][next_time].mean(axis=0)
                        - adata.obsm["X_umap"][i]
                    )
            else:
                if global_traj:
                    times_index = []
                    for t in times:
                        times_index.append(
                            np.where(
                                (adata.obs.time_info == t)
                                & (adata.obs.clonei == 1)
                            )[0]
                        )

                    consecutive_flag = np.array(
                        [int(time.shape[0] > 0) for time in times_index]
                    )
                    consecutive = np.diff(consecutive_flag)
                    if np.sum(consecutive_flag == 1) >= 2 and np.any(
                        consecutive == 0
                    ):  # Must be consecutive time points
                        # print('centroid:', consecutive, times_index)
                        adata_new = anndata.AnnData(
                            np.vstack(
                                [
                                    np.array(
                                        adata[time].X.mean(axis=0)
                                    ).squeeze()
                                    for time in times_index
                                    if time.shape[0] > 0
                                ]
                            ),
                            # layers={
                            #     "spliced": np.vstack(
                            #         [
                            #             np.array(
                            #                 adata[time]
                            #                 .layers["spliced"]
                            #                 .mean(axis=0)
                            #             )
                            #             for time in times_index
                            #             if time.shape[0] > 0
                            #         ]
                            #     ),
                            #     "unspliced": np.vstack(
                            #         [
                            #             np.array(
                            #                 adata[time]
                            #                 .layers["unspliced"]
                            #                 .mean(axis=0)
                            #             )
                            #             for time in times_index
                            #             if time.shape[0] > 0
                            #         ]
                            #     ),
                            # },
                            var=adata.var,
                        )
                        # print('----------cc------')
                        adata.obs.iloc[
                            np.hstack(
                                [
                                    time
                                    for time in times_index
                                    if time.shape[0] > 0
                                ]
                            ),
                            adata.obs.columns.get_loc("clones"),
                        ] = int(j)
                        adata_new.obs.loc[:, "time"] = [
                            t
                            for t, time in zip([2, 4, 6], times_index)
                            if time.shape[0] > 0
                        ]
                        adata_new.obs.loc[:, "clones"] = int(j)
                        adata_new.obs.loc[:, "state_info"] = "Centroid"
                        adata_new.obsm["X_emb"] = np.vstack(
                            [
                                adata[time].obsm["X_emb"].mean(axis=0)
                                for time in times_index
                                if time.shape[0] > 0
                            ]
                        )
                        # print('----------dd------')

                        # print(adata_new.shape)
                        # print(adata_new.obsm['X_umap'])
                        adata_new.obsm["clone_vector_emb"] = np.vstack(
                            [
                                adata_new.obsm["X_emb"][i + 1]
                                - adata_new.obsm["X_emb"][i]
                                for i in range(
                                    adata_new.obsm["X_emb"].shape[0] - 1
                                )
                            ]
                            + [np.zeros(2)]
                        )
                        # print('----------ee------')
                        # print(adata_new.obsm['clone_vector_emb'])
                    else:
                        # print('pass-------')
                        continue

                else:
                    time2 = np.where(
                        (adata.obs.time == t) & (adata.obs.clonei == 1)
                    )[0]
                    time4 = np.where(
                        (adata.obs.time == 4) & (adata.obs.clonei == 1)
                    )[0]
                    time6 = np.where(
                        (adata.obs.time == 6) & (adata.obs.clonei == 1)
                    )[0]
                    adata_new = anndata.AnnData(
                        np.vstack(
                            [
                                adata[time2].X.toarray().mean(axis=0),
                                adata[time4].X.toarray().mean(axis=0),
                                adata[time6].X.toarray().mean(axis=0),
                            ]
                        ),
                        layers={
                            "spliced": np.vstack(
                                [
                                    adata[time2]
                                    .layers["spliced"]
                                    .toarray()
                                    .mean(axis=0),
                                    adata[time4]
                                    .layers["spliced"]
                                    .toarray()
                                    .mean(axis=0),
                                    adata[time6]
                                    .layers["spliced"]
                                    .toarray()
                                    .mean(axis=0),
                                ]
                            ),
                            "unspliced": np.vstack(
                                [
                                    adata[time2]
                                    .layers["unspliced"]
                                    .toarray()
                                    .mean(axis=0),
                                    adata[time4]
                                    .layers["unspliced"]
                                    .toarray()
                                    .mean(axis=0),
                                    adata[time6]
                                    .layers["unspliced"]
                                    .toarray()
                                    .mean(axis=0),
                                ]
                            ),
                        },
                        var=adata.var,
                    )

                    print(adata_new.X.sum(axis=1))
                    adata_new.obs.loc[:, "time"] = [2, 4, 6]
                    adata_new.obs.loc[:, "Cell type annotation"] = "Centroid"
                    if not global_traj:
                        adata_new.obs.loc[:, "clonetype"] = (
                            adata[time6].obs.clonetype.unique()
                        )  # use cell fate from last time point
                    adata_new.obs.loc[:, "clones"] = j

                    if "noWell" in adata[time6].obs.columns:
                        adata_new.obs.loc[:, "Well"] = adata[
                            time6
                        ].obs.Well.unique()

                    adata_new.obsm["X_umap"] = np.vstack(
                        [
                            adata[time2].obsm["X_umap"].mean(axis=0),
                            adata[time4].obsm["X_umap"].mean(axis=0),
                            adata[time6].obsm["X_umap"].mean(axis=0),
                        ]
                    )
                    adata_new.obsm["velocity_umap"] = np.vstack(
                        [
                            adata.obsm["X_umap"][time4].mean(axis=0)
                            - adata.obsm["X_umap"][time2].mean(axis=0),
                            adata.obsm["X_umap"][time6].mean(axis=0)
                            - adata.obsm["X_umap"][time4].mean(axis=0),
                            np.zeros(2),
                        ]
                    )

                    # print(adata_new.obsm['velocity_umap'])
                    clone_new = anndata.AnnData(
                        np.vstack(
                            [
                                clone_adata[time2].X.toarray().mean(axis=0),
                                clone_adata[time4].X.toarray().mean(axis=0),
                                clone_adata[time6].X.toarray().mean(axis=0),
                            ]
                        ),
                        obs=adata_new.obs,
                    )
                    clone_new.var_names = clone_adata.var_names
                    clone_new.var = clone_adata.var
                    cen_clones.append(clone_new)
                centroids.append(adata_new)
        print(adata.shape)
        print(len(centroids))
        adata_new = adata.concatenate(
            centroids[0].concatenate(centroids[1:]), join="outer"
        )
        return adata_new


def align_trajectory_diff(
    adatas,
    velocity_embeds,
    density=0.3,
    smooth=0.5,
    input_grid=None,
    input_scale=None,
    min_mass=1.0,
    embed="umap",
    autoscale=False,
    length_cutoff=10,
):
    from scipy.stats import norm as normal
    from scvelo.tools.velocity_embedding import quiver_autoscale
    from sklearn.neighbors import NearestNeighbors

    if input_grid is None and input_scale is None:
        grs = []
        # align embedding points into shared grid across adata
        X_emb = np.vstack([a.obsm[f"X_{embed}"] for a in adatas])
        for dim_i in range(2):
            m, M = np.min(X_emb[:, dim_i]), np.max(X_emb[:, dim_i])
            m = m - 0.01 * np.abs(M - m)
            M = M + 0.01 * np.abs(M - m)
            gr = np.linspace(m, M, int(50 * density))
            grs.append(gr)

        meshes_tuple = np.meshgrid(*grs)
        scale = np.mean([(g[1] - g[0]) for g in grs]) * smooth
        X_grid = np.vstack([i.flat for i in meshes_tuple]).T
    else:
        scale = input_scale
        X_grid = input_grid

    n_neighbors = int(max([a.shape[0] for a in adatas]) / 50)

    results = [X_grid]
    p_mass_list = []
    for adata, velocity_embed in zip(adatas, velocity_embeds):
        nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
        nn.fit(adata.obsm[f"X_{embed}"])
        dists, neighs = nn.kneighbors(X_grid)
        weight = normal.pdf(x=dists, scale=scale)
        # how many cells around a grid points
        p_mass = weight.sum(1)
        V_grid = (velocity_embed[neighs] * weight[:, :, None]).sum(
            1
        ) / np.maximum(1, p_mass)[:, None]
        if autoscale:
            V_grid /= 3 * quiver_autoscale(X_grid, V_grid)
        results.append(V_grid)
        p_mass_list.append(p_mass)

    from functools import reduce

    if input_grid is None and input_scale is None:
        min_mass *= np.percentile(np.hstack(p_mass_list), 99) / 100
        mass_index = reduce(
            np.intersect1d,
            [np.where(p_mass > min_mass)[0] for p_mass in p_mass_list],
        )

    results = np.hstack(results)
    results = results[mass_index]
    print(results.shape)
    length_filter = np.sqrt((results[:, 2:4] ** 2).sum(1)) > length_cutoff
    return results[length_filter]
