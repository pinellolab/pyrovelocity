from pyrovelocity._velocity import PyroVelocity
from pyro.infer.autoguide.guides import AutoGuideList
from ._velocity_guide import VelocityAutoGuideList
from pyro.infer.autoguide import AutoDelta, AutoNormal, init_to_mean, AutoLowRankMultivariateNormal, AutoDiagonalNormal, AutoDiscreteParallel
from pyro import poutine
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


def train_model(adata,
                guide_type='auto',
                model_type='auto',
                svi_train=False, # svi_train alreadys turn off
                batch_size=-1, train_size=1.0, use_gpu=0,
                likelihood='Poisson', num_samples=30, log_every=100,
                cell_state='clusters', patient_improve=5e-4, patient_init=30,
                seed=99, lr=0.01, max_epochs=3000,
                include_prior=True,
                library_size=True, offset=False, input_type='raw',
                cell_specific_kinetics=None, kinetics_num=2):
    model = PyroVelocity(adata, likelihood=likelihood,
                         model_type=model_type,
                         guide_type=guide_type, correct_library_size=library_size,
                         add_offset=offset,
                         include_prior=include_prior, input_type=input_type, 
                         cell_specific_kinetics=cell_specific_kinetics, kinetics_num=kinetics_num)
    if svi_train and (guide_type=='velocity_auto' or guide_type == 'velocity_auto_t0_constraint'):
        if batch_size == -1:
            batch_size = adata.shape[0]
        model.train(max_epochs=max_epochs, lr=lr,
                    use_gpu=use_gpu, batch_size=batch_size,
                    train_size=train_size, valid_size=1-train_size,
                    check_val_every_n_epoch=1,
                    early_stopping=True,
                    patience=patient_init,
                    min_delta=patient_improve)
        import pandas as pd
        fig, ax = plt.subplots()
        fig.set_size_inches(2.5, 1.5)
        ax.scatter(model.history_['elbo_train'].index[:-1],
                   -model.history_['elbo_train'][:-1], label="Train")
        if train_size < 1:
            ax.scatter(model.history_['elbo_validation'].index[:-1],
                       -model.history_['elbo_validation'][:-1], label="Valid")
        #ax.set_yscale('log')
        ax.set_yscale('symlog')
        ax.set_xlabel("Epochs")
        ax.set_ylabel("-ELBO")
        pos = model.posterior_samples(model.adata,
                                      num_samples=num_samples,
                                      batch_size=512)
        return model, pos
    else:
        if train_size >= 1: ##support velocity_auto_depth
            if batch_size == -1:
                batch_size = adata.shape[0]

            if batch_size >= adata.shape[0]:
                losses = model.train_faster(max_epochs=max_epochs, lr=lr, use_gpu=use_gpu, seed=seed, patient_improve=patient_improve, patient_init=patient_init, log_every=log_every)
            else:
                losses = model.train_faster_with_batch(max_epochs=max_epochs,
                                                       batch_size=batch_size, log_every=log_every,
                                                       lr=lr, use_gpu=use_gpu, seed=seed, patient_improve=patient_improve, patient_init=patient_init)
            fig, ax = plt.subplots()
            fig.set_size_inches(2.5, 1.5)
            ax.scatter(np.arange(len(losses)), -np.array(losses), label='train', alpha=0.25)
            ax.set_yscale('symlog')
            ax.set_xlabel("Epochs")
            ax.set_ylabel("-ELBO")

            pos = model.posterior_samples(model.adata,
                                          num_samples=num_samples,
                                          batch_size=512)
            return model, pos
        else: # train validation procedure
            if guide_type == 'velocity_auto_depth': # velocity_auto, not support (velocity_auto_depth, failure by error)
                raise

            indices = np.arange(adata.shape[0])
            train_ind, test_ind, cluster_train, cluster_test = train_test_split(indices, adata.obs.loc[:, cell_state].values, test_size=1-train_size, random_state=seed, shuffle=False)
            if batch_size == -1:
                train_batch_size = train_ind.shape[0]
            else:
                train_batch_size = batch_size
            losses = model.train_faster_with_batch(max_epochs=max_epochs,
                                                   batch_size=train_batch_size,
                                                   indices=train_ind, log_every=log_every,
                                                   lr=lr, use_gpu=use_gpu, seed=seed, patient_improve=patient_improve, patient_init=patient_init)
            pos = model.posterior_samples(model.adata,
                                          num_samples=num_samples,
                                          indices=train_ind,
                                          batch_size=512)

            if batch_size == -1:
                test_batch_size = test_ind.shape[0]
            else:
                test_batch_size = batch_size

            if guide_type == 'auto' or guide_type == 'auto_t0_constraint':
                new_guide = AutoGuideList(model.module._model, create_plates=model.module._model.create_plates)
                new_guide.append(AutoNormal(poutine.block(model.module._model, expose=['cell_time', 'u_read_depth', 's_read_depth']), init_scale=0.1))
                new_guide.append(poutine.block(model.module._guide[-1], hide_types=['param']))
                losses_test = model.train_faster_with_batch(max_epochs=max_epochs, batch_size=test_batch_size,
                                                            indices=test_ind, new_valid_guide=new_guide, log_every=log_every,
                                                            lr=lr, use_gpu=use_gpu, seed=seed)
            elif guide_type == 'velocity_auto' or guide_type == 'velocity_auto_t0_constraint': # velocity_auto, not support (velocity_auto_depth, failure by error)
                print('valid new guide')
                #### not valid guide 
                ##new_guide = model.module._guide
                ##new_guide[0] = AutoNormal(poutine.block(model.module._model, expose=['u_read_depth', 's_read_depth']), init_scale=0.1)
                ##new_guide[-1] = poutine.block(model.module._guide[-1], hide_types=['param'])
                ##losses_test = model.train_faster_with_batch(max_epochs=max_epochs, batch_size=batch_size,
                ##                                            indices=test_ind, new_valid_guide=new_guide,
                ##                                            lr=lr, use_gpu=use_gpu, seed=seed)

                ## neural network guide for read depth
                losses_test = model.train_faster_with_batch(max_epochs=max_epochs, batch_size=test_batch_size,
                                                            indices=test_ind, log_every=log_every,
                                                            lr=lr, use_gpu=use_gpu, seed=seed)
            else:
                raise
            pos_test = model.posterior_samples(model.adata,
                                               num_samples=30,
                                               indices=test_ind,
                                               batch_size=512)
            fig, ax = plt.subplots()
            fig.set_size_inches(2.5, 1.5)
            ax.scatter(np.arange(len(losses)), -np.array(losses), label='train', alpha=0.25)
            ax.scatter(np.arange(len(losses_test)), -np.array(losses_test), label='validation', alpha=0.25)
            #ax.set_yscale('log')
            ax.set_yscale('symlog')
            ax.set_xlabel("Epochs")
            ax.set_ylabel("-ELBO")
        plt.legend()
        return pos, pos_test, train_ind, test_ind
