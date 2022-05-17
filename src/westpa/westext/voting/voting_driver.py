import logging

import numpy as np
import pandas as pd

import westpa

log = logging.getLogger(__name__)


class VotingDriver:
    '''
    This plugin implements an adaptive scheme using voronoi bins from
    Zhang 2010, J Chem Phys, 132. The options exposed to the configuration
    file are:

      - av_enabled (bool, default False): Enables adaptive binning
      - max_centers (int, default 10): The maximum number of voronoi centers to be placed
      - walk_count (integer, default 5): Number of walkers per voronoi center
      - center_freq (ingeter, default 1): Frequency of center placement
      - priority (integer, default 1): Priority in the plugin order
      - dfunc_method (function, non-optional, no default): Non-optional user defined
          function that will be used to calculate distances between voronoi centers and
          data points
      - mapper_func (function, optional): Optional user defined function for building bin
          mappers for more complicated binning schemes e.g. embedding the voronoi binning
          in a portion of the state space. If not defined the plugin will build a
          VoronoiBinMapper with the information it has.
    '''

    def __init__(self, sim_manager, plugin_config):

        if not sim_manager.work_manager.is_master:
            return

        self.sim_manager = sim_manager
        self.data_manager = sim_manager.data_manager
        self.system = sim_manager.system

        # Parameters from config file
        # enable the plugin
        self.do_voting = plugin_config.get('do_voting', False)
        # whether to use weighted votes
        self.use_weights = plugin_config.get('use_weights', False)
        # directional indicators
        self.pct_change_direct = plugin_config.get('pct_change_direct', None)
        # decision equation to apply to votes
        self.decision_eq = plugin_config.get('decision_eq', 'linear')
        # priority of the plugin (allows for order of execution)
        self.priority = plugin_config.get('priority', 0)

        # Register callback
        if self.do_voting:
            sim_manager.register_callback(sim_manager.pre_we, self.pre_we, self.priority)

    def pre_we(self):
        segments = westpa.rc.sim_manager.segments
        data = segments[0].data
        votes = np.zeros(len(segments))
        weights = np.zeros(len(segments))
        #        print(len(segments), len(data))
        pcoord_len = segments[0].pcoord.shape[0]
        #        print(self.pct_change_direct)
        for idx in segments:
            data = segments[idx].data
            df = pd.DataFrame.from_dict(data)
            #            print(df.head)
            pct_change = df.pct_change(periods=pcoord_len - 1)
            # print(pct_change)
            pct_change *= self.pct_change_direct
            # sorted_list = pct_change.iloc[-1].sort_values(ascending=False)
            #            print(sorted_list)
            values = pct_change.iloc[-1].values
            index = np.argmax(values)
            votes[idx] = index
            #            print(index, values[index])
            weight = segments[idx].weight
            weights[idx] = weight

        #        print("votes:", votes)
        #        print("weights:", weights)

        data_names = list(segments[0].data)
        print("voting results:")
        for vidx, vote in enumerate(votes):
            print(data_names[int(vote)], weights[vidx])

        if not self.use_weights:
            weights = np.ones(len(segments))

        for idx in segments:
            new_pcoord = 0
            data_names = list(segments[idx].data)
            for ivote, vote in enumerate(votes):
                vote_name = data_names[int(vote)]
                new_pcoord += segments[idx].data[vote_name] * weights[ivote]
            #                print(vote_name, segments[idx].data[vote_name], weights[ivote], new_pcoord)
            #            print(idx, new_pcoord)
            old_pcoord = segments[idx].pcoord[:, 1].reshape(pcoord_len, 1)
            new_pcoord = new_pcoord.reshape(pcoord_len, 1)
            combined_pcoord = np.concatenate((new_pcoord, old_pcoord), axis=1)
            segments[idx].pcoord = combined_pcoord
            # print(idx, segments[idx].pcoord)


#    def dfunc(self):
#        '''
#        Distance function to be used by the plugin. This function
#        will be used to calculate the distance between each point.
#        '''
#        raise NotImplementedError
#
#    def get_dfunc_method(self, plugin_config):
#        try:
#            methodname = plugin_config['dfunc_method']
#        except KeyError:
#            raise ConfigItemMissing('dfunc_method')
#
#        dfunc_method = extloader.get_object(methodname)
#
#        log.info('loaded adaptive voronoi dfunc method {!r}'.format(dfunc_method))
#
#        return dfunc_method
#
#    def get_mapper_func(self, plugin_config):
#        try:
#            methodname = plugin_config['mapper_func']
#        except KeyError:
#            return False
#
#        mapper_func = extloader.get_object(methodname)
#
#        log.info('loaded adaptive voronoi mapper function {!r}'.format(mapper_func))
#
#        return mapper_func
#
#    def get_initial_centers(self):
#        '''
#        This function pulls from the centers from either the
#        previous bin mapper  or uses the definition from the
#        system to calculate the number of centers
#        '''
#        self.data_manager.open_backing()
#
#        with self.data_manager.lock:
#            n_iter = max(self.data_manager.current_iteration - 1, 1)
#            iter_group = self.data_manager.get_iter_group(n_iter)
#
#            # First attempt to initialize voronoi centers
#            # from data rather than system
#            centers = None
#            try:
#                log.info('Voronoi centers from previous bin mapper')
#                binhash = iter_group.attrs['binhash']
#                bin_mapper = self.data_manager.get_bin_mapper(binhash)
#
#                centers = bin_mapper.centers
#
#            except Exception:
#                log.warning(
#                    'Initializing voronoi centers from data failed; \
#                        Using definition in system instead.'
#                )
#                centers = self.system.bin_mapper.centers
#
#        self.data_manager.close_backing()
#        return centers
#
#    def update_bin_mapper(self):
#        '''Update the bin_mapper using the current set of voronoi centers'''
#
#        westpa.rc.pstatus('westext.adaptvoronoi: Updating bin mapper\n')
#        westpa.rc.pflush()
#
#        # self.mapper_func = plugin_config.get('mapper_func', False)
#        try:
#            dfargs = getattr(self.system, 'dfargs', None)
#            dfkwargs = getattr(self.system, 'dfkwargs', None)
#            if self.mapper_func:
#                # The mapper should take in 1) distance function,
#                # 2) centers, 3) dfargs, 4) dfkwargs and return
#                # the mapper we want
#                self.system.bin_mapper = self.mapper_func(self.dfunc, self.centers, dfargs=dfargs, dfkwargs=dfkwargs)
#            else:
#                self.system.bin_mapper = VoronoiBinMapper(self.dfunc, self.centers, dfargs=dfargs, dfkwargs=dfkwargs)
#            self.ncenters = self.system.bin_mapper.nbins
#            new_target_counts = np.empty((self.ncenters,), np.int)
#            new_target_counts[...] = self.walk_count
#            self.system.bin_target_counts = new_target_counts
#        except (ValueError, TypeError) as e:
#            log.error(
#                'AdaptiveVoronoiDriver Error: \
#                    Failed updating the bin mapper: {}'.format(
#                    e
#                )
#            )
#            raise
#
#    def update_centers(self, iter_group):
#        '''
#        Update the set of Voronoi centers according to
#        Zhang 2010, J Chem Phys, 132. A short description
#        of the algorithm can be found in the text:
#
#        1) First reference structure is chosen randomly from
#        the first set of given structure
#        2) Given a set of n reference structures, for each
#        configuration in the iteration the distances to each
#        reference structure is calculated and the minimum
#        distance is found
#        3) The configuration with the minimum distance is
#        selected as the next reference
#        '''
#
#        westpa.rc.pstatus('westext.adaptvoronoi: Updating Voronoi centers\n')
#        westpa.rc.pflush()
#
#        # Pull the current coordinates to find distances
#        curr_pcoords = iter_group['pcoord']
#        # Initialize distance array
#        dists = np.zeros(curr_pcoords.shape[0])
#        for iwalk, walk in enumerate(curr_pcoords):
#            # Calculate distances using the provided function
#            # and find the distance to the closest center
#            dists[iwalk] = min(self.dfunc(walk[-1], self.centers))
#        # Find the maximum of the minimum distances
#        max_ind = np.where(dists == dists.max())
#        # Use the maximum progress coordinate as our next center
#        self.centers = np.vstack((self.centers, curr_pcoords[max_ind[0][0]][-1]))
#
#    def prepare_new_iteration(self):
#
#        n_iter = self.sim_manager.n_iter
#
#        with self.data_manager.lock:
#            iter_group = self.data_manager.get_iter_group(n_iter)
#
#        # Check if we are at the correct frequency for updating the bin mapper
#        if n_iter % self.center_freq == 0:
#            # Check if we still need to add more centers
#            if self.ncenters < self.max_centers:
#                # First find the center to add
#                self.update_centers(iter_group)
#                # Update the bin mapper with the new center
#                self.update_bin_mapper()
