import logging

import numpy as np
import pandas as pd

import math
from collections import Counter
import westpa

log = logging.getLogger(__name__)


class VotingDriver:
    '''
    This plugin implements a scheme for combining multiple candidate progress
    coordinates into a 1-D coordinate that relies on "voting" of the trajectories.
    Votes can be scaled by the raw weights, the log of the weights and there are
    other options for how the votes are combined (see the options below).
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
        # whether to use the log of the weights
        self.log_weights = plugin_config.get('log_weights', False)
        # whether to use one final vote instead of a combination
        self.one_vote = plugin_config.get('one_vote', False)
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
        pcoord_len = segments[0].pcoord.shape[0]
        for idx in segments:
            data = segments[idx].data
            df = pd.DataFrame.from_dict(data)
            pct_change = df.pct_change(periods=pcoord_len - 1)
            pct_change *= self.pct_change_direct
            values = pct_change.iloc[-1].values
            index = np.argmax(values)
            votes[idx] = index
            weight = segments[idx].weight
            if self.log_weights:
                weights[idx] = 1 / (-1 * math.log(weight))
            else:
                weights[idx] = weight

        data_names = list(segments[0].data)
        vote_names = []
        print("voting results:")
        for vidx, vote in enumerate(votes):
            vote_names.append(data_names[int(vote)])
            print(data_names[int(vote)], weights[vidx])
        print("vote names", vote_names)
        vote_names = np.array(vote_names)

        if not self.use_weights:
            weights = np.ones(len(segments))

        if self.one_vote:
            counter = Counter(vote_names)
            names = list(counter.keys())
            sum_weights = np.zeros((len(names)))
            for nameidx, name in enumerate(names):
                w = np.where(vote_names == name)[0]
                w_weights = weights[w]
                w_weights_sum = w_weights.sum()
                sum_weights[nameidx] = w_weights_sum

            data = np.column_stack((names, sum_weights))
            sorted_data = data[data[:, 1].argsort()][::-1]
            chosen_vote = sorted_data[:, 0][0]
            print("chosen vote:", chosen_vote)
            for idx in segments:
                new_pcoord = segments[idx].data[chosen_vote]
                old_pcoord = segments[idx].pcoord[:, 1].reshape(pcoord_len, 1)
                new_pcoord = new_pcoord.reshape(pcoord_len, 1)
                combined_pcoord = np.concatenate((new_pcoord, old_pcoord), axis=1)
                segments[idx].pcoord = combined_pcoord
        else:
            for idx in segments:
                new_pcoord = 0
                data_names = list(segments[idx].data)
                for ivote, vote in enumerate(votes):
                    vote_name = data_names[int(vote)]
                    new_pcoord += segments[idx].data[vote_name] * weights[ivote]
                old_pcoord = segments[idx].pcoord[:, 1].reshape(pcoord_len, 1)
                new_pcoord = new_pcoord.reshape(pcoord_len, 1)
                combined_pcoord = np.concatenate((new_pcoord, old_pcoord), axis=1)
                segments[idx].pcoord = combined_pcoord
