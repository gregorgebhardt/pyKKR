#     Python Implementation of the Algorithms presented in the publications
#     around the kernel Kalman rule.
#     Copyright (C) 2017  Gregor Gebhardt
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import pandas as pd
from scipy.stats import uniform


class Simulator:
    def __init__(self, environment, initial_state_generator=None):
        self.environment = environment

        if initial_state_generator is None:
            self.initial_state_generator = uniform(loc=0, scale=1)
        else:
            self.initial_state_generator = initial_state_generator

        self._episode_counter = 0

    def simulate(self, num_episodes, num_steps, reset_episode_counter=True) -> pd.DataFrame:
        if reset_episode_counter:
            self._episode_counter = 0

        # create DataFrame
        if num_episodes > 1:
            index = pd.MultiIndex.from_product([range(self._episode_counter, self._episode_counter + num_episodes),
                                                range(num_steps)], names=['episode', 'step'])
        else:
            index = range(num_steps)

        self._episode_counter += num_episodes

        df = pd.DataFrame(
            np.zeros((num_episodes * num_steps, self.environment.stateDim)),
            columns=self.environment.stateNames,
            index=index
        )

        # sample initial data
        df.loc[(slice(None), 0), :] = self.initial_state_generator((num_episodes, self.environment.stateDim))

        for t in range(1, num_steps):
            state = df.loc[(slice(None), t-1), :].values
            df.loc[(slice(None), t), :] = self.environment.transitionFunction(state, np.zeros((num_episodes, 1)))

        return df

    def reset_episode_counter(self):
        self._episode_counter = 0
