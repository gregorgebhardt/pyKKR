import numpy as np
import cma
from scipy import stats
from kkr import *

np.set_printoptions(linewidth=200)

# set some initial parameters
# regularization parameter for the inverses
alpha_t = -15
alpha_o = -15
alpha_q = -12

# bandwidth of the kernel function k and g
bandwidth_factor_k = np.log(1)
bandwidth_factor_g = np.log(1)

# Simulation parameters
num_steps = 30
num_episodes = 50
num_observations = 10

num_test_episodes = 10

process_noise_std = .1
observation_noise_std = .01

# Model parameters
# number of data points in the kernel matrices
kernel_size = 200
subspace_size = 50

window_size = 4

state_features = 'thetaNoisy'
obs_features = 'thetaNoisy'

# Simulate pendulum
pendulum = Pendulum()
pendulum.noise_std = process_noise_std
pendulum.dt = .1

simulator = Simulator(pendulum, stats.uniform(loc=np.array([-.25, -2]) * np.pi, scale=np.array([.5, 4]) * np.pi).rvs)

# simulate data
data = simulator.simulate(num_episodes, num_steps)
add_noise(data, 'theta', noise_generator=stats.norm(loc=.0, scale=observation_noise_std))
# Sample evaluation data
testData = simulator.simulate(num_test_episodes, num_steps)
add_noise(testData, 'theta', noise_generator=stats.norm(loc=.0, scale=observation_noise_std))

# Preprocess the data
windows = generate_data_windows(data[state_features], window_size)

# get initial states of the data
gb = windows.groupby(level='episode')
_X0 = gb.first()[state_features].values

# Choose bandwidth
bandwidths = compute_median_bandwidth(windows[state_features])

# select reference sets
# reference_set1, reference_set2 = select_reference_set_randomly(windows, size=kernel_size,
#                                                                consecutive_sets=2, group_by='episode')
kernel_ref = ExponentialQuadraticKernel()
kernel_ref.bandwidth = bandwidths
reference_set1, reference_set2 = select_reference_set_by_kernel_activation(windows, size=kernel_size,
                                                                           kernel_function=kernel_ref,
                                                                           # lambda a, b: a.dot(b.T),
                                                                           consecutive_sets=2, group_by='episode')
reference_set_subspace = select_reference_set_by_kernel_activation(windows.loc[reference_set1], size=subspace_size,
                                                                   kernel_function=kernel_ref)

# Select training data
_X1 = windows.loc[reference_set1, state_features].values
_X2 = windows.loc[reference_set2, state_features].values
_Y1 = data.loc[reference_set1, obs_features].values.reshape((-1, 1))
_Y2 = data.loc[reference_set2, obs_features].values.reshape((-1, 1))
_Xr = windows.loc[reference_set_subspace, state_features].values

train_data = {'states_1': _X1,
              'states_2': _X2,
              # 'subspace_states': _Xr,
              'observations': _Y2,
              'init_states': _X0,
              'preimage_states': _Y2}

observations = testData[obs_features].copy()
observation_dim = observations.ndim

observations.loc[(slice(None), slice(num_observations, None))] = np.nan

observations = observations.unstack(level=0).values
observations = observations.reshape(-1, observation_dim, num_test_episodes)

groundtruth = testData['theta'].unstack(level=0).values
groundtruth_dim = testData['theta'].ndim
groundtruth = groundtruth.reshape(-1, groundtruth_dim, num_test_episodes)

experiment = FilterEvaluation()
# experiment.model_class = SubspaceKernelKalmanFilter
experiment.setup_evaluation(train_data=train_data, test_observations=observations,
                            test_groundtruth=groundtruth)

parameter_names = ['alpha_t', 'alpha_o', 'alpha_q', 'bandwidth_k', 'bandwidth_g']


@parameter_naming(parameter_names)
@parameter_transform(np.exp)
@bandwidth_factor(bandwidth_k=bandwidths, bandwidth_g=bandwidths[0])
@exception_catcher(np.linalg.linalg.LinAlgError, 1e4)
def eval_experiment(**kwargs):
    return experiment.evaluate(**kwargs)


x_0 = [alpha_t, alpha_o, alpha_q,
       bandwidth_factor_k,
       bandwidth_factor_g]

cma_opt = cma.CMAEvolutionStrategy(x_0, 0.5)
cma_opt.optimize(objective_fct=eval_experiment, iterations=200, verb_disp=1)
