import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding

import pandapower.networks as pn
import pandapower as pp
import simbench as sb

import random
import matplotlib.pyplot as plt
from pandapower.plotting import simple_plot


class PwrCptingEnv(gym.Env):
    """
    Custom Power Computing Environment that follows gym interface.
    """

    # LunarLander example:
    # https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py
    def __init__(self, ts=True):

        self.ts = ts

        self.REWARD_CONSTANT = 2000
        # Debug Mode...
        self.LOAD_MAGNIFICATION_FACTOR = 2.5  # 1 -> 189.2 MW
        # import basic net...
        sb_code = '1-complete_data-mixed-all-2-sw'
        self.data_net = sb.get_simbench_net(sb_code)

        # Get valid load profile names...
        # profile_names = self.data_net.profiles['load'].columns.tolist()[1:]
        # valid_profile_names = []
        # for name in profile_names:
        #     if name[-5] == 'p' and len(name) <= 14:
        #         valid_profile_names.append(name[:-6])
        # self.valid_profile_names = valid_profile_names

        self.valid_profile_names = ['G0-A', 'G0-M', 'G1-A', 'G1-B', 'G1-C',
                                    'G3-A', 'G3-H', 'G4-A', 'G5-A', 'L2-M']

        # Set DER parameters...
        self.WIND_CAPACITY = 20
        self.WIND_CAPACITY_VAR = 1
        self.PV_CAPACITY = 10
        self.PV_CAPACITY_VAR = 1

        # Get valid DER profile names...
        # valid_wind_names = []
        # valid_pv_names = []
        # for name in self.data_net.profiles['renewables'].columns.to_list():
        #     if name[:2] == 'PV':
        #         valid_pv_names.append(name)
        #     elif name[:2] == 'WP':
        #         valid_wind_names.append(name)
        # self.valid_wind_names = valid_wind_names
        # self.valid_pv_names = valid_pv_names
        # Debug mode...
        self.valid_wind_names = ['WP1']
        self.valid_pv_names = ['PV1', 'PV2', 'PV3', 'PV4']

        # Set generation parameters...
        self.NEW_GEN_CAPACITY = 25
        self.NEW_GEN_CAPACITY_VAR = 5
        self.cp1_eur = 2.5
        self.cp2_eur = 0.0175

        net = pn.case30()
        # Define action space...
        # debug mode...
        # self.action_space = spaces.Box(low=0., high=1., shape=(net.bus.shape[0],))
        self.action_space = spaces.Box(low=0., high=1., shape=(5,))

        # Define observation space...
        # S = [x_t, p_{G,t-1}, P_L, P_W, P_P, P_ext, S_C]
        # space1 = spaces.Discrete(24)
        # space2 = spaces.Box(-1., 1., shape=(net.bus.shape[0], 5), dtype=np.float32)
        # space3 = spaces.Box(0., np.inf, shape=(net.bus.shape[0],), dtype=np.float32)
        # self.observation_space = spaces.Tuple((space1, space2, space3))
        Q_C_high = np.empty(net.bus.shape[0])
        Q_C_high.fill(np.inf)
        # self.observation_space = spaces.Box(
        #     low=np.concatenate(([0], np.zeros(net.bus.shape[0] * 3))),
        #     high=np.concatenate(([24], np.ones((net.bus.shape[0] * 2,),
        #                                       dtype=np.float32), Q_C_high))
        #     , )

        # Debug mode...
        self.observation_space = spaces.Box(
            low=np.append(np.array(0), np.zeros(net.bus.shape[0])),
            high=np.append(np.array(24), np.ones(net.bus.shape[0])))

        self.reset()

    def initialise_grid_powers(self):
        absolute_profiles_sgen = sb.get_absolute_profiles_from_relative_profiles(self.net, 'sgen', 'p_mw').iloc[
            list(range(self.day_of_year * 96 + self.year, self.day_of_year * 96 + 96 + self.year, 4))]
        absolute_profiles_load = sb.get_absolute_profiles_from_relative_profiles(self.net, 'load', 'p_mw').iloc[
            list(range(self.day_of_year * 96 + self.year, self.day_of_year * 96 + 96 + self.year, 4))]

        self.absolute_values_load_sgen = {'load': absolute_profiles_load, 'sgen': absolute_profiles_sgen}
        self.apply_absolute_values(self.absolute_values_load_sgen)

        self.net.ext_grid['max_p_mw'] = 800
        self.net.ext_grid['min_p_mw'] = -400

    def apply_absolute_values(self, absolute_values_dict):
        for element, this_profile in absolute_values_dict.items():
            self.net[element]['p_mw'] = this_profile.iloc[self.time_step]

    def step(self, action=None):
        """
        :param action: np.ndarray
        :return:
        """
        # First check the input is valid...
        # If action is None, it is the first of one episode
        # if action is not None:
        #     assert isinstance(action, np.ndarray), 'The action should be np.ndarray.'
        #     available_poses = self.net.gen.bus.values.tolist()
        #     # action[list(set(range(action.shape[0])) - set(available_poses))] = 0
        #     # print(action)
        #     # print('Q_C below')
        #     # print(available_poses)
        #     assert set(np.argwhere(~np.isclose(action, 0, atol=1e-4)).reshape(1, -1)[0]).issubset(set(available_poses)), \
        #         'Some nodes do not have generator.'

        # Check that the powers are within limits... NotImplemented

        # Update the grid powers according to action and the grid power dynamics...

        if action is None:
            P_G_t_1 = np.zeros(self.net.bus.shape[0])
            P_G_t_1[self.net.gen['bus'].values] = self.net.gen['p_mw']

            P_L = np.zeros(self.net.bus.shape[0])
            P_L[self.net.load['bus'].values] = self.net.load['p_mw']

            P_W = np.zeros(self.net.bus.shape[0])
            P_W[self.net.sgen[self.net.sgen['type'] == 'Wind_MV']['bus'].values] = \
                self.net.sgen[self.net.sgen['type'] == 'Wind_MV']['p_mw']

            P_P = np.zeros(self.net.bus.shape[0])
            P_P[self.net.sgen[self.net.sgen['type'] == 'PV_MV']['bus'].values] = \
                self.net.sgen[self.net.sgen['type'] == 'PV_MV']['p_mw']
            pp.rundcpp(self.net)
            P_ext = np.zeros(self.net.bus.shape[0])
            P_ext[self.net.res_ext_grid.index.values] = self.net.res_ext_grid['p_mw']

            # init_state = np.concatenate((np.array([self.time_step]), P_G_t_1, P_L + P_W + P_P + P_ext, self.S_C))
            # Debug mode...
            # init_state = np.concatenate((P_L + P_W + P_P, self.S_C))
            # init_state = P_L + P_W + P_P + P_ext
            init_state = P_L + P_W + P_P
            # print('-----\nInitial state: \n------')
            # print(init_state)
            # print('Load: ')
            # print(P_L)

            # print('load', self.net.load)
            # print('gen', self.net.gen)
            # print('sgen', self.net.sgen)

            return np.append(np.array(self.time_step), init_state)
        opp_flag = True
        try:
            pp.rundcopp(self.net)
        except BaseException as t:
            opp_flag = False
            print(t)
            # print('load', Env.net.load['p_mw'].sum())
            # print('gen_max', Env.net.gen['max_p_mw'].sum())
            # print('gen', Env.net.gen['p_mw'].sum())
            # print('ext_grid', Env.net.ext_grid['max_p_mw'].sum())

        opp_r = self.net.res_cost if opp_flag else self.REWARD_CONSTANT
        # print('Load: ', self.net.load['p_mw'].sum())
        if opp_flag:
            print('OPF: ')
            t = self.net.res_gen['p_mw'].copy()
            res_ext = self.net.res_ext_grid['p_mw'].values[0]
            print(self.net.res_gen['p_mw'].sum())
            print('Cost: ', self.net.res_cost)
            # print('ext: ', res_ext)

        # debug mode...
        # self.net.gen['p_mw'] = self.net.gen['max_p_mw'] * action[self.net.gen['bus'].values]
        # print('Before implementing RL: ')
        # print(self.net.gen['p_mw'])
        self.net.gen['p_mw'] = self.net.gen['max_p_mw'] * action
        print('RL: ')
        print(self.net.gen['p_mw'].sum())

        pp.rundcpp(self.net)

        # P_ext = np.zeros(self.net.bus.shape[0])
        # P_ext[self.net.res_ext_grid.index] = self.net.res_ext_grid['p_mw']
        # S = [x_t, p_{G,t-1}, P_L, P_W, P_P, P_ext, S_C]
        # Debug Mode...
        # state = np.concatenate((np.array([self.time_step]), P_G_t_1, P_L + P_W + P_P + P_ext, self.S_C))
        # state = np.concatenate((P_L + P_W + P_P + P_ext, self.S_C))

        # Calculate the reward using self.compute_gen_cost()...
        tot_cost = self.compute_gen_cost().sum()
        print('Cost: ', tot_cost)
        # print('\n----------------Step {}---------------'.format(self.time_step))
        # print('Current ext_cost: ', tot_cost[0])
        # print('Current load: ', self.net.load['p_mw'].sum())
        # print('Current gen: ', self.net.gen['p_mw'].sum())
        # print('Current sgen', self.net.sgen['p_mw'].sum())
        # print('current tot_gen: ', self.net.gen['p_mw'].sum() + self.net.sgen['p_mw'].sum())
        # print('Current tot_cost: ', tot_cost.sum())
        reward = -tot_cost + self.REWARD_CONSTANT
        # print('reward: ', reward)

        # Move the environment dynamic to the next time step...
        self.time_step += 1
        # print(self.time_step)
        if self.time_step == 24:
            done = True
        else:
            done = False

        # Debug mode...
        # done = True

        if not done:
            self.apply_absolute_values(self.absolute_values_load_sgen)

        P_G_t_1 = np.zeros(self.net.bus.shape[0])
        P_G_t_1[self.net.gen['bus'].values] = self.net.gen['p_mw']

        P_L = np.zeros(self.net.bus.shape[0])
        P_L[self.net.load['bus'].values] = self.net.load['p_mw']

        P_W = np.zeros(self.net.bus.shape[0])
        P_W[self.net.sgen[self.net.sgen['type'] == 'Wind_MV']['bus'].values] = \
            self.net.sgen[self.net.sgen['type'] == 'Wind_MV']['p_mw']

        P_P = np.zeros(self.net.bus.shape[0])
        P_P[self.net.sgen[self.net.sgen['type'] == 'PV_MV']['bus'].values] = \
            self.net.sgen[self.net.sgen['type'] == 'PV_MV']['p_mw']

        state = P_L + P_W + P_P

        return np.append(np.array(self.time_step), state), reward, done, (opp_r, t, res_ext)  # In accordance with gym API

    def reset(self):
        # Debug mode...
        # random.seed(0)
        # np.random.seed(0)

        net = pn.case30()
        self.net = net
        net.profiles = self.data_net.profiles
        self.net.load['p_mw'] = self.net.load['p_mw'] * self.LOAD_MAGNIFICATION_FACTOR
        self.net.line['max_i_ka'] = self.net.line['max_i_ka'] * self.LOAD_MAGNIFICATION_FACTOR * 100

        # Debug Mode...
        self.net.poly_cost.loc[0, 'cp1_eur_per_mw'] = 4
        self.net.poly_cost.loc[0, 'cp2_eur_per_mw2'] = 0.05

        # produce random load profiles...
        load_profile_belonging = ['G0-A',
                                  'G0-A',
                                  'G3-A',
                                  'G3-A',
                                  'G3-H',
                                  'G3-H',
                                  'G4-A',
                                  'G4-A',
                                  'G5-A',
                                  'G5-A',
                                  'L2-M',
                                  'L2-M',
                                  'G0-M',
                                  'G0-M',
                                  'G1-A',
                                  'G1-A',
                                  'G1-B',
                                  'G1-B',
                                  'G1-C',
                                  'G1-C']

        net.load['profile'] = load_profile_belonging

        # Randomly add DERs...
        wind_num = random.choice([1, 2])
        pv_num = random.choice(range(4, 6))
        # Debug Mode...
        wind_num = 1
        pv_num = 2
        wind_poses = np.random.choice(net.bus.index.tolist(), size=wind_num, replace=False)
        pv_poses = np.random.choice(net.bus.index.tolist(), size=pv_num, replace=False)

        for wind_pos in wind_poses:
            pp.create_sgen(net, wind_pos, max(0, self.WIND_CAPACITY + np.random.normal(scale=self.WIND_CAPACITY_VAR)),
                           type='Wind_MV', controllable=False)
        for pv_pos in pv_poses:
            pp.create_sgen(net, pv_pos, max(0, self.PV_CAPACITY + np.random.normal(scale=self.PV_CAPACITY_VAR)),
                           type='PV_MV', controllable=False)

        # Allocate profiles for DERs...

        wind_profiles = random.choices(self.valid_wind_names, k=wind_num)
        pv_profiles = random.choices(self.valid_pv_names, k=pv_num)
        # Debug mode...
        # wind_profiles = self.valid_wind_names[:1]
        # pv_profiles = self.valid_pv_names[:2]
        net.sgen.loc[net.sgen['type'] == 'Wind_MV', 'profile'] = wind_profiles
        net.sgen.loc[net.sgen['type'] == 'PV_MV', 'profile'] = pv_profiles

        # Randomly add redispatchable generations...
        new_gen_num = random.choice(range(2, 3))
        # Debug Mode...
        new_gen_num = 0
        available_poses = list(set(net.bus.index.tolist()) - set(net.gen.bus.tolist()))
        new_gen_poses = np.random.choice(available_poses, size=new_gen_num, replace=False)
        for gen_pos in new_gen_poses:
            maximum_gen = max(0, self.NEW_GEN_CAPACITY + np.random.normal(scale=self.NEW_GEN_CAPACITY_VAR))
            pp.create_gen(net, gen_pos, maximum_gen, max_p_mw=maximum_gen, min_p_mw=0)
            current_idx = max(net.gen.index)
            pp.create_poly_cost(net, current_idx, et='gen', cp1_eur_per_mw=self.cp1_eur, cp2_eur_per_mw2=self.cp2_eur)

        # Save S_C...
        self.S_C = np.zeros(self.net.bus.shape[0])
        self.S_C[self.net.gen['bus'].values] = self.net.gen['max_p_mw']

        # Prevent from invalid OPF...
        self.net.ext_grid['max_p_mw'] = 800
        self.net.ext_grid['min_p_mw'] = -400

        self.day_of_year = random.choice(range(366))
        self.year = random.choice(range(4))
        # self.year = 0
        # self.day_of_year = 347
        self.time_step = 0
        self.initialise_grid_powers()

        return self.step()

    def render(self, mode='human'):

        # 定义绘图函数
        def plot_with_DER(net):
            fig, ax = plt.subplots(figsize=[12, 8])
            for i in net.bus_geodata.iterrows():
                idx = i[0]
                x = i[1]['x']
                y = i[1]['y']
                ax.text(x - 0.3, y, idx, fontdict={'fontsize': 14})
            ax.scatter(net.bus_geodata.loc[net.gen['bus']]['x'], net.bus_geodata.loc[net.gen['bus']]['y'] + 0.15, s=200,
                       c='r', marker='*')
            simple_plot(net, plot_loads=True, ax=ax)
            fig.savefig('IEEE30节点系统', dpi=300, bbox_inches='tight')
            return fig, ax

        return plot_with_DER(self.net)

    def compute_gen_cost(self):
        """
        This function computes the total generation cost.
        *Parameter*:
        poly_cost: net.poly_cost
        res_gen: net.res_gen
        *return*:
        cost: The first element is ext_grid_cost, The left are gen_cost.
        """
        res1 = \
            self.net.poly_cost[self.net.poly_cost['et'] == 'gen'][['element', 'cp2_eur_per_mw2']].astype(
                {'element': int}).set_index(
                'element')['cp2_eur_per_mw2'] * self.net.res_gen['p_mw'] ** 2 + \
            self.net.poly_cost[self.net.poly_cost['et'] == 'gen'][['element', 'cp1_eur_per_mw']].astype(
                {'element': int}).set_index(
                'element')['cp1_eur_per_mw'] * self.net.res_gen['p_mw'] + \
            self.net.poly_cost[self.net.poly_cost['et'] == 'gen'][['element', 'cp0_eur']].astype(
                {'element': int}).set_index(
                'element')['cp0_eur']

        res2 = self.net.poly_cost[self.net.poly_cost['et'] == 'ext_grid'][['element', 'cp2_eur_per_mw2']].astype(
            {'element': int}).set_index('element')['cp2_eur_per_mw2'] * self.net.res_ext_grid['p_mw'] ** 2 + \
               self.net.poly_cost[self.net.poly_cost['et'] == 'ext_grid'][['element', 'cp1_eur_per_mw']].astype(
                   {'element': int}).set_index('element')['cp1_eur_per_mw'] * self.net.res_ext_grid['p_mw']

        if self.net.res_ext_grid['p_mw'].values[0] < 0:
            print('The external grid absorbs power.')
            res2.iloc[0] = 0

        return res2.append(res1)
