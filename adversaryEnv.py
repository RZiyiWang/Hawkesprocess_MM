import copy
import numpy as np

import tensorflow as tf
from tf_agents.environments import *
from tf_agents.specs import BoundedArraySpec, BoundedTensorSpec
from tf_agents.trajectories.time_step import StepType
from tf_agents.trajectories.time_step import TimeStep
from dynamics import *
from strategies import *

INV_BOUNDS = (-50.0, 50.0)
#SCALE_BOUNDS = (105.0, 175.0) #Poisson Process
BASELINERATE_BOUNDS = (7.5,12.5) # Hawkes Process
DRIFT_BOUNDS = (0.0, 1.0)
DECAY_BOUNDS = (1.125, 1.875)
WEALTH_BOUNDS = (-100000, 100000)
MAX_DRIFT = 5.0

#Adversary_RN_control_A

class AdversaryEnvironmentWithControllingA(py_environment.PyEnvironment):

    def __init__(self, eta=0.0, zeta=0.0, discount=1.0):
        super(AdversaryEnvironmentWithControllingA, self).__init__(handle_auto_reset=True)
        self.dt=0.005
        self.volatility=2
        self.decay=1.5
        self.drift=0

        self.baseline_arrival_rate = 10.0
        self.mean_reversion_speed = 60.0
        self.jump_size=40.0
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(self.dt, self.drift, self.volatility),HawkesArrivalModel(self.dt,self.baseline_arrival_rate,self.mean_reversion_speed,self.jump_size,self.decay))
        
        self.inv = 0.0
        self.inv_terminal = 0.0
        self.reward = 0.0
        self.wealth = 0.0
        self.eta=eta
        self.zeta = zeta
        self.inv_strategy = LinearUtilityTerminalPenaltyStrategy(self.decay, self.eta)
        self._discount = np.asarray(discount, dtype=np.float32)
        self._observation = None

    def action_spec(self):
        return BoundedArraySpec((1,), np.float32, name='action', minimum=BASELINERATE_BOUNDS[0], maximum=BASELINERATE_BOUNDS[1])

    def observation_spec(self):
        return BoundedArraySpec((3,), np.float32, name='observation', minimum=[0.0, INV_BOUNDS[0],WEALTH_BOUNDS[0]],
                                maximum=[1.0, INV_BOUNDS[1], WEALTH_BOUNDS[1]])

    def _get_observation(self):
        self._observation = np.asarray([self.dynamics.time, self.inv,self.wealth], dtype=np.float32)
        return self._observation

    def _reset(self):
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(0.005, 0.0, 2.0),
                                   HawkesArrivalModel(0.005,10.0,60.0,40.0,1.5))
        self.inv = 0.0
        self.inv_terminal = 0.0
        self.reward = 0.0
        self.wealth = 0.0
        return TimeStep(StepType.FIRST, np.asarray(0.0, dtype=np.float32), self._discount, self._get_observation())

    def _step(self, action: np.ndarray):
        self.baseline_arrival_rate = action[0]
        self.dynamics.execution_dynamics_ask = HawkesArrivalModel(self.dt,self.baseline_arrival_rate,self.mean_reversion_speed,self.jump_size,self.decay)
        self.dynamics.execution_dynamics_bid = HawkesArrivalModel(self.dt,self.baseline_arrival_rate,self.mean_reversion_speed,self.jump_size,self.decay)
        self.dynamics.price_dynamics=BrownianMotionWithDrift(self.dt, self.drift, self.volatility)
        self.inv_strategy = LinearUtilityTerminalPenaltyStrategy(self.decay, self.eta)

        (ask_offset, bid_offset) = self.inv_strategy.compute(self.dynamics.time, self.dynamics.price, self.inv)
        ask_price = self.dynamics.price + ask_offset
        bid_price = self.dynamics.price - bid_offset

        self.reward = -(self.inv * self.dynamics.innovate()-self.zeta * self.inv**2)

        self._do_executions(ask_price, bid_price)

        if self.is_terminal():
            self.wealth += self.dynamics.price * self.inv
            self.reward += self.eta * self.inv ** 2
            self.inv_terminal = self.inv
            return TimeStep(StepType.LAST, np.asarray(self.reward, dtype=np.float32), self._discount,
                            self._get_observation())

        return TimeStep(StepType.MID, np.asarray(self.reward, dtype=np.float32), self._discount,
                        self._get_observation())

    def _do_executions(self, ask_price, bid_price):
        if self.inv > INV_BOUNDS[0]:
            ask_offset = self.dynamics.try_execute_ask(ask_price)
            if ask_offset:
                self.inv -= 1.0
                self.reward -= ask_offset
                self.wealth += ask_price

        if self.inv < INV_BOUNDS[1]:
            bid_offset = self.dynamics.try_execute_bid(bid_price)
            if bid_offset:
                self.inv += 1.0
                self.reward -= bid_offset
                self.wealth -= bid_price
    
    def is_terminal(self):
        return self.dynamics.time >= 1.0

    def get_state(self):
        dynamics_state = self.dynamics.get_state()
        env_state = dict()
        env_state['_discount'] = copy.deepcopy(self._discount)
        env_state['inv'] = self.inv
        env_state['inv_terminal'] = self.inv_terminal
        env_state['reward'] = self.reward
        env_state['wealth'] = self.wealth
        return {**dynamics_state, **env_state}

    def set_state(self, state):
        self.dynamics.dt = state['dt']
        self.dynamics.time = state['time']
        self.dynamics.price = state['price']
        self.dynamics.price_initial = state['price_initial']
        self._discount = copy.deepcopy(state['_discount'])
        self.inv = state['inv']
        self.inv_terminal = state['inv_terminal']
        self.reward = state['reward']
        self.wealth = state['wealth']

#Adversary_RN_control_Drift

class AdversaryEnvironmentWithControllingDrift(py_environment.PyEnvironment):

    def __init__(self, eta=0.0, zeta=0.0, discount=1.0):
        super(AdversaryEnvironmentWithControllingDrift, self).__init__(handle_auto_reset=True)
        self.dt=0.005
        self.volatility=2
        self.decay=1.5
        self.drift=0
        self.baseline_arrival_rate = 10.0
        self.mean_reversion_speed = 60.0
        self.jump_size=40.0
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(self.dt, self.drift, self.volatility),HawkesArrivalModel(self.dt,self.baseline_arrival_rate,self.mean_reversion_speed,self.jump_size,self.decay))
      
        self.inv = 0.0
        self.inv_terminal = 0.0
        self.reward = 0.0
        self.wealth = 0.0
        self.eta=eta
        self.zeta = zeta
        self.inv_strategy = LinearUtilityTerminalPenaltyStrategy(self.decay, self.eta)
        self._discount = np.asarray(discount, dtype=np.float32)
        self._observation = None

    def action_spec(self):
        return BoundedArraySpec((1,), np.float32, name='action', minimum=DRIFT_BOUNDS[0], maximum=DRIFT_BOUNDS[1])

    def observation_spec(self):
        return BoundedArraySpec((3,), np.float32, name='observation', minimum=[0.0, INV_BOUNDS[0],WEALTH_BOUNDS[0]],
                                maximum=[1.0, INV_BOUNDS[1], WEALTH_BOUNDS[1]])

    def _get_observation(self):
        self._observation = np.asarray([self.dynamics.time, self.inv,self.wealth], dtype=np.float32)
        return self._observation

    def _reset(self):
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(0.005, 0.0, 2.0),
                                   HawkesArrivalModel(0.005,10.0,60.0,40.0,1.5))
        self.inv = 0.0
        self.inv_terminal = 0.0
        self.reward = 0.0
        self.wealth = 0.0
        return TimeStep(StepType.FIRST, np.asarray(0.0, dtype=np.float32), self._discount, self._get_observation())

    def _step(self, action: np.ndarray):
        drift = action[0]
        self.drift = MAX_DRIFT * (2.0 * drift - 1.0)
        self.dynamics.execution_dynamics_ask = HawkesArrivalModel(self.dt,self.baseline_arrival_rate,self.mean_reversion_speed,self.jump_size,self.decay)
        self.dynamics.execution_dynamics_bid = HawkesArrivalModel(self.dt,self.baseline_arrival_rate,self.mean_reversion_speed,self.jump_size,self.decay)
        self.dynamics.price_dynamics=BrownianMotionWithDrift(self.dt, self.drift, self.volatility)
        self.inv_strategy = LinearUtilityTerminalPenaltyStrategy(self.decay, self.eta)
        
        (ask_offset, bid_offset) = self.inv_strategy.compute(self.dynamics.time, self.dynamics.price, self.inv)
        ask_price = self.dynamics.price + ask_offset
        bid_price = self.dynamics.price - bid_offset

        self.reward = -(self.inv * self.dynamics.innovate()-self.zeta * self.inv**2)

        self._do_executions(ask_price, bid_price)

        if self.is_terminal():
            self.wealth += self.dynamics.price * self.inv
            self.reward += self.eta * self.inv ** 2
            self.inv_terminal = self.inv
            return TimeStep(StepType.LAST, np.asarray(self.reward, dtype=np.float32), self._discount,
                            self._get_observation())

        return TimeStep(StepType.MID, np.asarray(self.reward, dtype=np.float32), self._discount,
                        self._get_observation())

    def _do_executions(self, ask_price, bid_price):
        if self.inv > INV_BOUNDS[0]:
            ask_offset = self.dynamics.try_execute_ask(ask_price)
            if ask_offset:
                self.inv -= 1.0
                self.reward -= ask_offset
                self.wealth += ask_price

        if self.inv < INV_BOUNDS[1]:
            bid_offset = self.dynamics.try_execute_bid(bid_price)
            if bid_offset:
                self.inv += 1.0
                self.reward -= bid_offset
                self.wealth -= bid_price
    
    def is_terminal(self):
        return self.dynamics.time >= 1.0

    def get_state(self):
        dynamics_state = self.dynamics.get_state()
        env_state = dict()
        env_state['_discount'] = copy.deepcopy(self._discount)
        env_state['inv'] = self.inv
        env_state['inv_terminal'] = self.inv_terminal
        env_state['reward'] = self.reward
        env_state['wealth'] = self.wealth
        return {**dynamics_state, **env_state}

    def set_state(self, state):
        self.dynamics.dt = state['dt']
        self.dynamics.time = state['time']
        self.dynamics.price = state['price']
        self.dynamics.price_initial = state['price_initial']
        self._discount = copy.deepcopy(state['_discount'])
        self.inv = state['inv']
        self.inv_terminal = state['inv_terminal']
        self.reward = state['reward']
        self.wealth = state['wealth']

#Adversary_RN_control_K

class AdversaryEnvironmentWithControllingK(py_environment.PyEnvironment):

    def __init__(self, eta=0.0, zeta=0.0, discount=1.0):
        super(AdversaryEnvironmentWithControllingK, self).__init__(handle_auto_reset=True)
        self.dt=0.005
        self.volatility=2
        self.decay=1.5
        self.drift=0
        self.baseline_arrival_rate = 10.0
        self.mean_reversion_speed = 60.0
        self.jump_size=40.0
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(self.dt, self.drift, self.volatility),HawkesArrivalModel(self.dt,self.baseline_arrival_rate,self.mean_reversion_speed,self.jump_size,self.decay))
      
        self.inv = 0.0
        self.inv_terminal = 0.0
        self.reward = 0.0
        self.wealth = 0.0
        self.eta=eta
        self.zeta = zeta
        self.inv_strategy = LinearUtilityTerminalPenaltyStrategy(self.decay, self.eta)
        self._discount = np.asarray(discount, dtype=np.float32)
        self._observation = None

    def action_spec(self):
        return BoundedArraySpec((1,), np.float32, name='action', minimum=DECAY_BOUNDS[0], maximum=DECAY_BOUNDS[1])

    def observation_spec(self):
        return BoundedArraySpec((3,), np.float32, name='observation', minimum=[0.0, INV_BOUNDS[0],WEALTH_BOUNDS[0]],
                                maximum=[1.0, INV_BOUNDS[1], WEALTH_BOUNDS[1]])

    def _get_observation(self):
        self._observation = np.asarray([self.dynamics.time, self.inv,self.wealth], dtype=np.float32)
        return self._observation

    def _reset(self):
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(0.005, 0.0, 2.0),
                                   HawkesArrivalModel(0.005,10.0,60.0,40.0,1.5))
        self.inv = 0.0
        self.inv_terminal = 0.0
        self.reward = 0.0
        self.wealth = 0.0
        return TimeStep(StepType.FIRST, np.asarray(0.0, dtype=np.float32), self._discount, self._get_observation())

    def _step(self, action: np.ndarray):
        self.decay = action[0]
        self.dynamics.execution_dynamics_ask = HawkesArrivalModel(self.dt,self.baseline_arrival_rate,self.mean_reversion_speed,self.jump_size,self.decay)
        self.dynamics.execution_dynamics_bid = HawkesArrivalModel(self.dt,self.baseline_arrival_rate,self.mean_reversion_speed,self.jump_size,self.decay)
        self.dynamics.price_dynamics=BrownianMotionWithDrift(self.dt, self.drift, self.volatility)
        self.inv_strategy = LinearUtilityTerminalPenaltyStrategy(self.decay, self.eta)
        
        (ask_offset, bid_offset) = self.inv_strategy.compute(self.dynamics.time, self.dynamics.price, self.inv)
        ask_price = self.dynamics.price + ask_offset
        bid_price = self.dynamics.price - bid_offset

        self.reward = -(self.inv * self.dynamics.innovate()-self.zeta * self.inv**2)

        self._do_executions(ask_price, bid_price)

        if self.is_terminal():
            self.wealth += self.dynamics.price * self.inv
            self.reward += self.eta * self.inv ** 2
            self.inv_terminal = self.inv
            return TimeStep(StepType.LAST, np.asarray(self.reward, dtype=np.float32), self._discount,
                            self._get_observation())

        return TimeStep(StepType.MID, np.asarray(self.reward, dtype=np.float32), self._discount,
                        self._get_observation())

    def _do_executions(self, ask_price, bid_price):
        if self.inv > INV_BOUNDS[0]:
            ask_offset = self.dynamics.try_execute_ask(ask_price)
            if ask_offset:
                self.inv -= 1.0
                self.reward -= ask_offset
                self.wealth += ask_price

        if self.inv < INV_BOUNDS[1]:
            bid_offset = self.dynamics.try_execute_bid(bid_price)
            if bid_offset:
                self.inv += 1.0
                self.reward -= bid_offset
                self.wealth -= bid_price
    
    def is_terminal(self):
        return self.dynamics.time >= 1.0

    def get_state(self):
        dynamics_state = self.dynamics.get_state()
        env_state = dict()
        env_state['_discount'] = copy.deepcopy(self._discount)
        env_state['inv'] = self.inv
        env_state['inv_terminal'] = self.inv_terminal
        env_state['reward'] = self.reward
        env_state['wealth'] = self.wealth
        return {**dynamics_state, **env_state}

    def set_state(self, state):
        self.dynamics.dt = state['dt']
        self.dynamics.time = state['time']
        self.dynamics.price = state['price']
        self.dynamics.price_initial = state['price_initial']
        self._discount = copy.deepcopy(state['_discount'])
        self.inv = state['inv']
        self.inv_terminal = state['inv_terminal']
        self.reward = state['reward']
        self.wealth = state['wealth']

#Adversary_RN_control_All

class AdversaryEnvironmentWithControllingAll(py_environment.PyEnvironment):

    def __init__(self, eta=0.0, zeta=0.0, discount=1.0):
        super(AdversaryEnvironmentWithControllingAll, self).__init__(handle_auto_reset=True)
        self.dt=0.005
        self.volatility=2
        self.decay=1.5
        self.drift=0
        self.baseline_arrival_rate = 10.0
        self.mean_reversion_speed = 60.0
        self.jump_size=40.0
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(self.dt, self.drift, self.volatility),HawkesArrivalModel(self.dt,self.baseline_arrival_rate,self.mean_reversion_speed,self.jump_size,self.decay))
        
        self.inv = 0.0
        self.inv_terminal = 0.0
        self.reward = 0.0
        self.wealth = 0.0
        self.eta=eta
        self.zeta = zeta
        self.inv_strategy = LinearUtilityTerminalPenaltyStrategy(self.decay, self.eta)
        self._discount = np.asarray(discount, dtype=np.float32)
        self._observation = None

    def action_spec(self):
        return BoundedArraySpec((3,), np.float32, name = 'action', minimum = [DRIFT_BOUNDS[0], BASELINERATE_BOUNDS[0], DECAY_BOUNDS[0]], maximum = [DRIFT_BOUNDS[1], BASELINERATE_BOUNDS[1], DECAY_BOUNDS[1]])
    
    def observation_spec(self):
        return BoundedArraySpec((3,), np.float32, name='observation', minimum=[0.0, INV_BOUNDS[0],WEALTH_BOUNDS[0]],
                                maximum=[1.0, INV_BOUNDS[1], WEALTH_BOUNDS[1]])

    def _get_observation(self):
        self._observation = np.asarray([self.dynamics.time, self.inv,self.wealth], dtype=np.float32)
        return self._observation

    def _reset(self):
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(0.005, 0.0, 2.0),
                                   HawkesArrivalModel(0.005,10.0,60.0,40.0,1.5))
        self.inv = 0.0
        self.inv_terminal = 0.0
        self.reward = 0.0
        self.wealth = 0.0
        return TimeStep(StepType.FIRST, np.asarray(0.0, dtype=np.float32), self._discount, self._get_observation())

    def _step(self, action: np.ndarray):
        drift = action[0]
        self.drift = MAX_DRIFT * (2.0 * drift - 1.0)
        self.baseline_arrival_rate = action[1]
        self.decay = action[2]
        self.dynamics.execution_dynamics_ask = HawkesArrivalModel(self.dt,self.baseline_arrival_rate,self.mean_reversion_speed,self.jump_size,self.decay)
        self.dynamics.execution_dynamics_bid = HawkesArrivalModel(self.dt,self.baseline_arrival_rate,self.mean_reversion_speed,self.jump_size,self.decay)
        self.dynamics.price_dynamics=BrownianMotionWithDrift(self.dt, self.drift, self.volatility)
        self.inv_strategy = LinearUtilityTerminalPenaltyStrategy(self.decay, self.eta)
        
        (ask_offset, bid_offset) = self.inv_strategy.compute(self.dynamics.time, self.dynamics.price, self.inv)
        ask_price = self.dynamics.price + ask_offset
        bid_price = self.dynamics.price - bid_offset

        self.reward = -(self.inv * self.dynamics.innovate()-self.zeta * self.inv**2)

        self._do_executions(ask_price, bid_price)

        if self.is_terminal():
            self.wealth += self.dynamics.price * self.inv
            self.reward += self.eta * self.inv ** 2
            self.inv_terminal = self.inv
            return TimeStep(StepType.LAST, np.asarray(self.reward, dtype=np.float32), self._discount,
                            self._get_observation())

        return TimeStep(StepType.MID, np.asarray(self.reward, dtype=np.float32), self._discount,
                        self._get_observation())

    def _do_executions(self, ask_price, bid_price):
        if self.inv > INV_BOUNDS[0]:
            ask_offset = self.dynamics.try_execute_ask(ask_price)
            if ask_offset:
                self.inv -= 1.0
                self.reward -= ask_offset
                self.wealth += ask_price

        if self.inv < INV_BOUNDS[1]:
            bid_offset = self.dynamics.try_execute_bid(bid_price)
            if bid_offset:
                self.inv += 1.0
                self.reward -= bid_offset
                self.wealth -= bid_price
    
    def is_terminal(self):
        return self.dynamics.time >= 1.0

    def get_state(self):
        dynamics_state = self.dynamics.get_state()
        env_state = dict()
        env_state['_discount'] = copy.deepcopy(self._discount)
        env_state['inv'] = self.inv
        env_state['inv_terminal'] = self.inv_terminal
        env_state['reward'] = self.reward
        env_state['wealth'] = self.wealth
        return {**dynamics_state, **env_state}

    def set_state(self, state):
        self.dynamics.dt = state['dt']
        self.dynamics.time = state['time']
        self.dynamics.price = state['price']
        self.dynamics.price_initial = state['price_initial']
        self._discount = copy.deepcopy(state['_discount'])
        self.inv = state['inv']
        self.inv_terminal = state['inv_terminal']
        self.reward = state['reward']
        self.wealth = state['wealth']
