import copy
import numpy as np

import tensorflow as tf
from tf_agents.environments import *
from tf_agents.specs import BoundedArraySpec, BoundedTensorSpec
from tf_agents.trajectories.time_step import StepType
from tf_agents.trajectories.time_step import TimeStep
from dynamics import *
from strategies import *
from utils import *

INV_BOUNDS = (-50.0, 50.0)
WEALTH_BOUNDS = (-100000, 100000)
MAX_DRIFT = 5.0


#MM_RN_control_A
class MarketMakerEnvironmentAgainstStrategicAdversaryWithControllingA(py_environment.PyEnvironment):

    def __init__(self,adversary_policy,eta=0.0, zeta=0.0, discount=1.0):
        super(MarketMakerEnvironmentAgainstStrategicAdversaryWithControllingA, self).__init__(
            handle_auto_reset=True)

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
        self.eta = eta
        self.zeta = zeta

        self._discount = np.asarray(discount, dtype=np.float32)
        self._observation = None
        self.adversary_policy = adversary_policy
        self.adversary_policy_state = self.adversary_policy.get_initial_state(batch_size=1)

    def action_spec(self):
        return BoundedArraySpec((2,), np.float32, name='action', minimum=[0.0, 0.0], maximum=[3.0, 3.0])

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
        self.adversary_policy_state = self.adversary_policy.get_initial_state(batch_size=1)
        return TimeStep(StepType.FIRST, np.asarray(0.0, dtype=np.float32), self._discount, self._get_observation())

    def _step(self, action: np.ndarray):
        ask_offset = action[0]
        bid_offset = action[1]
        ask_price = self.dynamics.price + ask_offset
        bid_price = self.dynamics.price - bid_offset
        self.reward = self.inv * self.dynamics.innovate() -self.zeta * self.inv**2
        self._do_executions(ask_price, bid_price)

        time_step = TimeStep(converter(StepType.MID), converter(np.asarray(self.reward, dtype=np.float32)),
                             converter(self._discount), converter(self._get_observation()))
        policy_step = self.adversary_policy.action(time_step, self.adversary_policy_state)
        adversary_action = policy_step.action[0]

        self.baseline_arrival_rate = float(adversary_action[0])
        
        self.dynamics.execution_dynamics_ask = HawkesArrivalModel(self.dt,self.baseline_arrival_rate,self.mean_reversion_speed,self.jump_size,self.decay)
        self.dynamics.execution_dynamics_bid = HawkesArrivalModel(self.dt,self.baseline_arrival_rate,self.mean_reversion_speed,self.jump_size,self.decay)
        self.dynamics.price_dynamics=BrownianMotionWithDrift(self.dt, self.drift, self.volatility)

        self.adversary_policy_state = policy_step.state

        if self.is_terminal():
            self.wealth += self.dynamics.price * self.inv
            self.reward -= self.eta * self.inv ** 2

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
                self.reward += ask_offset
                self.wealth += ask_price

        if self.inv < INV_BOUNDS[1]:
            bid_offset = self.dynamics.try_execute_bid(bid_price)
            if bid_offset:
                self.inv += 1.0
                self.reward += bid_offset
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

#MM_RN_control_Drift
class MarketMakerEnvironmentAgainstStrategicAdversaryWithControllingDrift(py_environment.PyEnvironment):

    def __init__(self, adversary_policy,eta=0.0, zeta=0.0, discount=1.0):
        super(MarketMakerEnvironmentAgainstStrategicAdversaryWithControllingDrift, self).__init__(
            handle_auto_reset=True)

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
        self.eta = eta
        self.zeta = zeta

        self._discount = np.asarray(discount, dtype=np.float32)
        self._observation = None
        self.adversary_policy = adversary_policy
        self.adversary_policy_state = self.adversary_policy.get_initial_state(batch_size=1)

    def action_spec(self):
        return BoundedArraySpec((2,), np.float32, name='action', minimum=[0.0, 0.0], maximum=[3.0, 3.0])

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
        self.adversary_policy_state = self.adversary_policy.get_initial_state(batch_size=1)
        return TimeStep(StepType.FIRST, np.asarray(0.0, dtype=np.float32), self._discount, self._get_observation())

    def _step(self, action: np.ndarray):
        ask_offset = action[0]
        bid_offset = action[1]
        ask_price = self.dynamics.price + ask_offset
        bid_price = self.dynamics.price - bid_offset
        self.reward = self.inv * self.dynamics.innovate() -self.zeta * self.inv**2
        self._do_executions(ask_price, bid_price)

        time_step = TimeStep(converter(StepType.MID), converter(np.asarray(self.reward, dtype=np.float32)),
                             converter(self._discount), converter(self._get_observation()))
        policy_step = self.adversary_policy.action(time_step, self.adversary_policy_state)
        adversary_action = policy_step.action[0]
        drift = float(adversary_action[0])
        self.drift = MAX_DRIFT * (2.0 * drift - 1.0)
        self.dynamics.execution_dynamics_ask = HawkesArrivalModel(self.dt,self.baseline_arrival_rate,self.mean_reversion_speed,self.jump_size,self.decay)
        self.dynamics.execution_dynamics_bid = HawkesArrivalModel(self.dt,self.baseline_arrival_rate,self.mean_reversion_speed,self.jump_size,self.decay)
        self.dynamics.price_dynamics=BrownianMotionWithDrift(self.dt, self.drift, self.volatility)

        self.adversary_policy_state = policy_step.state

        if self.is_terminal():
            self.wealth += self.dynamics.price * self.inv
            self.reward -= self.eta * self.inv ** 2

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
                self.reward += ask_offset
                self.wealth += ask_price

        if self.inv < INV_BOUNDS[1]:
            bid_offset = self.dynamics.try_execute_bid(bid_price)
            if bid_offset:
                self.inv += 1.0
                self.reward += bid_offset
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

#MM_RN_control_K
class MarketMakerEnvironmentAgainstStrategicAdversaryWithControllingK(py_environment.PyEnvironment):

    def __init__(self, adversary_policy,eta=0.0, zeta=0.0, discount=1.0):
        super(MarketMakerEnvironmentAgainstStrategicAdversaryWithControllingK, self).__init__(
            handle_auto_reset=True)

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
        self.eta = eta
        self.zeta = zeta

        self._discount = np.asarray(discount, dtype=np.float32)
        self._observation = None
        self.adversary_policy = adversary_policy
        self.adversary_policy_state = self.adversary_policy.get_initial_state(batch_size=1)

    def action_spec(self):
        return BoundedArraySpec((2,), np.float32, name='action', minimum=[0.0, 0.0], maximum=[3.0, 3.0])

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
        self.adversary_policy_state = self.adversary_policy.get_initial_state(batch_size=1)
        return TimeStep(StepType.FIRST, np.asarray(0.0, dtype=np.float32), self._discount, self._get_observation())

    def _step(self, action: np.ndarray):
        ask_offset = action[0]
        bid_offset = action[1]
        ask_price = self.dynamics.price + ask_offset
        bid_price = self.dynamics.price - bid_offset
        self.reward = self.inv * self.dynamics.innovate() -self.zeta * self.inv**2
        self._do_executions(ask_price, bid_price)

        time_step = TimeStep(converter(StepType.MID), converter(np.asarray(self.reward, dtype=np.float32)),
                             converter(self._discount), converter(self._get_observation()))
        policy_step = self.adversary_policy.action(time_step, self.adversary_policy_state)
        adversary_action = policy_step.action[0]
        self.decay = float(adversary_action[0])

        self.dynamics.execution_dynamics_ask = HawkesArrivalModel(self.dt,self.baseline_arrival_rate,self.mean_reversion_speed,self.jump_size,self.decay)
        self.dynamics.execution_dynamics_bid = HawkesArrivalModel(self.dt,self.baseline_arrival_rate,self.mean_reversion_speed,self.jump_size,self.decay)
        self.dynamics.price_dynamics=BrownianMotionWithDrift(self.dt, self.drift, self.volatility)

        self.adversary_policy_state = policy_step.state

        if self.is_terminal():
            self.wealth += self.dynamics.price * self.inv
            self.reward -= self.eta * self.inv ** 2

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
                self.reward += ask_offset
                self.wealth += ask_price

        if self.inv < INV_BOUNDS[1]:
            bid_offset = self.dynamics.try_execute_bid(bid_price)
            if bid_offset:
                self.inv += 1.0
                self.reward += bid_offset
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

#MM_RN_control_All
class MarketMakerEnvironmentAgainstStrategicAdversaryWithControllingAll(py_environment.PyEnvironment):

    def __init__(self, adversary_policy,eta=0.0, zeta=0.0, discount=1.0):
        super(MarketMakerEnvironmentAgainstStrategicAdversaryWithControllingAll, self).__init__(
            handle_auto_reset=True)

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
        self.eta = eta
        self.zeta = zeta

        self._discount = np.asarray(discount, dtype=np.float32)
        self._observation = None
        self.adversary_policy = adversary_policy
        self.adversary_policy_state = self.adversary_policy.get_initial_state(batch_size=1)

    def action_spec(self):
        return BoundedArraySpec((2,), np.float32, name='action', minimum=[0.0, 0.0], maximum=[3.0, 3.0])

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
        self.adversary_policy_state = self.adversary_policy.get_initial_state(batch_size=1)
        return TimeStep(StepType.FIRST, np.asarray(0.0, dtype=np.float32), self._discount, self._get_observation())

    def _step(self, action: np.ndarray):
        ask_offset = action[0]
        bid_offset = action[1]
        ask_price = self.dynamics.price + ask_offset
        bid_price = self.dynamics.price - bid_offset
        self.reward = self.inv * self.dynamics.innovate() -self.zeta * self.inv**2
        self._do_executions(ask_price, bid_price)

        time_step = TimeStep(converter(StepType.MID), converter(np.asarray(self.reward, dtype=np.float32)),
                             converter(self._discount), converter(self._get_observation()))
        policy_step = self.adversary_policy.action(time_step, self.adversary_policy_state)
        adversary_action = policy_step.action[0]


        drift = float(adversary_action[0])
        self.drift = MAX_DRIFT * (2.0 * drift - 1.0)
        self.baseline_arrival_rate = float(adversary_action[1])
        self.decay = float(adversary_action[2])

        self.dynamics.execution_dynamics_ask = HawkesArrivalModel(self.dt,self.baseline_arrival_rate,self.mean_reversion_speed,self.jump_size,self.decay)
        self.dynamics.execution_dynamics_bid = HawkesArrivalModel(self.dt,self.baseline_arrival_rate,self.mean_reversion_speed,self.jump_size,self.decay)
        self.dynamics.price_dynamics=BrownianMotionWithDrift(self.dt, self.drift, self.volatility)

        self.adversary_policy_state = policy_step.state

        if self.is_terminal():
            self.wealth += self.dynamics.price * self.inv
            self.reward -= self.eta * self.inv ** 2

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
                self.reward += ask_offset
                self.wealth += ask_price

        if self.inv < INV_BOUNDS[1]:
            bid_offset = self.dynamics.try_execute_bid(bid_price)
            if bid_offset:
                self.inv += 1.0
                self.reward += bid_offset
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

#MM_RN_fix
class MarketMakerEnvironmentAgainstFixedAdversary(py_environment.PyEnvironment):

    def __init__(self, eta=0.0, zeta=0.0, discount=1.0):
        super(MarketMakerEnvironmentAgainstFixedAdversary, self).__init__(
            handle_auto_reset=True)

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
        self.eta = eta
        self.zeta = zeta

        self._discount = np.asarray(discount, dtype=np.float32)
        self._observation = None
        
    def action_spec(self):
        return BoundedArraySpec((2,), np.float32, name='action', minimum=[0.0, 0.0], maximum=[3.0, 3.0])

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
        ask_offset = action[0]
        bid_offset = action[1]
        ask_price = self.dynamics.price + ask_offset
        bid_price = self.dynamics.price - bid_offset
        self.reward = self.inv * self.dynamics.innovate() -self.zeta * self.inv**2
        self._do_executions(ask_price, bid_price)

        if self.is_terminal():
            self.wealth += self.dynamics.price * self.inv
            self.reward -= self.eta * self.inv ** 2

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
                self.reward += ask_offset
                self.wealth += ask_price

        if self.inv < INV_BOUNDS[1]:
            bid_offset = self.dynamics.try_execute_bid(bid_price)
            if bid_offset:
                self.inv += 1.0
                self.reward += bid_offset
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

#MM_RN_random
class MarketMakerEnvironmentAgainstRandomAdversary(py_environment.PyEnvironment):

    def __init__(self, eta=0.0, zeta=0.0, discount=1.0):
        super(MarketMakerEnvironmentAgainstRandomAdversary, self).__init__(handle_auto_reset=True)

        self.dt=0.005
        self.volatility=2
        self.decay=1.5
        self.drift=0
        self.baseline_arrival_rate = 10.0
        self.mean_reversion_speed = 60.0
        self.jump_size=40.0
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(self.dt, self.drift, self.volatility),HawkesArrivalModel(self.dt,self.baseline_arrival_rate,self.mean_reversion_speed,self.jump_size,self.decay))

        #self.dynamics = self.random_dynamics()

        self.inv = 0.0
        self.inv_terminal = 0.0
        self.reward = 0.0
        self.wealth = 0.0
        self.eta = eta
        self.zeta = zeta

        self._discount = np.asarray(discount, dtype=np.float32)
        self._observation = None

    def random_dynamics(self):
        '''randomize dynamics
        drift bounds: [-5.0, 5.0]
        baseline_arrival_rate bounds: [7.5, 12.5]
        decay bounds: [1.125, 1.875]
        '''
        epsilon = 1e-10
        drift = np.random.uniform(-5.0,5.0+epsilon)
        self.drift = min(drift,5.0)
        baseline_arrival_rate = np.random.uniform(7.5,12.5+epsilon)
        self.baseline_arrival_rate = min(baseline_arrival_rate,12.5)
        decay = np.random.uniform(1.125,1.875+epsilon)
        self.decay = min(decay,1.875)
        return ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(self.dt,self.drift, self.volatility), HawkesArrivalModel(self.dt,self.baseline_arrival_rate,self.mean_reversion_speed,self.jump_size,self.decay))
        
    def action_spec(self):
        return BoundedArraySpec((2,), np.float32, name='action', minimum=[0.0, 0.0], maximum=[3.0, 3.0])

    def observation_spec(self):
        return BoundedArraySpec((3,), np.float32, name='observation', minimum=[0.0, INV_BOUNDS[0],WEALTH_BOUNDS[0]],
                                maximum=[1.0, INV_BOUNDS[1], WEALTH_BOUNDS[1]])

    def _get_observation(self):
        self._observation = np.asarray([self.dynamics.time, self.inv,self.wealth], dtype=np.float32)
        return self._observation

    def _reset(self):
        self.dynamics = self.random_dynamics()
        self.inv = 0.0
        self.inv_terminal = 0.0
        self.reward = 0.0
        self.wealth = 0.0
        return TimeStep(StepType.FIRST, np.asarray(0.0, dtype=np.float32), self._discount, self._get_observation())

    def _step(self, action: np.ndarray):
        ask_offset = action[0]
        bid_offset = action[1]
        ask_price = self.dynamics.price + ask_offset
        bid_price = self.dynamics.price - bid_offset
        self.reward = self.inv * self.dynamics.innovate() -self.zeta * self.inv**2
        self._do_executions(ask_price, bid_price)

        if self.is_terminal():
            self.wealth += self.dynamics.price * self.inv
            self.reward -= self.eta * self.inv ** 2

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
                self.reward += ask_offset
                self.wealth += ask_price

        if self.inv < INV_BOUNDS[1]:
            bid_offset = self.dynamics.try_execute_bid(bid_price)
            if bid_offset:
                self.inv += 1.0
                self.reward += bid_offset
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

#Evaluate_MarketMakerEnvironmentAgainstFixedAdversary
class Evaluate_MarketMakerEnvironmentAgainstFixedAdversary(py_environment.PyEnvironment):

    def __init__(self, eta=0.0, zeta=0.0, discount=1.0):
        super(Evaluate_MarketMakerEnvironmentAgainstFixedAdversary, self).__init__(
            handle_auto_reset=True)

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
        self.eta = eta
        self.zeta = zeta

        self._discount = np.asarray(discount, dtype=np.float32)
        self._observation = None
        
    def action_spec(self):
        return BoundedArraySpec((2,), np.float32, name='action', minimum=[0.0, 0.0], maximum=[3.0, 3.0])

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
        ask_offset = action[0]
        bid_offset = action[1]
        ask_price = self.dynamics.price + ask_offset
        bid_price = self.dynamics.price - bid_offset
        self.reward = self.inv * self.dynamics.innovate() -self.zeta * self.inv**2
        self._do_executions(ask_price, bid_price)

        if self.is_terminal():
            self.wealth += self.dynamics.price * self.inv
            self.reward -= self.eta * self.inv ** 2

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
                self.reward += ask_offset
                self.wealth += ask_price

        if self.inv < INV_BOUNDS[1]:
            bid_offset = self.dynamics.try_execute_bid(bid_price)
            if bid_offset:
                self.inv += 1.0
                self.reward += bid_offset
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