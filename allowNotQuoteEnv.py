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
MAX_DRIFT = 5.0
WEALTH_BOUNDS = (-100000, 100000)


# Define: 2actions_ControlA
class AllowNotQuoteEnvironment_ControlA(py_environment.PyEnvironment):
    def __init__(self, quote_policy,adversary_policy,eta=0.0, zeta=0.0,discount = 1.0):
        super(AllowNotQuoteEnvironment_ControlA, self).__init__(handle_auto_reset=True)
        
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

        self._discount = np.asarray(discount, dtype = np.float32)
        self._observation = None
        self.quote_policy = quote_policy
        self.quote_policy_state = self.quote_policy.get_initial_state(batch_size=1)
        self.adversary_policy = adversary_policy
        self.adversary_policy_state = self.adversary_policy.get_initial_state(batch_size=1)
    
    def action_spec(self):
        # 0 -> not quote; 1 -> quote
        return BoundedArraySpec((), np.int32, name = 'action', minimum = 0, maximum = 1)
    
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
        self.quote_policy_state = self.quote_policy.get_initial_state(batch_size=1)
        self.adversary_policy_state = self.adversary_policy.get_initial_state(batch_size=1)
        return TimeStep(StepType.FIRST, np.asarray(0.0, dtype=np.float32), self._discount, self._get_observation())
    
    def _step(self, action: np.int32):
        quote = action
        time_step = TimeStep(converter(StepType.MID), converter(np.asarray(self.reward, dtype = np.float32)), converter(self._discount), converter(self._get_observation()))
        if quote:
            policy_step = self.quote_policy.action(time_step, self.quote_policy_state)
            self.quote_policy_state = policy_step.state
            quote_action = policy_step.action[0]
            ask_offset = float(quote_action[0])
            bid_offset = float(quote_action[1])
            ask_price = self.dynamics.price + ask_offset
            bid_price = self.dynamics.price - bid_offset
            self.reward = self.inv * self.dynamics.innovate() -self.zeta * self.inv**2
            self._do_executions(ask_price, bid_price)

        else:
            self.reward = self.inv * self.dynamics.innovate() -self.zeta * self.inv**2

        
        adversary_policy_step = self.adversary_policy.action(time_step, self.adversary_policy_state)
        self.adversary_policy_state = adversary_policy_step.state
        adversary_action = adversary_policy_step.action[0]
        self.baseline_arrival_rate = float(adversary_action[0])
        self.dynamics.execution_dynamics_ask = HawkesArrivalModel(self.dt,self.baseline_arrival_rate,self.mean_reversion_speed,self.jump_size,self.decay)
        self.dynamics.execution_dynamics_bid = HawkesArrivalModel(self.dt,self.baseline_arrival_rate,self.mean_reversion_speed,self.jump_size,self.decay)
        self.dynamics.price_dynamics=BrownianMotionWithDrift(self.dt, self.drift, self.volatility)

        if self.is_terminal():
            self.wealth += self.dynamics.price * self.inv
            self.reward -= self.eta * self.inv ** 2
            
            self.inv_terminal = self.inv
            return TimeStep(StepType.LAST, np.asarray(self.reward, dtype = np.float32), self._discount, self._get_observation())
        return TimeStep(StepType.MID, np.asarray(self.reward, dtype = np.float32), self._discount, self._get_observation())
    
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

#Define: 2actions_ControlDrfit
class AllowNotQuoteEnvironment_ControlDrfit(py_environment.PyEnvironment):
    def __init__(self, quote_policy,adversary_policy,eta=0.0, zeta=0.0,discount = 1.0):
        super(AllowNotQuoteEnvironment_ControlDrfit, self).__init__(handle_auto_reset=True)
        
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

        self._discount = np.asarray(discount, dtype = np.float32)
        self._observation = None
        self.quote_policy = quote_policy
        self.quote_policy_state = self.quote_policy.get_initial_state(batch_size=1)
        self.adversary_policy = adversary_policy
        self.adversary_policy_state = self.adversary_policy.get_initial_state(batch_size=1)
    
    def action_spec(self):
        # 0 -> not quote; 1 -> quote
        return BoundedArraySpec((), np.int32, name = 'action', minimum = 0, maximum = 1)
    
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
        self.quote_policy_state = self.quote_policy.get_initial_state(batch_size=1)
        self.adversary_policy_state = self.adversary_policy.get_initial_state(batch_size=1)
        return TimeStep(StepType.FIRST, np.asarray(0.0, dtype=np.float32), self._discount, self._get_observation())
    
    def _step(self, action: np.int32):
        quote = action
        time_step = TimeStep(converter(StepType.MID), converter(np.asarray(self.reward, dtype = np.float32)), converter(self._discount), converter(self._get_observation()))
        if quote:
            policy_step = self.quote_policy.action(time_step, self.quote_policy_state)
            self.quote_policy_state = policy_step.state
            quote_action = policy_step.action[0]
            ask_offset = float(quote_action[0])
            bid_offset = float(quote_action[1])
            ask_price = self.dynamics.price + ask_offset
            bid_price = self.dynamics.price - bid_offset
            self.reward = self.inv * self.dynamics.innovate() -self.zeta * self.inv**2
            self._do_executions(ask_price, bid_price)

        else:
            self.reward = self.inv * self.dynamics.innovate() -self.zeta * self.inv**2

        
        adversary_policy_step = self.adversary_policy.action(time_step, self.adversary_policy_state)
        self.adversary_policy_state = adversary_policy_step.state
        adversary_action = adversary_policy_step.action[0]
        drift = float(adversary_action[0])
        self.drift = MAX_DRIFT * (2.0 * drift - 1.0)
        self.dynamics.execution_dynamics_ask = HawkesArrivalModel(self.dt,self.baseline_arrival_rate,self.mean_reversion_speed,self.jump_size,self.decay)
        self.dynamics.execution_dynamics_bid = HawkesArrivalModel(self.dt,self.baseline_arrival_rate,self.mean_reversion_speed,self.jump_size,self.decay)
        self.dynamics.price_dynamics=BrownianMotionWithDrift(self.dt, self.drift, self.volatility)

        if self.is_terminal():
            self.wealth += self.dynamics.price * self.inv
            self.reward -= self.eta * self.inv ** 2
            
            self.inv_terminal = self.inv
            return TimeStep(StepType.LAST, np.asarray(self.reward, dtype = np.float32), self._discount, self._get_observation())
        return TimeStep(StepType.MID, np.asarray(self.reward, dtype = np.float32), self._discount, self._get_observation())
    
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

#Define: 2actions_ControlK
class AllowNotQuoteEnvironment_ControlK(py_environment.PyEnvironment):
    def __init__(self, quote_policy,adversary_policy,eta=0.0, zeta=0.0,discount = 1.0):
        super(AllowNotQuoteEnvironment_ControlK, self).__init__(handle_auto_reset=True)
        
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

        self._discount = np.asarray(discount, dtype = np.float32)
        self._observation = None
        self.quote_policy = quote_policy
        self.quote_policy_state = self.quote_policy.get_initial_state(batch_size=1)
        self.adversary_policy = adversary_policy
        self.adversary_policy_state = self.adversary_policy.get_initial_state(batch_size=1)
    
    def action_spec(self):
        # 0 -> not quote; 1 -> quote
        return BoundedArraySpec((), np.int32, name = 'action', minimum = 0, maximum = 1)
    
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
        self.quote_policy_state = self.quote_policy.get_initial_state(batch_size=1)
        self.adversary_policy_state = self.adversary_policy.get_initial_state(batch_size=1)
        return TimeStep(StepType.FIRST, np.asarray(0.0, dtype=np.float32), self._discount, self._get_observation())
    
    def _step(self, action: np.int32):
        quote = action
        time_step = TimeStep(converter(StepType.MID), converter(np.asarray(self.reward, dtype = np.float32)), converter(self._discount), converter(self._get_observation()))
        if quote:
            policy_step = self.quote_policy.action(time_step, self.quote_policy_state)
            self.quote_policy_state = policy_step.state
            quote_action = policy_step.action[0]
            ask_offset = float(quote_action[0])
            bid_offset = float(quote_action[1])
            ask_price = self.dynamics.price + ask_offset
            bid_price = self.dynamics.price - bid_offset
            self.reward = self.inv * self.dynamics.innovate() -self.zeta * self.inv**2
            self._do_executions(ask_price, bid_price)

        else:
            self.reward = self.inv * self.dynamics.innovate() -self.zeta * self.inv**2

        
        adversary_policy_step = self.adversary_policy.action(time_step, self.adversary_policy_state)
        self.adversary_policy_state = adversary_policy_step.state
        adversary_action = adversary_policy_step.action[0]

        self.decay = float(adversary_action[0])
        self.dynamics.execution_dynamics_ask = HawkesArrivalModel(self.dt,self.baseline_arrival_rate,self.mean_reversion_speed,self.jump_size,self.decay)
        self.dynamics.execution_dynamics_bid = HawkesArrivalModel(self.dt,self.baseline_arrival_rate,self.mean_reversion_speed,self.jump_size,self.decay)
        self.dynamics.price_dynamics=BrownianMotionWithDrift(self.dt, self.drift, self.volatility)


        if self.is_terminal():
            self.wealth += self.dynamics.price * self.inv
            self.reward -= self.eta * self.inv ** 2
            
            self.inv_terminal = self.inv
            return TimeStep(StepType.LAST, np.asarray(self.reward, dtype = np.float32), self._discount, self._get_observation())
        return TimeStep(StepType.MID, np.asarray(self.reward, dtype = np.float32), self._discount, self._get_observation())
    
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

#Define: 2actions_ControlAll
class AllowNotQuoteEnvironment_ControlAll(py_environment.PyEnvironment):
    def __init__(self, quote_policy,adversary_policy,eta=0.0, zeta=0.0,discount = 1.0):
        super(AllowNotQuoteEnvironment_ControlAll, self).__init__(handle_auto_reset=True)
        
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

        self._discount = np.asarray(discount, dtype = np.float32)
        self._observation = None
        self.quote_policy = quote_policy
        self.quote_policy_state = self.quote_policy.get_initial_state(batch_size=1)
        self.adversary_policy = adversary_policy
        self.adversary_policy_state = self.adversary_policy.get_initial_state(batch_size=1)
    
    def action_spec(self):
        # 0 -> not quote; 1 -> quote
        return BoundedArraySpec((), np.int32, name = 'action', minimum = 0, maximum = 1)
    
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
        self.quote_policy_state = self.quote_policy.get_initial_state(batch_size=1)
        self.adversary_policy_state = self.adversary_policy.get_initial_state(batch_size=1)
        return TimeStep(StepType.FIRST, np.asarray(0.0, dtype=np.float32), self._discount, self._get_observation())
    
    def _step(self, action: np.int32):
        quote = action
        time_step = TimeStep(converter(StepType.MID), converter(np.asarray(self.reward, dtype = np.float32)), converter(self._discount), converter(self._get_observation()))
        if quote:
            policy_step = self.quote_policy.action(time_step, self.quote_policy_state)
            self.quote_policy_state = policy_step.state
            quote_action = policy_step.action[0]
            ask_offset = float(quote_action[0])
            bid_offset = float(quote_action[1])
            ask_price = self.dynamics.price + ask_offset
            bid_price = self.dynamics.price - bid_offset
            self.reward = self.inv * self.dynamics.innovate() -self.zeta * self.inv**2
            self._do_executions(ask_price, bid_price)

        else:
            self.reward = self.inv * self.dynamics.innovate() -self.zeta * self.inv**2

        
        adversary_policy_step = self.adversary_policy.action(time_step, self.adversary_policy_state)
        self.adversary_policy_state = adversary_policy_step.state
        adversary_action = adversary_policy_step.action[0]

        drift = float(adversary_action[0])
        self.drift = MAX_DRIFT * (2.0 * drift - 1.0)
        self.baseline_arrival_rate = float(adversary_action[1])
        self.decay = float(adversary_action[2])
        self.dynamics.execution_dynamics_ask = HawkesArrivalModel(self.dt,self.baseline_arrival_rate,self.mean_reversion_speed,self.jump_size,self.decay)
        self.dynamics.execution_dynamics_bid = HawkesArrivalModel(self.dt,self.baseline_arrival_rate,self.mean_reversion_speed,self.jump_size,self.decay)
        self.dynamics.price_dynamics=BrownianMotionWithDrift(self.dt, self.drift, self.volatility)
       
        if self.is_terminal():
            self.wealth += self.dynamics.price * self.inv
            self.reward -= self.eta * self.inv ** 2
            
            self.inv_terminal = self.inv
            return TimeStep(StepType.LAST, np.asarray(self.reward, dtype = np.float32), self._discount, self._get_observation())
        return TimeStep(StepType.MID, np.asarray(self.reward, dtype = np.float32), self._discount, self._get_observation())
    
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

#Define: 2actions_FixedAdversary
class AllowNotQuoteEnvironment_Fix(py_environment.PyEnvironment):
    def __init__(self, quote_policy,eta=0.0, zeta=0.0,discount = 1.0):
        super(AllowNotQuoteEnvironment_Fix, self).__init__(handle_auto_reset=True)
        
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

        self._discount = np.asarray(discount, dtype = np.float32)
        self._observation = None
        self.quote_policy = quote_policy
        self.quote_policy_state = self.quote_policy.get_initial_state(batch_size=1)
    
    def action_spec(self):
        # 0 -> not quote; 1 -> quote
        return BoundedArraySpec((), np.int32, name = 'action', minimum = 0, maximum = 1)
    
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
        self.quote_policy_state = self.quote_policy.get_initial_state(batch_size=1)

        return TimeStep(StepType.FIRST, np.asarray(0.0, dtype=np.float32), self._discount, self._get_observation())
    
    def _step(self, action: np.int32):
        quote = action
        time_step = TimeStep(converter(StepType.MID), converter(np.asarray(self.reward, dtype = np.float32)), converter(self._discount), converter(self._get_observation()))
        if quote:
            policy_step = self.quote_policy.action(time_step, self.quote_policy_state)
            self.quote_policy_state = policy_step.state
            quote_action = policy_step.action[0]
            ask_offset = float(quote_action[0])
            bid_offset = float(quote_action[1])
            ask_price = self.dynamics.price + ask_offset
            bid_price = self.dynamics.price - bid_offset
            self.reward = self.inv * self.dynamics.innovate() -self.zeta * self.inv**2
            self._do_executions(ask_price, bid_price)

        else:
            self.reward = self.inv * self.dynamics.innovate() -self.zeta * self.inv**2

        

        if self.is_terminal():
            self.wealth += self.dynamics.price * self.inv
            self.reward -= self.eta * self.inv ** 2
            
            self.inv_terminal = self.inv
            return TimeStep(StepType.LAST, np.asarray(self.reward, dtype = np.float32), self._discount, self._get_observation())
        return TimeStep(StepType.MID, np.asarray(self.reward, dtype = np.float32), self._discount, self._get_observation())
    
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

#Define: 2actions_RandomAdversary
class AllowNotQuoteEnvironment_Random(py_environment.PyEnvironment):
    def __init__(self, quote_policy,eta=0.0, zeta=0.0,discount = 1.0):
        super(AllowNotQuoteEnvironment_Random, self).__init__(handle_auto_reset=True)
        
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

        self._discount = np.asarray(discount, dtype = np.float32)
        self._observation = None
        self.quote_policy = quote_policy
        self.quote_policy_state = self.quote_policy.get_initial_state(batch_size=1)


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
        # 0 -> not quote; 1 -> quote
        return BoundedArraySpec((), np.int32, name = 'action', minimum = 0, maximum = 1)
    
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
        self.quote_policy_state = self.quote_policy.get_initial_state(batch_size=1)

        return TimeStep(StepType.FIRST, np.asarray(0.0, dtype=np.float32), self._discount, self._get_observation())
    
    def _step(self, action: np.int32):
        quote = action
        time_step = TimeStep(converter(StepType.MID), converter(np.asarray(self.reward, dtype = np.float32)), converter(self._discount), converter(self._get_observation()))
        if quote:
            policy_step = self.quote_policy.action(time_step, self.quote_policy_state)
            self.quote_policy_state = policy_step.state
            quote_action = policy_step.action[0]
            ask_offset = float(quote_action[0])
            bid_offset = float(quote_action[1])
            ask_price = self.dynamics.price + ask_offset
            bid_price = self.dynamics.price - bid_offset
            self.reward = self.inv * self.dynamics.innovate() -self.zeta * self.inv**2
            self._do_executions(ask_price, bid_price)

        else:
            self.reward = self.inv * self.dynamics.innovate() -self.zeta * self.inv**2

        

        if self.is_terminal():
            self.wealth += self.dynamics.price * self.inv
            self.reward -= self.eta * self.inv ** 2
            
            self.inv_terminal = self.inv
            return TimeStep(StepType.LAST, np.asarray(self.reward, dtype = np.float32), self._discount, self._get_observation())
        return TimeStep(StepType.MID, np.asarray(self.reward, dtype = np.float32), self._discount, self._get_observation())
    
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

#Evaluate_2actions_MarketMakerEnvironmentAgainstFixedAdversary
class Evaluate_AllowNotQuoteEnvironment_Fix(py_environment.PyEnvironment):
    def __init__(self, quote_policy,eta=0.0, zeta=0.0,discount = 1.0):
        super(Evaluate_AllowNotQuoteEnvironment_Fix, self).__init__(handle_auto_reset=True)
        
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
        self.spread = None
        self.spread_list = []
        self.eta = eta
        self.zeta = zeta

        self._discount = np.asarray(discount, dtype = np.float32)
        self._observation = None
        self.quote_policy = quote_policy
        self.quote_policy_state = self.quote_policy.get_initial_state(batch_size=1)
    
    def action_spec(self):
        # 0 -> not quote; 1 -> quote
        return BoundedArraySpec((), np.int32, name = 'action', minimum = 0, maximum = 1)
    
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
        self.spread = None
        self.spread_list = []
        self.quote_policy_state = self.quote_policy.get_initial_state(batch_size=1)

        return TimeStep(StepType.FIRST, np.asarray(0.0, dtype=np.float32), self._discount, self._get_observation())
    
    def _step(self, action: np.int32):
        quote = action
        time_step = TimeStep(converter(StepType.MID), converter(np.asarray(self.reward, dtype = np.float32)), converter(self._discount), converter(self._get_observation()))
        if quote:
            policy_step = self.quote_policy.action(time_step, self.quote_policy_state)
            quote_action = policy_step.action[0]
            self.quote_policy_state = policy_step.state

            ask_offset = float(quote_action[0])
            bid_offset = float(quote_action[1])
            self.spread = ask_offset + bid_offset
            self.spread_list.append(self.spread)  
            ask_price = self.dynamics.price + ask_offset
            bid_price = self.dynamics.price - bid_offset
            self.reward = self.inv * self.dynamics.innovate() -self.zeta * self.inv**2
            self._do_executions(ask_price, bid_price)

        else:
            self.reward = self.inv * self.dynamics.innovate() -self.zeta * self.inv**2

    
        if self.is_terminal():
            self.wealth += self.dynamics.price * self.inv
            self.reward -= self.eta * self.inv ** 2
            
            self.inv_terminal = self.inv
            return TimeStep(StepType.LAST, np.asarray(self.reward, dtype = np.float32), self._discount, self._get_observation())
        return TimeStep(StepType.MID, np.asarray(self.reward, dtype = np.float32), self._discount, self._get_observation())
    
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

    def get_spread_list(self):
        return self.spread_list





#Define: 4actions_ControlA
class AllowNotQuoteEnvironment_ControlA_4actions(py_environment.PyEnvironment):
    def __init__(self, quote_policy,adversary_policy,eta=0.0, zeta=0.0,discount = 1.0):
        super(AllowNotQuoteEnvironment_ControlA_4actions, self).__init__(handle_auto_reset=True)
        
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

        self._discount = np.asarray(discount, dtype = np.float32)
        self._observation = None
        self.quote_policy = quote_policy
        self.quote_policy_state = self.quote_policy.get_initial_state(batch_size=1)
        self.adversary_policy = adversary_policy
        self.adversary_policy_state = self.adversary_policy.get_initial_state(batch_size=1)
    
    def action_spec(self):
        # 0 -> not quote; 1 -> quote(ask&bid); 2 -> only ask; 3 -> only bid
        return BoundedArraySpec((), np.int32, name = 'action', minimum = 0, maximum = 3)
       
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
        self.quote_policy_state = self.quote_policy.get_initial_state(batch_size=1)
        self.adversary_policy_state = self.adversary_policy.get_initial_state(batch_size=1)
        return TimeStep(StepType.FIRST, np.asarray(0.0, dtype=np.float32), self._discount, self._get_observation())
    
    def _step(self, action: np.int32):
        quote = action
        time_step = TimeStep(converter(StepType.MID), converter(np.asarray(self.reward, dtype = np.float32)), converter(self._discount), converter(self._get_observation()))
        if quote:
            policy_step = self.quote_policy.action(time_step, self.quote_policy_state)
            self.quote_policy_state = policy_step.state
            quote_action = policy_step.action[0]
            ask_offset = float(quote_action[0])
            bid_offset = float(quote_action[1])
            ask_price = self.dynamics.price + ask_offset
            bid_price = self.dynamics.price - bid_offset
            self.reward = self.inv * self.dynamics.innovate()-self.zeta * self.inv**2
            if quote ==1: 
                self._do_executions_ask(ask_price)
                self._do_executions_bid(bid_price)
            elif quote==2: 
                self._do_executions_ask(ask_price)
            elif quote==3:
                self._do_executions_bid(bid_price)
        else:
            self.reward = self.inv * self.dynamics.innovate()-self.zeta * self.inv**2
  
        adversary_policy_step = self.adversary_policy.action(time_step, self.adversary_policy_state)
        self.adversary_policy_state = adversary_policy_step.state
        adversary_action = adversary_policy_step.action[0]

        self.baseline_arrival_rate = float(adversary_action[0])
        self.dynamics.execution_dynamics_ask = HawkesArrivalModel(self.dt,self.baseline_arrival_rate,self.mean_reversion_speed,self.jump_size,self.decay)
        self.dynamics.execution_dynamics_bid = HawkesArrivalModel(self.dt,self.baseline_arrival_rate,self.mean_reversion_speed,self.jump_size,self.decay)
        self.dynamics.price_dynamics=BrownianMotionWithDrift(self.dt, self.drift, self.volatility)


        if self.is_terminal():
            self.wealth += self.dynamics.price * self.inv
            self.reward -= self.eta * self.inv ** 2
            
            self.inv_terminal = self.inv
            return TimeStep(StepType.LAST, np.asarray(self.reward, dtype = np.float32), self._discount, self._get_observation())
        return TimeStep(StepType.MID, np.asarray(self.reward, dtype = np.float32), self._discount, self._get_observation())
    
    def _do_executions_ask(self, ask_price):
        if self.inv > INV_BOUNDS[0]:
            ask_offset = self.dynamics.try_execute_ask(ask_price)
            if ask_offset:
                self.inv -= 1.0
                self.reward += ask_offset
                self.wealth += ask_price
    def _do_executions_bid(self, bid_price):
        if self.inv < INV_BOUNDS[1]:
            bid_offset = self.dynamics.try_execute_bid(bid_price)
            if bid_offset:
                self.inv += 1.0
                self.reward += bid_offset
                self.wealth -= bid_price
        
    def is_terminal(self):
        return self.dynamics.time >= 1.0

#Define: 4actions_ControlDrfit
class AllowNotQuoteEnvironment_ControlDrfit_4actions(py_environment.PyEnvironment):
    def __init__(self, quote_policy,adversary_policy,eta=0.0, zeta=0.0,discount = 1.0):
        super(AllowNotQuoteEnvironment_ControlDrfit_4actions, self).__init__(handle_auto_reset=True)
        
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

        self._discount = np.asarray(discount, dtype = np.float32)
        self._observation = None
        self.quote_policy = quote_policy
        self.quote_policy_state = self.quote_policy.get_initial_state(batch_size=1)
        self.adversary_policy = adversary_policy
        self.adversary_policy_state = self.adversary_policy.get_initial_state(batch_size=1)
    
    def action_spec(self):
        # 0 -> not quote; 1 -> quote(ask&bid); 2 -> only ask; 3 -> only bid
        return BoundedArraySpec((), np.int32, name = 'action', minimum = 0, maximum = 3)
       
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
        self.quote_policy_state = self.quote_policy.get_initial_state(batch_size=1)
        self.adversary_policy_state = self.adversary_policy.get_initial_state(batch_size=1)
        return TimeStep(StepType.FIRST, np.asarray(0.0, dtype=np.float32), self._discount, self._get_observation())
    
    def _step(self, action: np.int32):
        quote = action
        time_step = TimeStep(converter(StepType.MID), converter(np.asarray(self.reward, dtype = np.float32)), converter(self._discount), converter(self._get_observation()))
        if quote:
            policy_step = self.quote_policy.action(time_step, self.quote_policy_state)
            self.quote_policy_state = policy_step.state
            quote_action = policy_step.action[0]
            ask_offset = float(quote_action[0])
            bid_offset = float(quote_action[1])
            ask_price = self.dynamics.price + ask_offset
            bid_price = self.dynamics.price - bid_offset
            self.reward = self.inv * self.dynamics.innovate()-self.zeta * self.inv**2
            if quote ==1: 
                self._do_executions_ask(ask_price)
                self._do_executions_bid(bid_price)
            elif quote==2: 
                self._do_executions_ask(ask_price)
            elif quote==3:
                self._do_executions_bid(bid_price)
        else:
            self.reward = self.inv * self.dynamics.innovate()-self.zeta * self.inv**2
  
        adversary_policy_step = self.adversary_policy.action(time_step, self.adversary_policy_state)
        self.adversary_policy_state = adversary_policy_step.state
        adversary_action = adversary_policy_step.action[0]

        drift = float(adversary_action[0])
        self.drift = MAX_DRIFT * (2.0 * drift - 1.0)
        self.dynamics.execution_dynamics_ask = HawkesArrivalModel(self.dt,self.baseline_arrival_rate,self.mean_reversion_speed,self.jump_size,self.decay)
        self.dynamics.execution_dynamics_bid = HawkesArrivalModel(self.dt,self.baseline_arrival_rate,self.mean_reversion_speed,self.jump_size,self.decay)
        self.dynamics.price_dynamics=BrownianMotionWithDrift(self.dt, self.drift, self.volatility)


        if self.is_terminal():
            self.wealth += self.dynamics.price * self.inv
            self.reward -= self.eta * self.inv ** 2
            
            self.inv_terminal = self.inv
            return TimeStep(StepType.LAST, np.asarray(self.reward, dtype = np.float32), self._discount, self._get_observation())
        return TimeStep(StepType.MID, np.asarray(self.reward, dtype = np.float32), self._discount, self._get_observation())
    
    def _do_executions_ask(self, ask_price):
        if self.inv > INV_BOUNDS[0]:
            ask_offset = self.dynamics.try_execute_ask(ask_price)
            if ask_offset:
                self.inv -= 1.0
                self.reward += ask_offset
                self.wealth += ask_price
    def _do_executions_bid(self, bid_price):
        if self.inv < INV_BOUNDS[1]:
            bid_offset = self.dynamics.try_execute_bid(bid_price)
            if bid_offset:
                self.inv += 1.0
                self.reward += bid_offset
                self.wealth -= bid_price
        
    def is_terminal(self):
        return self.dynamics.time >= 1.0

#Define: 4actions_ControlK
class AllowNotQuoteEnvironment_ControlK_4actions(py_environment.PyEnvironment):
    def __init__(self, quote_policy,adversary_policy,eta=0.0, zeta=0.0,discount = 1.0):
        super(AllowNotQuoteEnvironment_ControlK_4actions, self).__init__(handle_auto_reset=True)
        
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

        self._discount = np.asarray(discount, dtype = np.float32)
        self._observation = None
        self.quote_policy = quote_policy
        self.quote_policy_state = self.quote_policy.get_initial_state(batch_size=1)
        self.adversary_policy = adversary_policy
        self.adversary_policy_state = self.adversary_policy.get_initial_state(batch_size=1)
    
    def action_spec(self):
        # 0 -> not quote; 1 -> quote(ask&bid); 2 -> only ask; 3 -> only bid
        return BoundedArraySpec((), np.int32, name = 'action', minimum = 0, maximum = 3)
       
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
        self.quote_policy_state = self.quote_policy.get_initial_state(batch_size=1)
        self.adversary_policy_state = self.adversary_policy.get_initial_state(batch_size=1)
        return TimeStep(StepType.FIRST, np.asarray(0.0, dtype=np.float32), self._discount, self._get_observation())
    
    def _step(self, action: np.int32):
        quote = action
        time_step = TimeStep(converter(StepType.MID), converter(np.asarray(self.reward, dtype = np.float32)), converter(self._discount), converter(self._get_observation()))
        if quote:
            policy_step = self.quote_policy.action(time_step, self.quote_policy_state)
            self.quote_policy_state = policy_step.state
            quote_action = policy_step.action[0]
            ask_offset = float(quote_action[0])
            bid_offset = float(quote_action[1])
            ask_price = self.dynamics.price + ask_offset
            bid_price = self.dynamics.price - bid_offset
            self.reward = self.inv * self.dynamics.innovate()-self.zeta * self.inv**2
            if quote ==1: 
                self._do_executions_ask(ask_price)
                self._do_executions_bid(bid_price)
            elif quote==2: 
                self._do_executions_ask(ask_price)
            elif quote==3:
                self._do_executions_bid(bid_price)
        else:
            self.reward = self.inv * self.dynamics.innovate()-self.zeta * self.inv**2
  
        adversary_policy_step = self.adversary_policy.action(time_step, self.adversary_policy_state)
        self.adversary_policy_state = adversary_policy_step.state
        adversary_action = adversary_policy_step.action[0]

        self.decay = float(adversary_action[0])
        self.dynamics.execution_dynamics_ask = HawkesArrivalModel(self.dt,self.baseline_arrival_rate,self.mean_reversion_speed,self.jump_size,self.decay)
        self.dynamics.execution_dynamics_bid = HawkesArrivalModel(self.dt,self.baseline_arrival_rate,self.mean_reversion_speed,self.jump_size,self.decay)
        self.dynamics.price_dynamics=BrownianMotionWithDrift(self.dt, self.drift, self.volatility)


        if self.is_terminal():
            self.wealth += self.dynamics.price * self.inv
            self.reward -= self.eta * self.inv ** 2
            
            self.inv_terminal = self.inv
            return TimeStep(StepType.LAST, np.asarray(self.reward, dtype = np.float32), self._discount, self._get_observation())
        return TimeStep(StepType.MID, np.asarray(self.reward, dtype = np.float32), self._discount, self._get_observation())
    
    def _do_executions_ask(self, ask_price):
        if self.inv > INV_BOUNDS[0]:
            ask_offset = self.dynamics.try_execute_ask(ask_price)
            if ask_offset:
                self.inv -= 1.0
                self.reward += ask_offset
                self.wealth += ask_price
    def _do_executions_bid(self, bid_price):
        if self.inv < INV_BOUNDS[1]:
            bid_offset = self.dynamics.try_execute_bid(bid_price)
            if bid_offset:
                self.inv += 1.0
                self.reward += bid_offset
                self.wealth -= bid_price
        
    def is_terminal(self):
        return self.dynamics.time >= 1.0

#Define: 4actions_ControlAll
class AllowNotQuoteEnvironment_ControlAll_4actions(py_environment.PyEnvironment):
    def __init__(self, quote_policy,adversary_policy,eta=0.0, zeta=0.0,discount = 1.0):
        super(AllowNotQuoteEnvironment_ControlAll_4actions, self).__init__(handle_auto_reset=True)
        
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

        self._discount = np.asarray(discount, dtype = np.float32)
        self._observation = None
        self.quote_policy = quote_policy
        self.quote_policy_state = self.quote_policy.get_initial_state(batch_size=1)
        self.adversary_policy = adversary_policy
        self.adversary_policy_state = self.adversary_policy.get_initial_state(batch_size=1)
    
    def action_spec(self):
        # 0 -> not quote; 1 -> quote(ask&bid); 2 -> only ask; 3 -> only bid
        return BoundedArraySpec((), np.int32, name = 'action', minimum = 0, maximum = 3)
       
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
        self.quote_policy_state = self.quote_policy.get_initial_state(batch_size=1)
        self.adversary_policy_state = self.adversary_policy.get_initial_state(batch_size=1)
        return TimeStep(StepType.FIRST, np.asarray(0.0, dtype=np.float32), self._discount, self._get_observation())
    
    def _step(self, action: np.int32):
        quote = action
        time_step = TimeStep(converter(StepType.MID), converter(np.asarray(self.reward, dtype = np.float32)), converter(self._discount), converter(self._get_observation()))
        if quote:
            policy_step = self.quote_policy.action(time_step, self.quote_policy_state)
            self.quote_policy_state = policy_step.state
            quote_action = policy_step.action[0]
            ask_offset = float(quote_action[0])
            bid_offset = float(quote_action[1])
            ask_price = self.dynamics.price + ask_offset
            bid_price = self.dynamics.price - bid_offset
            self.reward = self.inv * self.dynamics.innovate()-self.zeta * self.inv**2
            if quote ==1: 
                self._do_executions_ask(ask_price)
                self._do_executions_bid(bid_price)
            elif quote==2: 
                self._do_executions_ask(ask_price)
            elif quote==3:
                self._do_executions_bid(bid_price)
        else:
            self.reward = self.inv * self.dynamics.innovate()-self.zeta * self.inv**2
  
        adversary_policy_step = self.adversary_policy.action(time_step, self.adversary_policy_state)
        self.adversary_policy_state = adversary_policy_step.state
        adversary_action = adversary_policy_step.action[0]

        drift = float(adversary_action[0])
        self.drift = MAX_DRIFT * (2.0 * drift - 1.0)
        self.baseline_arrival_rate = float(adversary_action[1])
        self.decay = float(adversary_action[2])
        self.dynamics.execution_dynamics_ask = HawkesArrivalModel(self.dt,self.baseline_arrival_rate,self.mean_reversion_speed,self.jump_size,self.decay)
        self.dynamics.execution_dynamics_bid = HawkesArrivalModel(self.dt,self.baseline_arrival_rate,self.mean_reversion_speed,self.jump_size,self.decay)
        self.dynamics.price_dynamics=BrownianMotionWithDrift(self.dt, self.drift, self.volatility)
    

        if self.is_terminal():
            self.wealth += self.dynamics.price * self.inv
            self.reward -= self.eta * self.inv ** 2
            
            self.inv_terminal = self.inv
            return TimeStep(StepType.LAST, np.asarray(self.reward, dtype = np.float32), self._discount, self._get_observation())
        return TimeStep(StepType.MID, np.asarray(self.reward, dtype = np.float32), self._discount, self._get_observation())
    
    def _do_executions_ask(self, ask_price):
        if self.inv > INV_BOUNDS[0]:
            ask_offset = self.dynamics.try_execute_ask(ask_price)
            if ask_offset:
                self.inv -= 1.0
                self.reward += ask_offset
                self.wealth += ask_price
    def _do_executions_bid(self, bid_price):
        if self.inv < INV_BOUNDS[1]:
            bid_offset = self.dynamics.try_execute_bid(bid_price)
            if bid_offset:
                self.inv += 1.0
                self.reward += bid_offset
                self.wealth -= bid_price
        
    def is_terminal(self):
        return self.dynamics.time >= 1.0

# 4actions_FixedAdversary
class AllowNotQuoteEnvironment_Fix_4actions(py_environment.PyEnvironment):
    def __init__(self, quote_policy,eta=0.0, zeta=0.0,discount = 1.0):
        super(AllowNotQuoteEnvironment_Fix_4actions, self).__init__(handle_auto_reset=True)
        
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

        self._discount = np.asarray(discount, dtype = np.float32)
        self._observation = None
        self.quote_policy = quote_policy
        self.quote_policy_state = self.quote_policy.get_initial_state(batch_size=1)
  
    def action_spec(self):
        # 0 -> not quote; 1 -> quote(ask&bid); 2 -> only ask; 3 -> only bid
        return BoundedArraySpec((), np.int32, name = 'action', minimum = 0, maximum = 3)
       
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
        self.quote_policy_state = self.quote_policy.get_initial_state(batch_size=1)
  
        return TimeStep(StepType.FIRST, np.asarray(0.0, dtype=np.float32), self._discount, self._get_observation())
    
    def _step(self, action: np.int32):
        quote = action
        time_step = TimeStep(converter(StepType.MID), converter(np.asarray(self.reward, dtype = np.float32)), converter(self._discount), converter(self._get_observation()))
        if quote:
            policy_step = self.quote_policy.action(time_step, self.quote_policy_state)
            self.quote_policy_state = policy_step.state
            quote_action = policy_step.action[0]
            ask_offset = float(quote_action[0])
            bid_offset = float(quote_action[1])
            ask_price = self.dynamics.price + ask_offset
            bid_price = self.dynamics.price - bid_offset
            self.reward = self.inv * self.dynamics.innovate()-self.zeta * self.inv**2
            if quote ==1: 
                self._do_executions_ask(ask_price)
                self._do_executions_bid(bid_price)
            elif quote==2: 
                self._do_executions_ask(ask_price)
            elif quote==3:
                self._do_executions_bid(bid_price)
        else:
            self.reward = self.inv * self.dynamics.innovate()-self.zeta * self.inv**2


        if self.is_terminal():
            self.wealth += self.dynamics.price * self.inv
            self.reward -= self.eta * self.inv ** 2
            
            self.inv_terminal = self.inv
            return TimeStep(StepType.LAST, np.asarray(self.reward, dtype = np.float32), self._discount, self._get_observation())
        return TimeStep(StepType.MID, np.asarray(self.reward, dtype = np.float32), self._discount, self._get_observation())
    
    def _do_executions_ask(self, ask_price):
        if self.inv > INV_BOUNDS[0]:
            ask_offset = self.dynamics.try_execute_ask(ask_price)
            if ask_offset:
                self.inv -= 1.0
                self.reward += ask_offset
                self.wealth += ask_price
    def _do_executions_bid(self, bid_price):
        if self.inv < INV_BOUNDS[1]:
            bid_offset = self.dynamics.try_execute_bid(bid_price)
            if bid_offset:
                self.inv += 1.0
                self.reward += bid_offset
                self.wealth -= bid_price
        
    def is_terminal(self):
        return self.dynamics.time >= 1.0

#4actions_RandomAdversary
class AllowNotQuoteEnvironment_Random_4actions(py_environment.PyEnvironment):
    def __init__(self, quote_policy,eta=0.0, zeta=0.0,discount = 1.0):
        super(AllowNotQuoteEnvironment_Random_4actions, self).__init__(handle_auto_reset=True)
        
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

        self._discount = np.asarray(discount, dtype = np.float32)
        self._observation = None
        self.quote_policy = quote_policy
        self.quote_policy_state = self.quote_policy.get_initial_state(batch_size=1)

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
        # 0 -> not quote; 1 -> quote(ask&bid); 2 -> only ask; 3 -> only bid
        return BoundedArraySpec((), np.int32, name = 'action', minimum = 0, maximum = 3)
       
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
        self.quote_policy_state = self.quote_policy.get_initial_state(batch_size=1)
  
        return TimeStep(StepType.FIRST, np.asarray(0.0, dtype=np.float32), self._discount, self._get_observation())
    
    def _step(self, action: np.int32):
        quote = action
        time_step = TimeStep(converter(StepType.MID), converter(np.asarray(self.reward, dtype = np.float32)), converter(self._discount), converter(self._get_observation()))
        if quote:
            policy_step = self.quote_policy.action(time_step, self.quote_policy_state)
            self.quote_policy_state = policy_step.state
            quote_action = policy_step.action[0]
            ask_offset = float(quote_action[0])
            bid_offset = float(quote_action[1])
            ask_price = self.dynamics.price + ask_offset
            bid_price = self.dynamics.price - bid_offset
            self.reward = self.inv * self.dynamics.innovate()-self.zeta * self.inv**2
            if quote ==1: 
                self._do_executions_ask(ask_price)
                self._do_executions_bid(bid_price)
            elif quote==2: 
                self._do_executions_ask(ask_price)
            elif quote==3:
                self._do_executions_bid(bid_price)
        else:
            self.reward = self.inv * self.dynamics.innovate()-self.zeta * self.inv**2


        if self.is_terminal():
            self.wealth += self.dynamics.price * self.inv
            self.reward -= self.eta * self.inv ** 2
            
            self.inv_terminal = self.inv
            return TimeStep(StepType.LAST, np.asarray(self.reward, dtype = np.float32), self._discount, self._get_observation())
        return TimeStep(StepType.MID, np.asarray(self.reward, dtype = np.float32), self._discount, self._get_observation())
    
    def _do_executions_ask(self, ask_price):
        if self.inv > INV_BOUNDS[0]:
            ask_offset = self.dynamics.try_execute_ask(ask_price)
            if ask_offset:
                self.inv -= 1.0
                self.reward += ask_offset
                self.wealth += ask_price
    def _do_executions_bid(self, bid_price):
        if self.inv < INV_BOUNDS[1]:
            bid_offset = self.dynamics.try_execute_bid(bid_price)
            if bid_offset:
                self.inv += 1.0
                self.reward += bid_offset
                self.wealth -= bid_price
        
    def is_terminal(self):
        return self.dynamics.time >= 1.0

#Evaluate_4actions_MarketMakerEnvironmentAgainstFixedAdversary
class Evaluate_AllowNotQuoteEnvironment_Fix_4actions(py_environment.PyEnvironment):
    def __init__(self, quote_policy,eta=0.0, zeta=0.0,discount = 1.0):
        super(Evaluate_AllowNotQuoteEnvironment_Fix_4actions, self).__init__(handle_auto_reset=True)
        
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
        self.spread = None
        self.spread_list = []
        self.eta = eta
        self.zeta = zeta

        self._discount = np.asarray(discount, dtype = np.float32)
        self._observation = None
        self.quote_policy = quote_policy
        self.quote_policy_state = self.quote_policy.get_initial_state(batch_size=1)
  
    def action_spec(self):
        # 0 -> not quote; 1 -> quote(ask&bid); 2 -> only ask; 3 -> only bid
        return BoundedArraySpec((), np.int32, name = 'action', minimum = 0, maximum = 3)
       
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
        self.spread = None
        self.spread_list = []
        self.quote_policy_state = self.quote_policy.get_initial_state(batch_size=1)
  
        return TimeStep(StepType.FIRST, np.asarray(0.0, dtype=np.float32), self._discount, self._get_observation())
    
    def _step(self, action: np.int32):
        quote = action
        time_step = TimeStep(converter(StepType.MID), converter(np.asarray(self.reward, dtype = np.float32)), converter(self._discount), converter(self._get_observation()))
        if quote:
            policy_step = self.quote_policy.action(time_step, self.quote_policy_state)
            quote_action = policy_step.action[0]
            self.quote_policy_state = policy_step.state
            if quote ==1: 
                ask_offset = float(quote_action[0])
                bid_offset = float(quote_action[1])
                self.spread = ask_offset + bid_offset
                self.spread_list.append(self.spread)  
                ask_price = self.dynamics.price + ask_offset
                bid_price = self.dynamics.price - bid_offset
                self.reward = self.inv * self.dynamics.innovate()-self.zeta * self.inv**2
                self._do_executions_ask(ask_price)
                self._do_executions_bid(bid_price)
            elif quote ==2:
                ask_offset = float(quote_action[0])
                ask_price = self.dynamics.price + ask_offset
                self.reward = self.inv * self.dynamics.innovate()-self.zeta * self.inv**2
                self._do_executions_ask(ask_price)
            elif quote==3:
                bid_offset = float(quote_action[1])
                bid_price = self.dynamics.price - bid_offset
                self.reward = self.inv * self.dynamics.innovate()-self.zeta * self.inv**2
                self._do_executions_bid(bid_price)
        else:
            self.reward = self.inv * self.dynamics.innovate()-self.zeta * self.inv**2

        if self.is_terminal():
            self.wealth += self.dynamics.price * self.inv
            self.reward -= self.eta * self.inv ** 2
            
            self.inv_terminal = self.inv
            return TimeStep(StepType.LAST, np.asarray(self.reward, dtype = np.float32), self._discount, self._get_observation())
        return TimeStep(StepType.MID, np.asarray(self.reward, dtype = np.float32), self._discount, self._get_observation())
    
    def _do_executions_ask(self, ask_price):
        if self.inv > INV_BOUNDS[0]:
            ask_offset = self.dynamics.try_execute_ask(ask_price)
            if ask_offset:
                self.inv -= 1.0
                self.reward += ask_offset
                self.wealth += ask_price
    def _do_executions_bid(self, bid_price):
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

    def get_spread_list(self):
        return self.spread_list