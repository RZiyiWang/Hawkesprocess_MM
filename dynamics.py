#!/usr/bin/env python
import numpy as np
from math import exp
from math import sqrt
import copy

#ExecutionDynamics
class PoissonRate:
    def __init__(self, dt=0.005,scale = 140.0, decay = 1.5):
        self.dt = dt
        self.scale = scale
        self.decay = decay
    def match_prob(self, offset,*args):
        lamb = self.scale * exp(-self.decay * offset)
        return min(1, max(0, lamb * self.dt))
    

class HawkesArrivalModel:
    def __init__(self, dt=0.005,baseline_arrival_rate=10.0,mean_reversion_speed=60.0,jump_size=40.0,decay=1.5):

        self.dt = dt
        self.baseline_arrival_rate = baseline_arrival_rate
        self.mean_reversion_speed = mean_reversion_speed
        self.jump_size = jump_size
        self.decay = decay
        self.Hawkes_intensity = baseline_arrival_rate
    
    def match_prob(self, offset,last_match_result):
        self.Hawkes_intensity = (
            self.Hawkes_intensity 
            + self.mean_reversion_speed * (self.baseline_arrival_rate - self.Hawkes_intensity)* self.dt
            + self.jump_size * last_match_result )
        lamb = self.Hawkes_intensity * exp(-self.decay * offset)
        return min(1, max(0, lamb * self.dt))

#PriceDynamics

class BrownianMotionWithDrift:
    def __init__(self, dt = 0.005, drift = 0, volatility = 2.0):
        self.dt = dt
        self.volatility = volatility
        self.drift = drift
        
    def sample_increment(self, rng, x):
        w = rng.standard_normal()
        return self.drift * self.dt + self.volatility * sqrt(self.dt) * w
    
class BrownianMotion:
    def __init__(self, dt = 0.005, volatility = 2.0):
        self.dt = dt
        self.volatility = volatility
    
    def sample_increment(self, rng, x):
        w = rng.standard_normal()
        return self.volatility * sqrt(self.dt) * w
    
class OrnsteinUhlenbeck:
    def __init__(self, dt = 1.0, rate = 1.0, volatility = 1.0):
        self.dt = dt
        self.rate = rate
        self.volatility = volatility
        
    def sample_increment(self, rng, x):
        w = BrownianMotion(self.dt, self.volatility)
        return -self.rate * x * self.dt + w.sample_increment(rng, x)

class OrnsteinUhlenbeckWithDrift:
    def __init__(self, dt = 1.0, rate = 1.0, volatility = 1.0, drift = 0):
        self.dt = dt
        self.rate = rate
        self.volatility = volatility
        self.drift = drift
    
    def sample_increment(self, rng, x):
        w = BrownianMotion(self.dt, self.volatility)
        return -self.rate * (self.drift-x) * self.dt + w.sample_increment(rng, x)
    
#ASDynamics

class ASDynamics:
    def __init__(self, dt, price, rng, pd = None, ed = None):
        self._rng = rng
        self.dt = dt
        self.time = 0
        self.price = price
        self.price_initial = price
        self.price_dynamics = pd
        self.execution_dynamics_ask = copy.deepcopy(ed)
        self.execution_dynamics_bid = copy.deepcopy(ed)
        self.last_ask_match_result = False
        self.last_bid_match_result = False
    
    def innovate(self):
        rng = np.random.default_rng()
        price_inc = self.price_dynamics.sample_increment(rng, self.price)
        self.time += self.dt
        self.price += price_inc
        return price_inc
        
    def try_execute_ask(self, order_price):
        offset = order_price - self.price
        match_prob = self.execution_dynamics_ask.match_prob(offset,self.last_ask_match_result)
        self.last_ask_match_result = self._rng.random() < match_prob  
        return offset if self.last_ask_match_result else None
    
    def try_execute_bid(self, order_price):
        offset = self.price - order_price
        match_prob = self.execution_dynamics_bid.match_prob(offset,self.last_bid_match_result)
        self.last_bid_match_result = self._rng.random() < match_prob  
        return offset if self.last_bid_match_result else None

    def get_state(self):
        state = dict()
        state['dt'] = self.dt
        state['time'] = self.time
        state['price'] = self.price
        state['price_initial'] = self.price_initial
        return state

