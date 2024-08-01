#!/usr/bin/env python

from math import log

class LinearUtilityTerminalPenaltyStrategy:
    def __init__(self, k, eta):
        self.k = k
        self.eta = eta
    
    def compute(self, a, price, inventory):
        rp = price - 2.0 * inventory * self.eta
        sp = 2.0 / self.k + self.eta
        return (rp + sp / 2.0 - price, price - (rp - sp / 2.0))

class LinearUtilityStrategy:
    def __init__(self, k):
        self.k = k
    
    def compute(self, a, b, c):
        return (1.0 / self.k, 1.0 / self.k)


class ExponentialUtilityStrategy:
    def __init__(self, k, gamma, volatility):
        self.k = k
        self.gamma = gamma
        self.volatility = volatility
        
    def compute(self, time, price, inventory):
        gss = self.gamma * self.volatility * self.volatility
        rp = price - inventory * gss *(1.0 - time)
        sp = gss * (1.0 - time) + (2.0 / self.gamma) * log(1.0 + self.gamma / self.k)
        return (rp + sp / 2.0 - price, price - (rp - sp / 2.0))