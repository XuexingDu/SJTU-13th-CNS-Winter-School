import numpy as np
import brainpy as bp
import brainpy.math as bm
from scipy import stats as spstats

class HH_sbi(bp.NeuGroup):
    def __init__(self, size, ENa=53., gNa=50., EK=-107., gK=5., EL=-70., gL=0.1,
                 V_th= 10., C=1.0, gM=0.07, tau_max=6e2, Vt = -60.0, noise_factor = 0.1, **kwargs):
        # providing the group "size" information
        super(HH_sbi, self).__init__(size=size, **kwargs)

        # initialize parameters from HHsimulator
        self.ENa = ENa
        self.EK = EK
        self.EL = EL
        self.gNa = gNa
        self.gK = gK
        self.gL = gL
        self.C = C
        self.V_th = V_th
        self.gM = gM
        self.tau_max = tau_max
        self.Vt = Vt
        # self.nois_fact = 0.1  # noise factor
        self.noise     = noise_factor

        # initialize variables
        self.V = bm.Variable(bm.random.randn(self.num) - 70.)
        self.m = bm.Variable(0.00168 * bm.ones(self.num))
        self.h = bm.Variable(0.99968 * bm.ones(self.num))
        self.n = bm.Variable(0.00654 * bm.ones(self.num))
        self.p = bm.Variable(0.02931 * bm.ones(self.num))
        self.input = bm.Variable(bm.zeros(self.num))
        self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))
        self.t_last_spike = bm.Variable(bm.ones(self.num) * -1e7)

        # integral functions
        self.int_V = bp.odeint(f=self.dV, method='exp_auto')
        # self.int_V = bp.sdeint(f=self.dV, g =self.dg, method='exp_auto')
        self.int_m = bp.odeint(f=self.dm, method='exp_auto')
        self.int_h = bp.odeint(f=self.dh, method='exp_auto')
        self.int_n = bp.odeint(f=self.dn, method='exp_auto')
        self.int_p = bp.odeint(f=self.dp, method='exp_auto')

    def dV(self, V, t, m, h, n, p, Iext):
        I_Na = (self.gNa * m ** 3.0 * h) * (V - self.ENa)
        I_K = (self.gK * n ** 4.0) * (V - self.EK)
        I_leak = self.gL * (V - self.EL)
        I_M = self.gM * p * (V - self.EK)
        dVdt = (- I_Na - I_K - I_leak - I_M + Iext) / self.C
        return dVdt

    def dg(self, V, t):
        return self.noise

    def dm(self, m, t, V):
        v1    = -0.25 * (V - self.Vt - 13.)
        alpha = 0.32 * (v1/(bm.exp(v1)-1)) / 0.25
        v2    = 0.2 * (V - self.Vt - 40.)
        beta  = 0.28 * (v2/(bm.exp(v2)-1)) / 0.2
        dmdt = alpha * (1 - m) - beta * m
        return dmdt

    def dh(self, h, t, V):
        v1    = V - self.Vt - 17.0
        alpha = 0.128 * bm.exp(-v1 / 18.0)
        v2    = V - self.Vt - 40.
        beta = 4.0 / (1 + bm.exp(-0.2 * v2))
        dhdt = alpha * (1 - h) - beta * h
        return dhdt

    def dn(self, n, t, V):
        v1    = -0.2 * (V - self.Vt - 15.0)
        alpha = 0.032 * (v1/(bm.exp(v1) - 1)) / 0.2
        v2    = V - self.Vt - 10.0
        beta =  0.5 * bm.exp(-v2 / 40)
        dndt = alpha * (1 - n) - beta * n
        return dndt

    def p_inf(self,V):
        v1 = V + 35.0
        return 1.0 / (1.0 + bm.exp(-0.1 * v1))

    def tau_p(self,V):
        v1 = V + 35.0
        return self.tau_max / (3.3 * bm.exp(0.05 * v1) + bm.exp(-0.05 * v1))

    def dp(self, p, t, V):
        p_inf = self.p_inf(V)
        tau_p = self.tau_p(V)
        dpdt = (p_inf - p) / tau_p
        return dpdt

    def update(self, x=None):
        _t = bp.share.load('t')
        _dt= bp.share.load('dt')
        # compute V, m, h, n, p at the next time step
        noise_add    = self.noise * bm.random.randn(self.num) / bm.sqrt(_dt)
        V            = self.int_V(self.V, _t, self.m, self.h, self.n, self.p, self.input + noise_add, dt=_dt)
        self.h.value = self.int_h(self.h, _t, self.V, dt=_dt)
        self.m.value = self.int_m(self.m, _t, self.V, dt=_dt)
        # self.m.value = self.int_m(self.m, _t, self.V, dt=_dt)
        self.n.value = self.int_n(self.n, _t, self.V, dt=_dt)
        self.p.value = self.int_p(self.p, _t, self.V, dt=_dt)

        # update the spiking state and the last spiking time
        self.spike.value = bm.logical_and(self.V < self.V_th, V >= self.V_th)
        self.t_last_spike.value = bm.where(self.spike, _t, self.t_last_spike)

        # update V
        self.V.value = V

        # reset the external input
        self.input[:] = 0.