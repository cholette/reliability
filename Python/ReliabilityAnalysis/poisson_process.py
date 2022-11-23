from scipy import stats as stats
from scipy import optimize as opt
from scipy.integrate import quad
import numpy as np
import numdifftools as ndt
from ReliabilityAnalysis.utilities import _parameter_transform_log

class poisson_process:
    def __init__(self,intensity_function,parameters):
        self.parameters = parameters
        self.intensity = intensity_function
    
    def cumulative_intensity(self,t1,t0=0):
        intensity = lambda t: self.intensity(t,*self.parameters)
        LAMBDA = [0]*len(t1)
        for ii,_ in enumerate(t1):
            if len(list(t0))==1:
                t0ii = t0
            elif len(t0) != len(list(t1)):
                raise ValueError("t0 must be an integer or a list of the same length as t1")
            else:
                t0ii = t0[ii]
            LAMBDA[ii] = quad(intensity,t0ii,t1[ii])[0]
        
        return LAMBDA
    
    def log_intensity(self,t):
        return np.log(self.intensity(t,*self.parameters))

    def reliability(self,t,w):
        return np.exp(-self.cumulative_intensity(w,t0=t))
    
    def pdf(self,t,t_previous=0):
        w = t-t_previous
        return self.intensity(t,self.parameters)*\
            self.reliability(w,t0=t_previous)
    
    def nnlf(self,p,event_times,truncation_times=None):
        # event_times[asset][failure time index], truncation_time=None means that last index is a failure.

        original_parameters = self.parameters
        self.parameters = p

        # turn into a list if tim is a numpy array. Lists are preferred
        # since they can be ragged and have different numbers of event times. 
        if isinstance(event_times,np.ndarray):
            event_times = event_times.tolist()

        # check for valid truncation time
        if truncation_times != None:
            for m,_ in enumerate(event_times):
                assert truncation_times[m] > max(event_times[m]), "Invalid truncation time for asset "+str(m)

        like = 0
        for m,_ in enumerate(event_times): 
            for f in event_times[m]:
                like += self.log_intensity(f)
        
            if truncation_times[m] != None:
                T = truncation_times[m]
                like += -self.cumulative_intensity(T,t0=0)
        
        self.parameters = original_parameters
        return -like

    def fit(self,event_times,p0,truncation_times=None): # Overwriting scipy.stats fitting because it seems that it doesn't handle censoring
        y0 = self.transform_scale(p0,direction="forward")
        obj = lambda x: self.nnlf(self.transform_scale(x),event_times,truncation_times=truncation_times)
        result = opt.minimize(obj,y0)
        y_hat = result.x
        H = ndt.Hessian(obj)(y_hat)
        p_hat,p_cov = self.transform_scale(y_hat,likelihood_hessian=H,direction="inverse")
        s = np.sqrt(np.diag(p_cov))
        p_ci = p_hat + 1.96*np.array([-s,s])

        return p_hat, p_ci.transpose(),p_cov

    def transform_scale(self,x,likelihood_hessian=None,direction="inverse"):
         return _parameter_transform_log(x,likelihood_hessian=likelihood_hessian,\
            direction=direction)

class power_law_nhpp(poisson_process):
    def __init__(self,a,b):
        fun = lambda t: a*b*t**(b-1)
        super().__init__(fun,parameters=[a,b])
    
    def cumulative_intensity(self, t1, t0=0):
        a,b = [*self.parameters]
        return a*(t1**b-t0**b)
    
    def log_intensity(self, t):
        a,b = [*self.parameters]
        return np.log(a)+np.log(b)+(b-1)*np.log(t)
    
    def fit(self,event_times,truncation_times=None):

        # turn into a list if tim is a numpy array. Lists are preferred
        # since they can be ragged and have different numbers of event times. 
        if isinstance(event_times,np.ndarray):
            event_times = event_times.tolist()

        # check for valid truncation time
        tau = []
        if truncation_times != None:
            for m,_ in enumerate(event_times):
                assert truncation_times[m] > max(event_times[m]), "Invalid truncation time for asset "+str(m)
                tau.append(truncation_times[m])
        else:
            tau.append(max(event_times[m]))
        
        # analytical computation of MLE for observed failure times
        num_failures = 0
        den_beta_hat = 0
        for n,_ in enumerate(event_times):
            failures = event_times[n]
            num_failures += len(failures)
            for k,_ in enumerate(failures):
                den_beta_hat += (np.log(tau[n])-np.log(failures[k]))
        beta_hat = num_failures/den_beta_hat

        sum_truncation_times = 0
        for n,_ in enumerate(event_times):
            sum_truncation_times += tau[n]**beta_hat
        alpha_hat = num_failures/sum_truncation_times
        p_hat = [alpha_hat,beta_hat]

        # lazy numerical computation of the Hessian and parameter CIs
        y_hat = self.transform_scale(p_hat,direction='forward')
        obj = lambda x: self.nnlf(self.transform_scale(x),event_times,truncation_times=truncation_times)
        H = ndt.Hessian(obj)(y_hat)
        _,p_cov = self.transform_scale(y_hat,likelihood_hessian=H,direction="inverse")
        s = np.sqrt(np.diag(p_cov))
        p_ci = p_hat + 1.96*np.array([-s,s])

        return p_hat,p_ci.transpose(),p_cov

    def transform_scale(self,x,likelihood_hessian=None,direction="inverse"):
        return _parameter_transform_log(x,likelihood_hessian=likelihood_hessian,\
            direction=direction)

    def mcf_confidence_interval(self,t,p_cov,kind="mcf",c=1.96):
    
        assert kind.lower() in ["time",'mcf'],"kind must be ""time"" or ""mcf""."
        assert all([(t[ii+1]-t[ii])>=0 for ii in range(len(t)-1)]), "time vector must be sorted"
        assert t[0]>=0, "Negative time doesn't make sense!"
        if t[0] == 0:
            print('Warning: inserting nan for t==0 since logM(t) is undefined.')
            prependNaN = True
            t = t[1::]
        else:
            prependNaN = False

        # The below is a bit lazy and uses numerical gradients. Might use analytical gradients later.
        a,b = self.parameters
        p = np.array([a,b])
        M = self.cumulative_intensity(t)
        if kind.lower() == "time":
            u = np.log(t)
            fun = lambda x: 1/x[1] * (np.log(M)-np.log(x[0]))
            g = ndt.Gradient(fun)(p)
            w = c*np.sqrt( np.sum(g@p_cov*g,axis=1) )
            ML,MU = self.cumulative_intensity(np.exp(u+w)),self.cumulative_intensity(np.exp(u-w))

        elif kind.lower() == "mcf":
            u = np.log(M)
            fun = lambda x: x[1]*np.log(t)+np.log(x[0])
            g = ndt.Gradient(fun)(p)
            w = c*np.sqrt( np.sum(g@p_cov*g,axis=1) )
            ML,MU = np.exp(u-w),np.exp(u+w)
        
        if prependNaN:
            ML = np.insert(ML,0,np.nan)
            MU = np.insert(MU,0,np.nan)
        
        return ML,MU
