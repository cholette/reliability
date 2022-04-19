# note that all packages must have licenses that permit commercial use!
from scipy import stats as stats
from scipy import optimize as opt
from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt
import numdifftools as ndt

class reliability_distribution(stats.rv_continuous):
    def __init__(self,*args,**kwargs): #need *args and **kwargs so that I pass these into the methods inherited from the parent class!
        # self = stats.rv_continuous.__init__(self,*args,**kwargs)
        super(reliability_distribution,self).__init__()
        self.a=0  # alter support from 0 to \infty (support is self.a <= x <= self.b)    
    def reliability(self,x,*args,**kwargs):
        return self.sf(x,*args,**kwargs)
    def log_reliability(self,x,*args,**kwargs):
        return self.logsf(x,*args,**kwargs)
    def hazard(self,x,*args,**kwargs):
        return np.exp(self.logpdf(x,*args,**kwargs)-self.log_reliability(x,*args,**kwargs))
    def conditional_reliability(self,tau,t0):
        return np.exp(self.log_reliability(t0+tau) - self.log_reliability(t0))
    def fit(self,ti,p0,observed="all",bnds=None): # Overwriting scipy.stats fitting because it seems that it doesn't handle censoring
        result = opt.minimize(self.nnlf,p0,args=(ti,observed),bounds=bnds)
        p_hat = result.x

        if bnds==None or np.all(result.x>1e-6): # Not on bounds. Calculate confidence intervals based on Hessian
            H = ndt.Hessian(self.nnlf)(result.x,ti,observed)
            Hi = np.linalg.inv(H)
            s = np.sqrt(np.diag(Hi))
            p_ci = p_hat + 1.96*np.array([-s,s])
        else:
            print("Warning: Parameter estimates are on the boundary. Confidence intervals cannot be obtained.")
            p_ci = np.nan*np.ones((p_hat.shape[0],p_hat.shape[0]))
        return p_hat, p_ci
    def fit_interval(self,ti,ins,p0,observed="all",bnds=None): # Overwriting scipy.stats fitting because it seems that it doesn't handle censoring
        result = opt.minimize(self.nnlf_interval,p0,args=(ti,ins,observed),bounds=bnds)
        p_hat = result.x

        if bnds==None or np.all(result.x>1e-6): # Not on bounds. Calculate confidence intervals based on Hessian
            H = ndt.Hessian(self.nnlf)(result.x,ti,observed)
            Hi = np.linalg.inv(H)
            s = np.sqrt(np.diag(Hi))
            p_ci = p_hat + 1.96*np.array([-s,s])
        else:
            print("Warning: Parameter estimates are on the boundary. Confidence intervals cannot be obtained.")
            p_ci = np.nan*np.ones((p_hat.shape[0],p_hat.shape[0]))
        return p_hat, p_ci   

    def freeze(self, *args, **kwds):
        return reliability_distribution_frozen(self, *args, **kwds) # freeze using new reliabilty class, otherwise new functions won't be defined (e.g. reliability)

class reliability_distribution_frozen(stats._distn_infrastructure.rv_frozen):
    def reliability(self,x):
        return self.dist.sf(x,*self.args, **self.kwds)
    def log_reliability(self,x):
        return self.dist.logsf(x,*self.args, **self.kwds)
    def hazard(self,x):
        return self.dist.hazard(x,*self.args, **self.kwds)
    def conditional_reliability(self,tau,t0):
        return np.exp(self.dist.log_reliability(t0+tau,*self.args, **self.kwds) - self.dist.log_reliability(t0,*self.args, **self.kwds))

class reliability_from_hazard(reliability_distribution):
    def __init__(self,h,*args,**kwargs):
        self.hazard = h
        self.cumulative_hazard = None
        self = reliability_distribution.__init__(self,*args,**kwargs)
    def integrate_hazard(self,t,verb=False):
        if verb:
            print("Integrating hazard ... ")

        cdf_single = lambda t1,t2: quad(self.hazard,t1,t2)[0]
        if len(t)>1:
            L = np.vectorize(cdf_single)
            ints = np.array(L(t[0:-1],t[1::]))
            c = np.cumsum(np.append([0],ints))
        else:
            c = [cdf_single(0,t)]

        self.cumulative_hazard = np.stack((np.array(t),np.array(c)),axis=1)

        if verb:
            print("Done!")
        
    def _cdf(self,times):
        if len(times)<2 or np.all(self.cumulative_hazard == None) or (not np.all(times==self.cumulative_hazard[1::,0])): # not sure why, but times doesn't contain zero. There must be somthing about the cdf call that causes this.
            self.integrate_hazard(times)
        else:
            print("Cumulative hazard already computed")

        return 1-np.exp(-self.cumulative_hazard[:,1])

class expdist(reliability_distribution):
    def _pdf(self,t):
        return np.exp(-t)
    def _cdf(self,t):
        return 1-np.exp(-t)
    def nnlf(self,mu0,ti,observed="all"):
        loc = 0
        scale = mu0

        # handle case where all are observed
        if isinstance(observed,str) and observed == "all":
            observed = np.ones(ti.shape)
        
        loglike = sum(self.logpdf(ti[observed==1],loc,scale)) + \
            sum(self.logsf(ti[observed==0],loc,scale)) # deals with right censoring
        
        return -loglike
    def fit(self,ti,observed="all",bnds=None):
        if observed == "all":
            r = len(ti)
        else:
            r = np.sum(observed)

        p_hat = r/np.sum(ti)
        s = p_hat/np.sqrt(r)
        p_ci = p_hat + 1.96*np.array([-s,s])
        
        return p_hat, p_ci
        
class weibull(reliability_distribution):
    def _pdf(self,t,beta):
        return stats.distributions.weibull_min.pdf(t,beta)
    def _cdf(self,t,beta):
        return stats.distributions.weibull_min.cdf(t,beta)
    def _sf(self,t,beta):
        return stats.distributions.weibull_min.sf(t,beta)
    def _logsf(self,t,beta):
        return stats.distributions.weibull_min.logsf(t,beta)
    def _logpdf(self,t,beta):
        return stats.distributions.weibull_min.logpdf(t,beta)
    def nnlf(self,p,ti,observed="all"):
        loc = 0
        eta = p[0]
        beta = p[1]

        # handle case where all are observed
        if isinstance(observed,str) and observed == "all":
            observed = np.ones(ti.shape)
        
        loglike = sum(self.logpdf(ti[observed==1],beta,loc,eta)) + \
            sum(self.log_reliability(ti[observed==0],beta,loc,eta)) # deals with right censoring
        
        return -loglike
    def nnlf_interval(self,p,ti,ins,observed="all"):
        loc = 0
        eta = p[0]
        beta = p[1]

        # handle case where all are observed
        if isinstance(observed,str) and observed == "all":
            observed = np.ones(ti.shape)
        
        idxo, = np.where(observed==1)
        loglike = sum( np.log(self.cdf(ti[idxo],beta,loc,eta)-self.cdf(ins[idxo],beta,loc,eta)) ) +\
            sum(self.log_reliability(ti[observed==0],beta,loc,eta)) # deals with right censoring
        
        return -loglike
    
def ecdf(ti,observed,pos="midpoint",plot=True):
    ti = np.array(ti)
    observed = np.array(observed)
    idx = np.argsort(ti)
    ti = ti[idx]
    observed = observed[idx]
    i = np.arange(0,observed.sum())
    N = len(ti)
    if pos == "midpoint":
        x = ti[np.where(observed==1)]
        Fhat = (i+1-0.5)/N
    elif pos == "mean":
        x = ti[np.where(observed==1)]
        Fhat = (i+1)/(N+1)
    elif pos == "median":
        x = ti[np.where(observed==1)]
        Fhat = (i+1-0.3)/(N+0.4)

    x = np.insert(x,0,0)
    Fhat = np.insert(Fhat,0,0)

    if plot:
        fig, ax = plt.subplots()
        ax.step(x,Fhat,where="post")
        ax.set_xlabel("Time")
        ax.set_ylabel("$\hat{F}$")

    return x,Fhat

def kaplan_meier(ti,observed,plot=True,confidence_interval="greenwood"):
    
    # ensure that inputs are numpy arrays
    ti = np.array(ti)
    observed = np.array(observed)
    
    # sort times in ascending order
    idx = np.argsort(ti)
    ti = ti[idx]
    observed = observed[idx]
    
    # obtain unique time points
    uti,idxu = np.unique(ti,return_index=True)

    N = len(ti)
    Nu = len(uti) 
    ni = N #assets at risk
    Rhat = np.ones(Nu+1)
    S = np.zeros(Nu+1)
    S[0] = 0
    for i in range(0,Nu):
        droppedOut = (ti==uti[i]) # number dropped out at time uti 
        failed = np.logical_and(droppedOut,observed)  
        di = np.sum(failed) # number of failures at time uti

        Rhat[i+1] = Rhat[i]*(ni-di)/ni # Product limit estimation
        if Rhat[i+1]!=0:
            S[i+1] = S[i]+di/(ni*(ni-di)) # Greenwood formula
        else:
            print("Warning: Reliability is zero so no variance can be calculated.")

        ni -= np.sum(droppedOut)
    
    uti = np.insert(uti,0,0)
    Fhat = 1-Rhat

    if plot:
        fig, ax = plt.subplots()
        ax.step(ti,1-Rhat,where="post")
        ax.set_xlabel("Time")
        ax.set_ylabel("$\hat{F}$")    

    if confidence_interval.lower() == "greenwood":
        v = (Rhat**2)*S
        UB = np.clip(Fhat+1.96*np.sqrt(v),None,1)
        LB = np.clip(Fhat-1.96*np.sqrt(v),0,None)
    elif confidence_interval.lower() == "exponential":
        v = 1/np.log(Rhat)**2 * S
        cp = np.log(-np.log(Rhat))+1.96*np.sqrt(v)
        cm = np.log(-np.log(Rhat))-1.96*np.sqrt(v)
        LB = 1-np.exp(-np.exp(cm))
        UB = 1-np.exp(-np.exp(cp))

    return uti,Fhat,LB,UB