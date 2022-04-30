# note that all packages must have licenses that permit commercial use!
from scipy import stats as stats
from scipy import optimize as opt
from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt
import numdifftools as ndt
from matplotlib.ticker import AutoMinorLocator

class reliability_distribution(stats.rv_continuous):
    def __init__(self,*args,**kwargs): #need *args and **kwargs so that I pass these into the methods inherited from the parent class!
        stats.rv_continuous.__init__(self,*args,**kwargs)
        self.a = 0

    def reliability(self,x,*args,**kwargs):
        return self.sf(x,*args,**kwargs)
    
    def log_reliability(self,x,*args,**kwargs):
        return self.logsf(x,*args,**kwargs)
    
    def hazard(self,x,*args,**kwargs):
        return np.exp(self.logpdf(x,*args,**kwargs)-self.log_reliability(x,*args,**kwargs))
   
    def conditional_reliability(self,tau,t0):
        return np.exp(self.log_reliability(t0+tau) - self.log_reliability(t0))
    
    def fit(self,ti,p0,observed="all"): # Overwriting scipy.stats fitting because it seems that it doesn't handle censoring
        y0 = self.transform_scale(p0,direction="forward")
        obj = lambda x: self.nnlf(self.transform_scale(x),ti,observed)
        result = opt.minimize(obj,y0)
        y_hat = result.x
        H = ndt.Hessian(obj)(y_hat)
        p_hat,p_cov = self.transform_scale(y_hat,likelihood_hessian=H,direction="inverse")
        s = np.sqrt(np.diag(p_cov))
        p_ci = p_hat + 1.96*np.array([-s,s])
        
        return p_hat, p_ci.transpose(), p_cov
    
    def fit_interval(self,ti,ins,p0,observed="all",bnds=None): # Overwriting scipy.stats fitting because it seems that it doesn't handle censoring
        y0 = self.transform_scale(p0,direction="forward")
        obj = lambda x: self.nnlf_interval(self.transform_scale(x),ti,ins,observed)
        result = opt.minimize(obj,y0)
        y_hat = result.x
        H = ndt.Hessian(obj)(y_hat)
        p_hat,p_cov = self.transform_scale(y_hat,likelihood_hessian=H,direction="inverse")
        s = np.sqrt(np.diag(p_cov))
        p_ci = p_hat + 1.96*np.array([-s,s]) 

        return p_hat, p_ci.transpose()    
    
    def freeze(self, *args, **kwds):
        return reliability_distribution_frozen(self, *args, **kwds) # freeze using new reliabilty class, otherwise new functions won't be defined (e.g. reliability)
    
    def transform_scale(self,x,likelihood_hessian=None,direction="inverse"):
        # define as unity transform unless overwritten
         return _parameter_transform_identity(x,likelihood_hessian=likelihood_hessian,\
            direction=direction)

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
    
    def transform_scale(self,x,likelihood_hessian=None,direction="inverse"):
        return _parameter_transform_log(x,likelihood_hessian=likelihood_hessian,\
            direction=direction)

class poisson_process:
    def __init__(self,intensity_function,parameters):
        self.parameters = parameters
        self.intensity = intensity_function
    
    def cumulative_intensity(self,t1,t0=0):
        intensity = lambda t: self.intensity(t,*self.parameters)
        LAMBDA = [0]*len(t1)
        for ii in range(len(t1)):
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
            for m in range(len(event_times)):
                assert truncation_times[m] > max(event_times[m]), "Invalid truncation time for asset "+str(m)

        like = 0
        for m in range(len(event_times)): 
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
            for m in range(len(event_times)):
                assert truncation_times[m] > max(event_times[m]), "Invalid truncation time for asset "+str(m)
                tau.append(truncation_times[m])
        else:
            tau.append(max(event_times[m]))
        
        # analytical computation of MLE for observed failure times
        num_failures = 0
        den_beta_hat = 0
        for n in range(len(event_times)):
            failures = event_times[n]
            num_failures += len(failures)
            for k in range(len(failures)):
                den_beta_hat += (np.log(tau[n])-np.log(failures[k]))
        beta_hat = num_failures/den_beta_hat

        sum_truncation_times = 0
        for n in range(len(event_times)):
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
    else:
        raise ValueError("confidence_interval not recognized. Use ""greenwood"" or ""exponential"" """)

    if plot:
        fig, ax = plt.subplots()
        ax.step(uti,Fhat,where="post",label=r"$\hat{F}(t)$",color="blue")
        ax.fill_between(uti,LB,y2=UB,linestyle='--',color="blue",step="post",label="95% CI",alpha=0.1)
        ax.set_xlabel("Time")
        ax.set_ylabel("$\hat{F}(t)$")
        ax.set_ylim((0,ax.get_ylim()[1]))
        plt.legend()
        return uti,Fhat,LB,UB,fig,ax
    else:
        return uti,Fhat,LB,UB

def empirical_mean_cumulative_function(event_times,suspension_times,plot=True,confidence_interval="normal"):
    
    # [1] Chapter 12.1A of Tobias, P.A., Trindade, D., 2011. Applied Reliability, Third. ed. CRC Press LLC, London, United Kingdom.

    # ensure that we have a list of lists
    if (not isinstance(event_times,list)) or (not any(isinstance(el, list) for el in event_times)):
        raise ValueError("event_times must be a list of lists")
        
        
    n_systems = len(event_times)
    tau = suspension_times
    
    # create a single time grid from the flattened event times
    t = np.array([item for sublist in event_times for item in sublist])
    t.sort()
    t = np.unique(t)
    
    n = np.zeros((n_systems,len(t)))
    d = np.zeros((n_systems,len(t)))
    for ii in range(n_systems):
        d[ii,t<=tau[ii]] = 1
        for tij in event_times[ii]:
            n[ii,tij==t] = 1
    
    m_hat = n.sum(axis=0)/d.sum(axis=0)
    M_hat = m_hat.cumsum()
    
    if n_systems == 1:
        print('Warning: Confidence intervals cannot be estimated for a single asset')
        M_UCL = np.nan*np.ones(M_hat.shape)
        M_LCL = M_UCL
    else:
        V_hat = np.sum( np.cumsum( d/d.sum(axis=0)*(n-m_hat),axis=1)**2, axis=0)
        se_hat = np.sqrt(V_hat)
        if confidence_interval.lower() == "normal":
            M_UCL = M_hat + 1.96*se_hat
            M_LCL = M_hat - 1.96*se_hat
        elif confidence_interval == "logit":
            w = np.exp(1.96*se_hat/M_hat)
            M_UCL = w*M_hat
            M_LCL = M_hat/w
        else:
            raise ValueError("confidence_interval not recognized")

    # append zero
    t = np.insert(t,0,0)
    M_hat = np.insert(M_hat,0,0)

    # append last censoring time
    t = np.append(t,max(tau))
    M_hat = np.append(M_hat,M_hat[-1])
    M_LCL = np.append(M_LCL,M_LCL[-1])
    M_UCL = np.append(M_UCL,M_LCL[-1])

    if plot:
        fig, ax = plt.subplots()
        ax.step(t,M_hat,where="post",label=r"$\hat{M}(t)$",linewidth=2,color="blue")
        ax.fill_between(t[1::],M_LCL,y2=M_UCL,linestyle='--',linewidth=2,color="blue",step="post",label="95% CI",alpha=0.1)
        ax.set_xlabel("Time")
        ax.set_ylabel(r"$\hat{M}(t)$") 
        ax.set_ylim((0,ax.get_ylim()[1]))
        plt.legend()  
        return t,M_hat, M_LCL, M_UCL, fig, ax
    else:
        return t,M_hat, M_LCL, M_UCL

def weibull_probability_plot(dist,data=None,ax=None,confidence_bounds=None,parameter_covariance=None):  

    assert isinstance(dist,reliability_distribution_frozen),"Distribution must be frozen before using this function."
    assert type(dist.dist) in [weibull], "Distribution not supported. Must be Weibull for now."
        
    t = np.linspace( dist.ppf(1e-3),dist.ppf(1-1e-3),100 )
    Y = np.log(-np.log(dist.reliability(t)))

    ############################ Nominal plot ##################################
    if ax == None:
        fig, ax = plt.subplots()

    ax.plot(np.log(t),Y,linestyle="--",color="red",label="Distribution")
    ax.set_xlabel(r"Time")
    ax.set_ylabel(r"$F(t)$")

    if (data is not None):
        assert isinstance(data,dict), "Data needs to be a dict with keys [""times"",""ecdf""]."
        k = list(data.keys())
        assert k[0] in ['times','ecdf'], "Data must be a dict with keys [""times"",""ecdf""]."
        assert k[1] in ['times','ecdf'], "Data must be a dict with keys [""times"",""ecdf""]."
        assert len(data['times'])==len(data['ecdf']), "time and Fhat lists must be the same length"

        Yd = np.log(-np.log(1-data['ecdf']))
        ax.plot(np.log(data['times']),Yd,'.',color="blue",label="Data")
        plt.legend()
    
    ######################### confidence bounds ##########################
    if confidence_bounds!=None and confidence_bounds.lower() == "time":
        assert isinstance(parameter_covariance,np.ndarray), "You must supply parameter_covariance to get confidence bounds"
        assert parameter_covariance.shape[0]==2 and parameter_covariance.shape[1] == 2,"Parameter covariance must be 2-by-2"

        a,b = dist.kwds['scale'],dist.args[0]
        R = dist.reliability(t)
        u = np.log(t)

        fun = lambda x: 1/x[1] * np.log(-np.log(R))+np.log(x[0])
        p = np.array([a,b])
        g = ndt.Gradient(fun)(p)
        w = 1.96*np.sqrt( np.sum(g@parameter_covariance*g,axis=1) )
        log_TL,log_TU = u-w,u+w
        ax.fill_betweenx(Y,log_TL,log_TU,label="CI (time)",color='red',alpha=0.1)

    elif confidence_bounds!=None and confidence_bounds.lower() == "reliability":
        assert isinstance(parameter_covariance,np.ndarray), "You must supply parameter_covariance to get confidence bounds"
        assert parameter_covariance.shape[0]==2 and parameter_covariance.shape[1] == 2,"Parameter covariance must be 2-by-2"

        a,b = dist.kwds['scale'],dist.args[0]
        R = dist.reliability(t)
        u = np.log(-np.log(R))
        fun = lambda x: x[1]*(np.log(t)-np.log(x[0]))
        p = np.array([a,b])
        g = ndt.Gradient(fun)(p)
        w = 1.96*np.sqrt( np.sum(g@parameter_covariance*g,axis=1) )
        log_RL,log_RU = u-w,u+w
        ax.fill_between(np.log(t),log_RL,log_RU,label="CI (reliability)",color='red',alpha=0.1)

    ######################### format plot ################################
    yt = np.log(-np.log([0.001,0.01,0.1,0.2,0.4,0.6,0.8,0.9,0.99,0.9999]))
    ax.set_yticks(yt)
    ax.set_yticklabels(["{0:.3f}".format(1-np.exp(-np.exp(x))) for x in yt])  
    ax.grid(visible=True,which="major")
    ax.legend(loc='upper left')

    return ax

def _parameter_transform_log(x,likelihood_hessian=None,direction="inverse"):
        # direction is either "forward" (to log-scaled space) or "inverse" (back to original scale)
        x = np.array(x)
        if direction == "inverse":
            z = np.exp(x)
        elif direction == "forward":
            z = np.log(x)
        else:
            raise ValueError("Transformation direction not recognized.")

        if not isinstance(likelihood_hessian,np.ndarray): # can't use likelihood_hessian == None because it is an array if supplied
            # print("No valid Hessian supplied. Returning only parameter estimates")
            return z
        else:
            #Jacobian for transformation. See Reparameterization at https://en.wikipedia.org/wiki/Fisher_information 
            J = np.diag(np.exp(x))

            if direction == "inverse":
                Ji = np.linalg.inv(J)                            
                z_cov = np.linalg.inv( Ji.transpose() @ likelihood_hessian @ Ji ) 
            elif direction == "forward":
                z_cov = np.linalg.inv( J.transpose() @ likelihood_hessian @ J ) 

            return z,z_cov  

def _parameter_transform_identity(x,likelihood_hessian=None,direction="inverse"):
        z = np.array(x)

        if not isinstance(likelihood_hessian,np.ndarray): # can't use likelihood_hessian == None because it is an array if supplied
            # print("No valid Hessian supplied. Returning only parameter estimates")
            return z
        else:
            J = np.eye((len(z),len(z)))
            z_cov = np.linalg.inv( J.transpose() @ likelihood_hessian @ J ) 
            return z,z_cov 

# def _delta_method(fun,parameters,parameter_cov):
#     # https://en.wikipedia.org/wiki/Delta_method

#     grad = ndt.Gradient(fun)(parameters)
#     return grad.transpose()*parameter_cov*grad
