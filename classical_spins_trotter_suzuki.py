import numpy as np 

# from matplotlib import pyplot as plt 

#import runge-kutta 45 order from scipy
from scipy import integrate
from scipy.linalg import block_diag
import multiprocessing

from joblib import Parallel, delayed

num_cores=multiprocessing.cpu_count()
num_cores=1
if num_cores==None:
    num_cores=1

#----------------------------------------------------------------------------------------
#HELPER ROUTINES

#implement rotation for an angle theta about an arbitrary axis using the rodrigues formula
def rodrigues_rot(normal, vector, theta):

    """
    Rotation of a vector 'vector' about an arbitrary normal 'normal'
    for an angle theta, using the Rodrigues rotation formula given below. 

    See https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula for details

    This is needed because spin motion is actually precession around the mean spin of the two 
    spins in the pair. 


    normal: normal about which the precession takes place - must be of unit length

    vector: vector to be rotated

    theta: rotation angle

    In the case of precessing spins, the time dependence of angle is exp(i*J*t) where J is the exchange coupling.
    """
    normal*=1/np.linalg.norm(normal)

    return vector*np.cos(theta) + np.cross(normal, vector)*np.sin(theta) + (normal@vector)*normal*(1-np.cos(theta))


def step_index(t):
    """
    A routine that switches between H1 and H2 depending on the value 
    of time t.
    """

    return 1 + np.heaviside(t%1 - 0.5, 1)

#trotter_suzuki split-step propagation formula
def trotter_suzuki(t, dt, spin, J):

    """
    trotter suzuki split-step time propagation scheme
    
    Also returns energy at each step
    """
    #determine which trotter scheme is relevant (which spins are coupled)
    # step_index=1 + np.heaviside(t%1 - 0.5, 1)

    #adapt delta t if needed 

    before_step=step_index(t)
    after_step=step_index(t+dt)

    if before_step!=after_step:

        dt=before_step/2. - t%1        

    spin=spin.reshape(-1,3)
    N=len(spin)

    if before_step==2.0: #roll if step index is equal to two, so that mean field can be properly calculated
        spin=np.roll(spin,1, axis=0)
   
    #MEAN FIELD PART:
    mean_field=np.sum(spin.reshape(-1,2, spin.shape[-1]),axis=1) #get the mean field values
    mean_field=np.repeat(mean_field,2,axis=0)
    
    spin=np.array(list(map(lambda x,y: rodrigues_rot(x,y, J*dt), mean_field, spin )))

    #calculate energy
    engy=np.sum([spin[2*i,:]@spin[2*i+1,:] for i in range(0, int(N/2))])

    # print(rotate[:-1].ravel(),rotate[1:].ravel())
    if before_step==2.0:
        spin=np.roll(spin,-1,axis=0)

    return spin.ravel(), J*engy, dt


#------------------------------------------------------------------------------------------
#GRAM-SCHMIDT ORTHOGONALIZATION

def gs_coeff(v1,v2):

    return v2@v1/np.linalg.norm(v1)**2
    # return v2@v1

def gs_multiply(coeff, v):

    return list(map((lambda x: x*coeff), v))

def gs_proj(v1,v2):

    return gs_multiply(gs_coeff(v1,v2), v1)

def gs(X, delta, normalize=True):

    """
    Perform GS orthogonalization. 

    INPUT: 
    X - vector array
    delta - initial vector length
    """

    Y=[]
    
    for i in range(len(X)):

        temp_vec=X[i]
        for j,inY in enumerate(Y):

            proj_vec=gs_proj(inY, X[i])

            temp_vec-=proj_vec


        Y.append(temp_vec)

    
    if normalize:
        for Yin in Y:

            Yin/=np.linalg.norm(Yin)

    return np.array(Y)


#--------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------
#heis class implementation
class heis(object):

    """
    Implementation of the classical heisenberg class

    """

    def __init__(self, N, J=1.,t0=0, mode='RND',delta=1e-012, eps=1e04, error=1e-12):
        super(heis, self).__init__()

        #assert that there is an even number of spins
        assert N%2==0

        self.N=N
        self.J=J
        self.t0=t0
        self.t=t0
        self.dt=0
        self.error=error


        self.eps=eps

        #initial spin configuration
        self.set_spin0(mode=mode)
        #initialize the spin configuration vector
        self.spin=self.spin0

        self.get_engy0()
        self.engy=self.engy0

        #initial deviation vectors
        self.set_deviations(delta=delta)
        #initialize the deviation vector
        self.devs=self.devs0

        #benettin algorithm
        self.lyap_time=[]
        self.lyap_coeffs=[]

        #propagate
        self.time_evol=np.array([])
        self.spin_evol=np.array([])
        self.engy_evol=np.array([])


    def get_engy0(self):

        spin=self.spin0.reshape(-1,3)
        engy0=np.sum([spin[2*i,:]@spin[2*i+1,:] for i in range(0, int(self.N/2))])
        self.engy0=self.J*engy0


    def set_spin0(self, mode):

        """
        Sets the initial spin configuration

        input: mode

        Mode: 'FMA' - parallel along the z-axis (ferromagnet)
        Mode: 'AFM' - antiparallel along the z-axis (antiferromagnet)
        Mode: 'RND' - random on the unit sphere (random)
        """

        if (mode=='FMA') or (mode=='AFM'): 
            spin0=np.array([[1.,0.,0.],]*self.N)

            if mode=='AFM':
                #flip every second spin
                spin0[::2]*=-1.

        elif mode=='RND':

            phi=np.random.uniform(0.,2*np.pi, self.N)
            costheta=np.random.uniform(-1,1, self.N)
            theta=np.arccos(costheta)

            spin0=np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]).T
            # print('set_spin0 info: spin0', spin0)
        self.spin0=spin0.ravel()


    def set_deviations(self, delta=1e-013, random=True):

        """
        Delta: length of the initial deviation vector
        

        """

        assert delta>np.finfo(float).eps
        assert delta<1e-03
        self.delta=delta

        def get_deviations(normal):

            x_base = np.random.randn(3)  # take a random vector
            #make x_base orthogonal to normal

            x_base-= x_base.dot(normal)*normal       
            x_base/= np.linalg.norm(x_base)
            # get the second vector
            y_base = np.cross(normal, x_base)

            # x,y,z=np.random.uniform(-1,1,(3,3))

            # x=np.array([delta,0,0])
            # y=np.array([0,delta,0])
            # z=np.array([0,0,delta])


            return x_base*delta, y_base*delta, normal*delta

        # 2N linearly independent deviation vectors are possible, each 
        # has 3N components
        deviations=np.zeros((3*self.N, 3*self.N))

        if not random:
            print('set orthogonal deviations')
            for i, spin in enumerate(self.spin.reshape(-1,3)):

                base=get_deviations(spin)

                for j in range(3):
                    deviations[3*i+j, 3*i:3*(i+1)]=base[j]
        
        elif random:
            print('set random deviations')
            deviations=np.random.uniform(low=-1., high=1., size=(3*self.N, 3*self.N))

            for i, dev in enumerate(deviations):

                deviations[i]*=self.delta/np.linalg.norm(dev)

        self.devs0=deviations


    def trotter_suzuki(self, dt, spin):

        """
        trotter suzuki split-step time propagation scheme
        
        Also returns energy at each step
        """
        #determine which trotter scheme is relevant (which spins are coupled)
        # step_index=1 + np.heaviside(t%1 - 0.5, 1)

        #adapt delta t if needed 

        before_step=step_index(self.t)
        after_step=step_index(self.t+dt)

        # print('step index', before_step, after_step)
        #if the intervals change and we switch from H1 to H2:
        if before_step!=after_step:


            dt=before_step/2. - self.t%1 
            # print('switch needed', self.t, self.t+dt, dt)       

        spin=spin.reshape(-1,3)
        N=len(spin)

        if before_step==2.0: #roll if step index is equal to two, so that mean field can be properly calculated
            spin=np.roll(spin,1, axis=0)
       
        #MEAN FIELD PART:
        mean_field=np.sum(spin.reshape(-1,2, spin.shape[-1]),axis=1) #get the mean field values
        mean_field=np.repeat(mean_field,2,axis=0)
        
        spin=np.array(list(map(lambda x,y: rodrigues_rot(x,y, self.J*dt), mean_field, spin )))
        # spin=np.array([rodrigues_rot(el[0], el[1], self.J*dt) for el in zip(mean_field,spin)])

        #calculate energy
        engy=np.sum([spin[2*i,:]@spin[2*i+1,:] for i in range(0, int(N/2))])

        # print(rotate[:-1].ravel(),rotate[1:].ravel())
        if before_step==2.0:
            spin=np.roll(spin,-1,axis=0)

        self.dt=dt

        return spin.ravel(), self.J*engy





    def propagate(self, tmax, dt):

        """
        Time propagation incorporating the Trotter-Suzuki scheme
        """

        #initialize to default values
        self.t=self.t0
        self.spin=self.spin0

        dt0=dt

        spin_evol=[self.spin0]
        engy_evol=[self.engy0]
        time=[self.t0]

        #intial sums for checking   
        initsums=np.sum(self.spin.reshape(-1,3),axis=0)

        while self.t<=tmax:
            # print('time', self.t)
            dt=dt0
            #propagation
            self.spin, self.engy=self.trotter_suzuki(dt,self.spin)
            # print('spin')
            # print(self.spin)
            #SAFETY CHECK FOR CONSERVATION LAWS:
            checksums=np.sum(self.spin.reshape(-1,3),axis=0)
            checkval=np.linalg.norm(initsums-checksums)
            checknorm=np.linalg.norm(self.spin.reshape(-1,3),axis=1)

            #check norms
            if any(np.abs(checknorm-1)>self.error):
                print('heis.propagate info: Warning! Difference between spin norm and 1 larger than error!')
                #renorm
                spin_reshape=self.spin.reshape(-1,3)
                spin_reshape/=np.linalg.norm(spin_reshape,axis=1)[:,None]
                self.spin=spin_reshape.ravel()


            if checkval>self.error:
                print('heis.propagate info: Warning! Difference between initial and current sum equals {}'.format(checkval))

            
            spin_evol.append(self.spin)
            engy_evol.append(self.engy)

            # print(self.engy)
            self.t+=self.dt
            time.append(self.t)

        self.time_evol=np.array(time)
        self.spin_evol=np.array(spin_evol)
        self.engy_evol=np.array(engy_evol)
        # return np.array(time), np.array(spin_evol), np.array(engy_evol)

    def benettin(self, tmax, dt, eps=1e08):

        """
        Propagate the initial spin configuration along with 3N slightly perturbed ones 
        and track the time evolution of the deviation vectors. Perform GS ortogonalization 
        if needed
        """

        dt0=dt

        self.t=self.t0
        spin=self.spin0
        self.eps=eps

        initsums=np.sum(spin.reshape(-1,3),axis=0)


        #initial deviations
        self.devs=self.devs0

        # print(devs)

        #trajectories - perturbed initial spin configurations that are also time-evolved
        trajectories=spin + self.devs

        renorm_count=0
        while self.t<tmax:

            dt=dt0

            #propagate spin conf forward
            spin=self.trotter_suzuki(dt,spin)[0]
            #renormalize spin
            #check spin norms:
            print('spin norms')

            # print(np.log10(np.abs(1-np.linalg.norm(spin.reshape(-1,3),axis=1) )))

            checksums=np.sum(spin.reshape(-1,3),axis=0)
            checkval=np.linalg.norm(initsums-checksums)
            checknorm=np.linalg.norm(self.spin.reshape(-1,3),axis=1)

            #check norm preservation: 

            # if np.abs(conservation-self.N)>=self.error:
            #     print('Warning, conservation of spin might be violated! Difference:', np.log10(np.abs(self.N-conservation)))
            if any(np.abs(checknorm-1)>self.error):
                print('heis.propagate info: Warning! Difference between spin norm and 1 larger than error!')
                #renorm
                spin_reshape=spin.reshape(-1,3)
                spin_reshape/=np.linalg.norm(spin_reshape,axis=1)[:,None]
                spin=spin_reshape.ravel()  

                trajectories=spin + self.devs  

            if checkval>=self.error:
                print('heis.propagate info: Warning! Difference between initial and current sum equals {}'.format(checkval))


            #propagate deviated trajectories forward in time
            trajectories=np.array([self.trotter_suzuki(dt,trajectory)[0] for trajectory in trajectories     ])

            #obtain deviated vectors
            self.devs=trajectories - spin


            print('time:',self.t, 'renorm_count:', renorm_count)
            print(np.linalg.norm(self.devs, axis=1)/self.delta)



            if any((np.linalg.norm(self.devs, axis=1)/self.delta)>self.eps):
                print('GS NEEDED!')
                
                #perform GS on devs, then reset the trajectories

                #sort deviation vectors according to their norm
                self.devs=gs(self.devs, self.delta, normalize=False)


                coeffs=np.zeros(self.devs.shape[-1])

                for i,dev in enumerate(self.devs):

                    norm=np.linalg.norm(dev)/self.delta


                    self.devs[i]/=norm

                    coeffs[i]=norm

                    
               
                self.lyap_coeffs.append(coeffs)

                self.lyap_time.append(self.t)

                #reset trajectories

                trajectories=spin+self.devs
                renorm_count+=1


            self.t+=dt

        return trajectories



def calc_lyaps(lyap_time, lyap_coeffs):

    lyap_coeffs=np.cumsum(np.log(lyap_coeffs), axis=0).T

    return lyap_coeffs/lyap_time


def calc_lyaps_save(ham,tmax, dt, eps):


    ham.benettin(tmax, dt, eps)

    lyap_time=ham.lyap_time
    lyap_coeffs=ham.lyap_coeffs

    lyap_spec=calc_lyaps(lyap_time, lyap_coeffs)

    #save
    delta=np.log10(ham.delta)
    eps=np.log10(eps)
    savename='new_lyap_spectrum_N_{}_J_{}_delta_{}_eps_{}_.npy'.format(ham.N, ham.J, delta, eps)

    np.save(savename, np.array([lyap_time, lyap_spec]))


#--------------------------------------------------------------------------------------------
#plotting

#test:
if __name__=='__main__':

    dt=0.5
    delta=1e-011
    eps=1e04
    tmax=100

    N=10

    Jlist=[-100,-10,-1.-0.1,0.1,1,10,100]

    hamlist=[heis(N, delta=delta, eps=eps,J=J ) for J in Jlist]


    data=Parallel(n_jobs=num_cores)(delayed(calc_lyaps_save)(ham, tmax,dt, eps) for ham in hamlist)

