"""
Dynamic analysis homework assignment

"""


import numpy as np 

from matplotlib import pyplot as plt 

#import runge-kutta 45 order from scipy
from scipy import integrate
from scipy.linalg import block_diag

#------------------------------------------------------------------


#helper routines

def skew(a):

    """
    Returns the cross product as an array:

    a x b= A b, where A is an anti-symmetric array
    """
    a=np.array(a)

    skew=np.zeros((3,3), dtype=np.float)

    skew[0][1]=-a[2]
    skew[0][2]=a[1]
    skew[1][2]=-a[0]

    skew[1][0]=-skew[0][1]
    skew[2][0]=-skew[0][2]
    skew[2][1]=-skew[1][2]

    return skew

def jac_block(spin1,spin2):

    """
    Returns the spin block in the jacobian array
    
    input:

    spin1, spin2 - properly ordered spins in the 
    cross product
    """

    block=np.zeros((6,6), dtype=np.float)

    block[:3,:3]=-skew(spin1)
    block[3:,:3]=-block[:3,:3]

    block[:3,3:]=skew(spin2)
    block[3:,3:]=-block[:3,3:]

    return block

#gram schmidt
def gs_coeff(v1,v2):

    return v2@v1/np.linalg.norm(v1)**2
    # return v2@v1

def gs_multiply(coeff, v):

    return list(map((lambda x: x*coeff), v))

def gs_proj(v1,v2):

    return gs_multiply(gs_coeff(v1,v2), v1)

def gs(X, args_sort, delta, normalize=True):

    """
    Perform GS orthogonalization. 

    INPUT: 
    X - vector array
    args_sort: sorted arguments
    delta - initial vector length
    """

    Y=[]
    args_sort=args_sort[::-1]
    for i in range(len(X)):

        temp_vec=X[args_sort[i]]

        for j,inY in enumerate(Y):


            proj_vec=gs_proj(inY, X[args_sort[i]])

            # print('proj_vec')
            # print(proj_vec)

            # temp_vec=np.array(list(map(lambda x,y: x-y, temp_vec, proj_vec)))
            temp_vec-=proj_vec
            # print('number of inY in Y:',len(Y),'step num:', j+1)

        Y.append(temp_vec)

    
    if normalize:
        for Yin in Y:

            Yin/=np.linalg.norm(Yin)

    return np.array(Y)


#class implementation
class heis(object):

    """
    Implementation of the classical heisenberg class

    """

    def __init__(self, N, J=1.,t0=0, mode='RND',delta=1e-013, eps=1e04):
        super(heis, self).__init__()

        #assert that there is an even number of spins
        assert N%2==0

        self.N=N
        self.J=J
        self.t0=t0
        self.t=t0

        self.eps=eps

        #initial spin configuration
        self.set_spin0(mode=mode)
        #initialize the spin configuration vector
        self.spin=self.spin0

        #initial deviation vectors
        self.set_deviations(delta=delta)
        #initialize the deviation vector
        self.devs=self.devs0

        #benettin algorithm
        self.lyap_time=[]
        self.lyap_coeffs=[]




    def _reverse_arr(self, step_index):
        """
        Array that enables proper cross product multiplication
        of spins
        """
        block=[[0,1],[1,0]]

        _reverse_arr=np.zeros((self.N,self.N))
        if step_index==1:
            
            _reverse_arr=block_diag(*[block]*int(self.N/2))

        elif step_index==2:
            _reverse_arr[-1][0]=1
            _reverse_arr[0][-1]=1
            _reverse_arr[1:-1,1:-1]=block_diag(*[block]*int(self.N/2 -1))

        return _reverse_arr



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

        self.spin0=spin0.ravel()

    #implement equations of motion(EOM):
    def EOM(self, t, spin):

        """
        Which propagation hamiltonian to use. Switches between 
        H1 and H2 according to the value of t. If t>float(n/2),
        H2 is used, if not, H1 is used. 
        """

        #switches between H1 and H2
        step_index=1 + np.heaviside(t%1 - 0.5, 1)

        spin=spin.reshape(-1,3)
        eom=-self.J*np.cross(spin, self._reverse_arr(step_index)@spin)
        # print(eom)
        return eom.ravel()


    def set_deviations(self, delta):

        """
        Set the initial deviation vectors for the spin configuration

        delta - length of the initial deviation vector
        """

        assert delta>np.finfo(float).eps
        assert delta<0.1
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

        for i, spin in enumerate(self.spin.reshape(-1,3)):

            base=get_deviations(spin)

            for j in range(3):
                deviations[3*i+j, 3*i:3*(i+1)]=base[j]

        self.devs0=deviations





    def jacobian(self, t, spin):

        """
        Jacobian for the propagation of the deviation vectors,
        needed in the Benettin algorithm
        """

        #declare the jacobian array
        jac=np.zeros((3*self.N, 3*self.N), dtype=np.float)


        step_index=1 + np.heaviside(t%1 - 0.5, 1)

        spin=spin.reshape(-1,3)
        #create a block in the jacobian


        #properly order spins
        spin=self._reverse_arr(step_index)@spin


        if step_index==1:

            """
            The jacobian construction is straightforward for H1
            """
            #iterate over pairs 2*i, 2*i+1
            jac=-self.J*block_diag(*[jac_block(spin[2*i], spin[2*i+1]) for i in range(int(self.N/2))])

        elif step_index==2:

            """
            If the second hamiltonian H2 is to be considered, then the inner blocks of the jacobian can be
            created using a for loop, while PBC need to be considered separately
            """
            #iterate over pairs 2*i-1, 2*i, starting with i=1
            jac[3:-3, 3:-3]=-self.J*block_diag(*[jac_block(spin[2*i-1], spin[2*i]) for i in range(1,int(self.N/2))])

            #treat the PBC blocks separately

            jac[:3,:3]=-self.J*(-1.)*skew(spin[0])
            jac[-3:,:3]=-jac[:3,:3]

            jac[-3:,-3:]=-self.J*(-1.)*skew(spin[-1])
            jac[:3,-3:]=-jac[-3:,-3:]

        return jac

    def calc_engy(self):

        step_index=1 + np.heaviside(self.t%1 - 0.5, 1)

        spin1=self.spin.reshape(-1,3)
        spin2=self._reverse_arr(step_index)@spin1

        engy=self.J*(spin1.ravel()@spin2.ravel())/2

        return engy

        
    def jac_fun(self, t, devs, spin):

        jac=self.jacobian(t, spin)

        return devs@jac.T

    def propagate(self, tmax):

        """
        Propagation function implementing solve_ivp routine from scipy.integrate
        """

        return integrate.solve_ivp(self.EOM, (self.t0, tmax), self.spin0)



    def int_sys_fun(self, t, system):

        

        """
        Integrate EOM and deviation vectors together. The function is used in the integration
        routine. 
        """
        
        spin=system[:len(self.spin)]
        # print('time:',t)
        #get EOM
        EOM=self.EOM(t, spin )



        devs=system[len(self.spin):].reshape(len(self.devs),-1)

        devs=self.jac_fun(t, devs, spin)




        return np.append(EOM, devs.ravel())






    def prop_benet(self, tmax, dt=0.01, eps=30):

        # assert self.delta<eps
        # self.eps=eps
        # rescale_steps=int(rescale_tau/dt)

        # self.rescale_tau=rescale_tau
        self.lyap_coeffs=[]
        self.lyap_time=[]
        # self.t_eval=np.arange(self.t0, tmax, rescale_tau)
        sol_conf=np.append(self.spin0, self.devs.ravel())

        # self.terminate_benet.terminal=True
        self.t=self.t0

        while self.t<=tmax:

            #propagate for dt
            sol=integrate.solve_ivp(self.int_sys_fun, (self.t, self.t+dt),sol_conf,t_eval=[self.t+dt])
            #rescale devs

            sol_conf=sol.y.T[-1]

            self.spin=sol_conf[:3*self.N]
            self.devs=sol_conf[3*self.N:].reshape(3*self.N,-1)


            normvals=np.linalg.norm(self.devs, axis=1)/self.delta
            scale_factor=1
            
            if any(normvals>=eps):


                
                scale_factor=max(normvals)/eps
                print('GS NEEDED! t:',self.t, 'dt:', dt, 'scale_factor:', scale_factor)
                dt*=1/scale_factor
                # if any(normvals>=1000*eps):
                    # dt*=1/100


                # print('GS NEEDED.', max(np.linalg.norm(self.devs, axis=1))/self.delta)
                # print('norms', np.linalg.norm(self.devs, axis=1))
                sort_args=np.argsort(np.linalg.norm(self.devs, axis=1))
                # print('max dev', self.devs[sort_args[-1]])
                # print('max dev norm', np.linalg.norm(self.devs[sort_args[-1]]))
                # print('max dev norm rescaled wrt delta',np.linalg.norm(self.devs[sort_args[-1]])/self.delta)
                # print('sort args', sort_args)
                self.devs=gs(self.devs, sort_args, self.delta, normalize=False)


                coeffs=np.zeros(len(self.devs))

                for i,dev in enumerate(self.devs):

                    norm=np.linalg.norm(dev)/self.delta


                    self.devs[i]/=norm

                    coeffs[i]=norm


                sol_conf[3*self.N:]=self.devs.ravel()

                
                if int(100*self.t/tmax)%10==0:
                    print('{}%  done!'.format(int(100*self.t/tmax)))

                self.lyap_coeffs.append(coeffs)

                self.lyap_time.append(self.t)

            self.t+=dt
            
            

        # return sol






"""
Benettin algorithm: 

The equations of motion for the classical Heisenberg are solved using a numerical integration scheme, alongside with 
those the time evolution of the initial deviation vectors is calculated. Whenever deviations become too large, 
they are rescaled and reorthogonalized. Rescaling factors are stored in order to find the Lyapunov spectrum.

"""

if __name__=='__main__':

    #the number of spins
    J=1
    N=2
    tmax=30000
    delta=1e-013

    # instantiate class, random spin configuration
    ham=heis(N,J, mode='RND')

    ham.delta=delta


    print('devs')
    print(ham.devs0)
    print('norm devs 0')
    print(np.linalg.norm(ham.devs, axis=1))
    print('jacobian')
    print(ham.jacobian(ham.t0,ham.spin0))
    # ham.terminate_benet.terminal

    #first: normal propagation, plot energy per spin and total spin
    sol=ham.prop_benet(tmax,10000 ,1e07)

    lyap=np.cumsum(np.log(ham.lyap_coeffs),axis=0)


    # print(np.sum(sol.y[:6][2::3], axis=0))
    fig=plt.figure()
    plt.grid()
    plt.ylim(-0.3,0.3)
    lyapsum=0
    lyap_final=[]
    for el in lyap.T:

        plt.semilogx(ham.lyap_time, el/ham.lyap_time)
        print(el[-1]/ham.lyap_time[-1])
        lyapsum+=el[-1]/ham.lyap_time[-1]
        lyap_final.append(el[-1]/ham.lyap_time[-1])
    print(lyapsum)
    plt.show()
    

    fig=plt.figure()
    plt.grid()
    plt.plot(lyap_final, marker='o')

    plt.show()
    # fig=plt.figure()

    # sums=np.sum(sol.y[::3], axis=0)
    # plt.plot(sums)

    # plt.show()









