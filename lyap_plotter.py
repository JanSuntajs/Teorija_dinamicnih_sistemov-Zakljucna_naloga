import classical_spins_trotter_suzuki as cts

import numpy as np 
import sys,os
from matplotlib import pyplot as plt 

plt.rc('text', usetex = False)
# plt.rc('font',  family = 'sans-serif')
plt.rc('text', usetex = True)
plt.rc('font',family='serif',serif=['Computer Modern'])
plt.rc('xtick', labelsize = 'medium')
plt.rc('ytick', labelsize = 'medium')
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
fontsize=[17,20,24]



def prepare_ax(ax, legend=True, fontsize=fontsize,grid=True):
    
    ax.tick_params(axis='x',pad=5,direction='out')
    if legend:
        ax.legend(loc='best',prop={'size':fontsize[0]},fontsize=fontsize[0],framealpha=0.5)
    ax.tick_params(axis='x',which='major', labelsize=fontsize[1])
    ax.tick_params(axis='y',which='major', labelsize=fontsize[1])
    if grid:
        ax.grid(which='both')
   
def prepare_plt(savename='' ,plot_type='', top=0.89, save=True, show=True):

    plt.tight_layout()
    plt.subplots_adjust(top=top)
    if save:
        graphs_folder='./Graphs/'+plot_type+'/'
        if not os.path.isdir(graphs_folder):
            os.makedirs(graphs_folder)

        # plt.savefig(graphs_folder+'/' +'double_plot'+'{}{}_{}_{}.pdf'.format(nametag,file.syspar['size'], file.syspar['ne'],file.syspar['nu']))
        plt.savefig(graphs_folder+'/'+savename)
    if show:

        plt.show()

def prepare_axarr(nrows=1, ncols=2, sharex=True, sharey=True,fontsize=[18,21,25]):

    figsize=(ncols*8, nrows*7)

    fig, axarr=plt.subplots(nrows, ncols, figsize=figsize, sharex=sharex, sharey=sharey)

    return fig, axarr, fontsize


def plot_engy(hamlist, tmax=5, dt=0.001, savename='plot_engy.pdf'):

    fig, ax, fontsize=prepare_axarr(1,1)
    majorLocator = MultipleLocator(2)
    minorLocator = AutoMinorLocator(4)
    
    for ham in hamlist:

        ax.plot(ham.time_evol, ham.engy_evol/ham.N, label='$N={}$'.format(ham.N))

    ax.xaxis.set_major_locator(majorLocator)
    ax.xaxis.set_minor_locator(minorLocator)
    prepare_ax(ax)
    
    ax.set_xlabel('$t$', fontsize=fontsize[1])
    ax.set_ylabel('$E/N$', fontsize=fontsize[1])
    ax.set_title('Energija na posamezen spin, $J={}$'.format(hamlist[-1].J), fontsize=fontsize[-1])
    prepare_plt(savename=savename)


def plot_spin_cons(hamlist, savename='plot_spin_cons_{}_{}_{}.pdf'):

    fig, axarr, fontsize=prepare_axarr(1,3, sharey=True,fontsize=[28,31,38])
    majorLocator = MultipleLocator(2)
    minorLocator = AutoMinorLocator(4)
    axeslabels=['$\\mathrm{\\Gamma}^\\mathrm{{tot}}_x$','$\\mathrm{\\Gamma}^\\mathrm{{tot}}_y$','$\\mathrm{\\Gamma}^\\mathrm{{tot}}_z$']

    for i, ax in enumerate(axarr):
        
        ax.yaxis.set_major_locator(majorLocator)
        ax.yaxis.set_minor_locator(minorLocator)
        spinsum=hamlist[i].spin_evol
        # print(spinsum.shape)
        # print(spinsum[0])
        # print(np.linalg.norm(spinsum[0].reshape(-1,3),axis=1))
        spinsum=spinsum.reshape(spinsum.shape[0], -1,3)


        # # print(spinsum)
        spinsum=np.sum(spinsum ,axis=1) 
        for j,el in enumerate(spinsum.T):
            # print(el)

            ax.semilogx(ham.time_evol, np.log10(np.abs(el-el[0])), label=axeslabels[j])

        prepare_ax(ax, fontsize=fontsize)
        ax.set_xlabel('$t$', fontsize=fontsize[1])
        ax.set_title('$N={}$'.format(hamlist[i].N), fontsize=fontsize[-1])
        if i==0:
            ax.set_ylabel('$\\log_{{10}}|\\mathrm{\\vec{\\Gamma}}^\\mathrm{{tot}}_\\alpha (t) - \\mathrm{\\vec{\\Gamma}}^\\mathrm{{tot}}_\\alpha (0)|$', fontsize=fontsize[1])
    # ax.set_ylabel('$E/N$', fontsize=fontsize[1])

    savename=savename.format(hamlist[0].N, hamlist[1].N, hamlist[2].N)
    fig.suptitle('Ohranitev skupnega spina, $J={}$'.format(hamlist[-1].J), fontsize=fontsize[-1])
    prepare_plt(savename=savename, top=0.83)

    


def plot_norm_cons(hamlist, savename='plot_norm_cons_{}_{}_{}.pdf'):

    fig, axarr, fontsize=prepare_axarr(1,3, sharey=True, fontsize=[28,31,38])
    majorLocator = MultipleLocator(2)
    minorLocator = AutoMinorLocator(4)
    # axarr=list(axarr)

    # axeslabels=['$\\mathrm{S}^\\mathrm{{tot}}_x$','$\\mathrm{S}^\\mathrm{{tot}}_y$','$\\mathrm{S}^\\mathrm{{tot}}_z$']

    for i, ax in enumerate(axarr):
        
        ax.yaxis.set_major_locator(majorLocator)
        ax.yaxis.set_minor_locator(minorLocator)
        spinsum=hamlist[i].spin_evol
        # print(spinsum.shape)
        # print(spinsum[0])
        # print(np.linalg.norm(spinsum[0].reshape(-1,3),axis=1))
        spinsum=spinsum.reshape(spinsum.shape[0], -1,3)


        # # print(spinsum)
        norm=np.linalg.norm(spinsum ,axis=2) 
        for j,el in enumerate(norm.T):
            # print(el)

            ax.semilogx(ham.time_evol, np.log10(np.abs(el-1)))

        prepare_ax(ax, fontsize=fontsize)
        ax.set_xlabel('$t$', fontsize=fontsize[1])
        ax.set_title('$N={}$'.format(hamlist[i].N), fontsize=fontsize[-1])
        if i==0:
            ax.set_ylabel('$\\log_{{10}}||\\mathrm{\\vec{\\Gamma}}_i (t)| - 1|$', fontsize=fontsize[1])
    # ax.set_ylabel('$E/N$', fontsize=fontsize[1])

    savename=savename.format(hamlist[0].N, hamlist[1].N, hamlist[2].N)
    fig.suptitle('Ohranitev norme posameznih spinov, $J={}$'.format(hamlist[-1].J), fontsize=fontsize[-1])
    prepare_plt(savename=savename, top=0.85)


def plot_xyz(hamlist, savename='plot_xyz_{}_.pdf'):

    axeslabels=['$\\mathrm{\\Gamma}_x$','$\\mathrm{\\Gamma}_y$','$\\mathrm{\\Gamma}_z$']
    fig, axarr, fontsize=prepare_axarr(2,2, sharey=True,sharex='col', fontsize=[24,30,34])


    ham=hamlist[0]

    spins=ham.spin_evol
    spins=spins.reshape(spins.shape[0],-1,3)
    time=ham.time_evol


    for i, ax in enumerate(axarr.flatten()):

        ax.set_xlim(0,100)
        for j in range(3):
            ax.plot(time, spins[:,i,j], label=axeslabels[j])
    

        # spinsum=spinsum.reshape(spinsum.shape[0], -1,3)


        # # print(spinsum)

        prepare_ax(ax, legend=False, fontsize=fontsize)
        ax.set_ylim(-1.4,1.1)
        if ((i==2) or (i==3)):
            ax.set_xlabel('$t$', fontsize=fontsize[1])

        if i==3:
            ax.legend(loc='best',prop={'size':fontsize[0]},fontsize=fontsize[0],framealpha=0.5, ncol=3)
        # ax.set_title('$N={}$'.format(hamlist[i].N), fontsize=fontsize[-1])
        # if i==0:
        if ((i==0) or (i==2)):
            ax.set_ylabel('$\\Gamma_\\alpha(t)$', fontsize=fontsize[1])

    savename=savename.format(hamlist[0].N)
    fig.suptitle('Primer dinamike posameznih spinov, $J={}$, $N={}$'.format(hamlist[0].J, hamlist[0].N), fontsize=fontsize[-1])
    prepare_plt(savename=savename, top=0.92)
    plt.show()


def plot_phase_space(hamlist, savename='plot_phase_{}_.pdf'):

    axeslabels=['$\\mathrm{\\Gamma}_x$','$\\mathrm{\\Gamma}_y$','$\\mathrm{\\Gamma}_z$']
    fig, axarr, fontsize=prepare_axarr(1,3, sharey=True,sharex=True, fontsize=[26,29,37])
    titlelist=['$z=0$','$x=0$','$y=0$']

    labellist=[['$x$', '$y$'], ['$y$', '$z$'], ['$x$','$z$']]

    ham=hamlist[0]

    spins=ham.spin_evol
    spins=spins.reshape(spins.shape[0],-1,3)
    time=ham.time_evol

    poin_z0=np.array([val[:2] for val in spins[:,0,:] if np.abs(val[2])<1e08])
    poin_x0=np.array([val[1:] for val in spins[:,0,:] if np.abs(val[0])<1e08])
    poin_y0=np.array([np.array([val[0],val[2]]) for val in spins[:,0,:] if np.abs(val[1])<1e08])

    vals=[poin_z0, poin_x0, poin_y0]

    for i, ax in enumerate(axarr.flatten()):

        ax.plot(vals[i][:,0], vals[i][:,1], marker='o', linestyle=None, ms=0.3)

        prepare_ax(ax, legend=False, fontsize=fontsize)
        ax.set_ylim(-1.3,1.3)
        ax.set_xlim(-1.3,1.3)
        ax.set_xlabel(labellist[i][0], fontsize=fontsize[1])
        ax.set_ylabel(labellist[i][1], fontsize=fontsize[1])
        

        ax.set_title('Ravnina {}'.format(titlelist[i]), fontsize=fontsize[-1],y=1.01)


    savename=savename.format(hamlist[0].N)
    fig.suptitle('Primer dinamike posameznega spina, $J={}$, $N={}$'.format(hamlist[0].J, hamlist[0].N), fontsize=fontsize[-1])
    prepare_plt(savename=savename, top=0.83)
    plt.show()


def plot_spin_pair(hamlist, savename='plot_spin_pair_{}_.pdf'):

    axeslabels=['$\\mathrm{\\Gamma}_x$','$\\mathrm{\\Gamma}_y$','$\\mathrm{\\Gamma}_z$']
    fig, axarr, fontsize=prepare_axarr(2,2, sharey=False,sharex='col', fontsize=[24,29,34])


    ham=hamlist[0]

    spins=ham.spin_evol
    spins=spins.reshape(spins.shape[0],-1,3)
    time=ham.time_evol


    for i, ax in enumerate(axarr.flatten()):

        ax.set_xlim(0,2)
        next_ind=i+1
        if next_ind==4:
            next_ind=0

        ax.set_title('Spin {} in spin {}'.format(i+1, next_ind+1), fontsize=fontsize[1])
        for j in range(3):


            ax.plot(time, spins[:,i,j]+spins[:, next_ind,j], label=axeslabels[j])
    

        # spinsum=spinsum.reshape(spinsum.shape[0], -1,3)


        # # print(spinsum)

        prepare_ax(ax, legend=False, fontsize=fontsize)
        # ax.set_ylim(-1.4,1.1)
        if ((i==2) or (i==3)):
            ax.set_xlabel('$t$', fontsize=fontsize[1])

        if i==3:
            ax.legend(loc='best',prop={'size':fontsize[0]},fontsize=fontsize[0],framealpha=0.5, ncol=3)
        # ax.set_title('$N={}$'.format(hamlist[i].N), fontsize=fontsize[-1])
        # if i==0:
        if ((i==0) or (i==2)):
            ax.set_ylabel('$\\Gamma^\\alpha_i(t)+\\Gamma^\\alpha_{i+1}(t)$', fontsize=fontsize[1])

    savename=savename.format(hamlist[0].N)
    fig.suptitle('Dinamika skupnega spina para, $J={}$, $N={}$'.format(hamlist[0].J, hamlist[0].N), fontsize=fontsize[-1])
    prepare_plt(savename=savename, top=0.89)
    plt.show()





def plot_lyap_coeffs(ham, savename='plot_lyap_coeffs_J_{:d}_N_{:d}_.pdf',plot_saved=True,):


    #3 x 1 plot, first has time evolution of the lyap coeffs, 
    #second has the time evolution of the sum 
    #third has the final coeffs 

    fig, axarr, fontsize=prepare_axarr(1,3, sharey=False,sharex=False, fontsize=[26,28,35] )
    majorLocator = MultipleLocator(int(ham.N/2))
    minorLocator = AutoMinorLocator(int(ham.N/2))
    majorLocator_y = MultipleLocator(0.05)
    minorLocator_y = AutoMinorLocator(5)


    if plot_saved:
        #load data
        lyap_time, lyap_spec=np.load('calc_lyap_coeffs_{}_{}_eps_{}_delta_{}_.npy'.format(ham.N, ham.J, int(np.log10(ham.eps)), int(np.abs(np.log10(ham.delta)))))
        # lyap_time, lyap_spec= np.load('calc_lyap_coeffs_{}_{}_.npy'.format(ham.N, ham.J))
        # lyap_time=sol[0]
        # lyap_coeffs=sol[1]
    else:
        lyap_time=np.array(ham.lyap_time)
        lyap_coeffs=np.array(ham.lyap_coeffs)

        lyap_spec=cts.calc_lyaps(lyap_time, lyap_coeffs)

    print(lyap_time.shape)
    print(lyap_spec.shape)
    #save
    np.save('calc_lyap_coeffs_{}_{}_eps_{}_delta_{}_.npy'.format(ham.N, ham.J, int(np.log10(ham.eps)), int(np.abs(np.log10(ham.delta)))), np.array([lyap_time, lyap_spec]))
    # np.save('calc_lyap_coeffs_{}_{}_.npy'.format(ham.N, ham.J), np.array([lyap_time, lyap_spec]))

    for spec in lyap_spec:

        axarr[0].semilogx(lyap_time, spec)
    
    axarr[1].semilogx(lyap_time, np.log10(np.abs(np.sum(lyap_spec, axis=0))), marker='o')

    final_spectrum=np.sort(lyap_spec[:,-1])[::-1]
    axarr[2].scatter(np.linspace(1, 3*ham.N, 3*ham.N), np.sort(lyap_spec[:,-1])[::-1])



    axarr[0].set_xlabel('$t$', fontsize=fontsize[1])

    axarr[0].set_ylabel('$\\lambda_i(t)$', fontsize=fontsize[1])
    axarr[0].set_title('\\v{C}asovni razvoj spektra', fontsize=fontsize[1])

    axarr[1].set_xlabel('$t$', fontsize=fontsize[1])
    axarr[1].set_ylabel('$\\log_{10}\\left|\\sum\\limits_{i=1}^{3N} \\lambda_i\\right|$', fontsize=fontsize[1])
    axarr[1].set_title('Test natan\\v{c}nosti', fontsize=fontsize[1])

    axarr[2].set_xlabel('Indeks $i$', fontsize=fontsize[1])
    axarr[2].set_ylabel('$\\lambda_i$', fontsize=fontsize[1])
    axarr[2].set_title('Ljapunov spekter', fontsize=fontsize[1])
    axarr[2].xaxis.set_major_locator(majorLocator)
    axarr[2].xaxis.set_minor_locator(minorLocator)
    axarr[2].yaxis.set_major_locator(majorLocator_y)
    axarr[2].yaxis.set_minor_locator(minorLocator_y)
    axarr[2].set_xlim(0.5, 3*ham.N+0.5)
    axarr[2].text(0.77, 0.8, '$\\lambda_\\mathrm{{max}}={:.3f}$'.format(np.max(final_spectrum)),ha='center',va='center',transform = axarr[2].transAxes, fontsize=fontsize[1])
    axarr[2].text(0.77, 0.6, '$\\lambda_\\mathrm{{min}}={:.3f}$'.format(np.min(final_spectrum)),ha='center',va='center',transform = axarr[2].transAxes, fontsize=fontsize[1])
    

    for i, ax in enumerate(axarr):

        prepare_ax(ax, fontsize=fontsize)

    axarr[2].grid(which='minor', linestyle='--')
    savename=savename.format(int(ham.J), ham.N)
    if ((ham.N==4) or (ham.N==10)):
        fig.suptitle('Izra\\v{{c}}un Ljapunovega spektra, $J={}$, $N={}$'.format(ham.J, ham.N), fontsize=fontsize[-1])
    else:
        fig.suptitle('$J={}$, $N={}$'.format(ham.J, ham.N), fontsize=fontsize[-1])
    prepare_plt(savename=savename, top=0.85)

def plot_J_dependence(hamlist, savename='plot_J_dependence_N_{:d}_.pdf' ):

    fig, axarr, fontsize=prepare_axarr(1,2, sharey=False,sharex=False, fontsize=[26,28,35] )

    majorLocator = MultipleLocator(int(ham.N/2))
    minorLocator = AutoMinorLocator(int(ham.N/2))
    majorLocator_y = MultipleLocator(0.05)
    minorLocator_y = AutoMinorLocator(5)


    for ham in hamlist:
        delta=np.log10(ham.delta)
        eps=np.log10(ham.eps)
        lyap_time, lyap_spec= np.load('Data/new_lyap_spectrum_N_{}_J_{}_delta_{}_eps_{}_.npy'.format(ham.N, ham.J, np.log10(ham.delta), np.log10(ham.eps)))
        axarr[0].scatter(np.linspace(1, 3*ham.N, 3*ham.N), np.sort(lyap_spec[:,-1])[::-1])
        axarr[1].semilogx(ham.J*np.ones(3*ham.N), np.sort(lyap_spec[:,-1])[::-1], marker='o', ls='')


    for ax in axarr.flatten():
        prepare_ax(ax, fontsize_fontsize)
        ax.grid(which='minor', linestyle='--')
        ax.yaxis.set_major_locator(majorLocator_y)
        ax.yaxis.set_minor_locator(minorLocator_y)

    axarr[0].set_xlabel('Indeks $i$', fontsize=fontsize[1])
    axarr[0].set_ylabel('$\\lambda_i$', fontsize=fontsize[1])
    axarr[1].set_xlabel('$J$', fontsize=fontsize[1])
    axarr[1].set_ylabel('$\\lambda_i(J)$', fontsize=fontsize[1])
    axarr[0].xaxis.set_major_locator(majorLocator)
    axarr[0].xaxis.set_minor_locator(minorLocator)



    fig.suptitle('Ljapunov spekter v odvisnosti od $J$, $N={}$'.format(ham.N), fontsize=fontsize[-1])
    savename=savename.format(hamlist[0].N)
    prepare_plt(savename=savename, top=0.85)


def plot_N_dependence(hamlist,savename='plot_N_dependence_J_{:d}_.pdf'):

    fig, axarr, fontsize=prepare_axarr(1,2, sharey=False,sharex=False, fontsize=[26,28,31] )

    majorLocator = MultipleLocator(int(hamlist[0].N/2))
    minorLocator = AutoMinorLocator(int(hamlist[0].N/2))
    majorLocator_y = MultipleLocator(0.05)
    minorLocator_y = AutoMinorLocator(5)


    for ham in hamlist:
        delta=np.log10(ham.delta)
        eps=np.log10(ham.eps)
        lyap_time, lyap_spec= np.load('calc_lyap_coeffs_{}_{}_eps_{}_delta_{}_.npy'.format(ham.N, ham.J, int(np.log10(ham.eps)), int(np.abs(np.log10(ham.delta)))))
        axarr[0].scatter(ham.N*np.ones(3*ham.N), np.sort(lyap_spec[:,-1])[::-1])

        zero_lims=int((3*ham.N - (ham.N+4))/2)
        axarr[1].scatter(ham.N*np.ones(ham.N+4), np.log10(np.abs(np.sort(lyap_spec[:,-1])[::-1][zero_lims:-zero_lims])))

    for ax in axarr.flatten():
        prepare_ax(ax, fontsize=fontsize)
        ax.grid(which='minor', linestyle='--')
        ax.set_xlabel('$N$', fontsize=fontsize[1])
        
        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_minor_locator(minorLocator)
    axarr[0].yaxis.set_major_locator(majorLocator_y)
    axarr[0].yaxis.set_minor_locator(minorLocator_y)
    axarr[0].set_title('Spekter', fontsize=fontsize[1])
    axarr[1].set_title('Ujemanje ni\\v{c}elnih vrednosti', fontsize=fontsize[1])
    axarr[0].set_ylabel('$\\lambda_i(N)$', fontsize=fontsize[1])
    axarr[0].set_ylabel('$\\lambda_i(N)$', fontsize=fontsize[1])
    axarr[1].set_ylabel('$\\log_{10}|\\lambda_i(N)|$', fontsize=fontsize[1])

    delta_str=''
    fig.suptitle('Ljapunov spekter v odvisnosti od $N$, $J={}$, $\\tilde{{\\delta}}=10^{}$, $\\tilde{{\\varepsilon}}_{{gs}}=10^{}$'.format(ham.J,{int(delta)}, int(eps)), fontsize=fontsize[-1])
    savename=savename.format(int(hamlist[0].J))
    prepare_plt(savename=savename, top=0.84)




if __name__=='__main__':

    # plot_engy(nlist=[4,8,12], tmax=10, dt=0.001)

    # ham=cts.heis(4)

    # print(ham.spin0)
    # print(np.log10(np.abs(1-np.linalg.norm(ham.spin0.reshape(-1,3),axis=1))))
    hamlist=[]
    J=1.0
    nlist=[4,6,8,10]
    tmax=1e05
    dt=0.5
    eps=1e04
    delta=1e-11
    for N in nlist:

        ham=cts.heis(N, J)
        ham.delta=delta

        # ham.benettin(tmax, dt, eps)

        hamlist.append(ham)
        plot_lyap_coeffs(ham, plot_saved=True)



    # plot_N_dependence(hamlist)
    # plot_lyap_coeffs(hamlist[0], plot_saved=False)
    # print(hamlist[0].delta)
    # print(hamlist[0].eps)
    # plot_spin_cons(hamlist, )
    # plot_norm_cons(hamlist, )

    # plot_xyz(hamlist)
    # plot_phase_space(hamlist, savename='plot_phase_{}_new.pdf')
    # plot_spin_pair(hamlist)



