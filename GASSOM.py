import numpy as np
import math
import numpy.matlib
class GASSOM:
    def __init__(this,PARAM):
        this.patchwidth = PARAM["win_size"]
        this.topo_subspace = PARAM["topo_subspace"]
        this.num_subspace = np.prod(this.topo_subspace)
        this.transProb = PARAM["transProb"].copy()
        this.dim_bases = 100
        this.bases = np.random.randn(2,this.dim_bases,this.num_subspace)
        # this.bases = PARAM["initialbases"][0].copy()

        this.alpha_A = 8e-3
        this.sigma_A = 2
        this.alpha_C = 1e-4
        this.sigma_C = 0.2
        this.tconst = 10000
        this.sigma_n = 0.2
        this.sigma_w = 2
        this.batch_size = 280
        this.winners = []
        this.coef = np.zeros((2,this.num_subspace,this.batch_size))
        this.Proj = []
        this.nodeProb = np.random.rand(this.num_subspace,this.batch_size)
        this.nodeProb = this.nodeProb/this.nodeProb.sum(axis=0)
        
        this.iter = 1

    #####################################################################
    ### Function discription:
    ###     encode the input images with the best matched subspace
    ### Input argument discription:
    ###     imageBatch is the input images batch
    ### Returns
    ###     coef is the output coefficient matrix
    ###     error is the reconstructin error using current coefficients
    #####################################################################
    def sparseEncode(this,X):
        # this.batch_size = X.shape[1]
        
        this.coef[0,:,:] = np.matmul(this.bases[0,:,:].T,X)      #[num_subspace batch_size]
        this.coef[1,:,:] = np.matmul(this.bases[1,:,:].T,X)
        
        this.Proj  = this.coef[0,:,:]**2 + this.coef[1,:,:]**2  #[num_subspace,batch_size]
        Perr = np.ones((this.num_subspace,this.batch_size))-this.Proj
        emissProb = np.exp(-this.Proj/(2*this.sigma_w**2))*np.exp(-Perr/(2*this.sigma_n**2))
        
        this.nodeProb = np.matmul(this.transProb.T,this.nodeProb)*emissProb
        this.nodeProb /= np.sum(this.nodeProb,axis=0)
        
        this.winners = np.argmax(this.nodeProb, axis=0)
        winner = np.argmax(this.Proj, axis=0)
        proj_max = np.amax(this.Proj, axis=0)
        #print(proj_max)
        win_coef = this.Proj.mean(axis=1)
        win_err =  1-proj_max.mean()

        return win_coef, win_err

    #####################################################################
    ### Function discription:
    ###     Update the bases
    ### Input argument discription:
    ###     imageBatch is the input images batch
    #####################################################################
    def updateBasis(this,X):

        alpha = this.alpha_A*math.exp(-this.iter/this.tconst)+this.alpha_C
        sigma_h = this.sigma_A*math.exp(-this.iter/this.tconst)+this.sigma_C
        
        cj,ci = this.ind2sub(this.topo_subspace,this.winners)
        k = np.arange(this.num_subspace)
        kj,ki = this.ind2sub(this.topo_subspace,k)

        kj = np.matlib.repmat(kj,this.batch_size,1).T
        ki = np.matlib.repmat(ki,this.batch_size,1).T

        cj = np.matlib.repmat(cj.T, this.num_subspace,1)
        ci = np.matlib.repmat(ci.T, this.num_subspace,1)

        func_h = np.exp((-(ki-ci)**2-(kj-cj)**2)/(2*(sigma_h)**2)) # gaussian [n_subspace,batchsize]
        n_const = 1/(np.sqrt(this.Proj)+np.finfo(float).eps)
        weights = func_h*n_const
        w_c = np.zeros((2,this.num_subspace,this.batch_size))
        winput = np.zeros((2,this.dim_bases,this.num_subspace))
        diff = np.zeros((2,this.dim_bases,this.num_subspace))
        w_c[0,:,:] = weights*this.coef[0,:,:]
        w_c[1,:,:] = weights*this.coef[1,:,:]
        winput[0,:,:] = np.matmul(X,w_c[0,:,:].T)
        winput[1,:,:] = np.matmul(X,w_c[1,:,:].T)
        diff[0,:,:] =  winput[0,:,:]-this.bases[0,:,:]*np.sum(w_c[0,:,:]*this.coef[0,:,:],axis=1).T-this.bases[1,:,:]*np.sum(w_c[0,:,:]*this.coef[1,:,:],axis=1).T
        diff[1,:,:] =  winput[1,:,:]-this.bases[0,:,:]*np.sum(w_c[1,:,:]*this.coef[0,:,:],axis=1).T-this.bases[1,:,:]*np.sum(w_c[1,:,:]*this.coef[1,:,:],axis=1).T
        
        this.bases[0,:,:] += alpha*diff[0,:,:]
        this.bases[1,:,:] += alpha*diff[1,:,:]	

        this.bases[0,:,:] /= np.sqrt(np.sum(this.bases[0,:,:]**2,axis=0))
        this.bases[1,:,:] -= this.bases[0,:,:]*np.sum(this.bases[0,:,:]*this.bases[1,:,:],axis=0)
        this.bases[1,:,:] /= np.sqrt(np.sum(this.bases[1,:,:]**2,axis=0))            
        
        this.iter += 1

    #####################################################################
    ### Function discription:
    ###     Subscripts from linear index
    ### Input argument discription:
    ###     the shape of the array, linear index
    ### Returns
    ###     corresponding row and column index
    #####################################################################
    def ind2sub(this,array_shape, ind):
        ind[ind < 0] = -1
        ind[ind >= array_shape[0]*array_shape[1]] = -1
        rows = np.floor(ind / array_shape[1]).astype('int')
        cols = ind % array_shape[1]
        return cols, rows
