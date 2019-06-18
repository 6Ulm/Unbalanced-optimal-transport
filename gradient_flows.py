import autograd.numpy as np
from autograd import grad
from scipy.optimize import minimize, Bounds

import torch, tqdm
import torch.optim as optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'

################################################
######### Support functions ##################
################################################

def cost_matrix(x,y):
    C = x.unsqueeze(1) - y.unsqueeze(0) # C_ij = x_i - y_j
    C = C.abs()
    return C

def lse(A, dim = 1):
    '''
    dim = 1: return log of sum_j exp(A_ij)
    dim = 0: return log of sum_i exp(A_ij)
    '''
    A_max = torch.max(A, dim)[0].view(-1,1)
    return A_max + (A-A_max).exp().sum(dim).log().view(-1,1)

##################################
####### Use scipy.optimize #######
##################################

class Proximal():
    def __init__(self, q, norm_q, functional, lamb):
        self.q = q        
        self.norm_q = norm_q
        self.functional = functional
        self.lamb = lamb
    
    def KL(self, p):
        KL_div = np.dot(p, np.log(p)) - np.dot(p, np.log(self.q)) - np.sum(p) + np.sum(self.q)
        return KL_div

    def lagrangien(self, p):
        lagrange = self.KL(p) + self.lamb * self.functional(p*self.norm_q) / self.norm_q
        return lagrange
    
    def solver(self):
        obj = self.lagrangien
        bounds = Bounds([0]*len(self.q), [np.inf]*len(self.q))
        dobj = grad(obj)
        res = minimize(obj, x0=self.q, method='TNC', jac=dobj, bounds=bounds)

        return torch.from_numpy(res.x).float().to(device).view(-1,1)

def normalise(log_vec):
    log_norm = lse(log_vec, 0).view(-1)
    log_vec -= log_norm
    norm = log_norm.exp().numpy()
    vec = log_vec.exp().view(-1).numpy()
    return (norm, vec, log_vec)

def sinkhorn(x, y, b, functional, lamb, eps, n_iter):
    C = cost_matrix(x,y)
    C_eps = C/eps
    log_b = b.log().view(-1,1)
    log_u, log_v = torch.zeros_like(log_b), torch.zeros_like(log_b)
    
    for _ in tqdm.tqdm(range(n_iter)):
        log_Kv = lse(-C_eps.t() + log_v.view(1,-1))
        norm_Kv, Kv, log_Kv = normalise(log_Kv)

        log_u = Proximal(Kv, norm_Kv, functional, lamb).solver().log() - log_Kv
        log_v = log_b - lse(-C_eps + log_u.view(1,-1))

    log_pi = log_u.view(-1).unsqueeze(1) - C_eps + log_v.view(-1).unsqueeze(0)
    pi = log_pi.exp()
    neg_entropy = (pi*log_pi).sum() - pi.sum()
    objective = (C*pi).sum() - eps * neg_entropy + lamb * functional(pi.sum(1))

    return (objective, pi)

##################################
####### Use pure torch ###########
##################################

class Proximal_torch():
    def __init__(self, log_q, norm_q, functional, lamb):
        self.log_q = log_q.view(-1)
        self.q = self.log_q.exp()
        self.norm_q = norm_q
        self.functional = functional
        self.lamb = lamb
    
    def KL(self, log_p):
        p = log_p.exp()
        KL_div = p.dot(log_p) - p.dot(self.log_q) - p.sum() + self.q.sum()
        return KL_div

    def scale_functional(self, log_p):
        return self.functional(log_p.exp()*self.norm_q) / self.norm_q

    def lagrangien(self, log_p):
        lagrange = self.KL(log_p) + self.lamb * self.scale_functional(log_p)
        return lagrange
    
    def solver(self, n_epochs = 1000, lr = 1e-1):
        log_p = self.log_q.clone().requires_grad_(True)

        for _ in range(n_epochs):
            obj = self.lagrangien(log_p)
            obj.backward()
            with torch.no_grad():
                log_p -= lr * log_p.grad
            log_p.grad.zero_()

        return log_p.view(-1,1)

def normalise_torch(log_vec):
    log_norm = lse(log_vec, 0).view(-1)[0]
    norm_vec = log_norm.exp()
    log_vec -= log_norm
    return (norm_vec, log_vec)

def sinkhorn_torch(x, y, b, functional, lamb, eps, n_iter):
    C = cost_matrix(x,y)
    C_eps = C/eps
    log_b = b.log().view(-1,1)
    log_u, log_v = torch.zeros_like(log_b), torch.zeros_like(log_b)
    
    for _ in tqdm.tqdm(range(n_iter)):
        log_Kv = lse(-C_eps.t() + log_v.view(1,-1))
        norm_Kv, log_Kv = normalise_torch(log_Kv)
    
        log_u = Proximal_torch(log_Kv, norm_Kv, functional, lamb).solver(n_epochs = 100, lr = 1e-1) - log_Kv
        log_v = log_b - lse(-C_eps + log_u.view(1,-1))

        log_u.detach_()
        log_v.detach_()

    log_pi = log_u.view(-1).unsqueeze(1) - C_eps + log_v.view(-1).unsqueeze(0)
    pi = log_pi.exp()
    neg_entropy = (pi*log_pi).sum() - pi.sum()
    objective = (C*pi).sum() - eps * neg_entropy + lamb * functional(pi.sum(1))
    
    return (objective, pi)

##############################

def sinkhorn_test(x, y, b, a, eps, n_iter):
    C_eps = cost_matrix(x,y)/eps
    log_a, log_b = a.log().view(-1,1), b.log().view(-1,1)
    log_u, log_v = torch.zeros_like(log_b), torch.zeros_like(log_a)
    
    for _ in tqdm.tqdm(range(n_iter)):
        log_u = log_a - lse(-C_eps.t() + log_v.view(1,-1))
        log_v = log_b - lse(-C_eps + log_u.view(1,-1))

    log_pi = log_u.view(-1).unsqueeze(1) - C_eps + log_v.view(-1).unsqueeze(0)
    return log_pi.exp()
