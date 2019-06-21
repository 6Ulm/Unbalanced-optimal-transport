import torch, tqdm
device = 'cuda' if torch.cuda.is_available() else 'cpu'

'''
Pytorch implementation of the following discrete unbalanced OT problem:
    min <pi, C> - eps * H(pi) + f(pi * vec_1) + g(pi.T * vec_1) + Dom(pi)
    where:
        + C: cost matrix
        + eps: regularisation parameter
        + H: KL divergence of matrix
        + vec_1: vector of 1s
        + f, g: convex functions
        + Dom: domain of pi
'''

def cost_matrix(x, y, p):
    '''
    Calculate cost matrix, much faster than double loops
    '''
    C = x.unsqueeze(1) - y.unsqueeze(0) # C_ij = x_i - y_j
    if p == 1:
        C.abs_()
    elif p == 2:
        C.pow_(2)
    return C

def lse(A, dim = 1):
    '''
    Implementation of log_sum_exp trick
    dim = 1: return log of sum_j exp(A_ij)
    dim = 0: return log of sum_i exp(A_ij)
    '''
    A_max = torch.max(A, dim)[0].view(-1,1)
    return A_max + (A-A_max).exp().sum(dim).log().view(-1,1)

######################################
####### Stable scaling algorithm #####
######################################

class Proximal():
    '''
    Calculate proximal operator
    '''

    def __init__(self, functional, eps):
        self.function = functional['function']
        self.log_domain = functional['log_domain']
        self.eps = eps

    def update(self, log_q):
        self.q_sum = lse(log_q,0).view(-1)[0].exp()
        self.log_q = log_q.view(-1)

    def KL(self, log_p):
        p = log_p.exp()
        p_sum = lse(log_p.view(-1,1),0).view(-1)[0].exp()
        KL_div = p.dot(log_p) - p.dot(self.log_q) - p_sum + self.q_sum
        return KL_div

    def solver(self, n_epochs, torch_optimiser, **optim_kwarg):
        obj = lambda log_p: self.KL(log_p) + self.function(log_p.exp()) / self.eps

        # if function is continous/differentiable EVERYWHERE, then use autograd
        if self.log_domain == None:
            log_p = self.log_q.clone().requires_grad_(True)
            optimiser = torch_optimiser([log_p], **optim_kwarg)
            for _ in range(n_epochs):
                obj_value = obj(log_p)
                obj_value.backward()
                optimiser.step()
                optimiser.zero_grad()
            log_p.detach_()

        # otherwise, manually search over the log domain
        else:
            objs = {log_p: obj(log_p) for log_p in self.log_domain}
            log_p = min(objs, key=objs.get)

        return log_p.view(-1,1)

def generalised_sinkhorn(C, f, g, thres, eps, n_iter, prox_n_iter, torch_optimiser, **optim_kwarg):
    '''
    Implementation of stable scaling algorithm in [Chizat, 2018].
    https://arxiv.org/abs/1607.05816
    Work almost exclusively with log values in order to:
    + Achieve stability
    + Convert the (constrained) proximal problem (where proximal operator must be positive) 
        to the unconstrained one (calculate log of proximal operator instead),
        so that torch's autograd can be used

    Input:
        - C: cost matrix of size (M x N)
        - f,g: dictionaries whose keys are:
            + 'function': convex function, must be either continuous and defined EVERYWHERE, 
                        or defined at only FINITELY many points (e.g. indicator function).
                        f must take value in R^M and g must take value in R^N.
            + 'log_domain': None if the function is continuous everywhere, otherwise 
                            a FINITE list of LOG of values in the domain. Currently do not support 
                            functions which are continuous and defined on a convex and proper subset.
        - thres: threshold above which sinkhorn iterations are modified to guarantee stability
        - n_iter: number of sinkhorn iterations
        - prox_n_iter: number of torch optimiser iterations
        - torch_optimiser: NAME of torch.optim optimiser: SGD, Adam, Adadelta, etc...
        - optim_kwarg: parameters of torch_optimiser
    Output:
        - Unbalanced regularised OT cost and plan
    '''

    C_eps = C/eps
    log_u = torch.zeros(C.shape[0]).to(device).double().view(-1,1)
    log_v = torch.zeros(C.shape[1]).to(device).double().view(-1,1)

    err_u = torch.zeros(C.shape[0]).to(device).double().view(-1,1)
    err_v = torch.zeros(C.shape[1]).to(device).double().view(-1,1)

    prox_f = Proximal(f, eps)
    prox_g = Proximal(g, eps)
    
    for _ in tqdm.tqdm(range(n_iter)):
        log_Kv = lse(-C_eps + log_v.t())
        log_Kv_scale = log_Kv - err_u/eps
        prox_f.update(log_Kv_scale)
        log_u = prox_f.solver(prox_n_iter, torch_optimiser, **optim_kwarg) - log_Kv

        log_Ktu = lse(-C_eps.t() + log_u.t())
        log_Ktu_scale = log_Ktu - err_v/eps
        prox_g.update(log_Ktu_scale)
        log_v = prox_g.solver(prox_n_iter, torch_optimiser, **optim_kwarg) - log_Ktu

        if torch.max(log_u.abs().max(), log_v.abs().max()) > thres:
            print('Threshold violation')
            err_u += (eps * log_u)
            err_v += (eps * log_v)
            C_eps -= (err_u.view(-1).unsqueeze(1) + err_v.view(-1).unsqueeze(0))/eps
            log_v = torch.zeros(C.shape[1]).to(device).double().view(-1,1)

    log_pi = log_u.view(-1).unsqueeze(1) - C_eps + log_v.view(-1).unsqueeze(0)    
    pi = log_pi.exp()
    neg_entropy = (pi*log_pi).sum() - pi.sum()
    objective = (C*pi).sum() - eps * neg_entropy + prox_f.function(pi.sum(1)) + prox_g.function(pi.sum(0))
    
    return (objective, pi)

######################################

def standard_sinkhorn(C, a, b, eps, n_iter):
    '''
    Basic torch implementation of classical sinkhorn
    Input:
        + C: cost matrix of shape (M x N)
        + a: probability vector in R^M
        + b: probability vector in R^N
        + eps: regularisation parameter
        + n_iter: number of sinkhorn iterations
    Output:
        + Unbalanced regularised OT cost and plan
    '''

    C_eps = C/eps
    log_a, log_b = a.log().view(-1,1), b.log().view(-1,1)
    log_u, log_v = torch.zeros_like(log_a), torch.zeros_like(log_b)
    
    for _ in tqdm.tqdm(range(n_iter)):
        log_u = log_a - lse(-C_eps + log_v.t())
        log_v = log_b - lse(-C_eps.t() + log_u.t())

    log_pi = log_u.view(-1).unsqueeze(1) - C_eps + log_v.view(-1).unsqueeze(0)
    pi = log_pi.exp()
    neg_entropy = (pi*log_pi).sum() - pi.sum()
    objective = (C*pi).sum() - eps * neg_entropy

    return (objective, pi)
