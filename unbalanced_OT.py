import torch, tqdm
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
####### Generalised sinkhorn #####
##################################

class Proximal():
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

        if self.log_domain == None:
            log_p = self.log_q.clone().requires_grad_(True)
            optimiser = torch_optimiser([log_p], **optim_kwarg)
            for _ in range(n_epochs):
                obj_value = obj(log_p)
                obj_value.backward()
                optimiser.step()
                optimiser.zero_grad()
            log_p.detach_()

        else:
            objs = {log_p: obj(log_p) for log_p in self.log_domain}
            log_p = min(objs, key=objs.get)

        return log_p.view(-1,1)

def generalised_sinkhorn(C, f, g, thres, eps, n_iter, prox_n_iter, torch_optimiser, **optim_kwarg):
    C_eps = C/eps
    log_u = torch.zeros(C.shape[0]).to(device).float().view(-1,1)
    log_v = torch.zeros(C.shape[1]).to(device).float().view(-1,1)

    err_u = torch.zeros(C.shape[0]).to(device).float().view(-1,1)
    err_v = torch.zeros(C.shape[1]).to(device).float().view(-1,1)

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
            log_v = torch.zeros(C.shape[1]).to(device).float().view(-1,1)

    log_pi = log_u.view(-1).unsqueeze(1) - C_eps + log_v.view(-1).unsqueeze(0)    
    pi = log_pi.exp()
    neg_entropy = (pi*log_pi).sum() - pi.sum()
    objective = (C*pi).sum() - eps * neg_entropy + prox_f.function(pi.sum(1)) + prox_g.function(pi.sum(0))
    
    return (objective, pi)

##############################

def standard_sinkhorn(C, a, b, eps, n_iter):
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
