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
    def __init__(self, functional):
        self.function = functional['function']
        self.function_support = functional['support']

    def update(self, log_q, norm_q):
        self.log_q = log_q.view(-1)
        self.q_sum = self.log_q.exp().sum()
        self.norm_q = norm_q

    def KL(self, log_p):
        p = log_p.exp()
        KL_div = p.dot(log_p) - p.dot(self.log_q) - p.sum() + self.q_sum
        return KL_div

    def solver(self, n_epochs = 500, lr = 1e-2):
        lagrangien = lambda log_p: self.KL(log_p) + self.function(log_p.exp()*self.norm_q) / self.norm_q

        if self.function_support == None:
            log_p = self.log_q.clone().requires_grad_(True)
            for _ in range(n_epochs):
                obj = lagrangien(log_p)
                obj.backward()
                with torch.no_grad():
                    log_p -= lr * log_p.grad
                log_p.grad.zero_()
            log_p.detach_()

        else:
            objs = {log_p: lagrangien(log_p) for log_p in self.function_support}
            log_p = min(objs, key=objs.get) - self.norm_q.log()

        return log_p.view(-1,1)

def normalise(log_vec):
    log_norm = lse(log_vec, 0).view(-1)[0]
    norm_vec = log_norm.exp()
    log_vec -= log_norm
    return (norm_vec, log_vec)

def generalised_sinkhorn(C, f, g, eps, n_iter):
    C_eps = C/eps
    log_u = torch.zeros(C.shape[0]).to(device).float().view(-1,1)
    log_v = torch.zeros(C.shape[1]).to(device).float().view(-1,1)

    prox_f = Proximal(f)
    prox_g = Proximal(g)
    
    for _ in tqdm.tqdm(range(n_iter)):
        log_Kv = lse(-C_eps + log_v.t())
        norm_Kv, log_Kv = normalise(log_Kv)

        prox_f.update(log_Kv, norm_Kv)
        log_u = prox_f.solver() - log_Kv

        log_Ktu = lse(-C_eps.t() + log_u.t())
        norm_Ktu, log_Ktu = normalise(log_Ktu)

        prox_g.update(log_Ktu, norm_Ktu)
        log_v = prox_g.solver() - log_Ktu

    log_pi = log_u.view(-1).unsqueeze(1) - C_eps + log_v.view(-1).unsqueeze(0)
    log_pi -= lse(log_pi.view(-1,1),0)
    
    pi = log_pi.exp()
    neg_entropy = (pi*log_pi).sum() - pi.sum()
    objective = (C*pi).sum() - eps * neg_entropy + f['function'](pi.sum(1)) + g['function'](pi.sum(0))
    
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
