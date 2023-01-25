import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["maxcone_loss", "memory_maxcone_loss"]


def maxcone_loss(feat_q, targets):

    feat_q = nn.functional.normalize(feat_q, dim=1)
        # feat_k = nn.functional.normalize(feat_k, dim=1)
    bs_size = feat_q.size(0)
    N_id = len(torch.unique(targets))
    N_ins = bs_size // N_id
    
    scale = 8.0
    norm_type = 2
    loss_type = 'diag_sum'
    
    sim_qq = torch.matmul(feat_q, feat_q.T)
    # sf_sim_qq = scale*F.softplus(sim_qq, beta = scale)
    # sf_sim_qq = scale*(sim_qq+1)
    sf_sim_qq = torch.exp(scale*sim_qq)

    hard_pos = torch.eye(N_id,device=targets.device)
    hard_neg = 1 - torch.eye(N_id,device=targets.device)
    for i in range(N_id):
        for j in range(N_id):
            logits = sf_sim_qq[(N_ins*i):(N_ins*(i+1)),(N_ins*j):(N_ins*(j+1))]

            if i == j:
                soft_hard = F.softplus(-1*torch.logsumexp(-1.*logits,dim=1))
                v,_ = torch.min(logits,1)
                hard_pos[i,j] = soft_hard.mean()

            else:
                soft_hard = F.softplus(torch.logsumexp(logits,dim = 1))
                v,_ = torch.max(logits,1)
                hard_neg[i,j] = soft_hard.mean()
    
    if loss_type == 'max_cone':
        max_neg,_ = torch.max(hard_neg, dim = 1)
        max_neg_mat = torch.diag(max_neg)
        alpha_hard_pos = hard_pos + 2*max_neg_mat.detach()

        hard_sim = alpha_hard_pos + hard_neg

        ## normalize by L2-norm
        norm_hard_sim = nn.functional.normalize(hard_sim, dim = 1)
        # norm_hard_sim = nn.functional.normalize(hard_sim, p=1,dim = 1)

        try:
            _,singular_value,_ = torch.svd(norm_hard_sim.float())
        except:                     # torch.svd may have convergence issues for GPU and CPU.
            _,singular_value,_ = torch.svd(norm_hard_sim.float() + 1e-4*norm_hard_sim.float().mean()*torch.rand(norm_hard_sim.shape[0],norm_hard_sim.shape[1],device=targets.device))

        maxcone = torch.prod(singular_value)
        loss = -1* torch.log(maxcone)
    elif loss_type == 'nuc_norm':
        orth_mat = hard_pos + hard_neg
        orth_mat_sum1 = nn.functional.normalize(orth_mat, p=1,dim = 1)
        orth_mat_sum1_eye = orth_mat_sum1 + torch.eye(N_id,device=targets.device)

        nuc_norm = torch.norm(orth_mat_sum1_eye,'nuc')/(2*N_id)
        if nuc_norm > 1:
            print('Wrong normalization')
        loss = -1*torch.log(nuc_norm)
    elif loss_type == 'diag_prod':
        orth_mat = hard_pos + hard_neg
        #orth_mat_sum1 = nn.functional.normalize(orth_mat, p=norm_type, dim = 1)
        #orth_mat_norm = orth_mat/torch.norm(orth_mat,'fro') * torch.sqrt(torch.tensor(float(N_id),device=targets.device))
        orth_mat_norm = orth_mat/orth_mat.sum().sum() * N_id
        
        loss = -1.*torch.log(torch.prod(torch.diag(orth_mat_norm)))
    elif loss_type == 'diag_sum':
        orth_mat = hard_pos + hard_neg
        #orth_mat_norm = orth_mat/torch.norm(orth_mat,'fro') * torch.sqrt(torch.tensor(float(N_id),device=targets.device))
        orth_mat_norm = orth_mat/orth_mat.sum().sum() * N_id
        
        #orth_mat_sum1 = nn.functional.normalize(orth_mat,p=norm_type,dim = 1)
        loss = -1.*torch.log(torch.diag(orth_mat_norm).sum()/N_id)
        

    orth_mat = hard_pos.detach() + hard_neg.detach()
    nuc_norm = torch.norm(orth_mat.float(),'nuc')
    f_norm = torch.norm(orth_mat.detach(),'fro')
    rank =  torch.tensor(N_id,dtype=torch.float32,device=targets.device)

    orth_measure = (nuc_norm - f_norm)/(rank - torch.sqrt(rank))/f_norm

    return loss, orth_measure


def memory_maxcone_loss(feat_q, feat_k, memory_queue, targets, memory_id, memory_ins_num):
        
    queue_size = memory_queue.size(0)
    bs_size = feat_q.size(0)

    feat_q = nn.functional.normalize(feat_q, dim=1)
    feat_k = nn.functional.normalize(feat_k, dim=1)
    queue = nn.functional.normalize(memory_queue, dim=1)

    N_id = len(torch.unique(targets))
    N_m_id = len(torch.unique(memory_id))
    N_ins = bs_size // N_id

    scale = 8.0

    ## q_k similarity
    sim_qk = torch.matmul(feat_q, feat_k.T)

    sf_sim_qk = torch.exp(scale*sim_qk)

    ## q_m similarity
    sim_qm = torch.matmul(feat_q, queue.T)

    sf_sim_qm = torch.exp(scale*sim_qm)

    is_neg = targets.view(bs_size, 1).expand(bs_size, queue_size).ne(memory_id.expand(bs_size, queue_size)).float()

    sf_sim_qm = sf_sim_qm * is_neg + (-9999.) * (1 - is_neg) # (bs_zise * K)

    hard_pos = torch.eye(N_id,device=targets.device)
    hard_neg = 1 - torch.eye(N_id,device=targets.device)

    m_hard_neg = torch.zeros((N_id,N_m_id-1), device=targets.device)
    for i in range(N_id):
        m_k_count = 0
        for k in range(N_m_id):
            logits = sf_sim_qm[(N_ins*i):(N_ins*(i+1)),(memory_ins_num*k):(memory_ins_num*(k+1))]
            if logits.mean() < 0:
                continue
            else:
                soft_hard = F.softplus(1*torch.logsumexp(logits,dim = 1))
                v,_ = torch.max(logits,1)
                m_hard_neg[i, m_k_count] = soft_hard.mean()
                m_k_count += 1

    for i in range(N_id):
        sorted,_ = torch.sort(m_hard_neg[i,],-1, descending=True)
        top_n = 0
        for j in range(N_id):
            logits = sf_sim_qk[(N_ins*i):(N_ins*(i+1)),(N_ins*j):(N_ins*(j+1))]

            if i == j:
                soft_hard = F.softplus(-1*torch.logsumexp(-1.*logits,dim=1))
                v,_ = torch.min(logits,1)
                hard_pos[i,j] = soft_hard.mean()
            else:
                hard_neg[i,j] = sorted[top_n]
                top_n += 1              


    max_neg,_ = torch.max(hard_neg, dim = 1)
    max_neg_mat = torch.diag(max_neg)
    alpha_hard_pos = hard_pos + 2*max_neg_mat.detach()

    hard_sim = alpha_hard_pos + hard_neg

    ## normalize by L2-norm
    norm_hard_sim = nn.functional.normalize(hard_sim, dim = 1)
    # norm_hard_sim = nn.functional.normalize(hard_sim, p=1,dim = 1)

    try:
        _,singular_value,_ = torch.svd(norm_hard_sim.float())
    except:                     # torch.svd may have convergence issues for GPU and CPU.
        _,singular_value,_ = torch.svd(norm_hard_sim.float() + 1e-4*norm_hard_sim.float().mean()*torch.rand(norm_hard_sim.shape[0],norm_hard_sim.shape[1],device=targets.device))

    maxcone = torch.prod(singular_value)

    # orth_mat = hard_pos.detach() + hard_neg.detach()
    # nuc_norm = torch.norm(orth_mat.float(),'nuc')
    # f_norm = torch.norm(orth_mat.detach(),'fro')
    # rank =  torch.tensor(N_id,dtype=torch.float32,device=targets.device)

     # orth_measure = (nuc_norm - f_norm)/(rank - torch.sqrt(rank))/f_norm
    loss = -1*torch.log(maxcone)
    return loss
