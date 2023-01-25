import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__all__ = ["maxcone_loss", "memory_maxcone_loss"]

def maxcone_loss(feat_q, targets, temp, epochs = None, all_epochs = None, loss_type = None):
    
    feat_q = nn.functional.normalize(feat_q, dim=1)
    # feat_k = nn.functional.normalize(feat_k, dim=1)
    bs_size = feat_q.size(0)
    N_id = len(torch.unique(targets))
    N_ins = bs_size // N_id

    scale = 1./temp
    m = 0.35

    sim_qq = torch.matmul(feat_q, feat_q.T)
    # sf_sim_qq = scale*F.softplus(sim_qq, beta = scale)

    ## margin
    margin_mask = torch.from_numpy(np.kron(np.eye(N_id),(1-m)*np.ones((N_ins,N_ins)))).cuda()
    margin_mask[margin_mask==0]=m

    sim_qq_margin = sim_qq #- margin_mask

    sf_sim_qq = sim_qq_margin*scale
    # sf_sim_qq = torch.exp(scale*sim_qq)

    right_factor = torch.from_numpy(np.kron(np.eye(N_id),np.ones((N_ins,1)))).cuda()

    pos_mask = torch.from_numpy(np.kron(np.eye(N_id),np.ones((N_ins,1)))).cuda()

    left_factor = torch.from_numpy(np.kron(np.eye(N_id), np.ones((1,N_ins)))).cuda()


    ## hard-hard mining for pos
    mask_HH = torch.from_numpy(np.kron(np.eye(N_id),-1.*np.ones((N_ins,N_ins)))).cuda()
    mask_HH[mask_HH==0]=1.

    ID_sim_HH = torch.exp(sf_sim_qq.mul(mask_HH))

    ID_sim_HH = ID_sim_HH.mm(right_factor)

    ID_sim_HH = left_factor.mm(ID_sim_HH)

    pos_mask_id = torch.eye(N_id).cuda()
    pos_sim_HH = ID_sim_HH.mul(pos_mask_id)
    pos_sim_HH[pos_sim_HH==0]=1.
    pos_sim_HH = 1./pos_sim_HH
    ID_sim_HH = ID_sim_HH.mul(1-pos_mask_id) + pos_sim_HH.mul(pos_mask_id)

    ID_sim_HH_L1 = nn.functional.normalize(ID_sim_HH,p = 1, dim = 1)   
    
    ## loss hhh ==> triplet hardmining
    pos = 1./(1./torch.diag(ID_sim_HH)).sum()
    neg = ID_sim_HH.mul(1-pos_mask_id).sum()
    loss_hhh = -1*torch.log(pos/(pos+neg))

    ## loss hhe
    pos = torch.diag(ID_sim_HH).sum()
    neg = ID_sim_HH.mul(1-pos_mask_id).sum()
    loss_hhe = -1*torch.log(pos/(pos+neg))
   
    ## loss mean_hh using all neg
    pos = torch.diag(ID_sim_HH)
    neg = ID_sim_HH.mul(1-pos_mask_id).sum().expand(pos.size())
    loss_mean_hh = -1*torch.log(pos.mul(1./(pos+neg))).mean()

    ## hard-easy mining for pos
    mask_HE = torch.from_numpy(np.kron(np.eye(N_id),-1.*np.ones((N_ins,N_ins)))).cuda()
    mask_HE[mask_HE==0]=1.

    ID_sim_HE = torch.exp(sf_sim_qq.mul(mask_HE))

    ID_sim_HE = ID_sim_HE.mm(right_factor)

    pos_sim_HE = ID_sim_HE.mul(pos_mask)
    pos_sim_HE[pos_sim_HE==0]=1.
    pos_sim_HE = 1./pos_sim_HE
    ID_sim_HE = ID_sim_HE.mul(1-pos_mask) + pos_sim_HE.mul(pos_mask)

    # hard-hard for neg
    ID_sim_HE = left_factor.mm(ID_sim_HE)

    ID_sim_HE_L1 = nn.functional.normalize(ID_sim_HE,p = 1, dim = 1)
    
    ## loss heh
    pos = 1./(1./torch.diag(ID_sim_HE)).sum()
    neg = ID_sim_HE.mul(1-pos_mask_id).sum()
    loss_heh = -1*torch.log(pos/(pos+neg))

    ## loss hee
    pos = torch.diag(ID_sim_HE).sum()
    neg = ID_sim_HE.mul(1-pos_mask_id).sum()
    loss_hee = -1*torch.log(pos/(pos+neg))
    
    ## loss mean_he using all neg
    pos = torch.diag(ID_sim_HE)
    neg = ID_sim_HE.mul(1-pos_mask_id).sum().expand(pos.size())
    loss_mean_he = -1*torch.log(pos.mul(1./(pos+neg))).mean()

    ## center_hard for neg
    # val,row_id = torch.max(pos_sim_HE.mul(pos_mask),0)
    # row_mask = torch.zeros(pos_mask.size()).cuda()
    # row_mask[row_id,:] = 1
    # row_mask = row_mask+pos_mask
    # row_mask[row_mask==2] = 1
    # ID_sim_CH = ID_sim_HE.mul(row_mask)

    # ID_sim_CH = left_factor.mm(ID_sim_CH)

    # ID_sim_CH_sym = ID_sim_CH + ID_sim_CH.T
    # ID_sim_CH_sym = torch.eye(N_id).cuda().mul(ID_sim_CH) + ID_sim_CH_sym.mul(1-torch.eye(N_id).cuda())

    # ID_sim_CH_L1 = nn.functional.normalize(ID_sim_CH_sym,p = 1, dim = 1)


    ## hard-easy mining for neg
    # ID_sim_HE_neg = torch.exp(sf_sim_qq.mul(mask_HE))
    # ID_sim_HE_neg = ID_sim_HE_neg.mm(right_factor)
    # ID_sim_HE_neg = 1./ID_sim_HE_neg

    # ID_sim_HE_neg = left_factor.mm(ID_sim_HE_neg)

    # neg_mask = 1. - torch.eye(N_id).cuda()
    # neg_mask[neg_mask==0] = -1.
    # neg_sim = 1./ID_sim_HE_neg.mul(neg_mask)

    # ID_sim_HE_neg = torch.eye(N_id).cuda().mul(ID_sim_HE_neg) + neg_sim.mul(1-torch.eye(N_id).cuda())
    # ID_sim_HE_neg_sym = ID_sim_HE_neg + ID_sim_HE_neg.T
    # ID_sim_HE_neg_sym = torch.eye(N_id).cuda().mul(ID_sim_HE_neg) + ID_sim_HE_neg_sym.mul(1-torch.eye(N_id).cuda())

    # ID_sim_HE_neg_L1 = nn.functional.normalize(ID_sim_HE_neg_sym,p = 1, dim = 1)   


    ## easy-hard mining for weighting
    mask_EH = torch.from_numpy(np.kron(np.eye(N_id),1.*np.ones((N_ins,N_ins)))).cuda()
    mask_EH = mask_EH.mul(-9999*torch.eye(bs_size,bs_size).cuda())
    mask_EH[mask_EH==0]=1.

    EH_sim_qq = sim_qq*scale
    ID_sim_EH = torch.exp(EH_sim_qq.mul(mask_EH))

    ID_sim_EH = ID_sim_EH.mm(right_factor)

    # # if mode['intra'] == 'soft' and mode['inter'] == 'hard':
    pos_sim_EH = ID_sim_EH.mul(pos_mask)
    pos_sim_EH[pos_sim_EH==0]=1.
    pos_sim_EH = 1./pos_sim_EH
    ID_sim_EH = ID_sim_EH.mul(1-pos_mask) + pos_sim_EH.mul(pos_mask)

    # # left_factor = pos_weight.T

    ID_sim_EH = left_factor.mm(ID_sim_EH)

    pos_mask_2 = torch.eye(N_id).cuda()
    pos_sim_2 = ID_sim_EH.mul(pos_mask_2)
    pos_sim_2[pos_sim_2==0]=1.
    pos_sim_2 = 1./pos_sim_2
    ID_sim_EH = ID_sim_EH.mul(1-pos_mask_2) + pos_sim_2.mul(pos_mask_2)
    ID_sim_EH_L1 = nn.functional.normalize(ID_sim_EH,p = 1, dim = 1)
    test = torch.log(ID_sim_EH)
    

    ##  loss construction
    weight_sim_HH = torch.log(torch.diag(ID_sim_HH)).detach()/scale
    weight_sim_HE = torch.log(torch.diag(ID_sim_HE)).detach()/scale
    # weight_sim_EH = torch.log(torch.diag(ID_sim_EH)).detach()/scale

    mask_sim_HE = torch.zeros(weight_sim_HE.size()).cuda()
    mask_sim_HE[weight_sim_HE >= (weight_sim_HE.mean() - weight_sim_HE.std())] = 1

    mask_sim_HH = torch.zeros(weight_sim_HH.size()).cuda()
    mask_sim_HH[weight_sim_HH >= (weight_sim_HH.mean() - weight_sim_HH.std())] = 1

    loss_HH = -1*torch.log(torch.diag(ID_sim_HH_L1))
    loss_HE = -1*torch.log(torch.diag(ID_sim_HE_L1))

    loss_HH_cut = (-1*torch.log(torch.diag(ID_sim_HH_L1)).mul(mask_sim_HH)).sum()/mask_sim_HH.sum()
    loss_HE_cut = (-1*torch.log(torch.diag(ID_sim_HE_L1)).mul(mask_sim_HE)).sum()/mask_sim_HE.sum()
    # loss_EH = -1*torch.log(torch.diag(ID_sim_EH_L1))

    weight_up = (1+epochs/all_epochs)/2  # weighting with epochs
    weight_down = (1-epochs/all_epochs)/2

    loss_HH_up = weight_up*loss_HH + weight_down*loss_HE
    loss_HE_up = weight_down*loss_HH + weight_up*loss_HE
    
    mask_sim_HH = torch.zeros(weight_sim_HE.size()).cuda()
    
    ## circle border
    #pi = 3.1415926
    #theta = 15
    #theta_arc = torch.tensor(theta/180*pi).cuda()

    #fix_ag = torch.tensor(45/180*pi).cuda()
    #x_0 = torch.sin(fix_ag-theta_arc)/torch.sin(theta_arc)*torch.sqrt(torch.tensor(2.).cuda())
    #x = -1-x_0

    #y_0 = torch.cos(fix_ag-theta_arc)/torch.sin(theta_arc)*torch.sqrt(torch.tensor(2.).cuda())
    #y = y_0-1

    #radius = torch.sqrt(torch.tensor(2.).cuda())/torch.sin(theta_arc)

    #x_cut = torch.sqrt(radius**2-y**2) + x
    #y_cut = y - torch.sqrt(radius**2-x**2)

    #border = weight_sim_HE**2 #y - torch.sqrt(radius**2-(weight_sim_HE-x)**2) #(r_sim_HE+1)**2/2 - 1
    
    #mask_sim_HH[weight_sim_HH > border] = 1
    #mask_sim_HH[weight_sim_HE < 0] = 1

    border = (weight_sim_HE+1)**2/2 - 1
    mask_sim_HH[weight_sim_HH > border] = 1
    
    #print(loss_HH)
    #loss_adaptive = (mask_sim_HH.mul(loss_HH) + (1-mask_sim_HH).mul(loss_HE)).mean()
    #loss_adaptive = (loss_HH.mean()*mask_sim_HH.sum() + loss_HE.mean()*(1-mask_sim_HH).sum())/mask_sim_HH.shape[0]
    
    l_sim = torch.log(torch.diag(ID_sim_HH))
    s_sim = torch.log(torch.diag(ID_sim_HE))

    delta_sim = weight_sim_HE - weight_sim_HH
    sum_sim = (weight_sim_HE + weight_sim_HH)/2
    #wt_l = 1 - delta_sim
    #wt_s = delta_sim

    # ratio
    #both_sim = mask_sim_HH.mean()*l_sim + (1-mask_sim_HH).mean()*s_sim
    mu_delta_sim = delta_sim.mean()

    #mu_wt = 1-mask_sim_HH.sum()/mask_sim_HH.shape[0]
    mu_sum = torch.clamp(sum_sim.mean(), min = 0)
    mu_wt = mu_sum
    
    #wt_l = torch.clamp(delta_sim/mu_delta_sim * (1-mu_wt), max = 1) 
    
    #wt_l = torch.clamp(-1*delta_sim/2/mu_delta_sim + 1, min = 0)
    #wt_l = torch.clamp(-1*delta_sim/mu_delta_sim*mu_wt + 1, min = 0)
    
    wt_l = 2*weight_sim_HE.mul(weight_sim_HH)/(weight_sim_HH + weight_sim_HE)
    wt_l[weight_sim_HH < 0] = 0
    #wt_l[weight_sim_HE < 0] = 1  # add 2022.11.11

    ## adaptive weight by distance ratio
    #p1_x = torch.ones(weight_sim_HE.size()).cuda().double()
    #p1_y = -1 + weight_sim_HH + weight_sim_HE

    #p2_x = 1 + weight_sim_HH + weight_sim_HE
    #p2_y = -1*torch.ones(weight_sim_HE.size()).cuda().double()

    #p_x = p1_x
    #p_y = p1_y
    #p_x[p1_y < -1] = p2_x[p1_y < -1]
    #p_y[p1_y < -1] = p2_y[p1_y < -1]

    #wt_l = torch.abs(weight_sim_HE-weight_sim_HH)/torch.abs(p_y-p_x)

    both_sim = l_sim.mul(wt_l) + s_sim.mul(1-wt_l) 
    
    new_pos = torch.diag(torch.exp(both_sim))

    pos_mask_id = torch.eye(N_id).cuda()
    new_sim_mat = new_pos.mul(pos_mask_id) + ID_sim_HE.mul(1-pos_mask_id)

    new_sim_mat_L1 = nn.functional.normalize(new_sim_mat,p = 1, dim = 1)

    loss_adaptive = -1*torch.log(torch.diag(new_sim_mat_L1)).mean()
    
    if loss_type == 'all_hh': 
        loss = loss_HH.mean()
    elif loss_type == 'cut_hh':
        loss = loss_HH_cut
    elif loss_type == 'all_he':
        loss = loss_HE.mean()
    elif loss_type == 'cut_he':
        loss = loss_HE_cut
    elif loss_type == 'hh_up':
        loss = loss_HH_up
    elif loss_type == 'he_up':
        loss = loss_HE_up
    elif loss_type == 'adaptive':
        loss = loss_adaptive
    elif loss_type == 'margin':
        loss = loss_margin

    loss = loss

    #orth_mat = ID_sim_HE_L1.detach()
    #nuc_norm = torch.norm(orth_mat.float(),'nuc')
    #f_norm = torch.norm(orth_mat.detach(),'fro')
    #rank =  torch.tensor(N_id,dtype=torch.float32,device=targets.device)

    #orth_measure = (nuc_norm - f_norm)/(rank - torch.sqrt(rank))/f_norm

    return loss, mask_sim_HH.sum()/mask_sim_HH.shape[0]

def maxcone_loss_classic(feat_q, targets, temp):

    mode = {'intra':'soft','inter':'hard'}

    feat_q = nn.functional.normalize(feat_q, dim=1)
    # feat_k = nn.functional.normalize(feat_k, dim=1)
    bs_size = feat_q.size(0)
    N_id = len(torch.unique(targets))
    N_ins = bs_size // N_id

    #temp = 0.05

    sim_qq = torch.matmul(feat_q, feat_q.T)
    # sf_sim_qq = scale*F.softplus(sim_qq, beta = scale)
    sf_sim_qq = sim_qq/temp
    # sf_sim_qq = torch.exp(scale*sim_qq)

    mask = torch.from_numpy(np.kron(np.eye(N_id),-1.*np.ones((N_ins,N_ins)))).cuda()
    mask[mask==0]=1.

    ID_sim_mat = torch.exp(sf_sim_qq.mul(mask))

    right_factor = torch.from_numpy(np.kron(np.eye(N_id),np.ones((N_ins,1)))).cuda()

    ID_sim_mat = ID_sim_mat.mm(right_factor)

    if mode['intra'] == 'soft' and mode['inter'] == 'hard':
        pos_mask = torch.from_numpy(np.kron(np.eye(N_id),np.ones((N_ins,1)))).cuda()
        pos_sim = ID_sim_mat.mul(pos_mask)
        pos_sim[pos_sim==0]=1.
        pos_sim = 1./pos_sim
        ID_sim_mat = ID_sim_mat.mul(1-pos_mask) + pos_sim.mul(pos_mask)

    if mode['intra'] == 'soft' and mode['inter'] == 'soft':
        ID_sim_mat = 1./ID_sim_mat
    
    if mode['intra'] == 'hard' and mode['inter'] == 'soft':
        neg_mask = 1. - torch.from_numpy(np.kron(np.eye(N_id),np.ones((N_ins,1)))).cuda()
        neg_mask[neg_mask==0] = -1.
        neg_sim = 1./ID_sim_mat.mul(neg_mask)

        ID_sim_mat = (1-neg_mask).mul(ID_sim_mat) + neg_sim.mul(neg_mask)

    # ## compute weight for each instance
    # mask_pos = torch.from_numpy(np.kron(np.eye(N_id),np.ones((N_ins,N_ins)))).cuda()
    # mask_eye = torch.eye(bs_size,bs_size).cuda()
    # mask_pos = mask_pos.mul(1 - mask_eye)
    # sim_mat_pos = (torch.clamp(sim_qq.detach(),max = 0.5)/temp).mul(mask_pos)
    # sim_mat_pos[sim_mat_pos==0] = -1* 1e9
    # sim_mat_pos = torch.exp(sim_mat_pos)

    # right_factor = torch.from_numpy(np.kron(np.eye(N_id),np.ones((N_ins,1)))).cuda()
    # pos_weight = sim_mat_pos.mm(right_factor)
    # pos_weight = nn.functional.normalize(pos_weight,p = 1, dim = 0)
    # ######

    left_factor = torch.from_numpy(np.kron(np.eye(N_id), np.ones((1,N_ins)))).cuda()
    # left_factor = pos_weight.T

    ID_sim_mat = left_factor.mm(ID_sim_mat)

    # ## max max pos
    if mode['intra'] == 'hard' and mode['inter'] == 'hard':
        pos_mask = torch.eye(N_id).cuda()
        pos_sim = ID_sim_mat.mul(pos_mask)
        pos_sim[pos_sim==0]=1.
        pos_sim = 1./pos_sim
        ID_sim_mat = ID_sim_mat.mul(1-pos_mask) + pos_sim.mul(pos_mask)

    ## min max neg
    if mode['intra']== 'soft' and mode['inter'] == 'soft':
        neg_mask = 1. - torch.eye(N_id).cuda()
        neg_mask[neg_mask==0] = -1.
        neg_sim = 1./ID_sim_mat.mul(neg_mask)

        pos_mask = torch.eye(N_id).cuda()
        ID_sim_mat = pos_mask.mul(ID_sim_mat) + neg_sim.mul(1-pos_mask)

        ID_sim_mat_sym = ID_sim_mat + ID_sim_mat.T
        ID_sim_mat_sym = pos_mask.mul(ID_sim_mat) + ID_sim_mat_sym.mul(1-pos_mask)
        ID_sim_mat = ID_sim_mat_sym
    ##########
    
    if mode['intra'] == 'hard' and mode['inter'] == 'soft':
        ID_sim_mat = 1./ID_sim_mat
        
        pos_mask = torch.eye(N_id).cuda()
        ID_sim_mat_sym = ID_sim_mat + ID_sim_mat.T
        ID_sim_mat_sym = pos_mask.mul(ID_sim_mat) + ID_sim_mat_sym.mul(1-pos_mask)
        ID_sim_mat = ID_sim_mat_sym

    orth_mat_L1 = nn.functional.normalize(ID_sim_mat,p = 1, dim = 1)    
    # orth_mat_Fro = ID_sim_mat/torch.norm(ID_sim_mat,'fro') * torch.sqrt(torch.tensor(N_id,device=targets.device))
    orth_mat_L1sum = ID_sim_mat/ID_sim_mat.sum() * N_id
    # orth_mat_norm = (orth_mat_L2 + orth_mat_Fro)/2
    #orth_mat_norm = (orth_mat_L1+orth_mat_L1sum)/2
    orth_mat_norm = orth_mat_L1
    #orth_mat_norm = orth_mat_L1sum

    diag_pos = torch.diag(orth_mat_norm)
    #diag_pos = torch.clamp(diag_pos, max=1.0)
    loss = -1*torch.log(diag_pos).mean()

    orth_mat = ID_sim_mat.detach()
    nuc_norm = torch.norm(orth_mat.float(),'nuc')
    f_norm = torch.norm(orth_mat.detach(),'fro')
    rank =  torch.tensor(N_id,dtype=torch.float32,device=targets.device)

    orth_measure = (nuc_norm - f_norm)/(rank - torch.sqrt(rank))/f_norm

    return loss, orth_measure



def maxcone_loss_old(feat_q, targets):

    feat_q = nn.functional.normalize(feat_q, dim=1)
        # feat_k = nn.functional.normalize(feat_k, dim=1)
    bs_size = feat_q.size(0)
    N_id = len(torch.unique(targets))
    N_ins = bs_size // N_id
    
    temp = 0.05

    sim_qq = torch.matmul(feat_q, feat_q.T)
    # sf_sim_qq = scale*F.softplus(sim_qq, beta = scale)
    sf_sim_qq = sim_qq/temp

    mask = torch.from_numpy(np.kron(np.eye(N_id),-1.*np.ones((N_ins,N_ins)))).cuda()
    mask[mask==0]=1.
    
    ID_sim_mat = torch.exp(sf_sim_qq.mul(mask))
    
    right_factor = torch.from_numpy(np.kron(np.eye(N_id),np.ones((N_ins,1)))).cuda()
    
    ID_sim_mat = ID_sim_mat.mm(right_factor)
    
    pos_mask = torch.from_numpy(np.kron(np.eye(N_id),np.ones((N_ins,1)))).cuda()
    pos_sim = ID_sim_mat.mul(pos_mask)
    pos_sim[pos_sim==0]=1.
    pos_sim = 1./pos_sim
    
    ID_sim_mat = ID_sim_mat.mul(1-pos_mask) + pos_sim.mul(pos_mask)
    
    ## compute weight for each instance
    mask_pos = torch.from_numpy(np.kron(np.eye(N_id),np.ones((N_ins,N_ins)))).cuda()
    mask_eye = torch.eye(bs_size,bs_size).cuda()
    mask_pos = mask_pos.mul(1 - mask_eye)
    sim_mat_pos = (torch.clamp(sim_qq.detach(),max = 0.5)/temp).mul(mask_pos)
    sim_mat_pos[sim_mat_pos==0] = -1* 1e9
    sim_mat_pos = torch.exp(sim_mat_pos)

    right_factor = torch.from_numpy(np.kron(np.eye(N_id),np.ones((N_ins,1)))).cuda()
    pos_weight = sim_mat_pos.mm(right_factor)
    pos_weight = nn.functional.normalize(pos_weight,p = 1, dim = 0)
    ######

    #left_factor = torch.from_numpy(np.kron(np.eye(N_id),1./N_ins * np.ones((1,N_ins)))).cuda()
    left_factor = pos_weight.T

    ID_sim_mat = left_factor.mm(ID_sim_mat)

    #orth_mat_L2 = nn.functional.normalize(ID_sim_mat,p = 2, dim = 1)
    orth_mat_L1 = nn.functional.normalize(ID_sim_mat,p = 1, dim = 1)
    orth_mat_Fro = ID_sim_mat/torch.norm(ID_sim_mat,'fro') * torch.sqrt(torch.tensor(float(N_id),device=targets.device))
    #orth_mat_norm = (orth_mat_L1 + orth_mat_Fro)/2
    orth_mat_norm = orth_mat_L1
    
    diag_pos = torch.diag(orth_mat_norm)
    diag_pos[diag_pos<=0] = 1e-4

    loss = -1*torch.log(diag_pos).mean()

    orth_mat = ID_sim_mat.detach()
    nuc_norm = torch.norm(orth_mat.float(),'nuc')
    f_norm = torch.norm(orth_mat.detach(),'fro')
    rank =  torch.tensor(N_id,dtype=torch.float32,device=targets.device)

    orth_measure = (nuc_norm - f_norm)/(rank - torch.sqrt(rank))/f_norm

    # temp = 0.1
    # loss_type = 'diag_prod'
    
    # sim_qq = torch.matmul(feat_q, feat_q.T)
    # # sf_sim_qq = scale*F.softplus(sim_qq, beta = scale)
    # sf_sim_qq = sim_qq/temp
    # #sf_sim_qq = torch.exp(scale*sim_qq)

    # hard_pos = torch.eye(N_id,device=targets.device)
    # hard_neg = 1 - torch.eye(N_id,device=targets.device)
    # for i in range(N_id):
    #     for j in range(N_id):
    #         logits = sf_sim_qq[(N_ins*i):(N_ins*(i+1)),(N_ins*j):(N_ins*(j+1))]

    #         if i == j:
    #             soft_hard = 1/torch.exp(-1.*logits).sum(1)
    #             v,_ = torch.min(logits,1)
    #             hard_pos[i,j] = soft_hard.mean()

    #         else:
    #             soft_hard = torch.exp(logits).sum(1)
    #             v,_ = torch.max(logits,1)
    #             hard_neg[i,j] = soft_hard.mean()
    
    # if loss_type == 'max_cone':
    #     max_neg,_ = torch.max(hard_neg, dim = 1)
    #     max_neg_mat = torch.diag(max_neg)
    #     alpha_hard_pos = hard_pos + 2*max_neg_mat.detach()

    #     hard_sim = alpha_hard_pos + hard_neg

    #     ## normalize by L2-norm
    #     norm_hard_sim = nn.functional.normalize(hard_sim, dim = 1)
    #     # norm_hard_sim = nn.functional.normalize(hard_sim, p=1,dim = 1)

    #     try:
    #         _,singular_value,_ = torch.svd(norm_hard_sim.float())
    #     except:                     # torch.svd may have convergence issues for GPU and CPU.
    #         _,singular_value,_ = torch.svd(norm_hard_sim.float() + 1e-4*norm_hard_sim.float().mean()*torch.rand(norm_hard_sim.shape[0],norm_hard_sim.shape[1],device=targets.device))

    #     maxcone = torch.prod(singular_value)
    #     loss = -1* torch.log(maxcone)
    # elif loss_type == 'nuc_norm':
    #     orth_mat = hard_pos + hard_neg
    #     orth_mat_sum1 = nn.functional.normalize(orth_mat, p=1,dim = 1)
    #     orth_mat_sum1_eye = orth_mat_sum1 + torch.eye(N_id,device=targets.device)

    #     nuc_norm = torch.norm(orth_mat_sum1_eye,'nuc')/(2*N_id)
    #     if nuc_norm > 1:
    #         print('Wrong normalization')
    #     loss = -1*torch.log(nuc_norm)
    # elif loss_type == 'diag_prod':
    #     orth_mat = hard_pos + hard_neg
    #     orth_mat_norm = nn.functional.normalize(orth_mat, p=1, dim = 1) # L1
    #     #orth_mat_norm = nn.functional.normalize(orth_mat, p=2, dim = 1) # L2
    #     #orth_mat_norm = orth_mat/orth_mat.sum().sum() * N_id  ## L1-sum
    #     #orth_mat_norm = orth_mat/torch.norm(orth_mat,'fro') * torch.sqrt(torch.tensor(float(N_id),device=targets.device)) # Fro
    #     loss = -1.*torch.log(torch.diag(orth_mat_norm)).mean()
    # elif loss_type == 'diag_sum':
    #     orth_mat = hard_pos + hard_neg
    #     #orth_mat_norm = orth_mat/torch.norm(orth_mat,'fro') * torch.sqrt(torch.tensor(float(N_id),device=targets.device))
    #     orth_mat_norm = orth_mat/orth_mat.sum().sum() * N_id
        
    #     #orth_mat_sum1 = nn.functional.normalize(orth_mat,p=norm_type,dim = 1)
    #     loss = -1.*torch.log(torch.diag(orth_mat_norm).sum()/N_id)
    

    return loss, orth_measure

def memory_maxcone_loss(feat_q, feat_k, memory_queue, targets, memory_id, memory_ins_num):
    
    queue_size = memory_queue.size(0)
    bs_size = feat_q.size(0)

    feat_q = nn.functional.normalize(feat_q, dim=1)
    feat_k = nn.functional.normalize(feat_k, dim=1)
    queue = nn.functional.normalize(memory_queue, dim=1)

    N_id = len(torch.unique(targets))
    N_ins = bs_size // N_id
    N_m_id = queue_size // memory_ins_num  #len(torch.unique(memory_id[memory_id>0]))

    temp = 0.1

    ## q_k similarity
    sim_qk = torch.matmul(feat_q, feat_k.T) # torch.matmul(feat_q, feat_k.T)

    sf_sim_qk = sim_qk/temp

    mask = torch.from_numpy(np.kron(np.eye(N_id,dtype=np.float32),-1.*np.ones((N_ins,N_ins),dtype=np.float32))).cuda()
    mask[mask==0]=1.

    ID_sim_mat = torch.exp(sf_sim_qk.mul(mask))

    right_factor = torch.from_numpy(np.kron(np.eye(N_id,dtype=np.float32),np.ones((N_ins,1),dtype=np.float32))).cuda()

    ID_sim_mat = ID_sim_mat.mm(right_factor)

    pos_mask = torch.from_numpy(np.kron(np.eye(N_id,dtype=np.float32),np.ones((N_ins,1),dtype=np.float32))).cuda()
    pos_sim = ID_sim_mat.mul(pos_mask)
    pos_sim[pos_sim==0]=1.
    pos_sim = 1./pos_sim

    ID_sim_mat = ID_sim_mat.mul(1-pos_mask) + pos_sim.mul(pos_mask)

    left_factor = torch.from_numpy(np.kron(np.eye(N_id,dtype=np.float32),1./N_ins * np.ones((1,N_ins),dtype=np.float32))).cuda()

    ID_sim_mat = left_factor.mm(ID_sim_mat)  # contain pos and neg, pos is located in diagnal entries
    
    if torch.any(torch.isnan(ID_sim_mat)):
        #test = 1
        ID_sim_mat = torch.where(torch.isnan(ID_sim_mat),torch.full_like(ID_sim_mat,1.),ID_sim_mat)

    # hard_pos = torch.diag(ID_sim_mat).reshape((N_id,1))

    # hard_neg generated from memory

    ## q_m similarity
    sim_qm = torch.matmul(feat_q, queue.T)

    sf_sim_qm = sim_qm/temp

    is_neg = targets.view(bs_size, 1).expand(bs_size, queue_size).ne(memory_id.expand(bs_size, queue_size))

    queue_neg_sim_mat_0 = torch.exp(sf_sim_qm[is_neg>0].reshape(sf_sim_qm.shape[0],-1)).float()


    queue_right_factor = torch.from_numpy(np.kron(np.eye(N_m_id-1,dtype=np.float32),np.ones((memory_ins_num,1),dtype=np.float32))).cuda()

    queue_neg_sim_mat_1 = queue_neg_sim_mat_0.mm(queue_right_factor).float()


    queue_left_factor = torch.from_numpy(np.kron(np.eye(N_id,dtype=np.float32),1./memory_ins_num * np.ones((1,memory_ins_num),dtype=np.float32))).cuda()

    queue_neg_sim_mat_2 = queue_left_factor.mm(queue_neg_sim_mat_1).float()  # only contains neg ID in memory

    if torch.any(torch.isnan(queue_neg_sim_mat_2)):
        queue_neg_sim_mat_2 = torch.where(torch.isnan(queue_neg_sim_mat_2),torch.full_like(queue_neg_sim_mat_2,1.),queue_neg_sim_mat_2)

    concate_sim_rect = torch.cat((ID_sim_mat.float(), queue_neg_sim_mat_2), dim=1)

    orth_mat_L2 = nn.functional.normalize(concate_sim_rect,p = 2, dim = 1)
    orth_mat_Fro = concate_sim_rect/torch.norm(concate_sim_rect,'fro') * torch.sqrt(torch.tensor(float(N_id),device=targets.device))
    orth_mat_norm = (orth_mat_L2 + orth_mat_Fro)/2

    diag_pos = torch.diag(orth_mat_norm)
    diag_pos[diag_pos==0] += 1e-4
    loss = -1*torch.log(diag_pos).mean()

    # orth_mat = concate_sim_rect.detach()
    # nuc_norm = torch.norm(orth_mat.float(),'nuc')
    # f_norm = torch.norm(orth_mat.detach(),'fro')
    # rank =  torch.tensor(N_id,dtype=torch.float32,device=targets.device)

    # orth_measure = (nuc_norm - f_norm)/(rank - torch.sqrt(rank))/f_norm

    # if isnan(sim_qk):
    #     test = 1
    # if isnan(sim_qk):
    #     test = 1

    return loss
    
