def lyap_loss(V, Vdot, circle):  
  return Vdot

def lyap_verif(connectives, variables, V, Vdot):
  lyap_negated = Vdot >= 0                       
  
  not_origin = connectives["Not"](connectives["And"](*[xi == 0 for xi in variables]))
  return connectives["And"](lyap_negated, not_origin)

lyapunov= {"loss": lyap_loss, "verif": lyap_verif}

def sign_loss_neg(V, Vdot, circle):  
  return V

def sign_verif_neg(connectives, variables, V, Vdot):
  return V>=0

negative= {"loss": sign_loss_neg, "verif": sign_verif_neg}

def sign_loss_pos(V, Vdot, circle):  
  return V

def sign_verif_pos(connectives, variables, V, Vdot):
  return V<=0

positive = {"loss": sign_loss_pos, "verif": sign_verif_pos}

def barrier_loss_belt(B_d, Bdot_d, circle):            
        margin = 0
        belt_index = torch.nonzero(torch.abs(B_d) <= 0.5)

        if belt_index.nelement() != 0:
            dB_belt = torch.index_select(Bdot_d, dim=0, index=belt_index[:, 0])          # choose the elements of Bdot_d for which B_d is close to zero
            loss= (torch.relu(dB_belt + 0 * margin)).mean()
            percent_belt = (
                100 * ((dB_belt <= -margin).count_nonzero()).item() / dB_belt.shape[0]
            )            
        else:
            loss= 0
            percent_belt= 0
          
        return loss, percent_belt

def barrier_verif(connectives, variables, V, Vdot):
        return _And(V == 0, Vdot >= 0)

barrier = {"loss": barrier_loss_belt, "verif": barrier_verif }
