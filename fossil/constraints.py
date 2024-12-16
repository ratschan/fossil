def lyap_loss(V, Vdot, circle):  
  return Vdot

def lyap_verif(connectives, variables, V, Vdot):
  lyap_negated = Vdot >= 0                       
  
  not_origin = connectives["Not"](connectives["And"](*[xi == 0 for xi in variables]))
  return connectives["And"](lyap_negated, not_origin)

lyapunov= {"loss": lyap_loss, "verif": lyap_verif}

def sign_neg_loss(V, Vdot, circle):  
  return V

def sign_neg_strict_verif(connectives, variables, V, Vdot):
  return V>=0

def sign_neg_nonstrict_verif(connectives, variables, V, Vdot):
  return V>0

negative_strict= {"loss": sign_neg_loss, "verif": sign_neg_strict_verif}
negative_nonstrict= {"loss": sign_neg_loss, "verif": sign_neg_nonstrict_verif}

def sign_pos_loss(V, Vdot, circle):  
  return V

def sign_pos_strict_verif(connectives, variables, V, Vdot):
  return V<=0

def sign_pos_nonstrict_verif(connectives, variables, V, Vdot):
  return V<0

positive_strict = {"loss": sign_pos_loss, "verif": sign_pos_strict_verif }
positive_nonstrict = {"loss": sign_pos_loss, "verif": sign_pos_nonstrict_verif }

def barrier_belt_loss(B_d, Bdot_d, circle):            
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

barrier = {"loss": barrier_belt_loss, "verif": barrier_verif }

def sign_boundary_pos_verif(connectives, variables, V, Vdot):
  assert(false)
  return _And(V<=0, _not(self.goal), Vdot>=0)

positive_boundary = { "loss": sign_pos_loss , "verif" : sign_boundary_pos_verif }

def safe_progress_loss(V, Vdot, circle):
  return lyap_loss(V, Vdot, circle)    # this is way too strong, but used in examples

def safe_progress_verif(connectives, variables, V, Vdot):
  return _And(V<=0, _not(self.goal), Vdot>=0)

safe_progress = { "loss": safe_progress_loss, "verif": safe_progress_verif }

