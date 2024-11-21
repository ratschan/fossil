def lyap_loss(V, Vdot, circle):  
  return Vdot

def lyap_verif(connectives, variables, V, Vdot):
  lyap_negated = Vdot >= 0                       
  
  not_origin = connectives["Not"](connectives["And"](*[xi == 0 for xi in variables]))
  return connectives["And"](lyap_negated, not_origin)


lyapunov= {"loss": lyap_loss, "verif": lyap_verif}

