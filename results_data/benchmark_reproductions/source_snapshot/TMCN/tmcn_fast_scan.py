"""Associative scan for the exact recurrence x[t] = a[t]*x[t-1] + b[t].

Reassociates floating-point operations; see scan_validation.json and full-model validation.
"""
import torch
def selective_scan(self,u,delta,A,B,C,D):
 a=torch.exp(torch.einsum('bld,dn->bldn',delta,A))
 state=torch.einsum('bld,bln,bld->bldn',delta,B,u)
 offset=1
 while offset<u.shape[1]:
  state=torch.cat((state[:,:offset],state[:,offset:]+a[:,offset:]*state[:,:-offset]),dim=1)
  a=torch.cat((a[:,:offset],a[:,offset:]*a[:,:-offset]),dim=1)
  offset*=2
 return torch.einsum('bldn,bln->bld',state,C)+u*D
