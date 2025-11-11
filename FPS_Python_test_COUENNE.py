#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monte-Carlo Compromise Search (COUENNE stable version)
  • Parse le graphe (.dot)
  • Lit l’Excel (P0, ε, coûts)
  • Résout MINLP global via COUENNE
  • Simulation Monte-Carlo (P0, ε)
  • Génère : frontier.png, risk_hist.png, chemins_critiques.png, frontier_summary.json
"""

# --------------------- IMPORTS ---------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition, SolverStatus
from scipy.stats import beta, triang, gaussian_kde
import pydot
from tqdm import trange
from pathlib import Path
import re, json
from collections import defaultdict

# --------------------- PARAMÈTRES UTILISATEUR ------------------------------
DOT_PATH   = r"Good_graph.dot"              # fichier DOT
EXCEL_PATH = r"Attack_Measures1.xlsx"       # fichier Excel
OUT_DIR    = r"OUT"                         # dossier de sortie

# === SOLVEUR MINLP (COUENNE recommandé pour non-convexe) ===
SOLVER_NAME  = "couenne"
COUENNE_EXEC = "ampl.macos64/couenne"    # chemin vers couenne (ajuste si besoin)
BONMIN_EXEC  = r"/usr/local/bin/bonmin"

# === Monte-Carlo / Risque ===
theta_global   = 3500.0
target_prob    = 0.85
budgets_to_try = list(range(40_000, 200_000, 10_000))
N_MC           = 3000
SEED           = 42
CONCENTRATION  = 50
TRI_WIDTH      = 0.20
DEFAULT_P0     = 0.30

impact = {"C":12000,"I":10000,"A":7500,"Au":1500,"R":2500,"AC":5000,"Authz":5000}
ATTR_KEYS = list(impact.keys())
RNG = np.random.default_rng(SEED)

# --------------------- OUTILS & PARSING -----------------------------------
def strip_quotes(s): return s.strip('"').strip() if isinstance(s,str) else s

def extract_tid_from_label(label):
    if not isinstance(label,str): return None
    m = re.search(r'\[(T\d{4}(?:\.\d{3})?)\]', label)
    return m.group(1).upper() if m else None

def parse_dot(dot_path):
    g = pydot.graph_from_dot_file(dot_path)[0]
    node_tid, parents, leaves = {}, defaultdict(list), {k:[] for k in ATTR_KEYS}
    nodes=set()
    for n in g.get_nodes():
        nid=strip_quotes(n.get_name())
        if nid in ('graph','node','edge'): continue
        nodes.add(nid)
        tid=extract_tid_from_label(n.get_label() or "")
        node_tid[nid]=tid
    for e in g.get_edges():
        u,v=strip_quotes(e.get_source()),strip_quotes(e.get_destination())
        if v in ATTR_KEYS: leaves[v].append(u)
        else: parents[v].append(u); nodes|={u,v}
    roots=[n for n in nodes if n not in ATTR_KEYS and not parents.get(n)]
    return [n for n in nodes if n not in ATTR_KEYS], parents, node_tid, leaves, roots

def read_measures(excel_path):
    xls=pd.ExcelFile(excel_path)
    df=pd.concat([pd.read_excel(xls,s) for s in xls.sheet_names],ignore_index=True)
    def getcol(*keys):
        for c in df.columns:
            if any(k.lower() in c.lower() for k in keys): return c
    c_tid,c_p0,c_mid,c_eff,c_cost=[getcol("tech","tid"),getcol("p0","prob"),
                                   getcol("mitig","control"),getcol("eff"),getcol("cost")]
    def extract_tid(x):
        if isinstance(x,str):
            m=re.search(r'(T\d{4}(?:\.\d{3})?)',x)
            return m.group(1).upper() if m else None
    P0={}
    if c_tid and c_p0:
        t=df[[c_tid,c_p0]].copy()
        t["tid"]=t[c_tid].apply(extract_tid)
        t["p"]=pd.to_numeric(t[c_p0],errors="coerce")
        P0=t.groupby("tid")["p"].mean().fillna(DEFAULT_P0).clip(0,1).to_dict()
    controls={}
    for _,r in df.iterrows():
        mid=str(r.get(c_mid,"")).strip()
        if not mid: continue
        controls.setdefault(mid,{"targets":set(),"eff":{},"cost":0})
        if pd.notna(r.get(c_cost)): controls[mid]["cost"]=max(controls[mid]["cost"],float(r[c_cost]))
        tid=extract_tid(r.get(c_tid,""))
        if tid and pd.notna(r.get(c_eff)):
            eff=float(r[c_eff]); eff=np.clip(eff,0,1)
            controls[mid]["targets"].add(tid)
            controls[mid]["eff"][tid]=max(controls[mid]["eff"].get(tid,0),eff)
    for m in controls:
        if controls[m]["cost"]<=0: controls[m]["cost"]=15000
    return P0,controls

# --------------------- SOLVEUR MINLP (COUENNE) -----------------------------
def solve_minlp(nodes,parents,node_tid,leaves,P0_by_tid,controls,theta,budget_max):
    P0_node={t:float(np.clip(P0_by_tid.get(node_tid.get(t),DEFAULT_P0),0,1)) for t in nodes}
    T,M,K=sorted(nodes),sorted(controls.keys()),[k for k in ATTR_KEYS if leaves.get(k)]
    tot=sum(impact[k] for k in K) or 1
    TH={k:theta*(impact[k]/tot) for k in K}
    m=pyo.ConcreteModel("ControlSelection")
    m.T,m.M,m.K=pyo.Set(initialize=T),pyo.Set(initialize=M),pyo.Set(initialize=K)
    m.P0=pyo.Param(m.T,initialize=P0_node)
    m.C=pyo.Param(m.M,initialize={x:controls[x]["cost"] for x in M})
    def eps(_,t,x): tid=node_tid.get(t);return float(controls[x]["eff"].get(tid,0)) if tid else 0
    m.eps=pyo.Param(m.T,m.M,initialize=eps)
    par={t:tuple(parents.get(t,[])) for t in T}; m.Par=pyo.Set(m.T,initialize=lambda _,t:par.get(t,()))
    Lk={k:tuple(set(leaves.get(k,[]))&set(T)) for k in K}; m.L=pyo.Set(m.K,initialize=lambda _,k:Lk[k])
    m.I,m.TH=pyo.Param(m.K,initialize={k:impact[k] for k in K}),pyo.Param(m.K,initialize=TH)
    m.x=pyo.Var(m.M,within=pyo.Binary); m.p=pyo.Var(m.T,bounds=(0,1))
    m.q=pyo.Var(m.T,bounds=(0,1)); m.r=pyo.Var(m.K,within=pyo.NonNegativeReals)
    def pr(_,t):return m.p[t]==m.P0[t]*pyo.prod((1-m.eps[t,x]*m.x[x]) for x in m.M)
    m.pdef=pyo.Constraint(m.T,rule=pr)
    roots=[t for t in T if not parents.get(t)]
    def qr(_,t):return m.q[t]==m.p[t] if t in roots else m.q[t]==m.p[t]*(1-pyo.prod((1-m.q[p]) for p in m.Par[t]))
    m.qdef=pyo.Constraint(m.T,rule=qr)
    def rr(_,k):L=list(m.L[k]);return m.r[k]==0 if not L else m.r[k]==m.I[k]*(1-pyo.prod((1-m.q[l]) for l in L))
    m.rdef=pyo.Constraint(m.K,rule=rr)
    m.thr=pyo.Constraint(m.K,rule=lambda _,k:m.r[k]<=m.TH[k])
    m.budget=pyo.Constraint(expr=sum(m.C[x]*m.x[x] for x in m.M)<=budget_max)
    m.obj=pyo.Objective(expr=sum(m.C[x]*m.x[x] for x in m.M))
    try:
        opt=pyo.SolverFactory("couenne",executable=COUENNE_EXEC)
        # ⛔️ pas d'options ici (Couenne 0.5.8 ne les supporte pas via Pyomo)
        results=opt.solve(m,tee=True)
        tc=results.solver.termination_condition
        if tc not in (TerminationCondition.optimal,TerminationCondition.feasible):
            print(f"[WARN] Couenne not optimal ({tc})")
            return []
    except Exception as e:
        print(f"[WARN] Couenne error: {e}")
        return []
    return [x for x in m.M if (m.x[x].value or 0)>0.5]

# --------------------- MONTE-CARLO SIMULATION -------------------------------
def sample_params(P0_node,controls):
    P0_s={t:beta.rvs(p*CONCENTRATION,(1-p)*CONCENTRATION,random_state=RNG) for t,p in P0_node.items()}
    eff_s={}
    for m,obj in controls.items():
        for tid,mode in obj["eff"].items():
            low,high=0,min(1,mode+TRI_WIDTH)
            if high<=low+1e-9: eff_s[(m,tid)]=mode;continue
            c=(mode-low)/(high-low)
            eff_s[(m,tid)]=triang.rvs(c,loc=low,scale=high-low,random_state=RNG)
    return P0_s,eff_s

def risk_calc(xbin,nodes,parents,node_tid,leaves,P0s,effs,controls):
    p={}
    for t in nodes:
        tid=node_tid.get(t);prod=1
        if tid:
            for m,on in xbin.items():
                if on and tid in controls[m]["targets"]:
                    prod*=(1-effs.get((m,tid),0))
        p[t]=P0s[t]*prod
    qcache={}
    def q(u):
        if u in qcache:return qcache[u]
        par=parents.get(u,[])
        val=p[u] if not par else p[u]*(1-np.prod([1-q(pp) for pp in par]))
        qcache[u]=val;return val
    R={a:impact[a]*(1-np.prod([1-q(l) for l in leaves.get(a,[]) if l in nodes])) for a in impact}
    return R,sum(R.values())

# --------------------- PIPELINE PRINCIPAL ----------------------------------
def main():
    Path(OUT_DIR).mkdir(exist_ok=True)
    nodes,parents,node_tid,leaves,roots=parse_dot(DOT_PATH)
    P0,controls=read_measures(EXCEL_PATH)
    P0n={t:float(np.clip(P0.get(node_tid.get(t),DEFAULT_P0),0,1)) for t in nodes}
    print("\nBudget(k)\tCost(k)\tMeanR\tP(R≤θ)\tControls(with nodes)")
    frontier=[]
    for B in budgets_to_try:
        x=solve_minlp(nodes,parents,node_tid,leaves,P0,controls,theta_global,B)
        xd={m:1 for m in x}
        totals=[]
        for _ in trange(N_MC,desc=f"MC {B//1000}k",leave=False):
            P0s,effs=sample_params(P0n,controls)
            _,R=risk_calc(xd,nodes,parents,node_tid,leaves,P0s,effs,controls)
            totals.append(R)
        Rarr=np.array(totals)
        prob=(Rarr<=theta_global).mean();meanR=Rarr.mean()
        cost=sum(controls[m]["cost"] for m in x)
        frontier.append((B,cost,meanR,prob,x,Rarr))
        ctrl=", ".join([f"{m} ({','.join([n for n in nodes if node_tid.get(n) in controls[m]['targets']])})" for m in x])
        print(f"{B/1e3:.0f}\t{cost/1e3:.1f}\t{meanR:.0f}\t{prob:.1%}\t{ctrl}")
    # best plan
    best=max(frontier,key=lambda r:r[3])
    B_star,c_star,mR_star,p_star,x_star,R_star=best
    print(f"\n✅ Best compromise: {B_star/1e3:.0f}k | Cost {c_star/1e3:.1f}k | P(R≤θ)={p_star:.1%}")
    # figures
    fig,ax=plt.subplots(figsize=(6,4))
    ax.hist(R_star,bins=30,density=True,alpha=.7,edgecolor="white")
    kde=gaussian_kde(R_star);xs=np.linspace(R_star.min(),R_star.max(),200)
    ax.plot(xs,kde(xs));ax.axvline(theta_global,color="r",ls="--")
    plt.tight_layout();plt.savefig(Path(OUT_DIR)/"risk_hist.png",dpi=300)
    plt.close()
    # frontier
    fig,ax=plt.subplots(figsize=(6,4))
    ax.plot([B for B,_,_,_,_,_ in frontier],[r for _,_,r,_,_,_ in frontier],'o-')
    plt.tight_layout();plt.savefig(Path(OUT_DIR)/"frontier.png",dpi=300);plt.close()
    # JSON summary
    out=Path(OUT_DIR)/"frontier_summary.json"
    payload=[{"budget":B,"cost_selected":c,"E_R":mR,"P_R_le_theta":p,"selected_controls":x} for B,c,mR,p,x,_ in frontier]
    json.dump({"theta_global":theta_global,"frontier":payload},open(out,"w"),indent=2)
    print(f"\n→ Results saved in {OUT_DIR}")

if __name__=="__main__":
    main()
