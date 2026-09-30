
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor

ROOT=Path(__file__).resolve().parent
BASE=ROOT/"processed"
RESULTS=ROOT/"results"
TABLES=ROOT/"tables"
FIG=ROOT/"figures"
for p in [TABLES,FIG]:p.mkdir(parents=True,exist_ok=True)

FEATURES=(BASE/"exact_174_feature_list.txt").read_text().splitlines()
SEED=42

plt.rcParams.update({
    "font.family":"Arial","font.size":9,
    "axes.labelsize":10,"axes.titlesize":10,
    "legend.fontsize":8,"figure.dpi":120,"savefig.dpi":600
})

def data():
    d=pd.read_csv(BASE/"official_1681_with_exact_174_descriptors.csv")
    return d[d.split=="train"].copy(),d[d.split=="val"].copy(),d[d.split=="test"].copy()

def prep(tr,va,te,features=FEATURES):
    A=tr[features].replace([np.inf,-np.inf],np.nan)
    B=va[features].replace([np.inf,-np.inf],np.nan)
    C=te[features].replace([np.inf,-np.inf],np.nan)
    med=A.median().fillna(0)
    return A.fillna(med),B.fillna(med),C.fillna(med)

def make_xgb():
    return XGBRegressor(
        n_estimators=1200,max_depth=5,learning_rate=.02,
        min_child_weight=4,subsample=.85,colsample_bytree=.75,
        reg_alpha=.10,reg_lambda=2.0,objective="reg:absoluteerror",
        eval_metric="mae",tree_method="hist",random_state=SEED,n_jobs=-1)

def save(fig,name):
    fig.savefig(FIG/name,dpi=600,bbox_inches="tight",facecolor="white")
    plt.close(fig)

def main():
    tr,va,te=data()
    Xtr,Xv,Xte=prep(tr,va,te)

    model=make_xgb()
    model.fit(Xtr,tr.em_dft)
    p_tr=model.predict(Xtr)
    p_va=model.predict(Xv)
    p_te=model.predict(Xte)

    # Error table
    pred=pd.DataFrame({
        "material_id":te.material_id.values,
        "edge_id":te.edge_id.values,
        "y_true_eV":te.em_dft.values,
        "prediction_eV":p_te,
    })
    pred["residual_eV"]=pred.prediction_eV-pred.y_true_eV
    pred["abs_error_eV"]=np.abs(pred.residual_eV)
    qs=pred.y_true_eV.quantile([.25,.5,.75]).to_numpy()
    pred["barrier_bin"]=pd.cut(
        pred.y_true_eV,[-np.inf,*qs,np.inf],
        labels=["Q1_low","Q2","Q3","Q4_high"]
    )
    pred.groupby("barrier_bin",observed=True).agg(
        n=("abs_error_eV","size"),
        MAE_eV=("abs_error_eV","mean"),
        RMSE_eV=("residual_eV",lambda x:np.sqrt(np.mean(x*x))),
        bias_eV=("residual_eV","mean"),
    ).reset_index().to_csv(TABLES/"error_by_barrier_bin.csv",index=False)
    pred.sort_values("abs_error_eV",ascending=False).to_csv(TABLES/"worst_test_events.csv",index=False)

    # Permutation importance on validation.
    pi=permutation_importance(
        model,Xv,va.em_dft,n_repeats=25,random_state=SEED,
        scoring="neg_mean_absolute_error",n_jobs=-1
    )
    fi=pd.DataFrame({
        "feature":FEATURES,
        "importance_mean":pi.importances_mean,
        "importance_std":pi.importances_std
    }).sort_values("importance_mean",ascending=False)
    fi.to_csv(TABLES/"permutation_importance.csv",index=False)

    # Bootstrap CIs.
    y=te.em_dft.to_numpy(float); p=p_te
    rng=np.random.default_rng(SEED); rows=[]
    estimates={
        "MAE":np.mean(np.abs(y-p)),
        "RMSE":np.sqrt(np.mean((y-p)**2)),
        "R2":1-np.sum((y-p)**2)/np.sum((y-y.mean())**2)
    }
    for metric in ["MAE","RMSE","R2"]:
        vals=[]
        for _ in range(10000):
            ix=rng.integers(0,len(y),len(y))
            yy=y[ix];pp=p[ix]
            if metric=="MAE":
                vals.append(np.mean(np.abs(yy-pp)))
            elif metric=="RMSE":
                vals.append(np.sqrt(np.mean((yy-pp)**2)))
            else:
                den=np.sum((yy-yy.mean())**2)
                vals.append(1-np.sum((yy-pp)**2)/den if den>0 else np.nan)
        vals=np.asarray(vals);vals=vals[np.isfinite(vals)]
        rows.append({
            "metric":metric,
            "estimate":estimates[metric],
            "CI95_low":np.quantile(vals,.025),
            "CI95_high":np.quantile(vals,.975)
        })
    pd.DataFrame(rows).to_csv(TABLES/"bootstrap_CI95.csv",index=False)

    # Conformal calibration with validation residuals.
    abs_cal=np.abs(va.em_dft.to_numpy(float)-p_va)
    unc=[]
    for cov in [.80,.90,.95]:
        q=float(np.quantile(abs_cal,cov,method="higher"))
        covered=(y>=p-q)&(y<=p+q)
        unc.append({
            "nominal_coverage":cov,
            "half_width_eV":q,
            "empirical_test_coverage":covered.mean(),
            "mean_interval_width_eV":2*q
        })
    pd.DataFrame(unc).to_csv(TABLES/"conformal_uncertainty.csv",index=False)

    # Learning curve.
    rng=np.random.default_rng(SEED)
    lc=[]
    for n in [100,200,400,600,800,1000,1220]:
        ix=np.sort(rng.choice(len(tr),size=n,replace=False))
        m=make_xgb()
        m.fit(Xtr.iloc[ix],tr.em_dft.iloc[ix])
        pv=m.predict(Xv)
        lc.append({
            "train_size":n,
            "validation_MAE_eV":np.mean(np.abs(va.em_dft.to_numpy(float)-pv))
        })
    pd.DataFrame(lc).to_csv(TABLES/"learning_curve.csv",index=False)

    # Parity
    fig,ax=plt.subplots(figsize=(5.2,4.5))
    ax.scatter(y,p,s=25,alpha=.78,color="#0072B2",edgecolor="white",linewidth=.3)
    lo=min(y.min(),p.min());hi=max(y.max(),p.max())
    ax.plot([lo,hi],[lo,hi],color="#222222",lw=1)
    ax.set(xlabel="DFT migration barrier (eV)",ylabel="Predicted migration barrier (eV)",
           title=f"Test parity | MAE={np.mean(np.abs(y-p)):.3f} eV")
    save(fig,"Figure_1_parity_test.png")

    # Residual
    fig,ax=plt.subplots(figsize=(5.2,4.2))
    ax.hist(pred.residual_eV,bins=30,color="#E69F00",edgecolor="white")
    ax.axvline(0,color="#222222",lw=1)
    ax.set(xlabel="Residual, prediction − DFT (eV)",ylabel="Count",title="Test residual distribution")
    save(fig,"Figure_2_residuals.png")

    # Error
    fig,ax=plt.subplots(figsize=(5.2,4.2))
    ax.scatter(y,np.abs(y-p),s=24,alpha=.75,color="#009E73",edgecolor="white",linewidth=.3)
    ax.set(xlabel="DFT migration barrier (eV)",ylabel="Absolute error (eV)",title="Absolute error versus target")
    save(fig,"Figure_3_absolute_error.png")

    # Importance
    top=fi.head(15).sort_values("importance_mean")
    fig,ax=plt.subplots(figsize=(6,5.2))
    ax.barh(top.feature,top.importance_mean,color="#CC79A7")
    ax.set_xlabel("Permutation importance (MAE increase)")
    ax.set_title("Top 15 features")
    save(fig,"Figure_4_feature_importance.png")

    # Calibration
    u=pd.DataFrame(unc)
    fig,ax=plt.subplots(figsize=(5.2,4.2))
    ax.plot(u.nominal_coverage,u.empirical_test_coverage,"o-",color="#56B4E9",lw=2)
    ax.plot([.8,.9,.95],[.8,.9,.95],"--",color="#222222")
    ax.set(xlabel="Nominal coverage",ylabel="Empirical coverage",title="Conformal calibration")
    save(fig,"Figure_5_uncertainty_calibration.png")

    # Learning curve
    lc=pd.read_csv(TABLES/"learning_curve.csv")
    fig,ax=plt.subplots(figsize=(5.2,4.2))
    ax.plot(lc.train_size,lc.validation_MAE_eV,"o-",color="#D55E00",lw=2)
    ax.set(xlabel="Training events",ylabel="Validation MAE (eV)",title="Learning curve")
    save(fig,"Figure_6_learning_curve.png")

    print("ANALYSIS COMPLETE")
    print("Outputs:",TABLES,FIG)

if __name__=="__main__":
    main()
