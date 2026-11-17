"""Chronological bike-share demand forecast with lag and calendar features."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd,joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--input",required=True);p.add_argument("--date",default="dteday");p.add_argument("--target",default="cnt");p.add_argument("--out",default="outputs");a=p.parse_args()
 d=pd.read_csv(a.input); d[a.date]=pd.to_datetime(d[a.date],errors="coerce");d=d.dropna(subset=[a.date,a.target]).sort_values(a.date).copy();source_dates=[d[a.date].min().date().isoformat(),d[a.date].max().date().isoformat()]
 d["hour"]=d[a.date].dt.hour;d["weekday"]=d[a.date].dt.day_of_week;d["month"]=d[a.date].dt.month;d["dayofyear"]=d[a.date].dt.day_of_year
 d["lag_1"]=d[a.target].shift(1);d["lag_7"]=d[a.target].shift(7);d["lag_24"]=d[a.target].shift(24);d["rolling_7"]=d[a.target].shift(1).rolling(7,min_periods=1).mean();d=d.dropna(subset=["lag_24"])
 features=[c for c in d.select_dtypes(include="number") if c!=a.target and c not in {"casual","registered"}]
 cut=max(1,int(len(d)*.8));tr,te=d.iloc[:cut],d.iloc[cut:]
 if len(te)<10:raise SystemExit("Need enough chronologically ordered rows for a holdout.")
 m=HistGradientBoostingRegressor(max_iter=200,l2_regularization=1,random_state=42);m.fit(tr[features],tr[a.target]);pred=m.predict(te[features])
 metrics={"rows":int(len(d)),"input_sha256":hashlib.sha256(Path(a.input).read_bytes()).hexdigest(),"source_date_range":source_dates,"train_rows":len(tr),"test_rows":len(te),"mae":float(mean_absolute_error(te[a.target],pred)),"rmse":float(mean_squared_error(te[a.target],pred)**.5),"lag_1_baseline_mae":float(mean_absolute_error(te[a.target],te["lag_1"])),"lag_24_baseline_mae":float(mean_absolute_error(te[a.target],te["lag_24"])),"split":"last 20% by time; one-step-ahead lag features use previously observed counts"}
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);(out/"metrics.json").write_text(json.dumps(metrics,indent=2));pd.DataFrame({"date":te[a.date],"actual":te[a.target],"prediction":pred}).to_csv(out/"forecast.csv",index=False);joblib.dump({"model":m,"features":features},out/"model.joblib");print(json.dumps(metrics,indent=2))
if __name__=="__main__":main()

