"""Small, synthetic unit tests only. Does not train or replace experiment results."""
import importlib.util, sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('test_cv_driver',ROOT/'scripts/26_five_fold_experiments.py')
cv=importlib.util.module_from_spec(spec);spec.loader.exec_module(cv)

def main():
    rng=np.random.default_rng(6)
    groups=np.repeat(np.arange(50),4).astype(str);y=rng.integers(0,2,(200,8))
    a=cv.balanced_group_folds(groups,y,5,100);b=cv.balanced_group_folds(groups,y,5,100)
    assert np.array_equal(a,b)
    assert set(a)==set(range(5))
    assert pd.DataFrame({'g':groups,'f':a}).groupby('g').f.nunique().max()==1
    d=pd.DataFrame({'sequence_sha256':['a','a','b','b','c','d']})
    joined=cv.union_groups(d,['s1','s2','s2','s3','s4','s5'])
    assert len(set(joined[:4]))==1 and len(set(joined))==3
    sample=np.array([.8,.9,.7,.6,1.])
    assert np.isclose(sample.mean(),.8) and np.isclose(sample.std(ddof=1),np.sqrt(.025))
    for f in range(5):
        te=np.flatnonzero(a==f);rest=np.flatnonzero(a!=f)
        inner=cv.balanced_group_folds(groups[rest],y[rest],5,101+f)
        tr=rest[inner!=0];va=rest[inner==0]
        assert not(set(groups[tr])&set(groups[va]) or set(groups[te])&set(groups[tr]) or set(groups[te])&set(groups[va]))
    print('PASS: deterministic grouped folds; transitive duplicate union; inner/outer group isolation; sample SD formula')

if __name__=='__main__':main()
