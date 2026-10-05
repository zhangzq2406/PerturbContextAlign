"""Small synthetic software fixtures; no empirical analysis is rerun."""
import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/fine/code'))
from effect_alignment_metrics import rank_correlation,fractional_topk,expected_dcg,cosine_matrix
from landmark_decoder import LandmarkRidge,build_landmark_features

class MetricContracts(unittest.TestCase):
    def test_constant_rank_is_nan(self):self.assertTrue(np.isnan(rank_correlation(np.ones(4),np.arange(4))))
    def test_rank_exact_positive_negative(self):
        self.assertAlmostEqual(rank_correlation(np.arange(5),np.arange(5)),1)
        self.assertAlmostEqual(rank_correlation(np.arange(5),-np.arange(5)),-1)
    def test_fractional_topk_mass(self):np.testing.assert_allclose(fractional_topk([5,4,4,4,1],2),[1,1/3,1/3,1/3,0])
    def test_all_tied_expected_dcg(self):
        rel=np.array([1,1,0,0,0.]);d=expected_dcg(np.ones(5),rel,3)
        self.assertAlmostEqual(d,rel.mean()*np.sum(1/np.log2(np.arange(2,5))))
    def test_cosine_scale_and_zero(self):np.testing.assert_allclose(cosine_matrix([[1,0],[2,0],[0,0]]),[[1,1,0],[1,1,0],[0,0,0]])
    def test_product_interaction(self):np.testing.assert_array_equal(build_landmark_features([[1,2]],[[3,4]],mode='interaction'),[[3,8]])
    def test_ridge_matches_formula(self):
        x=np.array([[0.,1,1e-10],[1,1,2e-10],[2,1,3e-10],[3,1,4e-10]]);y=np.array([[1.,2],[4,3],[3,5],[2,6]]);q=x[[0,3]]
        m=LandmarkRidge(10).fit(x,y);scale=x.std(0);scale[scale<1e-8]=1;z=(x-x.mean(0))/scale
        w=np.linalg.solve(z.T@z+10*np.eye(3),z.T@(y-y.mean(0)))
        np.testing.assert_allclose(m.predict(q),(q-x.mean(0))/scale@w+y.mean(0),atol=1e-14)
        with self.assertRaises(RuntimeError):m.fit(x,y)
    def test_ridge_has_training_only_transform(self):
        x=np.arange(12.).reshape(4,3);y=np.arange(4.);m=LandmarkRidge(10).fit(x,y);mu=m.feature_mean_.copy();m.predict(x+1000);np.testing.assert_array_equal(mu,m.feature_mean_)
    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):LandmarkRidge(10).fit([[1,np.nan]],[1])
        with self.assertRaises(ValueError):fractional_topk([1,2],2)
if __name__=='__main__':unittest.main()
