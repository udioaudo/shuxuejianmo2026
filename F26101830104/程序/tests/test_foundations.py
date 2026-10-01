"""Analytical/manufactured fixtures only. These are never contest datasets."""

import unittest
from pathlib import Path
import sys
from contextlib import contextmanager
from uuid import uuid4
import json
import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"03_src"))
from fmodel.quality import Indicator,score_quality,summarize_domains,semantic_conflict
from fmodel.mixture import validate_simplex,MixtureRidge,regression_metrics
from fmodel.scaling import (classical_loss,fit_classical,quality_penalty,
                            quality_penalty_grads,generalized_loss)
from fmodel.costs import (compute_costs,quality_cost,critical_context_length,
                         analytical_classic_allocation,optimize_classic_fixed_context)
from fmodel.audit import audit_project


@contextmanager
def test_directory():
    # Keep tiny fixtures in the project log tree for inspection. Windows sandbox
    # permissions can make tempfile's restrictive system-temp directories unusable.
    root=Path(__file__).resolve().parents[1]/'08_logs'/'test_fixtures'/uuid4().hex
    root.mkdir(parents=True)
    yield root


class QualityTests(unittest.TestCase):
    def setUp(self):
        self.rules=[Indicator("value",1,0,10),Indicator("advertising",-1,0,100)]

    def test_negative_direction(self):
        r=score_quality([[10,0],[0,100],[5,50]],self.rules)
        np.testing.assert_allclose(r['scores'],[1,0,.5])

    def test_missing_not_zero(self):
        r=score_quality([[10,np.nan],[np.nan,np.nan]],self.rules)
        self.assertTrue(np.isnan(r['scores']).all())
        np.testing.assert_allclose(r['coverage'],[.5,0])

    def test_partial_requires_explicit_policy(self):
        r=score_quality([[10,np.nan]],self.rules,min_coverage=.5)
        self.assertEqual(r['scores'][0],1)
        self.assertEqual(r['coverage'][0],.5)

    def test_bounds_not_silently_clipped(self):
        with self.assertRaises(ValueError):score_quality([[11,0]],self.rules)
        self.assertEqual(score_quality([[11,0]],self.rules,clip=True)['clipped_values'],1)

    def test_all_records_accounted_for(self):
        r=summarize_domains([.4,np.nan,.8],['a','a','b'])
        self.assertEqual(sum(x['n_total'] for x in r),3)
        self.assertEqual(r[0]['n_missing'],1)

    def test_conflict_is_semantic_and_reports_coverage(self):
        r=semantic_conflict([[.9,.1],[.1,.9],[np.nan,.1]],0,1)
        np.testing.assert_array_equal(r['conflict'],[True,False,False])
        np.testing.assert_array_equal(r['evaluable'],[True,True,False])


class MixtureTests(unittest.TestCase):
    def test_simplex_rejects_negative(self):
        with self.assertRaises(ValueError):validate_simplex([[1.1,-.1]],2)

    def test_simplex_rejects_bad_total(self):
        with self.assertRaises(ValueError):validate_simplex([[.2,.3]],2)

    def test_simplex_accepts_boundary(self):
        np.testing.assert_array_equal(validate_simplex([[1,0]],2),[[1,0]])

    def test_regression_on_held_out_fixture(self):
        rng=np.random.default_rng(28)
        p=rng.dirichlet(np.ones(17),size=220)
        coef=np.arange(17)/10
        y=2+p@coef
        model=MixtureRidge().fit(p[:180],y[:180],np.repeat(np.arange(18),10),alphas=(1e-8,1e-4))
        self.assertLess(regression_metrics(y[180:],model.predict(p[180:]))['rmse'],1e-5)

    def test_group_tuning_requires_independent_groups(self):
        with self.assertRaises(ValueError):
            MixtureRidge(2).fit([[.2,.8],[.4,.6]],[1,2],[0,0])


class ScalingTests(unittest.TestCase):
    def test_manufactured_surface_parameter_recovery(self):
        n,d=np.meshgrid(np.logspace(7,10,9),np.logspace(8,12,10))
        n,d=n.ravel(),d.ravel()
        params=[1.2,.7,.9,.3,.25]
        y=classical_loss(n,d,params)
        fit=fit_classical(n,d,y,starts=5)
        np.testing.assert_allclose(fit['params'],params,rtol=2e-5,atol=2e-5)
        self.assertEqual(fit['jacobian_rank'],5)

    def test_nonpositive_counts_rejected(self):
        with self.assertRaises(ValueError):classical_loss(0,100,[1,1,1,.3,.2])


class QualityTermTests(unittest.TestCase):
    q={'c':.43,'kappa':.99,'theta_N':.16,'theta_D':.04}

    def test_theta_zero_is_scale_free(self):
        old={'c':.3622,'gamma':.9907}
        for N,D in [(1e8,1e9),(1e10,1e12)]:
            self.assertAlmostEqual(float(quality_penalty(N,D,.675,old)),.3622*.325**.9907,places=12)

    def test_perfect_quality_recovers_classical(self):
        law=[1.69,.354,1.24,.34,.28]
        self.assertAlmostEqual(float(generalized_loss(3e9,6e10,1.0,law,self.q)),
                               float(classical_loss(3e9,6e10,law,1e9,1e9)),places=12)

    def test_gradients_match_finite_differences(self):
        N,D,Q,e=2e9,4e10,.7,1e-6
        hN,hD,hQ=quality_penalty_grads(N,D,Q,self.q)
        h=lambda n,d,q:float(quality_penalty(n,d,q,self.q))
        self.assertAlmostEqual(float(hN),(h(N*(1+e),D,Q)-h(N,D,Q))/e,places=5)
        self.assertAlmostEqual(float(hD),(h(N,D*(1+e),Q)-h(N,D,Q))/e,places=5)
        self.assertAlmostEqual(float(hQ),(h(N,D,Q+e)-h(N,D,Q))/e,places=4)

    def test_penalty_decreases_with_scale(self):
        self.assertGreater(float(quality_penalty(1e8,1e10,.5,self.q)),float(quality_penalty(1e10,1e10,.5,self.q)))


class CostTests(unittest.TestCase):
    def test_critical_ratio(self):
        costs=compute_costs(1e8,1e9,.5,.5,critical_context_length(),'power')
        self.assertEqual(costs['quality'],0)
        self.assertAlmostEqual(costs['attention']/costs['train'],1)

    def test_downgrading_quality_does_not_refund(self):
        for kind in ['exponential','power','logarithmic']:
            self.assertEqual(compute_costs(1e8,1e9,.3,.5,2048,kind)['quality'],0)

    def test_given_cost_parameters(self):
        self.assertAlmostEqual(quality_cost(1,'power'),5e9)
        self.assertAlmostEqual(quality_cost(1,'exponential'),1e7*np.exp(6))
        self.assertAlmostEqual(quality_cost(1,'logarithmic'),2e9*np.log(11))

    def test_analytic_optimum_against_independent_minimization(self):
        C,a,b,alpha,beta=1e20,400,700,.32,.28
        exact=analytical_classic_allocation(C,a,b,alpha,beta)
        K=C/6
        fit=minimize_scalar(lambda z:a*np.exp(-alpha*z)+b*(K/np.exp(z))**(-beta),
                            bounds=(np.log(exact['N']/10),np.log(exact['N']*10)),method='bounded')
        self.assertAlmostEqual(np.exp(fit.x)/exact['N'],1,places=5)
        self.assertAlmostEqual(6*exact['N']*exact['D']/C,1,places=12)

    def test_fixed_context_numeric_optimum(self):
        C=1e20
        expected=analytical_classic_allocation(C/2,400,700,.32,.28)
        actual=optimize_classic_fixed_context(C,30000,400,700,.32,.28,(1e3,1e15),(1e3,1e18))
        self.assertAlmostEqual(actual['N']/expected['N'],1,places=5)
        self.assertLess(abs(actual['relative_budget_residual']),1e-12)

    def test_infeasible_bounds(self):
        with self.assertRaises(ValueError):
            optimize_classic_fixed_context(1e6,1000,1,1,.3,.3,(1e9,1e10),(1e9,1e10))

    def test_invalid_quality(self):
        with self.assertRaises(ValueError):quality_cost(0,'power')


class AuditTests(unittest.TestCase):
    def project(self,root,paths):
        (root/'configs').mkdir();(root/'02_data'/'raw').mkdir(parents=True)
        config={'datasets':{'A1':{'paths':paths}},'requirements':{'q1':{'all_of':['A1']}}}
        (root/'configs'/'datasets.json').write_text(json.dumps(config),encoding='utf-8')

    def test_missing_registration_is_explicit(self):
        with test_directory() as tmp:
            root=Path(tmp);self.project(root,[])
            r=audit_project(root)
            self.assertEqual(r['registered_available_count'],0)
            self.assertEqual(r['requirements']['q1']['missing_registered_inputs'],['A1'])

    def test_file_is_not_semantic_validation(self):
        with test_directory() as tmp:
            root=Path(tmp);self.project(root,['sample.json'])
            (root/'02_data'/'raw'/'sample.json').write_text('{}')
            r=audit_project(root)
            self.assertEqual(r['registered_available_count'],1)
            self.assertFalse(r['competition_results_ready'])

    def test_path_escape_rejected(self):
        with test_directory() as tmp:
            root=Path(tmp);self.project(root,['../../README.md'])
            with self.assertRaises(ValueError):audit_project(root)

    def test_registered_directory_counts_files_without_semantic_approval(self):
        with test_directory() as tmp:
            root=Path(tmp);self.project(root,[])
            config_path=root/'configs/datasets.json'
            config=json.loads(config_path.read_text())
            config['datasets']['A1']={'paths':[],'directory':'tasks','pattern':'*.json','minimum_files':2}
            config_path.write_text(json.dumps(config),encoding='utf-8')
            tasks=root/'02_data/raw/tasks'
            tasks.mkdir()
            (tasks/'one.json').write_text('{}')
            self.assertEqual(audit_project(root)['registered_available_count'],0)
            (tasks/'two.json').write_text('{}')
            result=audit_project(root)
            self.assertEqual(result['registered_available_count'],1)
            self.assertFalse(result['competition_results_ready'])


if __name__=='__main__':
    unittest.main(verbosity=2)
