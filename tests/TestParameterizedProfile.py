"""Continuous-profile chain rule and fixed-grid convergence."""
import unittest
from unittest.mock import Mock
import importlib.util
import numpy as np
from nu_waves.matter.ParameterizedProfile import ParameterizedMatterProfile
from nu_waves.propagation.gradient import ProfileParameter, MatterDensity, ElectronFraction, MixingAngle
from nu_waves.globals.backend import Backend
from nu_waves.models.mixing import Mixing
from nu_waves.models.spectrum import Spectrum
from nu_waves.hamiltonian.matter import Hamiltonian
from nu_waves.propagation.oscillator import Oscillator, NeutrinoEventBatch


class TestParameterizedProfile(unittest.TestCase):
    parameters = dict(scale=1., slope=.8, ye=.5)

    def tearDown(self):
        Backend.set_api(np)

    def profile(self, parameters=None, n=8, absolute=False):
        return ParameterizedMatterProfile(np.linspace(0, 1300 if absolute else 1, n+1),
            self.parameters if parameters is None else parameters,
            density=lambda x, p: p['scale'] * (2.6 + p['slope'] * x / (1300 if absolute else 1)),
            electronFraction=lambda x, p: p['ye'],
            densityGradient=lambda x, p: {'scale': 2.6 + p['slope'] * x / (1300 if absolute else 1),
                                          'slope': p['scale'] * x / (1300 if absolute else 1)},
            electronFractionGradient=lambda x, p: {'ye': 1.},
            slicing='absolute' if absolute else 'fraction')

    def oscillator(self, profile):
        h = Hamiltonian(Mixing(3, {(1, 2): .59, (1, 3): .15, (2, 3): .81}, {(1, 3): -1.6}),
                        Spectrum(3, {(2, 1): 7.5e-5, (3, 2): 2.5e-3}), False)
        h.set_matter_profile(profile)
        return Oscillator(h)

    def events(self):
        return NeutrinoEventBatch(np.full(6, 1300.), np.full(6, .8), np.ones(6, dtype=int),
                                 np.tile(np.arange(3), 2), np.repeat([False, True], 3))

    def testChainRuleAndFiniteDifferences(self):
        names = ['ye', 'scale', 'slope', 'scale']
        for absolute in [False, True]:
            osc = self.oscillator(self.profile(absolute=absolute))
            p, g = osc.probability(self.events(), parameterBasis=[ProfileParameter(name) for name in names])
            np.testing.assert_allclose(p, osc.probability(self.events()), atol=1e-14)
            np.testing.assert_allclose(g[:, 1], g[:, 3], atol=1e-14)
            for column, name in enumerate(names[:3]):
                step = 1e-3
                def shifted(offset):
                    parameters = dict(self.parameters)
                    parameters[name] += offset * step
                    return self.oscillator(self.profile(parameters, absolute=absolute)).probability(self.events())
                numerical = (shifted(-2) - 8*shifted(-1) + 8*shifted(1) - shifted(2)) / (12*step)
                np.testing.assert_allclose(g[:, column], numerical, rtol=1e-6, atol=1e-9)
            basis = [MatterDensity(k) for k in range(8)] + [ElectronFraction(k) for k in range(8)]
            _, local = osc.probability(self.events(), parameterBasis=basis)
            dr, dy = osc.hamiltonian._matter_profile.parameterGradients(names)['scale']
            np.testing.assert_allclose(g[:, 1], local[:, :8] @ dr + local[:, 8:] @ dy, atol=1e-13)
            _, mixed = osc.probability(self.events(), parameterBasis=[MixingAngle(2, 3), ProfileParameter('ye')])
            np.testing.assert_allclose(mixed[:, 1], g[:, 0], atol=1e-13)
            np.testing.assert_allclose(g.reshape(2, 3, 4).sum(axis=1), 0, atol=1e-12)

    def testConvergence(self):
        results = [self.oscillator(self.profile(n=n)).probability(self.events(), parameterBasis=[ProfileParameter('scale')])
                   for n in [8, 16, 32, 128]]
        for component in [0, 1]:
            errors = [np.max(np.abs(result[component] - results[-1][component])) for result in results[:-1]]
            self.assertLess(errors[1], errors[0] / 3)
            self.assertLess(errors[2], errors[1] / 3)

    def testParameterAffectingBothDensityAndFraction(self):
        def profile(q):
            return ParameterizedMatterProfile([0., .4, 1.], {'q': q},
                lambda x, p: 2.6*p['q'], lambda x, p: .5*p['q'],
                lambda x, p: {'q': 2.6}, lambda x, p: {'q': .5})
        _, gradient = self.oscillator(profile(.8)).probability(self.events(), parameterBasis=[ProfileParameter('q')])
        step = 1e-4
        numerical = (self.oscillator(profile(.8+step)).probability(self.events())
                     - self.oscillator(profile(.8-step)).probability(self.events())) / (2*step)
        np.testing.assert_allclose(gradient[:, 0], numerical, rtol=1e-6, atol=1e-9)
        with self.assertRaises(TypeError):
            profile(.8).parameters['q'] = .9

    def testLazyDerivativesAndValidation(self):
        profile = self.profile()
        callback = Mock(side_effect=AssertionError('Derivative callback was not requested'))
        profile._densityGradient = callback
        self.oscillator(profile).probability(self.events())
        callback.assert_not_called()
        with self.assertRaises(ValueError):
            self.oscillator(self.profile()).probability(self.events(), parameterBasis=[ProfileParameter('unknown')])
        profile = self.profile()
        profile._densityGradient = None
        with self.assertRaises(ValueError):
            self.oscillator(profile).probability(self.events(), parameterBasis=[ProfileParameter('scale')])
        for edges in [[0, 0, 1], [0, .5], [1, 2], [0, np.nan, 1]]:
            with self.assertRaises(ValueError):
                ParameterizedMatterProfile(edges, {}, lambda x, p: 2.6, lambda x, p: .5)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Torch is not installed')
    def testTorch(self):
        import torch
        expected = self.oscillator(self.profile()).probability(self.events(), parameterBasis=[ProfileParameter('scale'), ProfileParameter('ye')])
        for device in ['cpu'] + (['mps'] if torch.backends.mps.is_available() else []):
            Backend.set_api(torch, device=device)
            actual = self.oscillator(self.profile()).probability(self.events(), parameterBasis=[ProfileParameter('scale'), ProfileParameter('ye')])
            for a, b in zip(actual, expected):
                np.testing.assert_allclose(a, b, rtol=3e-3, atol=3e-6)
            Backend.set_api(np)
