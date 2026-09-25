"""Independent PDG references for amplitudes and all probability paths."""

import unittest

import numpy as np

from nu_waves.hamiltonian import matter, vacuum
from nu_waves.correlations import GaussianWavePacket
from nu_waves.models.mixing import Mixing
from nu_waves.models.spectrum import Spectrum
from nu_waves.propagation.oscillator import NeutrinoEvent, Oscillator
from nu_waves.utils.units import GEV_TO_EV, KM_TO_EVINV, VCOEFF_EV


def pdgMatrix(delta):
    """Explicit PDG formula, independent of Mixing's rotation construction."""
    s12, s13, s23 = np.sin(np.deg2rad([33.4, 8.6, 49.0]))
    c12, c13, c23 = np.cos(np.deg2rad([33.4, 8.6, 49.0]))
    phase = np.exp(1j * delta)
    return np.array([
        [c12*c13, s12*c13, s13/phase],
        [-s12*c23-c12*s23*s13*phase, c12*c23-s12*s23*s13*phase, s23*c13],
        [s12*s23-c12*c23*s13*phase, -c12*s23-s12*c23*s13*phase, c23*c13],
    ])


def matrixExponential(matrix):
    """Scaling/squaring Taylor reference, independent of the eigensolver."""
    scale = max(0, int(np.ceil(np.log2(max(1.0, np.linalg.norm(matrix, ord=1))))))
    reduced = matrix / 2**scale
    result = np.eye(3, dtype=complex)
    term = result.copy()
    for order in range(1, 40):
        term = term @ reduced / order
        result += term
    for _ in range(scale):
        result = result @ result
    return result


def referencePropagator(delta, isAntiNu, energy, layers):
    mixing = pdgMatrix(delta)
    if isAntiNu:
        mixing = mixing.conj()
    masses = np.array([0.0, 7.42e-5, 7.42e-5 + 2.4428e-3])
    hVacuum = mixing @ np.diag(masses) @ mixing.conj().T / (2 * energy * GEV_TO_EV)
    result = np.eye(3, dtype=complex)
    for length, density in layers:
        potential = (-1 if isAntiNu else 1) * VCOEFF_EV * density * 0.5
        hamiltonian = hVacuum + np.diag([potential, 0, 0])
        result = matrixExponential(-1j * hamiltonian * length * KM_TO_EVINV) @ result
    return result


def buildHamiltonian(delta, mode, isAntiNu=False):
    module = vacuum if mode == "vacuum" else matter
    hamiltonian = module.Hamiltonian(
        mixing=Mixing(3, mixing_angles=dict(zip(
            [(1, 2), (1, 3), (2, 3)], np.deg2rad([33.4, 8.6, 49.0])
        )), dirac_phases={(1, 3): delta}),
        spectrum=Spectrum(3, dm2={(2, 1): 7.42e-5, (3, 2): 2.4428e-3}),
        antineutrino=isAntiNu,
    )
    if mode in ("matter", "optimized"):
        hamiltonian.set_constant_density(2.8)
        hamiltonian.enableConstantMatterBatchOptimization = mode == "optimized"
    elif mode == "layered":
        hamiltonian.set_matter_profile(matter.MatterProfile.from_segments(
            rho_gcm3=[1.0, 6.0], Ye=[0.5, 0.5], lengths_km=[300.0, 1000.0]
        ))
    return hamiltonian


class TestPdgConvention(unittest.TestCase):
    def testDecoherenceUsesPdgMassBasis(self):
        for isAntiNu in (False, True):
            oscillator = Oscillator(buildHamiltonian(-np.pi/2, "vacuum", isAntiNu))
            mixing = pdgMatrix(np.pi/2 if isAntiNu else -np.pi/2)
            actual = oscillator.density_matrix(
                1e6, 0.6, flavor_emit=1, coherence_model=GaussianWavePacket(sigma_x_m=1e-15)
            )[0, 0]
            expected = mixing @ np.diag(abs(mixing[1])**2) @ mixing.conj().T
            np.testing.assert_allclose(actual, expected, atol=1e-12)

    def testAllPathsAgainstIndependentReference(self):
        for delta in (0.0, -np.pi / 2, np.pi / 2):
            for mode in ("vacuum", "zeroDensity", "matter", "optimized", "layered"):
                layers = [(300.0, 1.0), (1000.0, 6.0)] if mode == "layered" else [
                    (295.0, 2.8 if mode in ("matter", "optimized") else 0.0)
                ]
                baseline = sum(length for length, _ in layers)
                events, expectedEvents = [], []
                for isAntiNu in (False, True):
                    hamiltonian = buildHamiltonian(delta, mode, isAntiNu)
                    oscillator = Oscillator(hamiltonian=hamiltonian)
                    energies = np.array([0.4, 0.6, 1.7])
                    expected = np.array([
                        referencePropagator(delta, isAntiNu, energy, layers)
                        for energy in energies
                    ])
                    with self.subTest(delta=delta, mode=mode, isAntiNu=isAntiNu):
                        np.testing.assert_allclose(
                            hamiltonian.mixing.build_mixing_matrix(), pdgMatrix(delta), atol=1e-14
                        )
                        actual = hamiltonian.get_barger_propagator(
                            L=np.full(3, baseline * KM_TO_EVINV), E=energies * GEV_TO_EV
                        )
                        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=2e-13)
                        states = oscillator.propagate_state(baseline, energies).values
                        np.testing.assert_allclose(states, expected.transpose(0, 2, 1), atol=2e-13)
                        probabilities = oscillator.probability(baseline, energies)
                        np.testing.assert_allclose(probabilities, abs(expected.transpose(0, 2, 1))**2, atol=2e-13)
                    for index, energy in enumerate(energies):
                        for emitted in range(3):
                            for detected in range(3):
                                events.append(NeutrinoEvent(baseline, energy, emitted, detected, isAntiNu))
                                expectedEvents.append(abs(expected[index, detected, emitted])**2)
                for useExecutor in (False, True):
                    oscillator = Oscillator(buildHamiltonian(delta, mode), useExecutor=useExecutor)
                    with self.subTest(delta=delta, mode=mode, useExecutor=useExecutor):
                        np.testing.assert_allclose(oscillator.probability(events), expectedEvents, atol=2e-13)
                        compiled = oscillator.compileEvents(events)
                        np.testing.assert_allclose(oscillator.probability(compiled), expectedEvents, atol=2e-13)

    def testVacuumAnalyticAmplitudeAndCpSign(self):
        masses = np.array([0.0, 7.42e-5, 7.42e-5 + 2.4428e-3])
        for delta in (0.0, -np.pi / 2, np.pi / 2):
            for isAntiNu in (False, True):
                mixing = pdgMatrix(-delta if isAntiNu else delta)
                phases = np.exp(-1j * masses * 295 * KM_TO_EVINV / (2 * 0.6 * GEV_TO_EV))
                amplitude = sum(mixing[0, i] * mixing[1, i].conjugate() * phases[i] for i in range(3))
                reference = referencePropagator(delta, isAntiNu, 0.6, [(295, 0)])
                np.testing.assert_allclose(reference[0, 1], amplitude, atol=1e-14)
        enhanced = abs(referencePropagator(-np.pi/2, False, 0.6, [(295, 0)])[0, 1])**2
        suppressed = abs(referencePropagator(np.pi/2, False, 0.6, [(295, 0)])[0, 1])**2
        self.assertGreater(enhanced, suppressed)

    def testAsymmetricProfileIsSensitiveToLayerOrder(self):
        layers = [(300.0, 1.0), (1000.0, 6.0)]
        forward = referencePropagator(-np.pi/2, False, 0.6, layers)
        reverse = referencePropagator(-np.pi/2, False, 0.6, layers[::-1])
        self.assertGreater(np.max(abs(abs(forward)**2 - abs(reverse)**2)), 1e-4)


if __name__ == "__main__":
    unittest.main()
