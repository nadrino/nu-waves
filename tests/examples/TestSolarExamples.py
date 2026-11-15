"""Physical limits of the examples' phase-averaged adiabatic approximation."""
from types import SimpleNamespace
from pathlib import Path
import importlib.util
import unittest

import numpy as np
# Load the standalone example helper without adding examples to sys.path.
supportPath = Path(__file__).resolve().parents[2] / 'examples' / 'Support' / 'ExampleSupport.py'
supportSpec = importlib.util.spec_from_file_location('ExampleSupport', supportPath)
support = importlib.util.module_from_spec(supportSpec)
supportSpec.loader.exec_module(support)
from nu_waves.models.mixing import Mixing
from nu_waves.models.spectrum import Spectrum
from nu_waves.hamiltonian.vacuum import Hamiltonian
from nu_waves.propagation.oscillator import Oscillator


class TestSolarExamples(unittest.TestCase):
    def testVacuumMassFractions(self):
        mixing = Mixing(3, {(1, 2): .59, (1, 3): .15, (2, 3): .81}, {(1, 3): -1.6})
        oscillator = Oscillator(Hamiltonian(mixing, Spectrum(3, {(2, 1): 7.5e-5, (3, 2): .0025}), False))
        profile = SimpleNamespace(rho_gcm3=lambda radius: np.zeros_like(radius),
                                  Ye=lambda radius: np.full_like(radius, .5))
        for anti in (False, True):
            fractions = support.adiabaticMassFractions(oscillator, .01, profile, 0., np.linspace(0., 1e5, 32),
                                               antineutrino=anti)
            expected = np.abs(mixing.build_mixing_matrix()[0]) ** 2
            np.testing.assert_allclose(fractions, np.broadcast_to(expected, fractions.shape), atol=1e-12)
            np.testing.assert_allclose(support.initialMassComposition(oscillator), expected, atol=1e-12)
            matrices = support.solarHamiltonians(oscillator, .01, profile, [0., 1e5], anti)
            np.testing.assert_allclose(matrices, matrices.conj().transpose(0, 2, 1), atol=1e-25)


if __name__ == '__main__':
    unittest.main()
