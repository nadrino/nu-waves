from Support.ExampleSupport import sampleCount, probabilityOnDevice
import importlib.util
import numpy as np
from nu_waves.models.mixing import Mixing
from nu_waves.models.spectrum import Spectrum
from nu_waves.hamiltonian.matter import Hamiltonian, MatterProfile
from nu_waves.propagation.oscillator import Oscillator


def makeOscillator():
    mixing = Mixing(3, {(1, 2): np.deg2rad(33.4), (1, 3): np.deg2rad(8.6),
                        (2, 3): np.deg2rad(49)}, {(1, 3): np.deg2rad(195)})
    spectrum = Spectrum(3, {(2, 1): 7.42e-5, (3, 2): .0024428})
    hamiltonian = Hamiltonian(mixing, spectrum, False)
    hamiltonian.set_matter_profile(MatterProfile.from_fractions([2.8, 11.], [.5, .467], [.7, .3]))
    return Oscillator(hamiltonian)


def main():
    energy = np.linspace(.3, 5., sampleCount(240))
    reference = makeOscillator().probability(1300., energy)
    np.testing.assert_allclose(reference.sum(axis=-1), 1., atol=1e-12)
    if not importlib.util.find_spec('torch'):
        print('NumPy checks passed; Torch is not installed.')
        return
    import torch
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    if torch.backends.mps.is_available():
        devices.append('mps')
    for device in devices:
        actual = probabilityOnDevice(makeOscillator(), 1300., energy, device)
        np.testing.assert_allclose(actual, reference, atol=5e-5 if device == 'mps' else 1e-12,
                                   rtol=5e-4 if device == 'mps' else 1e-10)
        print(f'{device}: layered probability parity OK')


if __name__ == '__main__':
    main()
