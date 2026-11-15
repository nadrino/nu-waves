from Support.ExampleSupport import sampleCount, probabilityOnDevice
import importlib.util
import numpy as np
import matplotlib.pyplot as plt
from nu_waves.models.mixing import Mixing
from nu_waves.models.spectrum import Spectrum
from nu_waves.hamiltonian.vacuum import Hamiltonian
from nu_waves.propagation.oscillator import Oscillator


def main():
    mixing = Mixing(3, {(1, 2): np.deg2rad(33.4), (1, 3): np.deg2rad(8.6),
                        (2, 3): np.deg2rad(49)}, {(1, 3): np.deg2rad(195)})
    spectrum = Spectrum(3, {(2, 1): 7.42e-5, (3, 2): .0024428})
    oscillator = Oscillator(Hamiltonian(mixing, spectrum, False))
    energy = np.linspace(.2, 3., sampleCount(200))
    reference = oscillator.probability(295., energy)
    np.testing.assert_allclose(reference.sum(axis=-1), 1., atol=1e-12)
    np.testing.assert_allclose(oscillator.probability(295., 1e6), np.eye(3), atol=1e-10)
    if importlib.util.find_spec('torch'):
        import torch
        devices = ['cpu']
        if torch.cuda.is_available(): devices.append('cuda')
        if torch.backends.mps.is_available(): devices.append('mps')
        for device in devices:
            actual = probabilityOnDevice(oscillator, 295., energy, device)
            np.testing.assert_allclose(actual, reference, atol=5e-5 if device == 'mps' else 1e-12,
                                       rtol=5e-4 if device == 'mps' else 1e-10)
            print(f'{device}: vacuum probability parity OK')
    else:
        print('NumPy checks passed; Torch is not installed.')
    for flavor, label in enumerate(['electron', 'muon', 'tau']):
        plt.plot(energy, reference[:, 1, flavor], label=label)
    plt.xlabel('Neutrino energy [GeV]')
    plt.ylabel('Muon-neutrino transition probability')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
