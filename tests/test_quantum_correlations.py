import numpy as np

from nu_waves.correlations import GaussianWavePacket, flavor_mode_correlations
from nu_waves.hamiltonian import matter, vacuum
from nu_waves.models.mixing import Mixing
from nu_waves.models.spectrum import Spectrum
from nu_waves.propagation.oscillator import Oscillator
from nu_waves.utils.flavors import electron, muon


def _two_flavor_oscillator():
    return Oscillator(
        vacuum.Hamiltonian(
            mixing=Mixing(n_neutrinos=2, mixing_angles={(1, 2): np.pi / 4}),
            spectrum=Spectrum(n_neutrinos=2, dm2={(2, 1): 2.4e-3}),
            antineutrino=False,
        )
    )


def test_density_matrix_diagonal_matches_probability():
    osc = _two_flavor_oscillator()
    energy = np.linspace(0.3, 1.5, 8)
    rho = osc.density_matrix(L_km=295.0, E_GeV=energy, flavor_emit=electron)
    probability = osc.probability(
        L_km=295.0,
        E_GeV=energy,
        flavor_emit=electron,
        flavor_det=[electron, muon],
    )

    assert rho.shape == (energy.size, 1, 2, 2)
    np.testing.assert_allclose(np.diagonal(rho[:, 0], axis1=-2, axis2=-1).real, probability)
    np.testing.assert_allclose(np.trace(rho[:, 0], axis1=-2, axis2=-1), 1.0)


def test_flavor_mode_metrics_match_two_flavor_probabilities():
    osc = _two_flavor_oscillator()
    rho = osc.density_matrix(L_km=295.0, E_GeV=np.array([0.6]), flavor_emit=electron)[0, 0]
    metrics = flavor_mode_correlations(rho)
    pee, pem = metrics.populations

    np.testing.assert_allclose(metrics.linear_entropy, 4.0 * metrics.populations * (1.0 - metrics.populations))
    np.testing.assert_allclose(metrics.pairwise_concurrence[electron, muon], 2.0 * np.sqrt(pee * pem))
    np.testing.assert_allclose(metrics.pairwise_concurrence[muon, electron], 2.0 * np.sqrt(pee * pem))


def test_gaussian_wave_packet_preserves_trace_and_damps_mass_coherence():
    osc = _two_flavor_oscillator()
    coherence = GaussianWavePacket(sigma_x_m=1e-15)
    rho_coherent = osc.density_matrix(L_km=np.array([0.0, 1e6]), E_GeV=np.array([0.6, 0.6]), flavor_emit=electron)
    rho_damped = osc.density_matrix(
        L_km=np.array([0.0, 1e6]),
        E_GeV=np.array([0.6, 0.6]),
        flavor_emit=electron,
        coherence_model=coherence,
    )

    np.testing.assert_allclose(rho_damped[0], rho_coherent[0], atol=1e-12)
    np.testing.assert_allclose(np.trace(rho_damped[:, 0], axis1=-2, axis2=-1), 1.0, atol=1e-12)

    U = osc.hamiltonian.mixing.build_mixing_matrix()
    rho_mass_coherent = U.T @ rho_coherent[1, 0] @ U.conj()
    rho_mass_damped = U.T @ rho_damped[1, 0] @ U.conj()
    assert abs(rho_mass_damped[0, 1]) < abs(rho_mass_coherent[0, 1]) * 1e-6


def test_gaussian_wave_packet_rejects_matter_hamiltonian():
    osc = Oscillator(
        matter.Hamiltonian(
            mixing=Mixing(n_neutrinos=2, mixing_angles={(1, 2): np.pi / 4}),
            spectrum=Spectrum(n_neutrinos=2, dm2={(2, 1): 2.4e-3}),
            antineutrino=False,
        )
    )
    try:
        osc.density_matrix(295.0, 0.6, electron, GaussianWavePacket(sigma_x_m=1e-6))
        assert False, "Matter wave-packet channel must not silently use a vacuum approximation."
    except ValueError:
        pass
