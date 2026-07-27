import numpy as np
from nu_waves.models.mixing import Mixing
from nu_waves.models.spectrum import Spectrum
from nu_waves.hamiltonian import vacuum
from nu_waves.propagation.oscillator import CompiledEventBatch, NeutrinoEvent, NeutrinoEventBatch, Oscillator
from nu_waves.utils.flavors import electron, muon, tau
from nu_waves.globals.backend import Backend

# import torch
# Backend.set_api(torch, device='mps')

angles = {(1, 2): np.deg2rad(33.4), (1, 3): np.deg2rad(8.6), (2, 3): np.deg2rad(49)}
phases = {(1, 3): np.deg2rad(195)}
dm2 = {(2, 1): 7.42e-5, (3, 2): 0.0024428}
h = vacuum.Hamiltonian(
    mixing=Mixing(n_neutrinos=3, mixing_angles=angles, dirac_phases=phases),
    spectrum=Spectrum(n_neutrinos=3, m_lightest=0, dm2=dm2),
    antineutrino=False
)

osc = Oscillator(hamiltonian=h)

def test_syntax():
    print("test_syntax test...")
    P = osc.probability(L_km=[0], E_GeV=[1])
    print(f"P = {P}")
    assert P.shape == (3, 3)
    P = osc.probability(L_km=0, E_GeV=np.linspace(0.2, 3.0, 10))
    assert P.shape == (10, 3, 3)
    P = osc.probability(L_km=0, E_GeV=np.linspace(0.2, 3.0, 10), flavor_emit=muon)
    assert P.shape == (10, 3)
    P = osc.probability(L_km=0, E_GeV=np.linspace(0.2, 3.0, 10), flavor_det=muon)
    assert P.shape == (10, 3)
    P = osc.probability(L_km=0, E_GeV=np.linspace(0.2, 3.0, 10), flavor_emit=muon, flavor_det=[muon, electron])
    assert P.shape == (10, 2)
    P = osc.probability(
        L_km=np.linspace(0, 300, 10),
        E_GeV=np.linspace(0.2, 3.0, 10),
        flavor_emit=muon, flavor_det=[muon, electron]
    )
    assert P.shape == (10, 2)
    P = osc.probability(
        L_km=np.linspace(0, 300, 10),
        E_GeV=1,
        flavor_emit=muon, flavor_det=[muon, electron]
    )
    assert P.shape == (10, 2)
    try:
        P = osc.probability(
            L_km=np.linspace(0, 300, 11),
            E_GeV=np.linspace(0.2, 3.0, 10),
            flavor_emit=muon, flavor_det=[muon, electron]
        )
        # SHOULD PRODUCE AN ERROR
        assert False
    except ValueError:
        pass
    print("test_syntax test success.")

def test_zero_baseline_identity():
    print("zero_baseline_identity test...")
    P = osc.probability(
        flavor_emit=muon, flavor_det=[electron, muon, tau],
        L_km=0, E_GeV=np.linspace(0.2, 3.0, 10),
    )
    print(P[:, electron])
    assert np.allclose(P[:, electron], 0.0, atol=1e-14) # no electron appearance
    assert np.allclose(P[:, tau], 0.0, atol=1e-14) # no tau appearance
    assert np.allclose(P[:, muon], 1.0, atol=1e-14) # all muons
    print("zero_baseline_identity: success.")


def test_probability_conservation():
    print("test_probability_conservation test...")
    E_min, E_max = 0.3, 3.0
    Enu_list = np.linspace(E_min, E_max, 3)
    P = osc.probability(
        flavor_emit=muon, flavor_det=[electron, muon, tau],
        L_km=295, E_GeV=Enu_list, # t2k baseline
    )
    try:
        # for any energy, sum of P for each flavor should be 1.
        assert np.allclose(np.sum(P, axis=1), 1.0, atol=1E-18)
    except AssertionError:
        pass
        for iE, flavor_prob in enumerate(P):
            print(f"E({Enu_list[iE]:.3f} GeV):", flavor_prob, f"sum={np.sum(flavor_prob)}")
        print("Assertion failed.")

    print("test_probability_conservation: success.")


def test_event_probability_list_matches_legacy_channels():
    print("test_event_probability_list_matches_legacy_channels test...")
    events = [
        NeutrinoEvent(L_km=295, E_GeV=0.6, flavor_emit=muon, flavor_det=electron),
        NeutrinoEvent(L_km=295, E_GeV=1.0, flavor_emit=muon, flavor_det=muon),
        NeutrinoEvent(L_km=295, E_GeV=2.0, flavor_emit=electron, flavor_det=tau),
    ]

    P_events = osc.probability(events)
    P_all = osc.probability(
        L_km=[event.L_km for event in events],
        E_GeV=[event.E_GeV for event in events],
        flavor_emit=None,
        flavor_det=None,
    )
    P_expected = np.array([
        P_all[i, event.flavor_emit, event.flavor_det]
        for i, event in enumerate(events)
    ])

    assert P_events.shape == (len(events),)
    np.testing.assert_allclose(P_events, P_expected, atol=1e-14)
    print("test_event_probability_list_matches_legacy_channels: success.")


def test_event_probability_batch_matches_legacy_channels():
    print("test_event_probability_batch_matches_legacy_channels test...")
    batch = NeutrinoEventBatch(
        L_km=np.array([295, 295, 295]),
        E_GeV=np.array([0.6, 1.0, 2.0]),
        flavor_emit=np.array([muon, muon, electron]),
        flavor_det=np.array([electron, muon, tau]),
    )

    P_events = osc.probability(batch)
    P_all = osc.probability(
        L_km=batch.L_km,
        E_GeV=batch.E_GeV,
        flavor_emit=None,
        flavor_det=None,
    )
    P_expected = P_all[np.arange(batch.L_km.shape[0]), batch.flavor_emit, batch.flavor_det]

    assert P_events.shape == (batch.L_km.shape[0],)
    np.testing.assert_allclose(P_events, P_expected, atol=1e-14)
    print("test_event_probability_batch_matches_legacy_channels: success.")


def test_event_probability_mixed_antinu_batch_matches_legacy_channels():
    print("test_event_probability_mixed_antinu_batch_matches_legacy_channels test...")
    batch = NeutrinoEventBatch(
        L_km=np.array([295, 295, 295, 295]),
        E_GeV=np.array([0.6, 0.8, 1.0, 2.0]),
        flavor_emit=np.array([muon, muon, muon, electron]),
        flavor_det=np.array([electron, electron, muon, tau]),
        isAntiNu=np.array([False, True, True, False]),
    )

    original_antineutrino = h._antineutrino
    P_events = osc.probability(batch)
    assert h._antineutrino == original_antineutrino

    P_expected = np.zeros(batch.L_km.shape[0])
    for antineutrino in (False, True):
        mask = batch.isAntiNu == antineutrino
        h.set_antineutrino(antineutrino)
        P_all = osc.probability(
            L_km=batch.L_km[mask],
            E_GeV=batch.E_GeV[mask],
            flavor_emit=None,
            flavor_det=None,
        )
        P_expected[mask] = P_all[
            np.arange(np.count_nonzero(mask)),
            batch.flavor_emit[mask],
            batch.flavor_det[mask],
        ]

    h.set_antineutrino(original_antineutrino)
    np.testing.assert_allclose(P_events, P_expected, atol=1e-14)
    print("test_event_probability_mixed_antinu_batch_matches_legacy_channels: success.")


def test_event_probability_list_antinu_defaults_to_global_flag():
    print("test_event_probability_list_antinu_defaults_to_global_flag test...")
    events = [
        NeutrinoEvent(L_km=295, E_GeV=0.6, flavor_emit=muon, flavor_det=electron),
        NeutrinoEvent(L_km=295, E_GeV=1.0, flavor_emit=muon, flavor_det=muon),
    ]

    original_antineutrino = h._antineutrino
    try:
        h.set_antineutrino(True)
        P_events = osc.probability(events)
        P_all = osc.probability(
            L_km=[event.L_km for event in events],
            E_GeV=[event.E_GeV for event in events],
            flavor_emit=None,
            flavor_det=None,
        )
    finally:
        h.set_antineutrino(original_antineutrino)

    P_expected = np.array([
        P_all[i, event.flavor_emit, event.flavor_det]
        for i, event in enumerate(events)
    ])
    np.testing.assert_allclose(P_events, P_expected, atol=1e-14)
    print("test_event_probability_list_antinu_defaults_to_global_flag: success.")


def test_compiled_event_batch_preserves_input_order():
    print("test_compiled_event_batch_preserves_input_order test...")
    events = [
        NeutrinoEvent(L_km=295, E_GeV=1.2, flavor_emit=muon, flavor_det=muon, isAntiNu=False),
        NeutrinoEvent(L_km=295, E_GeV=0.6, flavor_emit=muon, flavor_det=electron, isAntiNu=False),
        NeutrinoEvent(L_km=295, E_GeV=1.8, flavor_emit=electron, flavor_det=tau, isAntiNu=False),
        NeutrinoEvent(L_km=295, E_GeV=0.9, flavor_emit=muon, flavor_det=muon, isAntiNu=False),
    ]

    compiled = osc.compile_events(events)
    assert isinstance(compiled, CompiledEventBatch)

    P_events = osc.probability(events)
    P_compiled = osc.probability(compiled)

    np.testing.assert_allclose(P_compiled, P_events, atol=1e-14)
    print("test_compiled_event_batch_preserves_input_order: success.")


def test_vacuum_executor_matches_disabled_legacy_path():
    print("test_vacuum_executor_matches_disabled_legacy_path test...")
    batch = NeutrinoEventBatch(
        L_km=np.array([295, 295, 295, 295, 295, 295]),
        E_GeV=np.array([0.4, 0.6, 0.9, 1.3, 1.7, 2.2]),
        flavor_emit=np.array([muon, muon, electron, tau, muon, electron]),
        flavor_det=np.array([electron, muon, tau, electron, tau, muon]),
        isAntiNu=np.array([False, False, False, False, True, True]),
    )

    h.enableExecutor = True
    P_executor = osc.probability(batch)

    h.enableExecutor = False
    try:
        P_legacy = osc.probability(batch)
    finally:
        h.enableExecutor = True

    np.testing.assert_allclose(P_executor, P_legacy, atol=1e-14)
    print("test_vacuum_executor_matches_disabled_legacy_path: success.")


def test_event_probability_rejects_extra_arguments():
    print("test_event_probability_rejects_extra_arguments test...")
    events = [NeutrinoEvent(L_km=295, E_GeV=0.6, flavor_emit=muon, flavor_det=electron)]

    try:
        osc.probability(events, flavor_emit=muon)
        assert False
    except TypeError:
        pass
    print("test_event_probability_rejects_extra_arguments: success.")


test_syntax()
test_zero_baseline_identity()
test_probability_conservation()
test_event_probability_list_matches_legacy_channels()
test_event_probability_batch_matches_legacy_channels()
test_event_probability_mixed_antinu_batch_matches_legacy_channels()
test_event_probability_list_antinu_defaults_to_global_flag()
test_compiled_event_batch_preserves_input_order()
test_vacuum_executor_matches_disabled_legacy_path()
test_event_probability_rejects_extra_arguments()
