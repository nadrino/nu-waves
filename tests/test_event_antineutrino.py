import numpy as np

from nu_waves.models.mixing import Mixing
from nu_waves.models.spectrum import Spectrum
from nu_waves.hamiltonian import matter
from nu_waves.propagation.oscillator import NeutrinoEvent, NeutrinoEventBatch, Oscillator
from nu_waves.utils.flavors import electron, muon, tau


angles = {(1, 2): np.deg2rad(33.4), (1, 3): np.deg2rad(8.6), (2, 3): np.deg2rad(49)}
phases = {(1, 3): np.deg2rad(195)}
dm2 = {(2, 1): 7.42e-5, (3, 2): 0.0024428}

h = matter.Hamiltonian(
    mixing=Mixing(n_neutrinos=3, mixing_angles=angles, dirac_phases=phases),
    spectrum=Spectrum(n_neutrinos=3, m_lightest=0, dm2=dm2),
    antineutrino=False,
)
h.set_constant_density(rho_in_g_per_cm3=2.8, Ye=0.5)
osc = Oscillator(hamiltonian=h)


def test_mixed_event_batch_antineutrino_matches_split_legacy_calls():
    print("test_mixed_event_batch_antineutrino_matches_split_legacy_calls test...")
    batch = NeutrinoEventBatch(
        L_km=np.array([1300.0, 1300.0, 1300.0, 1300.0, 1300.0]),
        E_GeV=np.array([0.8, 1.6, 2.4, 3.0, 4.2]),
        flavor_emit=np.array([muon, muon, muon, electron, tau]),
        flavor_det=np.array([electron, muon, electron, muon, electron]),
        isAntiNu=np.array([False, True, False, True, True]),
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
    np.testing.assert_allclose(P_events, P_expected, atol=1e-12)
    print("test_mixed_event_batch_antineutrino_matches_split_legacy_calls: success.")


def test_event_list_supports_per_event_antineutrino_flags():
    print("test_event_list_supports_per_event_antineutrino_flags test...")
    events = [
        NeutrinoEvent(L_km=1300.0, E_GeV=2.5, flavor_emit=muon, flavor_det=electron, isAntiNu=False),
        NeutrinoEvent(L_km=1300.0, E_GeV=2.5, flavor_emit=muon, flavor_det=electron, isAntiNu=True),
    ]

    P_events = osc.probability(events)
    assert P_events.shape == (2,)
    assert not np.isclose(P_events[0], P_events[1])
    print("test_event_list_supports_per_event_antineutrino_flags: success.")


def test_compiled_matter_event_batch_preserves_input_order():
    print("test_compiled_matter_event_batch_preserves_input_order test...")
    events = [
        NeutrinoEvent(L_km=1300.0, E_GeV=2.0, flavor_emit=muon, flavor_det=electron, isAntiNu=True),
        NeutrinoEvent(L_km=1300.0, E_GeV=0.8, flavor_emit=muon, flavor_det=electron, isAntiNu=False),
        NeutrinoEvent(L_km=1300.0, E_GeV=4.2, flavor_emit=tau, flavor_det=electron, isAntiNu=True),
        NeutrinoEvent(L_km=1300.0, E_GeV=1.6, flavor_emit=muon, flavor_det=muon, isAntiNu=True),
    ]

    compiled = osc.compile_events(events)
    P_events = osc.probability(events)
    P_compiled = osc.probability(compiled)

    np.testing.assert_allclose(P_compiled, P_events, atol=1e-12)
    print("test_compiled_matter_event_batch_preserves_input_order: success.")


def test_constant_matter_executor_matches_legacy_path():
    print("test_constant_matter_executor_matches_legacy_path test...")
    events = [
        NeutrinoEvent(L_km=1300.0, E_GeV=0.7, flavor_emit=muon, flavor_det=electron, isAntiNu=False),
        NeutrinoEvent(L_km=1300.0, E_GeV=1.2, flavor_emit=muon, flavor_det=muon, isAntiNu=False),
        NeutrinoEvent(L_km=1300.0, E_GeV=2.1, flavor_emit=muon, flavor_det=electron, isAntiNu=True),
        NeutrinoEvent(L_km=1300.0, E_GeV=3.4, flavor_emit=tau, flavor_det=electron, isAntiNu=True),
        NeutrinoEvent(L_km=1300.0, E_GeV=4.8, flavor_emit=electron, flavor_det=muon, isAntiNu=False),
    ]

    compiled = osc.compile_events(events)
    original_use_executor = osc.useExecutor
    try:
        osc.useExecutor = False
        P_legacy = osc.probability(compiled)

        osc.useExecutor = True
        P_executor = osc.probability(compiled)
    finally:
        osc.useExecutor = original_use_executor

    np.testing.assert_allclose(P_executor, P_legacy, atol=1e-12)
    print("test_constant_matter_executor_matches_legacy_path: success.")


def test_constant_matter_optimized_executor_matches_standard_executor():
    print("test_constant_matter_optimized_executor_matches_standard_executor test...")
    events = [
        NeutrinoEvent(L_km=1300.0, E_GeV=0.7, flavor_emit=muon, flavor_det=electron, isAntiNu=False),
        NeutrinoEvent(L_km=1300.0, E_GeV=1.2, flavor_emit=muon, flavor_det=muon, isAntiNu=False),
        NeutrinoEvent(L_km=1300.0, E_GeV=2.1, flavor_emit=muon, flavor_det=electron, isAntiNu=True),
        NeutrinoEvent(L_km=1300.0, E_GeV=3.4, flavor_emit=tau, flavor_det=electron, isAntiNu=True),
        NeutrinoEvent(L_km=1300.0, E_GeV=4.8, flavor_emit=electron, flavor_det=muon, isAntiNu=False),
    ]

    compiled = osc.compile_events(events)
    original_use_executor = osc.useExecutor
    original_optimized = h.enableConstantMatterBatchOptimization
    try:
        osc.useExecutor = True

        h.enableConstantMatterBatchOptimization = False
        P_standard = osc.probability(compiled)

        h.enableConstantMatterBatchOptimization = True
        P_optimized = osc.probability(compiled)
    finally:
        osc.useExecutor = original_use_executor
        h.enableConstantMatterBatchOptimization = original_optimized

    np.testing.assert_allclose(P_optimized, P_standard, atol=1e-12)
    print("test_constant_matter_optimized_executor_matches_standard_executor: success.")


def test_constant_matter_eigh_profiling_collects_stats():
    print("test_constant_matter_eigh_profiling_collects_stats test...")
    events = [
        NeutrinoEvent(L_km=1300.0, E_GeV=0.7, flavor_emit=muon, flavor_det=electron, isAntiNu=False),
        NeutrinoEvent(L_km=1300.0, E_GeV=1.2, flavor_emit=muon, flavor_det=muon, isAntiNu=False),
        NeutrinoEvent(L_km=1300.0, E_GeV=2.1, flavor_emit=muon, flavor_det=electron, isAntiNu=True),
    ]

    compiled = osc.compile_events(events)
    original_optimized = h.enableConstantMatterBatchOptimization
    original_profile = h.enableEighProfiling
    try:
        h.enableConstantMatterBatchOptimization = True
        h.enableEighProfiling = True
        h.resetProfiling()
        osc.probability(compiled)
        stats = h.getProfiling()
    finally:
        h.enableConstantMatterBatchOptimization = original_optimized
        h.enableEighProfiling = original_profile
        h.resetProfiling()

    assert stats["eighCalls"] > 0
    assert stats["eighEvents"] == len(events)
    assert stats["eighTimeSeconds"] >= 0.0
    print("test_constant_matter_eigh_profiling_collects_stats: success.")


test_mixed_event_batch_antineutrino_matches_split_legacy_calls()
test_event_list_supports_per_event_antineutrino_flags()
test_compiled_matter_event_batch_preserves_input_order()
test_constant_matter_executor_matches_legacy_path()
test_constant_matter_optimized_executor_matches_standard_executor()
test_constant_matter_eigh_profiling_collects_stats()
