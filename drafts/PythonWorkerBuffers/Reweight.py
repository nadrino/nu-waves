"""GUNDAM PythonWorker: nu-waves writes directly into host or device buffers."""
import sys
import json
from pathlib import Path

# This draft uses the checkout; production workers can import an installed package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from nu_waves.hamiltonian import vacuum
from nu_waves.models.mixing import Mixing
from nu_waves.models.spectrum import Spectrum
from nu_waves.propagation.oscillator import Oscillator, NeutrinoEventBatch
from nu_waves.propagation.gradient import MixingAngle, DiracPhase, MassSquaredDifference


PARAMETER_NAMES = ['theta_12', 'theta_13', 'theta_23', 'delta_cp', 'delta_m2_21', 'delta_m2_32']
BASIS = (MixingAngle(1, 2), MixingAngle(1, 3), MixingAngle(2, 3),
         DiracPhase(1, 3), MassSquaredDifference(2, 1), MassSquaredDifference(3, 2))


def configure(arguments):
    global auditPath
    auditPath = Path(arguments[0])


def makeOscillator(parameterBuffer):
    # Slice before reshaping: NumPy indexing with an integer would copy a scalar.
    # These views follow GUNDAM's in-place parameter updates on every evaluation.
    p = [parameterBuffer[i:i + 1].reshape(()) for i in range(6)]
    mixing = Mixing(3, {(1, 2): p[0], (1, 3): p[1], (2, 3): p[2]}, {(1, 3): p[3]})
    spectrum = Spectrum(3, {(2, 1): p[4], (3, 2): p[5]})
    return Oscillator(vacuum.Hamiltonian(mixing, spectrum, False))


def run(command):
    global events
    action = command['command']
    if action == 'initialize':
        names = [entry['name'] for entry in command['parameterInfo']]
        if names != PARAMETER_NAMES:
            raise ValueError(f'Unexpected PMNS parameter order: {names}')
        inputs = command['inputs']
        events = NeutrinoEventBatch(inputs['L'], inputs['E'], inputs['flavorEmit'],
                                    inputs['flavorDetect'], inputs['isAntiNu'])
        print(f"nu-waves PythonWorker: device={command['bufferDevice']}, "
              f"dtype={command['weights'].dtype}", file=sys.stderr, flush=True)
        # GUNDAM currently exports gradients on host only. nu-waves also supports
        # device Jacobians, ready for a future device gradient transport.
        return {'status': 'ok', 'supportsGradient': command['bufferDevice'] == 'cpu'}
    if action == 'evaluate':
        oscillator = makeOscillator(command['parameterBuffer'])
        withGradient = command.get('withGradient', False)
        oscillator.probability(
            events, outputBuffer=command['weights'],
            parameterBasis=BASIS if withGradient else None,
            gradientOutputBuffer=command['gradients'] if withGradient else None)
        with auditPath.open('a') as stream:
            stream.write(json.dumps({'device': command['bufferDevice']}) + '\n')
        response = {'status': 'ok'}
        if withGradient:
            response['gradientGeneration'] = command['generation']
        return response
    if action == 'shutdown':
        events = None
        return {'status': 'ok'}
    raise ValueError(f'Unknown command: {action}')
