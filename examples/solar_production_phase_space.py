from Support.ExampleSupport import figureDirectory, dataDirectory, sampleCount
import os
import numpy as np
import matplotlib.pyplot as plt

from nu_waves.sources.sun import SolarModel, SolarSource, R_SUN, integrateTrapezoid

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def plot_phase_space(model, source: SolarSource, outdir="figures"):
    ensure_dir(outdir)
    R, E, f = model.grid(source, nr=300, nE=600)  # r[km], E[MeV], joint pdf
    # 2D density
    plt.figure(figsize=(6.5, 5.0))
    im = plt.pcolormesh(R / R_SUN, E, f, shading="auto")
    plt.xlabel(r"$r/R_\odot$")
    plt.ylabel(r"$E_\nu$ [MeV]")
    plt.title(f"{source.value} production density $f(r,E)$ (normalized)")
    cbar = plt.colorbar(im)
    cbar.set_label("pdf")
    plt.tight_layout()
    plt.show()

    # 1D marginals
    r = np.linspace(0, R_SUN, sampleCount(1000))
    fr = model.radial_pdf(source, r)
    Egrid = np.linspace(*model.sources[source].e_range, sampleCount(2000))
    fE = model.spectrum_pdf(source, Egrid)

    plt.figure(figsize=(6.5, 4.0))
    plt.plot(r / R_SUN, fr / integrateTrapezoid(fr, r), label="radial pdf")
    plt.xlabel(r"$r/R_\odot$")
    plt.ylabel("pdf")
    plt.title(f"{source.value} radial production pdf")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6.5, 4.0))
    plt.plot(Egrid, fE / integrateTrapezoid(fE, Egrid), label="energy pdf")
    plt.xlabel(r"$E_\nu$ [MeV]")
    plt.ylabel("pdf")
    plt.title(f"{source.value} energy spectrum (shape only)")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_multi_radial(model, sources, outdir="figures"):
    ensure_dir(outdir)
    r = np.linspace(0, R_SUN, sampleCount(1000))
    plt.figure(figsize=(6.8, 4.2))
    for s in sources:
        fr = model.radial_pdf(s, r)
        fr /= integrateTrapezoid(fr, r)
        plt.plot(r / R_SUN, fr, label=s.value)
    plt.xlabel(r"$r/R_\odot$")
    plt.ylabel("pdf")
    plt.title("Radial production pdf (comparative)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def describe_radial(model, source):
    r = np.linspace(0, R_SUN, sampleCount(2000))
    fr = model.radial_pdf(source, r)
    fr /= integrateTrapezoid(fr, r)
    peak_x = (r[np.argmax(fr)] / R_SUN)
    cdf = np.cumsum(fr);
    cdf /= cdf[-1]
    r90 = r[np.searchsorted(cdf, 0.90)] / R_SUN
    print(f"[{source.value}] peak at r/Rsun ≈ {peak_x:.3f}, 90% within r/Rsun ≈ {r90:.3f}")


def main():
    model = SolarModel.standard_model()

    describe_radial(model, SolarSource.B8)
    describe_radial(model, SolarSource.BE7)
    describe_radial(model, SolarSource.PP)

    # Focus first on B8 for the later upturn study
    plot_phase_space(model, SolarSource.B8)
    # plot_phase_space(model, SolarSource.BE7)
    # Helpful comparisons
    plot_multi_radial(model, [SolarSource.PP, SolarSource.BE7, SolarSource.B8, SolarSource.PEP])


if __name__ == "__main__":
    main()
