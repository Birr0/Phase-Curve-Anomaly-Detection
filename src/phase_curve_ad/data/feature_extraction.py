import light_curve as lc
import numpy as np

# Time values can be non-evenly separated but must be an ascending array
n = 101
t = np.linspace(0.0, 1.0, n)
perfect_m = 1e3 * t + 1e2
err = np.sqrt(perfect_m)
m = perfect_m + np.random.normal(0, err)

# Half-amplitude of magnitude
amplitude = lc.Amplitude()
# Fraction of points beyond standard deviations from mean
beyond_std = lc.BeyondNStd(nstd=1)
# Slope, its error and reduced chi^2 of linear fit
linear_fit = lc.LinearFit()
# Feature extractor, it will evaluate all features in more efficient way
extractor = lc.Extractor(amplitude, beyond_std, linear_fit)

# Array with all 5 extracted features
result = extractor(t, m, err, sorted=True, check=False)

print('\n'.join(f"{name} = {value:.2f}" for name, value in zip(extractor.names, result)))

# Run in parallel for multiple light curves:
results = amplitude.many(
    [(t[:i], m[:i], err[:i]) for i in range(n // 2, n)],
    n_jobs=-1,
    sorted=True,
    check=False,
)
print("Amplitude of amplitude is {:.2f}".format(np.ptp(results)))``