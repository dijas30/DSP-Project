import numpy as np 
import librosa
from scipy.stats import skew, kurtosis
from scipy.signal import hilbert

# Time domain features
def pre_emphasis_filter(signal, pre_emphasis=0.97):
    return np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

def frame_signal(y, sr, frame_size=0.025, frame_stride=0.01):
    frame_length = int(sr * frame_size)
    frame_step = int(sr * frame_stride)
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=frame_step).T
    frames = frames.copy()
    frames *= np.hamming(frame_length)
    return frames

def zero_crossing_rate(frame):
    return np.mean(np.abs(np.diff(np.sign(frame))))

def time_domain_moments(y):
    return np.array([np.mean(y), np.std(y), skew(y), kurtosis(y)])

def envelope_features(y):
    analytic_signal = hilbert(y)
    amplitude_envelope = np.abs(analytic_signal)
    return np.array([np.mean(amplitude_envelope), np.std(amplitude_envelope)])

def autocorr_features(y):
    autocorr = np.correlate(y, y, mode='full')
    mid = len(autocorr)//2
    autocorr = autocorr[mid:]
    return np.array([np.max(autocorr), np.mean(autocorr)])

def formant_frequencies(y, sr):
    try:
        N = len(y)
        y = y * np.hamming(N)
        A = librosa.lpc(y, order=12)
        rts = np.roots(A)
        rts = [r for r in rts if np.imag(r) >= 0]
        angles = np.angle(rts)
        freqs = angles * (sr/(2*np.pi))
        freqs = np.sort(freqs)
        return freqs[:3] if len(freqs) >= 3 else np.pad(freqs, (0, 3-len(freqs)), 'constant')
    except:
        return np.array([0,0,0])

# Frequency domain features
def frequency_domain_features(y, sr):
    N = len(y)
    Y = np.fft.rfft(y)
    Y_mag = np.abs(Y)
    freqs = np.fft.rfftfreq(N, 1/sr)

    # Avoid zeros for log calculations
    Y_mag_safe = Y_mag + 1e-12

    # Dominant frequency
    dom_freq = freqs[np.argmax(Y_mag)]

    # Magnitude statistics
    mag_mean = np.mean(Y_mag)
    mag_std = np.std(Y_mag)
    mag_skew = skew(Y_mag)
    mag_kurt = kurtosis(Y_mag)
    spectral_crest = np.max(Y_mag) / mag_mean

    # Spectral entropy
    psd = Y_mag_safe**2
    psd_norm = psd / np.sum(psd)
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm))

    # Band energies (low, mid, high, ultra)
    band_limits = [(0,500), (500,2000), (2000,4000), (4000, sr/2)]
    band_energy = [np.sum(psd[(freqs >= low) & (freqs < high)]) for low, high in band_limits]

    # Spectral slope (linear regression)
    slope = np.polyfit(freqs, Y_mag, 1)[0]

    # Spectral rolloff at multiple percentiles
    cumsum = np.cumsum(Y_mag)
    total = cumsum[-1]
    rolloff = []
    for perc in [0.85, 0.9, 0.95]:
        idx = np.searchsorted(cumsum, perc*total)
        rolloff.append(freqs[idx] if idx < len(freqs) else freqs[-1])

    return np.array([dom_freq, mag_mean, mag_std, mag_skew, mag_kurt,
                     spectral_crest, spectral_entropy, *band_energy, slope, *rolloff])

# Harmonic-to-Noise Ratio
def hnr_feature(y):
    y_harm = librosa.effects.harmonic(y)
    y_perc = y - y_harm
    hnr = 10 * np.log10(np.sum(y_harm**2) / (np.sum(y_perc**2)+1e-12))
    return np.array([hnr])

# Feature extraction
def extract_features(y, sr):
    y_preemph = pre_emphasis_filter(y)
    frames = frame_signal(y_preemph, sr)
    energy = np.sum(frames ** 2, axis=1)
    zcr = np.array([zero_crossing_rate(f) for f in frames])
    rms = librosa.feature.rms(y=y_preemph)[0]
    td_moments = time_domain_moments(y_preemph)
    env_feats = envelope_features(y_preemph)
    autocorr_feats = autocorr_features(y_preemph)
    spectral_centroid = librosa.feature.spectral_centroid(y=y_preemph, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y_preemph, sr=sr)[0]
    spectral_contrast = librosa.feature.spectral_contrast(y=y_preemph, sr=sr)
    spectral_flatness = librosa.feature.spectral_flatness(y=y_preemph)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y_preemph, sr=sr)[0]
    spectral_flux = librosa.onset.onset_strength(y=y_preemph, sr=sr)
    pitches, magnitudes = librosa.piptrack(y=y_preemph, sr=sr)
    pitch_values = pitches[magnitudes > np.median(magnitudes)]
    pitch_values = pitch_values[pitch_values > 0] if len(pitch_values) > 0 else np.array([0])
    formants = formant_frequencies(y_preemph, sr)

    mfcc = librosa.feature.mfcc(y=y_preemph, sr=sr, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    mfcc_delta3 = librosa.feature.delta(mfcc, order=3)

    def mfcc_stats(mfcc_feat):
        return np.hstack([
            np.mean(mfcc_feat, axis=1),
            np.std(mfcc_feat, axis=1),
            skew(mfcc_feat, axis=1),
            kurtosis(mfcc_feat, axis=1)
        ])

    mfcc_features = np.hstack([
        mfcc_stats(mfcc),
        mfcc_stats(mfcc_delta),
        mfcc_stats(mfcc_delta2),
        mfcc_stats(mfcc_delta3)
    ])

    chroma = librosa.feature.chroma_stft(y=y_preemph, sr=sr)
    chroma_features = np.hstack([
        np.mean(chroma, axis=1),
        np.std(chroma, axis=1),
        skew(chroma, axis=1),
        kurtosis(chroma, axis=1)
    ])

    freq_feats = frequency_domain_features(y_preemph, sr)
    hnr_feat = hnr_feature(y_preemph)

    feature_vector = np.hstack([
        mfcc_features, chroma_features, td_moments, env_feats, autocorr_feats,
        np.mean(energy), np.std(energy), np.mean(rms), np.std(rms),
        np.mean(zcr), np.std(zcr),
        np.mean(spectral_centroid), np.std(spectral_centroid),
        np.mean(spectral_bandwidth), np.std(spectral_bandwidth),
        np.mean(spectral_flatness), np.std(spectral_flatness),
        np.mean(spectral_contrast, axis=1), np.std(spectral_contrast, axis=1),
        np.mean(spectral_rolloff), np.std(spectral_rolloff),
        np.mean(spectral_flux), np.std(spectral_flux),
        np.mean(pitch_values), np.std(pitch_values),
        formants,
        freq_feats,
        hnr_feat
    ])
    return feature_vector
