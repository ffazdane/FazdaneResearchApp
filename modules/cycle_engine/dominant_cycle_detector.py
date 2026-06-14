import numpy as np
import pandas as pd
from scipy.signal import find_peaks, hilbert

def detrend_series(prices: pd.Series) -> np.ndarray:
    """Detrend prices by subtracting a rolling mean (linear filter) to isolate cycles."""
    if len(prices) < 30:
        # Fallback to subtracting simple mean if series is too short
        return (prices - prices.mean()).values
    rolling_mean = prices.rolling(window=20, min_periods=1).mean()
    return (prices - rolling_mean).values

def detect_fft_cycle(detrended: np.ndarray) -> tuple[float, float]:
    """Identify the peak frequency using Fast Fourier Transform."""
    n = len(detrended)
    if n < 10:
        return 20.0, 50.0 # Default fallback
    
    # Compute FFT
    fft_vals = np.fft.rfft(detrended)
    fft_freqs = np.fft.rfftfreq(n, d=1.0) # 1 day per sample
    amplitudes = np.abs(fft_vals)
    
    # Map frequencies to periods (period = 1/freq)
    periods = np.zeros_like(fft_freqs)
    non_zero = fft_freqs > 0
    periods[non_zero] = 1.0 / fft_freqs[non_zero]
    
    # Filter periods between 5 and 63 trading days
    mask = (periods >= 5) & (periods <= 63)
    if not np.any(mask):
        return 20.0, 50.0
        
    filtered_periods = periods[mask]
    filtered_amps = amplitudes[mask]
    
    peak_idx = np.argmax(filtered_amps)
    best_period = filtered_periods[peak_idx]
    
    # Compute score (0-100) based on relative amplitude of peak
    total_amp = np.sum(amplitudes)
    peak_amp = filtered_amps[peak_idx]
    fft_score = min((peak_amp / (total_amp + 1e-6)) * 400.0, 100.0)
    
    return float(best_period), float(fft_score)

def detect_lomb_scargle_cycle(detrended: np.ndarray) -> tuple[float, float]:
    """
    Fits sines and cosines across candidate cycle lengths to evaluate periodic fit.
    This behaves identically to Lomb-Scargle periodogram for evenly spaced data.
    """
    n = len(detrended)
    if n < 10:
        return 20.0, 50.0
        
    t = np.arange(n)
    candidate_periods = np.linspace(5, 63, 100)
    best_r2 = -1.0
    best_period = 20.0
    
    for period in candidate_periods:
        omega = 2 * np.pi / period
        # Construct regressor matrix [sin(omega*t), cos(omega*t)]
        X = np.column_stack([np.sin(omega * t), np.cos(omega * t), np.ones(n)])
        # Solve least squares
        try:
            coeffs, residuals, _, _ = np.linalg.lstsq(X, detrended, rcond=None)
            pred = X @ coeffs
            ss_tot = np.sum((detrended - np.mean(detrended))**2)
            if ss_tot > 1e-6:
                ss_res = np.sum((detrended - pred)**2)
                r2 = 1.0 - (ss_res / ss_tot)
                if r2 > best_r2:
                    best_r2 = r2
                    best_period = period
        except np.linalg.LinAlgError:
            continue
            
    lomb_score = max(min(best_r2 * 100.0 * 2.0, 100.0), 0.0) # Map R2 to 0-100 score
    return float(best_period), float(lomb_score)

def detect_autocorrelation_cycle(detrended: np.ndarray) -> tuple[float, float]:
    """Find the peak lag in the autocorrelation function between 5 and 63 days."""
    n = len(detrended)
    if n < 15:
        return 20.0, 50.0
        
    max_lag = min(63, n - 2)
    lags = np.arange(5, max_lag + 1)
    corrs = []
    
    mean = np.mean(detrended)
    var = np.var(detrended)
    if var < 1e-8:
        return 20.0, 0.0
        
    for lag in lags:
        cov = np.mean((detrended[:-lag] - mean) * (detrended[lag:] - mean))
        corr = cov / var
        corrs.append(corr)
        
    corrs = np.array(corrs)
    if len(corrs) == 0:
        return 20.0, 50.0
        
    # Find peaks in autocorrelation space
    peaks, _ = find_peaks(corrs, distance=3)
    if len(peaks) > 0:
        peak_idx = peaks[np.argmax(corrs[peaks])]
        best_period = lags[peak_idx]
        corr_val = corrs[peak_idx]
    else:
        # Fallback to absolute max if no formal peaks
        best_idx = np.argmax(corrs)
        best_period = lags[best_idx]
        corr_val = corrs[best_idx]
        
    ac_score = max(min(corr_val * 100.0, 100.0), 0.0)
    return float(best_period), float(ac_score)

def detect_swing_cycle(prices: pd.Series) -> tuple[float, float]:
    """Calculate the cycle period based on distances between price swing highs and lows."""
    if len(prices) < 20:
        return 20.0, 50.0
        
    # Smooth prices with simple moving average
    smoothed = prices.rolling(window=5, min_periods=1).mean().values
    
    # Detect peaks and troughs
    peaks, _ = find_peaks(smoothed, distance=5)
    troughs, _ = find_peaks(-smoothed, distance=5)
    
    peak_diffs = np.diff(peaks) if len(peaks) > 1 else np.array([])
    trough_diffs = np.diff(troughs) if len(troughs) > 1 else np.array([])
    
    all_diffs = np.concatenate([peak_diffs, trough_diffs])
    
    if len(all_diffs) == 0:
        return 20.0, 30.0 # Low swing confidence
        
    # Filter spacing to normal cycle lengths (5 to 63)
    valid_diffs = all_diffs[(all_diffs >= 5) & (all_diffs <= 63)]
    if len(valid_diffs) == 0:
        return 20.0, 30.0
        
    mean_period = np.mean(valid_diffs)
    # Score based on cycle consistency (lower coefficient of variation = higher score)
    std_period = np.std(valid_diffs)
    cv = std_period / mean_period if mean_period > 0 else 1.0
    
    swing_score = max(min((1.0 - cv) * 100.0, 100.0), 10.0)
    return float(mean_period), float(swing_score)

def detect_hilbert_cycle(detrended: np.ndarray) -> tuple[float, float]:
    """Extract instantaneous cycle length using Hilbert transform phase angle."""
    n = len(detrended)
    if n < 10:
        return 20.0, 50.0
        
    try:
        analytic_signal = hilbert(detrended)
        phase = np.unwrap(np.angle(analytic_signal))
        # Instantaneous frequency is derivative of phase
        inst_freq = np.diff(phase) / (2.0 * np.pi)
        
        # Convert to period (lags)
        periods = 1.0 / (inst_freq + 1e-8)
        
        # Take the median period of the last 40 days (recent dominant period)
        recent_periods = periods[-min(40, len(periods)):]
        valid_periods = recent_periods[(recent_periods >= 5) & (recent_periods <= 63)]
        
        if len(valid_periods) == 0:
            return 20.0, 50.0
            
        best_period = np.median(valid_periods)
        # Score based on how stable the frequency is
        stability = 1.0 - min(np.std(valid_periods) / (best_period + 1e-8), 1.0)
        hilbert_score = stability * 100.0
        return float(best_period), float(hilbert_score)
    except Exception:
        return 20.0, 50.0

def detect_dominant_cycle(prices: pd.Series) -> dict:
    """
    Blends FFT, Lomb-Scargle, Autocorrelation, Swing Cycle, and Hilbert calculations.
    Returns dominant cycle length and strength score (0-100).
    """
    if len(prices) < 15:
        return {
            "dominant_cycle_days": 20.0,
            "cycle_strength": 50.0,
            "secondary_cycle_days": 40.0,
            "secondary_cycle_strength": 30.0,
            "method_agreement_score": 50.0,
            "methods": {}
        }
        
    detrended = detrend_series(prices)
    
    # 1. Run all detectors
    fft_period, fft_score = detect_fft_cycle(detrended)
    lomb_period, lomb_score = detect_lomb_scargle_cycle(detrended)
    ac_period, ac_score = detect_autocorrelation_cycle(detrended)
    swing_period, swing_score = detect_swing_cycle(prices)
    hilbert_period, hilbert_score = detect_hilbert_cycle(detrended)
    
    periods = np.array([fft_period, lomb_period, ac_period, swing_period, hilbert_period])
    scores = np.array([fft_score, lomb_score, ac_score, swing_score, hilbert_score])
    
    # 2. Method Agreement Score
    # Measured as coefficient of variation across periods (low spread = high agreement)
    mean_period = np.mean(periods)
    std_period = np.std(periods)
    method_cv = std_period / mean_period if mean_period > 0 else 1.0
    agreement_score = max(min((1.0 - method_cv) * 100.0, 100.0), 0.0)
    
    # 3. Weighted Blending of Dominant Period
    # Give higher weights to high-scoring methods
    weights = scores / np.sum(scores + 1e-6)
    blended_period = np.sum(periods * weights)
    
    # Clamp blended period between 5 and 63 days
    dominant_period = max(min(blended_period, 63.0), 5.0)
    
    # 4. Cycle Strength Formula
    # cycle_strength = 0.30*fft + 0.25*lomb + 0.20*ac + 0.15*swing + 0.10*agreement
    strength = (
        0.30 * fft_score +
        0.25 * lomb_score +
        0.20 * ac_score +
        0.15 * swing_score +
        0.10 * agreement_score
    )
    
    # Compute secondary peak (usually the harmonic or the next strongest)
    secondary_period = 40.0
    for p in [40.0, 63.0, 10.0, 20.0]:
        if abs(p - dominant_period) > 8:
            secondary_period = p
            break
            
    return {
        "dominant_cycle_days": round(dominant_period, 1),
        "cycle_strength": round(strength, 1),
        "secondary_cycle_days": round(secondary_period, 1),
        "secondary_cycle_strength": round(strength * 0.7, 1),
        "method_agreement_score": round(agreement_score, 1),
        "methods": {
            "fft": {"period": round(fft_period, 1), "score": round(fft_score, 1)},
            "lomb": {"period": round(lomb_period, 1), "score": round(lomb_score, 1)},
            "autocorrelation": {"period": round(ac_period, 1), "score": round(ac_score, 1)},
            "swing": {"period": round(swing_period, 1), "score": round(swing_score, 1)},
            "hilbert": {"period": round(hilbert_period, 1), "score": round(hilbert_score, 1)}
        }
    }
