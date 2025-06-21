linear conv:
full:
    for n in range(N):
        for k in range(len(h)):
            if 0 <= n - k < len(x):
                y[n] += x[n - k] * h[k]

valid:
    for n in range(N):
        for k in range(len(h)):
            y[n] += x[n + k] * h[len(h) - 1 - k]
    return y

same:
    full = lin_convolution(x, h)
    N = len(x)
    start = (len(full) - N) // 2
    end = start + N
    y = full[start:end]

circular:
    for n in range (N):
        for m in range (len(h)):
            y[n] += x[m] * h[(n-m) % N]

    x_freq = DFT(x)
    h_freq = DFT(h)
    y_freq = x_freq * h_freq
    y_circ_from_freq = iDFT(y_freq)

DFT:
    for k in range (N):
        for n in range (N):
            X[k] += x[n] * np.exp(-1j * 2 * np.pi * k * n / N)
iDFT:
    for n in range (N):
        for k in range(N):
            x[n] += X[k] * np.exp(1j * 2 * np.pi * k * n / N)
        x[n] /= N

plot:
N = len(x)
t = np.arange(N) / fsx
f = np.arange(N) * fsx / N
X = DFT(x)
amp = np.abs(X)
ph = np.angle(X)
ph[amp < 0.01] = 0
ax[0].plot(t, x)
ax[1].plot(f, amp)
ax[2].plot(f, ph)

plot with shifting:
f = np.fft.fftshift(np.fft.fftfreq(N, 1/fsx))
X = np.fft.fftshift(X)

convolve wav:
if len(h.shape) > 1:
    h = np.mean(h, axis=1)
if len(x.shape) > 1:
    x = np.mean(x, axis=1)
h = h / np.max(np.abs(h))
x = x / np.max(np.abs(x))
if fsh != fsx:
    h = sig.resample(h, int(len(h) * fsx / fsh))
y = sig.convolve(x, h, mode='same')
y = y / np.max(np.abs(y))
wav.write('./audio/convolved_speech.wav', fsx, y.astype(np.float32))

load wav:
fsx, x = wav.read('./audio/speech.wav')
x = x / np.max(np.abs(x))

HPF:
f = np.fft.fftfreq(N, d=1/fsx)
H = np.zeros(N)
H[np.abs(f) >= 160] = 1
f_vis = np.fft.fftshift(f)
H_vis = np.fft.fftshift(H)
ax[0].plot(f_vis, H_vis)
h = iDFT(H).real
t_h = np.arange(N) / fsx
ax[1].plot(t_h, h)

AM:
carrier = np.cos(2 * np.pi * carrier_freq * t)
x_am = (1 + x) * carrier

hilbert:
analytic_signal = hilbert(x_am)
envelope = np.abs(analytic_signal)

plot AM:
plt.plot(t, x, label='Original Speech Signal')
plt.plot(t, envelope, label='Extracted Envelope')

2D image:
FFT:
F = np.fft.fft2(image)
F = np.fft.fftshift(F)
amplitude = np.abs(F)
phase = np.angle(F)
ax[0].imshow(np.log1p(amplitude), cmap='gray', vmin=0, vmax=10)
ax[1].imshow(phase, cmap='gray', vmin=-np.pi, vmax=np.pi)

DCT:
D = dct(dct(image.T, norm='ortho').T, norm='ortho')
plt.imshow(np.log1p(np.abs(D)), cmap='gray', vmin=0, vmax=10)

Window DCT:
k = 64
window = np.zeros_like(D)
window[:k, :k] = 1
plt.imshow(window, cmap='gray')

Reconstruc DCT:
D = D * window
reconstructed = idct(idct(D.T, norm='ortho').T, norm='ortho')

DFT:
F = np.fft.fft2(image)
F_filtered = F * window
reconstructed_dft = np.fft.ifft2(F_filtered).real
plt.imshow(reconstructed_dft, cmap='gray')

Design window: circular low-pass filter:
rows, cols = image.shape
cx, cy = rows // 2, cols // 2
radius = 80 
Y, X = np.ogrid[:rows, :cols]
distance = np.sqrt((X - cy)**2 + (Y - cx)**2)
new_window = np.zeros_like(image)
new_window[distance <= radius] = 1
F_new = np.fft.fftshift(np.fft.fft2(image))
F_new_filtered = F_new * new_window
reconstructed_new = np.fft.ifft2(np.fft.ifftshift(F_new_filtered)).real

analytic signal:
    N = len(x)
    X = DFT(x)
    X[1:N//2] *= 2
    X[N//2+1:] = 0
    return iDFT(X)

instantaneous_frequency = np.diff(np.unwrap(instantaneous_phase)) / (2 * np.pi) * fs

plt.scatter(np.real(z), np.imag(z), c=t, cmap='hsv', s=2)
plt.colorbar(label='Time [s]')

3D:
ax = fig.add_subplot(111, projection='3d')
ax.plot(np.real(z), np.imag(z), t, color='blue', linewidth=1)

Demodular:
x_demodulated = instantaneous_frequency - fc
plt.plot(t[1:], x_demodulated)

stft:
N = fs * T
for i in range(N):
    if i < fs:
        x[i] = np.sin(2 * np.pi * f1 * t[i])
    elif i < 2 * fs:
        x[i] = np.sin(2 * np.pi * f2 * t[i])
    else:
        x[i] = np.sin(2 * np.pi * f3 * t[i])

def stft(x, window_size, hop_size):
    window = hann(window_size)
    num_frames = 1 + (len(x) - window_size) // hop_size
    
    stft_matrix = []

    for i in range(num_frames):
        start = i * hop_size
        end = start + window_size
        segment = x[start:end] * window
        spectrum = np.fft.fft(segment)
        magnitude = np.abs(spectrum[:window_size // 2])
        stft_matrix.append(magnitude)

    return np.array(stft_matrix)

sampling:
f = 5
fs = 1000
T = 1
t = np.arange(0, T, 1/fs)
x = np.sin(2 * np.pi * f * t)
x_down = x[::M]
t_down = t[::M]
plt.plot(t, x, label='Original Signal')
plt.stem(t_down, x_down, linefmt='r-', markerfmt='ro', basefmt='k', label='Downsampled Signal (M=4)')

L = 3 
x_up = np.zeros(len(x) * L)
x_up[::L] = x
fs_up = fs * L
t_up = np.arange(0, len(x_up)) / fs_up
plt.stem(t_up[:100], x_up[:100])

plot sampling in frequency domain:
X = np.fft.fft(x)
X_down = np.fft.fft(x_down)
X_up = np.fft.fft(x_up)
f = np.fft.fftfreq(len(x), d=1/fs)
f_down = np.fft.fftfreq(len(x_down), d=1/(fs/4))  # fs/M
f_up = np.fft.fftfreq(len(x_up), d=1/(fs*3))      # fs*L
ax[0].plot(f[:len(f)//2], np.abs(X[:len(f)//2]), label='Original Signal')
ax[1].plot(f_down[:len(f_down)//2], np.abs(X_down[:len(f_down)//2]), label='Downsampled Signal', color='orange')
ax[2].plot(f_up[:len(f_up)//2], np.abs(X_up[:len(f_up)//2]), label='Upsampled Signal', color='green')

Lowpass filter:
cutoff = 1 / M

def ideal_lpf(cutoff_hz, fs, numtaps):
    n = np.arange(numtaps)
    middle = (numtaps - 1) / 2
    sinc_arg = 2 * cutoff_hz / fs * (n - middle)
    h = np.sinc(sinc_arg)
    window = np.hanning(numtaps)
    h = h * window
    h = h / np.sum(h)
    return h
lpf = ideal_lpf(cutoff, fs, numtaps) or: lpf = signal.firwin(numtaps, cutoff=cutoff_hz, fs=fs)
x_filtered = signal.lfilter(lpf, 1.0, x)
x_down = x[::M]
x_filtered_down = x_filtered[::M]
N = len(x_down)
f = np.fft.fftfreq(N, d=1 / (fs // M))
f = np.fft.fftshift(f)

up/down:
numtaps = 101
fs_new = fs * L / M
x_up = np.zeros(len(x) * L)
x_up[::L] = x
cutoff = 1 / max(L, M)
lpf = signal.firwin(numtaps, cutoff=cutoff, fs=fs * L)
lpf /= np.sum(lpf)
x_up_filtered = signal.lfilter(lpf, 1.0, x_up)
x_up_filtered = x_up_filtered[numtaps//2:]
x_resampled = x_up_filtered[::M] * L
t_resampled = np.arange(len(x_resampled)) / fs_new

from scipy.signal import resample_poly
x_resampled_scipy = resample_poly(x, up=L, down=M)
t_resampled = np.arange(len(x_resampled)) / fs_new
t_scipy = np.arange(len(x_resampled_scipy)) / fs_new

band pass, band stop (notch), high pass and low pass filters:
f = np.fft.fftfreq(N, d=1/fs)
lpf = np.zeros(N)
lpf[np.abs(f) <= 55] = 1
hpf = np.zeros(N)
hpf[np.abs(f) >= 45] = 1
bpf = lpf * hpf
bsf = 1 - bpf
h_bsf = np.fft.ifft(bsf).real
t = np.arange(N) / fs

DWT:
coeffs = pywt.wavedec(x, wavelet='db2', level=3)
cA3, cD3, cD2, cD1 = coeffs  # Order: [approx, detail3, detail2, detail1]
reconstruct_signal = pywt.waverec(coeffs, wavelet='db2')
reconstruct_signal = reconstruct_signal[:len(x)]

gaus_noise = np.random.normal(0, 0.5, len(x_test))
noisy_signal = x_test + gaus_noise
coeffs_noisy = pywt.wavedec(noisy_signal, wavelet='db4', level=4)
threshold = 0.1
coeffs_noisy_thresholded = [pywt.threshold(c, threshold, mode='soft') for c in coeffs_noisy]
denoised_signal = pywt.waverec(coeffs_noisy_thresholded, wavelet='db4')

total_coeffs = sum(len(c) for c in coeffs_noisy_thresholded)
zero_coeffs = sum(np.sum(c == 0) for c in coeffs_noisy_thresholded)

DCT:
t = np.arange(N) / fs
n = np.arange(N)
x_dct = dct(x, type=2, norm='ortho')
coeff_idx = np.arange(len(x_dct))
x_reconstructed = idct(x_dct, type=2, norm='ortho')
img_dct = dct(dct(img, axis=0, type=2, norm='ortho'), axis=1, type=2, norm='ortho')

square low-pass window (top-left corner only):
window_size = 64
window = np.zeros_like(img_dct)
window[:window_size, :window_size] = 1
img_dct_windowed = img_dct * window
img_reconstructed = idct(idct(img_dct_windowed, axis=0, type=2, norm='ortho'), axis=1, type=2, norm='ortho')

jpg window style:
rows, cols = img.shape
compressed_img = np.zeros_like(img)
for i in range(0, rows, block_size):
    for j in range(0, cols, block_size):
        block = img[i:i+block_size, j:j+block_size]
        block_dct = dct(dct(block.T, norm='ortho').T, norm='ortho')
        mask = np.zeros_like(block_dct)
        mask[:keep_size, :keep_size] = 1
        block_dct *= mask
        block_idct = idct(idct(block_dct.T, norm='ortho').T, norm='ortho')
        compressed_img[i:i+block_size, j:j+block_size] = block_idct

periodogram and windowing:
X = np.fft.fft(x)
psd_periodogram = (1/N) * np.abs(X)**2
f = np.fft.fftfreq(N, d=1/fs)
plt.plot(f[:N//2], psd_periodogram[:N//2])

window = np.hanning(N)
windowed_x = x * window
X_windowed = np.fft.fft(windowed_x)
psd_windowed = (1/N) * np.abs(X_windowed)**2
f = np.fft.fftfreq(N, d=1/fs)
plt.plot(f[:N//2], psd_periodogram[:N//2], label='Original Periodogram')
plt.plot(f[:N//2], psd_windowed[:N//2], label='Windowed Periodogram')

segments = np.split(x, 5)
psd_segments = []
for segment in segments:
    X_seg = np.fft.fft(segment, n=N)
    psd_seg = (1/N) * np.abs(X_seg)**2
    psd_segments.append(psd_seg)
psd_bartlett = np.mean(psd_segments, axis=0)
f = np.fft.fftfreq(N, d=1/fs)
plt.plot(f[:N//2], psd_periodogram[:N//2], label="Original Periodogram")
plt.plot(f[:N//2], psd_bartlett[:N//2], label="Bartlett Estimate")

f_scipy, psd_welch_scipy = signal.welch(x, fs=fs, window='hann', nperseg=256, noverlap=128, nfft=N, return_onesided=False)

Multitapers:
from scipy.signal.windows import dpss
tapers = dpss(N, NW, Kmax=K, return_ratios=False)
for i in range(K):
    plt.plot(tapers[i], label=f'Taper {i+1}')

psd_multitaper = []
for taper in tapers:
    tapered_x = x * taper
    X_tapered = np.fft.fft(tapered_x, n=N)
    psd = (1/N) * np.abs(X_tapered)**2
    psd_multitaper.append(psd)
psd_multitaper = np.mean(psd_multitaper, axis=0)
plt.plot(f[:N//2], psd_multitaper[:N//2], label='Multitaper Estimate')
plt.plot(f[:N//2], psd_welch[:N//2], label='Welch Estimate')
plt.plot(f[:N//2], psd_bartlett[:N//2], label='Bartlett Estimate')
