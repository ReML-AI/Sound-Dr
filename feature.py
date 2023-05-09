import os
import numpy as np

import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
import tensorflow_hub as hub

import librosa


frill_nofrontend_model = hub.load('https://tfhub.dev/google/nonsemantic-speech-benchmark/frill-nofrontend/1')

def stabilized_log(data, additive_offset, floor):
  """TF version of mfcc_mel.StabilizedLog."""
  return tf.math.log(tf.math.maximum(data, floor) + additive_offset)


def log_mel_spectrogram(data,
                        audio_sample_rate,
                        num_mel_bins=64,
                        log_additive_offset=0.001,
                        log_floor=1e-12,
                        window_length_secs=0.025,
                        hop_length_secs=0.010,
                        fft_length=None):
    """TF version of mfcc_mel.LogMelSpectrogram."""
    window_length_samples = int(round(audio_sample_rate * window_length_secs))
    hop_length_samples = int(round(audio_sample_rate * hop_length_secs))
    if not fft_length:
        fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))

    spectrogram = tf.abs(
        tf.signal.stft(
            tf.cast(data, tf.dtypes.float64),
            frame_length=window_length_samples,
            frame_step=hop_length_samples,
            fft_length=fft_length,
            window_fn=tf.signal.hann_window,
        )
    )

    to_mel = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=num_mel_bins,
        num_spectrogram_bins=fft_length // 2 + 1,
        sample_rate=audio_sample_rate,
        lower_edge_hertz=125.0,
        upper_edge_hertz=7500.0,
        dtype=tf.dtypes.float64
    )

    mel = spectrogram @ to_mel
    log_mel = stabilized_log(mel, log_additive_offset, log_floor)
    return log_mel

def compute_frontend_features(samples, sr, frame_hop, n_required=16000, num_mel_bins=64, frame_width=96):
    if samples.dtype == np.int16:
        samples = tf.cast(samples, np.float32) / np.iinfo(np.int16).max
    if samples.dtype == np.float64:
        samples = tf.cast(samples, np.float32)
    assert samples.dtype == np.float32, samples.dtype
    n = tf.size(samples)
    samples = tf.cond(
        n < n_required,
        lambda: tf.pad(samples, [(0, n_required - n)]),
        lambda: samples
    )
    mel = log_mel_spectrogram(samples, sr, num_mel_bins=num_mel_bins)
    mel = tf.signal.frame(mel, frame_length=frame_width, frame_step=frame_hop, axis=0)
    return mel

def make_nonsemantic_frill_nofrontend_feat(filename):
    waveform, _ = librosa.load(filename, sr=16000, mono=True, res_type="kaiser_fast")
    frontend_feats = tf.expand_dims(compute_frontend_features(waveform, 16000, frame_hop=17), axis=-1).numpy().astype(np.float32)
    assert frontend_feats.shape[1:] == (96, 64, 1)

    embeddings = frill_nofrontend_model(frontend_feats)['embedding']
    mean_emb = embeddings.numpy().mean(axis=0)
    std_emb = embeddings.numpy().std(axis=0)
    return np.concatenate((mean_emb, std_emb))

# # Extract Features
def get_features_of_list_audio(path, X):
    X_trill_features = []
    #extract train data features
    for index, row in X.iterrows():
        #get cough audio path
        cough_path = os.path.join(path, row['file_path'])
        X_trill_features.append(make_nonsemantic_frill_nofrontend_feat(cough_path))
        
    return np.array(X_trill_features)