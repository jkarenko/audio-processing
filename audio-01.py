import torch
import torchaudio


def load_wav(path):
    waveform, sample_rate = torchaudio.load(path)
    return waveform, sample_rate


def to_midside(waveform):  # sum + difference
    left = waveform[0]
    right = waveform[1]
    sum = (left + right) / 2
    difference = (left - right) / 2
    waveform_midside = torch.stack((sum, difference))
    return waveform_midside


def reverse_channel_polarity(waveform):  # flip left channel polarity
    waveform_left, waveform_right = -waveform[0], waveform[1]
    result = torch.stack((waveform_left, waveform_right))
    return result


def slice_and_split(waveform):
    result = torch.zeros(2, waveform.shape[1])
    for i in range(waveform.shape[1]):
        result[i % 2][i] = waveform[0][i]
    return result


def save_wav(waveform, sample_rate, path):
    torchaudio.save(filepath=path, src=waveform, sample_rate=sample_rate)


def mix_to_mono(waveform):
    mono = (waveform.split(1)[0] + waveform.split(1)[1]) / 2
    return mono


def shift_left(waveform, shift=500):
    if waveform.shape[0] == 1:  # make sure audio is stereo
        waveform = mono_to_stereo(waveform)
    waveform_left, waveform_right = waveform[0], waveform[1]
    waveform_left = torch.roll(waveform_left, shift)
    result = torch.stack((waveform_left, waveform_right))
    return result


def mono_to_stereo(waveform):  # TODO: refactor
    waveform_stereo = torch.zeros(2, waveform.shape[1])
    for i in range(waveform.shape[1]):
        waveform_stereo[0][i] = waveform[0][i]
        waveform_stereo[1][i] = waveform[0][i]
    return waveform_stereo


def phase_shift(waveform, shift=1):  # TODO: implement
    waveform_left, waveform_right = waveform[0, :], waveform[1, :]
    for i in range(waveform.shape[1]):
        waveform_left[i] = torch.sin(waveform_left[i] * shift)
    result = torch.stack((waveform_left, waveform_right))
    return result


def operate(waveform, fn):
    for i in range(len(fn)):
        waveform = fn[i](waveform)
    return waveform


WAV_IN = "./speech-sample.wav"
WAV_OUT = "./speech-sample-processed.wav"

if __name__ == "__main__":
    operations = [
        mono_to_stereo,
        reverse_channel_polarity,
    ]
    waveform, sample_rate = load_wav(WAV_IN)
    waveform = operate(waveform, operations)
    save_wav(waveform, sample_rate, WAV_OUT)
