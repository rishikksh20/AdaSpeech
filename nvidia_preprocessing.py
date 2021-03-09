import os
import glob
import tqdm
import torch
import argparse
import numpy as np
from utils.stft import TacotronSTFT
from utils.util import read_wav_np
from dataset.audio_processing import pitch
from utils.hparams import HParam
import torch.nn.functional as F
from utils.util import str_to_int_list

def _average_mel_by_duration(x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    #print(d.sum(), len(x))
    if d.sum() != x.shape[-1]:
        d[-1] += 1
    d_cumsum = F.pad(d.cumsum(dim=0), (1, 0))
    x_avg = [
            x[:, int(start):int(end)].sum(dim=1)//(end - start) if len(x[:, int(start):int(end)]) != 0 else x.zeros()
            for start, end in zip(d_cumsum[:-1], d_cumsum[1:])
        ]
    return torch.stack(x_avg)

def preprocess(data_path, hp, file):
    stft = TacotronSTFT(
        filter_length=hp.audio.n_fft,
        hop_length=hp.audio.hop_length,
        win_length=hp.audio.win_length,
        n_mel_channels=hp.audio.n_mels,
        sampling_rate=hp.audio.sample_rate,
        mel_fmin=hp.audio.fmin,
        mel_fmax=hp.audio.fmax,
    )


    mel_path = os.path.join(hp.data.data_dir, "mels")
    energy_path = os.path.join(hp.data.data_dir, "energy")
    pitch_path = os.path.join(hp.data.data_dir, "pitch")
    avg_mel_phon = os.path.join(hp.data.data_dir, "avg_mel_ph")

    os.makedirs(mel_path, exist_ok=True)
    os.makedirs(energy_path, exist_ok=True)
    os.makedirs(pitch_path, exist_ok=True)
    os.makedirs(avg_mel_phon, exist_ok=True)
    print("Sample Rate : ", hp.audio.sample_rate)

    with open("{}".format(file), encoding="utf-8") as f:
        _metadata = [line.strip().split("|") for line in f]
    for metadata in tqdm.tqdm(_metadata, desc="preprocess wav to mel"):
        wavpath = os.path.join(data_path, metadata[4])
        sr, wav = read_wav_np(wavpath, hp.audio.sample_rate)

        dur = str_to_int_list(metadata[2])
        dur = torch.from_numpy(np.array(dur))

        p = pitch(wav, hp)  # [T, ] T = Number of frames
        wav = torch.from_numpy(wav).unsqueeze(0)
        mel, mag = stft.mel_spectrogram(wav)  # mel [1, 80, T]  mag [1, num_mag, T]
        mel = mel.squeeze(0)  # [num_mel, T]
        mag = mag.squeeze(0)  # [num_mag, T]
        e = torch.norm(mag, dim=0)  # [T, ]
        p = p[: mel.shape[1]]

        avg_mel_ph = _average_mel_by_duration(mel, dur)  # [num_mel, L]
        assert (avg_mel_ph.shape[0] == dur.shape[-1])

        id = os.path.basename(wavpath).split(".")[0]
        np.save("{}/{}.npy".format(mel_path, id), mel.numpy(), allow_pickle=False)
        np.save("{}/{}.npy".format(energy_path, id), e.numpy(), allow_pickle=False)
        np.save("{}/{}.npy".format(pitch_path, id), p, allow_pickle=False)
        np.save("{}/{}.npy".format(avg_mel_phon, id), avg_mel_ph.numpy(), allow_pickle=False)

def main(args, hp):
    print("Preprocess Training dataset :")
    preprocess(args.data_path, hp, hp.data.train_filelist)
    print("Preprocess Validation dataset :")
    preprocess(args.data_path, hp, hp.data.valid_filelist)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data_path", type=str, required=True, help="root directory of wav files"
    )
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="yaml file for configuration"
    )
    args = parser.parse_args()

    hp = HParam(args.config)

    main(args, hp)
