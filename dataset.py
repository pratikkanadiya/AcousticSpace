import os
import json
from matplotlib.pylab import sample
import torch
import numpy as np
from torch.utils.data import Dataset


class ASVspoofDataset(Dataset):

    def __init__(self, protocol_path, preprocessed_dir, acoustic_stats=None):

        self.samples = []

        label_map = {}

        with open(protocol_path, "r") as f:
            for line in f:
                parts = line.strip().split()

                if len(parts) >= 5:
                    base_name = parts[1]
                    label = 0 if parts[4] == "bonafide" else 1
                    label_map[base_name] = label

        for file in os.listdir(preprocessed_dir):

            if not file.endswith("_analysis.json"):
                continue

            json_path = os.path.join(preprocessed_dir, file)

            with open(json_path, "r") as f:
                report = json.load(f)

            base_name = report["metadata"]["filename"].split(".")[0]

            if base_name not in label_map:
                continue

            label = label_map[base_name]

            for seg in report["segments"]:

                self.samples.append({

                    "spectrogram": seg["spectrogram_path"],

                    "room_descriptor": seg["room_descriptor_path"],

                    "acoustic_features": seg["acoustic_features"],

                    "label": label

                })

        if acoustic_stats is not None:
            self.acoustic_mean, self.acoustic_std = acoustic_stats
        else:
            if len(self.samples) == 0:
                raise ValueError(
                    f"No samples found for protocol '{protocol_path}' in "
                    f"'{preprocessed_dir}' -- cannot compute acoustic stats."
                )
            all_acoustic = np.stack(
                [
                    np.asarray(
                        s["acoustic_features"],
                        dtype=np.float32
                    )
                    for s in self.samples
                ],
                axis=0
            )


            if not np.isfinite(all_acoustic).all():

                print(
                    "[WARNING] Invalid acoustic values detected "
                    "in dataset."
                )

                print(
                    "Invalid values:",
                    np.sum(~np.isfinite(all_acoustic))
                )

                all_acoustic = np.nan_to_num(
                    all_acoustic,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0
                )


            self.acoustic_mean = np.mean(
                all_acoustic,
                axis=0
            )

            self.acoustic_std = np.std(
                all_acoustic,
                axis=0
            )

            # Prevent zero standard deviation
            self.acoustic_std = np.maximum(
                self.acoustic_std,
                1e-8
            )

            # Final safety check
            if not np.isfinite(self.acoustic_mean).all():
                raise ValueError(
                    "acoustic_mean still contains NaN or Inf."
                )

            if not np.isfinite(self.acoustic_std).all():
                raise ValueError(
                    "acoustic_std still contains NaN or Inf."
                )

            print("Acoustic Mean:")
            print(self.acoustic_mean)

            print("Acoustic Std:")
            print(self.acoustic_std)

    def get_acoustic_stats(self):
        
        return self.acoustic_mean, self.acoustic_std

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]

        spec = torch.from_numpy(
            np.load(sample["spectrogram"])
        ).float()

        room = torch.from_numpy(
            np.load(sample["room_descriptor"])
        ).float()

        acoustic = torch.tensor(
            sample["acoustic_features"],
            dtype=torch.float32
        )

        spec = (spec - spec.mean()) / (spec.std() + 1e-8)

        room = (room - room.mean()) / (room.std() + 1e-8)

        self.acoustic_mean = torch.tensor(
            self.acoustic_mean,
            dtype=torch.float32
        )

        self.acoustic_std = torch.tensor(
            self.acoustic_std,
            dtype=torch.float32
        )
        
        acoustic = (
            acoustic - self.acoustic_mean
        ) / (
            self.acoustic_std + 1e-8
        )

        spec = torch.from_numpy(spec).unsqueeze(0)

        room = torch.from_numpy(room)

        acoustic = torch.from_numpy(acoustic)

        label = torch.tensor(
            sample["label"],
            dtype=torch.long
        )

        return spec, room, acoustic, label