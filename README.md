# AcousticSpace

AcousticSpace is a deepfake / synthetic-speech detector that flags spoofed audio using room acoustics and breathing-pattern artifacts rather than pure spectral cues. The idea: synthesized or replayed speech rarely reproduces a physically consistent room impulse response or natural breathing behavior, so a multi-modal model trained on ASVspoof-style data can catch what waveform-only detectors miss.
