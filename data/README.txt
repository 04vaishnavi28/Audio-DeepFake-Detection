You may download the datasets used in the project from the following URLs:





(Real) Human Voice Dataset: LJ Speech (v1.1)

https://keithito.com/LJ-Speech-Dataset/

This dataset consists of 13,100 short audio clips of a single speaker reading passages from 7 non-fiction books.








(Fake) Synthetic Voice Dataset: WaveFake (v1.20)

https://zenodo.org/records/5642694

The dataset consists of 104,885 generated audio clips (16-bit PCM wav).









After downloading the datasets, you may extract them under data/real and data/fake respectively. In the end, the data directory should look like this:

data
├── real
│   └── wavs
└── fake
    ├── common_voices_prompts_from_conformer_fastspeech2_pwg_ljspeech
    ├── jsut_multi_band_melgan
    ├── jsut_parallel_wavegan
    ├── ljspeech_full_band_melgan
    ├── ljspeech_hifiGAN
    ├── ljspeech_melgan
    ├── ljspeech_melgan_large
    ├── ljspeech_multi_band_melgan
    ├── ljspeech_parallel_wavegan
    └── ljspeech_waveglow