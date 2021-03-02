# AdaSpeech: Adaptive Text to Speech for Custom Voice [WIP]
Unofficial Pytorch implementation of [AdaSpeech](https://arxiv.org/pdf/2103.00993.pdf).

## Note:
* I am not considering multi-speaker use case, Iam much more focus only on single speaker.
* I will use only `Utterance level encoder` and `Phoneme level encoder` not condition layer norm (which is the soul of AdaSpeech paper), it definelty restrict the adaptive nature of AdaSpeech but my focus is to improve FastSpeech 2 acoustic generalization rather than adaptation.


# Citations
```bibtex
@misc{chen2021adaspeech,
      title={AdaSpeech: Adaptive Text to Speech for Custom Voice}, 
      author={Mingjian Chen and Xu Tan and Bohan Li and Yanqing Liu and Tao Qin and Sheng Zhao and Tie-Yan Liu},
      year={2021},
      eprint={2103.00993},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}
```
