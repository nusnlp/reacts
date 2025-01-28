## CREST and REACTS
This repository provides the data and code to perform constrained timeline summarization, as reported in this paper:
> Just What You Desire: Constrained Timeline Summarization with Self-Reflection for Enhanced Relevance <br>
> [Muhammad Reza Qorib](https://mrqorib.github.io/), [Qisheng Hu](https://openreview.net/profile?id=~Qisheng_Hu1) and [Hwee Tou Ng](https://www.comp.nus.edu.sg/~nght/). <br>
> 2025. The 39th Annual AAAI Conference on Artificial Intelligence (to appear) [[PDF](https://arxiv.org/abs/2412.17408)].

Please cite our paper if you use its source code or data.

### Dataset
We provide the proposed CREST dataset under the folder `data`. The data is derived from the [ENTITIES](https://github.com/complementizer/news-tls) dataset. We do not own the copyright of the articles and the timeline summaries. Please contact the respective data owners for usage other than evaluating models on the constrained timeline summarization task.

### Code
We provide the code for our method, REACTS, under the folder `code`.

### Security Notice
Be cautious when using unknown LLM models. Check the HuggingFace [security guidelines](https://huggingface.co/docs/hub/en/security-malware) before loading a model from HuggingFace.

### License
This repository is licensed under the GNU General Public License Version 3 (see [License](./LICENSE.txt)).
