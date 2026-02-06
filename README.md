# A Partition Cover Approach to Tokenization

---

[NeurIPS 2025](https://neurips.cc/virtual/2025/poster/115918) | [arXiv](https://arxiv.org/abs/2501.06246)

---

In this work, we formulate tokenization as an optimization objective, show that it is NP-hard via a simple reduction from vertex cover, and propose a polynomial-time greedy algorithm **GreedTok**.
Our formulation naturally relaxes to the well-studied weighted maximum coverage problem which has a simple $(1 - 1/e)$-approximation greedy algorithm.

| Dataset       | $k$ | #uniq words | #candidates | Time         | Gain from lazy updates |
| ------------- | --- | ----------: | ----------: | ------------ | :--------------------: |
| UN            | 5K  |     105,505 |     884,630 | ~6 seconds   |          x23           |
| arXiv         | 5K  |     881,233 |   7,625,530 | ~63 seconds  |          x26           |
| Wiki          | 10K |   8,769,943 |  93,243,449 | ~11 minutes  |          x68           |
| PubMed        | 10K |   6,527,614 |  97,870,366 | ~11 minutes  |          x133          |
| Wiki-chinese  | 10K |   7,035,544 |  69,728,860 | ~8.5 minutes |          x90           |
| Wiki-japanese | 10K |   2,737,555 |  60,410,961 | ~8.5 minutes |          x74           |
| Wiki-korean   | 10K |   5,459,833 | 130,927,124 | ~18 minutes  |          x86           |

Table results shows time to solve (obtain a $k$-sized token set) from word counts. Since most of the compute is front-heavy, solving for larger $k$ size is trivial.
For detailed logs, compare cpp_logs/{\$data}/{\$data}.log versus cpp_logs/{\$data}/{\$data}\_fast.log.

## Huggingface AutoTokenizer interface

Install the v0.14.x version (for transformers >= 4):

```bash
wget "https://github.com/ncduy0303/pcatt/archive/refs/tags/v0.14.2.zip"
unzip v0.14.2.zip
cd pcatt-0.14.2
pip install -r requirements.txt
pip install .
```

For "training" either:

```python
from pcatt.hf.greedtok import GreedTok
greedtok = GreedTok().train_new_from_iterator(word_iterator, 100, max_token_length=5, min_word_count=1)
```

or

```python
from pcatt.hf.greedtok import GreedTok
greedtok = GreedTok().train_new_from_counts(word_count_dict, 100, max_token_length=5, min_word_count=1)
```

To use either:

```python
from pcatt.hf.greedtok import GreedTok
greedtok = GreedTok.from_pretrained(greedtok_file_directory)
```

or

```python
import pcatt.hf
greedtok = AutoTokenizer.from_pretrained("greedtok_file_directory")
```

### Citation

```bibtex
@inproceedings{lim2025partitioncoverapproachtokenization,
      title={A Partition Cover Approach to Tokenization},
      author={Jia Peng Lim and Shawn Tan and Davin Choo and Hady W. Lauw},
      year={2025}
      booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
}
```
