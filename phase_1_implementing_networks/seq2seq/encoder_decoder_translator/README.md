## seq2seq Translator

This is an implementation of [Cho et al., 2014](https://arxiv.org/abs/1406.1078).

## Usage

```
python main.py data/processed/ en vi tmp/checkpoint.ckpt-103 load
```

## Files

* `main.py`: logic for training models
* `dataset.py`: logic for parsing billingual corpuses and batching
* `model.py`: the seq2seq model 