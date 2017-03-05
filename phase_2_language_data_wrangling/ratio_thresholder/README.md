## Ratio Thresholder


As per [this paper](https://wit3.fbk.eu/papers/WIT3-EAMT2012.pdf), the `thresholder.py` script filters out innapropriate alignments by discarding pairs whose length ratio is an outlier (assuming normal distribution and 95% ci).


## Usage

```
python thresholder.py [corpus file 1] [corpus file 2]
```

The above command will generate two files, both with their original filename and suffixed with "cleaned".

## Preconditions 

This script assumes that each corpus file is a raw list of sentences, 1 per line, and that the sentences in matching lines are themselves matched.

