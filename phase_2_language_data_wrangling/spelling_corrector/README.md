## Spellchecker

This is a spellchecker that we might want to use. It's a little smarter than other python spellcheckers I could find on the web. Those just do edit distance and spit out the closest word. This does dfs through the space of possible reconstructions, with a combined bigram+unigram language model. 

Note that it only corrects words within edit distance 1. 

## Usage

Make a `reconstructor` object, then use it however you want!

```
r = Reconstructor()
r.reconstruct("ehllo my name is Jack")     # ===> hello my name is Jack
```






