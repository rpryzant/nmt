

## EN tokenization 

Using moses:

```
git clone https://github.com/moses-smt/mosesdecoder.git

mosesdecoder/scripts/tokenizer/tokenizer.perl -l en -threads 3 < combined.en > combined.en.tok
```



## JP tokenization

Using KyTea:

```
wget http://www.phontron.com/kytea/download/kytea-0.4.7.tar.gz
tar -xzf kytea-0.4.7.tar.gz
cd kytea-0.4.7
./configure
make
make install
kytea --help # make sure installation went swimmingly
cd ..
rm kytea-0.4.7.tar.gz
rm -rf kytea-0.4.7
kytea -nounk -notags -wordbound ' ' -tagbound '' < corpus.ja.cleaned | sed 's/\\//g' >  corpus.ja.cleaned.tok
```
