# Pytropic

[![Python package](https://github.com/willf/pytropic/actions/workflows/test.yml/badge.svg)](https://github.com/willf/pytropic/actions/workflows/test.yml)

<img alt="An python with a lot of entropy" src="https://user-images.githubusercontent.com/37049/192400489-7a2fdc49-b29a-4299-a1c6-97c8b97b2eaf.png" width=150>

Train and predict string entropy based on character n-grams

## Features

- Train a model on a corpus of text
- multiple n-gram sizes
- Can name models

## Example

```python
>>> from pytropic import pytropic

>>> en = pytropic.Model(name='English 3-gram', size=3)
>>> fr = pytropic.Model(name='French 3-gram', size=3)

>>> with open('./corpora/bible-english.txt') as f:
        en.train(f)
>>> with open('./corpora/bible-french.txt') as f:
        fr.train(f)

>>> t = {'en': en, 'fr': fr}

>>> min(t, key=lambda x: t[x].entropy("this is a test"))
Out: 'en'

>>> min(t, key=lambda x: t[x].entropy("c'est un test"))
Out: 'fr'
```
