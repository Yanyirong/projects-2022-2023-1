#!/usr/bin/python3

import re
import itertools
import functools
import math

from typing import List, Dict, Tuple
from typing import Iterator

import pickle as pkl

Sentence = List[str]
IntSentence = List[int]

Corpus = List[Sentence]
IntCorpus = List[IntSentence]

Gram = Tuple[int]

_splitor_pattern = re.compile(r"[^a-zA-Z']+|(?=')")
_digit_pattern = re.compile(r"\d+")
def normaltokenize(corpus: List[str]) -> Corpus:
    """
    Normalizes and tokenizes the sentences in `corpus`. Turns the letters into
    lower case and removes all the non-alphadigit characters and splits the
    sentence into words and added BOS and EOS marks.

    Args:
        corpus - list of str

    Return:
        list of list of str where each inner list of str represents the word
          sequence in a sentence from the original sentence list
    """

    tokeneds = [ ["<s>"]
               + list(
                   filter(lambda tkn: len(tkn)>0,
                       _splitor_pattern.split(
                           _digit_pattern.sub("N", stc.lower()))))
               + ["</s>"]
                    for stc in corpus
               ]
    return tokeneds

def extract_vocabulary(corpus: Corpus) -> Dict[str, int]:
    """
    Extracts the vocabulary from `corpus` and returns it as a mapping from the
    word to index. The words will be sorted by the codepoint value.

    Args:
        corpus - list of list of str

    Return:
        dict like {str: int}
    """

    vocabulary = set(itertools.chain.from_iterable(corpus))
    vocabulary = dict(
            map(lambda itm: (itm[1], itm[0]),
                enumerate(
                    sorted(vocabulary))))
    return vocabulary

def words_to_indices(vocabulary: Dict[str, int], sentence: Sentence) -> IntSentence:
    """
    Convert sentence in words to sentence in word indices.

    Args:
        vocabulary - dict like {str: int}
        sentence - list of str

    Return:
        list of int
    """

    return list(map(lambda tkn: vocabulary.get(tkn, len(vocabulary)), sentence))

# OPTIONAL: implement a DictTree instead of using a flattern dict
#class DictTree:
    #def __init__(self):
        #pass
#
    #def __len__(self) -> int:
        #pass
#
    #def __iter__(self) -> Iterator[Gram]:
        #pass
#
    #def __contains__(self, key: Gram):
        #pass
#
    #def __getitem__(self, key: Gram) -> int:
        #pass
#
    #def __setitem__(self, key: Gram, frequency: int):
        #pass
#
    #def __delitem__(self, key: Gram):
        #pass

class NGramModel:
    def __init__(self, vocab_size: int, n: int = 4):
        """
        Constructs `n`-gram model with a `vocab_size`-size vocabulary.

        Args:
            vocab_size - int
            n - int
        """

        self.vocab_size: int = vocab_size
        self.n: int = n

        self.frequencies: List[Dict[Gram, int]]\
            = [{} for _ in range(n)]
        self.disfrequencies: List[Dict[Gram, int]]\
            = [{} for _ in range(n)]
        #self.frequencies: DictTree = DictTree()

        self.ncounts: Dict[ Gram
                          , Dict[int, int]
                          ] = {}

        self.discount_threshold:int = 7
        self._d: Dict[Gram, Tuple[float, float]] = {}
        self._alpha: List[Dict[Gram, float]]\
            = [{} for _ in range(n)]

        self.eps = 1e-10

    def learn(self, corpus: IntCorpus):
        """
        Learns the parameters of the n-gram model.

        Args:
            corpus - list of list of int
        """

        for stc in corpus:
            for i in range(1, len(stc)+1):
                for j in range(min(i, self.n)):
                    # TODO: count the frequencies of the grams
                    gram = stc[i-j-1:i]
                    gram = tuple(gram)
                    if gram not in self.frequencies[j].keys():
                        self.frequencies[j][gram]=1
                    else:
                        self.frequencies[j][gram]=self.frequencies[j][gram]+1

        for i in range(1, self.n):
            grams = itertools.groupby(
                    sorted(
                        sorted(
                            map(lambda itm: (itm[0][:-1], itm[1]),
                                 self.frequencies[i].items()),
                               key=(lambda itm: itm[1])),
                        key=(lambda itm: itm[0])))
            # TODO: calculates the value of $N_r$
            for key, l in grams:
                gram = key[0]
                num = key[1]
                if gram not in self.ncounts.keys():
                    self.ncounts[gram]={}
                self.ncounts[gram][num]=len(list(l))

    def d(self, gram: Gram) -> float:
        """
        Calculates the interpolation coefficient.

        Args:
            gram - tuple of int

        Return:
            float
        """

        if gram not in self._d:
            # TODO: calculates the value of $d'$
            n = len(gram)-1
            theta = self.discount_threshold
            r = self.frequencies[n][gram]
            if self.frequencies[n][gram]>theta:
                self._d[gram]=1
            else:
                if 1 not in self.ncounts[gram[:-1]].keys():
                    self._d[gram]=1
                    return self._d[gram]
                if (theta+1) not in self.ncounts[gram[:-1]].keys():
                    self.ncounts[gram[:-1]][theta+1]=0
                if (r+1) not in self.ncounts[gram[:-1]].keys():
                    self.ncounts[gram[:-1]][r+1]=0    
                division = (self.ncounts[gram[:-1]][1]-(theta+1)*self.ncounts[gram[:-1]][theta+1])
                if division==0:
                    division = 1
                lamda = self.ncounts[gram[:-1]][1]/division
                self._d[gram] = lamda*(r+1)*self.ncounts[gram[:-1]][r+1]/(r*self.ncounts[gram[:-1]][r])+1-lamda
            self._d[gram] = max(self._d[gram],self.eps * 10)
        return self._d[gram]

    def alpha(self, gram: Gram) -> float:
        """
        Calculates the back-off weight alpha(`gram`)

        Args:
            gram - tuple of int

        Return:
            float
        """

        n = len(gram)
        if gram not in self._alpha[n]:
            if gram in self.frequencies[n-1]:
                # TODO: calculates the value of $\alpha$
                numerator = 1.
                denominator = 1.
                for key in self.frequencies[n].keys():
                    if key[:-1]==gram:
                        numerator = numerator - self[key]
                        denominator = denominator - self[key[1:]]
                self._alpha[n][gram] = numerator/denominator
            else:
                self._alpha[n][gram] = 1.
            self._alpha[n][gram] = max(self._alpha[n][gram],self.eps * 10)
        return self._alpha[n][gram]

    def __getitem__(self, gram: Gram) -> float:
        """
        Calculates smoothed conditional probability P(`gram[-1]`|`gram[:-1]`).

        Args:
            gram - tuple of int

        Return:
            float
        """

        n = len(gram)-1
        if gram not in self.disfrequencies[n]:
            if n>0:
                # TODO: calculates the smoothed probability value according to the formulae
                if gram in self.frequencies[n].keys():
                    self.disfrequencies[n][gram] = self.d(gram)*self.frequencies[n][gram]/self.frequencies[n-1][gram[:-1]]
                else:
                    self.disfrequencies[n][gram] = self.alpha(gram[:-1])*self[gram[1:]]
            else:
                self.disfrequencies[n][gram] = self.frequencies[n].get(gram, self.eps)/float(sum([item[1] for item in self.frequencies[0].items()]))
            if self.disfrequencies[n][gram]>1: 
                self.disfrequencies[n][gram] = 1
            if self.disfrequencies[n][gram]<0:
                self.disfrequencies[n][gram] = self.eps * 10
        return self.disfrequencies[n][gram]

    def log_prob(self, sentence: IntSentence) -> float:
        """
        Calculates the log probability of the given sentence. Assumes that the
        first token is always "<s>".

        Args:
            sentence: list of int

        Return:
            float
        """

        log_prob = 0.
        for i in range(2, len(sentence)+1):
            # TODO: calculates the log probability
            if i == 2:
                result = self[tuple([sentence[1]])]
            elif i==3:
                result = self[tuple(sentence[1:3])]
            elif i==4:
                result = self[tuple(sentence[1:4])]
            else:
                result = self[tuple(sentence[i-4:i])]
            result = max(result,self.eps * 10)
            log_prob = log_prob + math.log2(result)
        log_prob = log_prob/(len(sentence)-1)
        return log_prob

    def ppl(self, sentence: IntSentence) -> float:
        """
        Calculates the PPL of the given sentence. Assumes that the first token
        is always "<s>".

        Args:
            sentence: list of int

        Return:
            float
        """

        # TODO: calculates the PPL
        return 2**(-self.log_prob(sentence))

if __name__ == "__main__":
    action = "eval"

    if action=="train":
        with open("data/news.2007.en.shuffled.deduped.train", encoding="utf-8") as f:
            texts = list(map(lambda l: l.strip(), f.readlines()))

        print("Loaded training set.")

        corpus = normaltokenize(texts)
        vocabulary = extract_vocabulary(corpus)
        #print(vocabulary)
        corpus = list(
                map(functools.partial(words_to_indices, vocabulary),
                    corpus))

        print("Preprocessed training set.")

        model = NGramModel(len(vocabulary))
        model.learn(corpus)
        with open("model.pkl", "wb") as f:
            pkl.dump(vocabulary, f)
            pkl.dump(model, f)

        print("Dumped model.")

    elif action=="eval":
        with open("model.pkl", "rb") as f:
            vocabulary = pkl.load(f)
            model = pkl.load(f)
        print("Loaded model.")

        with open("data/news.2007.en.shuffled.deduped.test", encoding="utf-8") as f:
            test_set = list(map(lambda l: l.strip(), f.readlines()))
        test_corpus = normaltokenize(test_set)
        test_corpus = list(
                map(functools.partial(words_to_indices, vocabulary),
                    test_corpus))
        ppls = []
        for t in test_corpus:
            #print(t)
            ppls.append(model.ppl(t))
            print(ppls[-1])
        print("Avg: ", sum(ppls)/len(ppls))
