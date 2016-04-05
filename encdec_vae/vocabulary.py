EOS = u'<EOS>'
UNK = u'<UNK>'


class Vocab(object):
    """Mapping between words and IDs."""

    def __init__(self):
        self.i2w = []
        self.w2i = {}
        self.add_word(EOS)
        self.add_word(UNK)
        self.eos_id = self.get_id(EOS)
        self.unk_id = self.get_id(UNK)

    def add_word(self, word):
        assert isinstance(word, unicode)
        if word not in self.w2i:
            new_id = self.size()
            self.i2w.append(word)
            self.w2i[word] = new_id

    def get_id(self, word):
        assert isinstance(word, unicode)
        return self.w2i.get(word)

    def get_word(self, w_id):
        return self.i2w[w_id]

    def convert_ids(self, words):
        ids = map(self.get_id, words)
        for i in range(len(ids)):
            if ids[i] is None:
                ids[i] = self.unk_id
        return ids

    def convert_words(self, ids):
        words = map(self.get_word, ids)
        return words

    def size(self):
        return len(self.i2w)

    def save(self, path):
        with open(path, 'w') as f:
            for i, w in enumerate(self.i2w):
                print >> f, str(i) + '\t' + w.encode('utf-8')

    @classmethod
    def load(cls, path, size=-1):
        vocab = Vocab()
        with open(path) as f:
            for line in f:
                if size >= 0 and vocab.size() >= size:
                    break
                w = line.strip().split('\t')[1].decode('utf-8')
                vocab.add_word(w)
        return vocab

