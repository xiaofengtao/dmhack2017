import gensim
from nltk.corpus import stopwords

workpath = "c:\\users\\maschram\\downloads\\"
usephrase = False
outsize=200
minimum=10
workers=4
txtoutput = True

class queryiterator(object):
    def __init__ (self, filename):
        self.filename = filename
        self.stop = set(stopwords.words('english'))

    def __iter__(self):
        with open(self.filename, encoding="utf8") as modelfile:
            for line in modelfile:
                for phrase in line.split("\t")[0].split("|"):
                    keywords = phrase.split(" ")
                    keywords = [k for k in keywords if k not in self.stop]
                    yield keywords


sourcedata = queryiterator(workpath+"usen_queries_20170619_v2.tsv")


if usephrase == False:
    model = gensim.models.Word2Vec(sourcedata, min_count=minimum, workers=workers, size=outsize)
else:
    phrasebin = gensim.models.Phrases(sourcedata)
    model = gensim.models.Word2Vec(phrasebin[sourcedata], min_count=minimum, workers=workers, size=outsize)

model.wv.save_word2vec_format(workpath+"testmodel.bin",binary=True)
if txtoutput==True:
    model.wv.save_word2vec_format(workpath+"testmodel.txt",binary=False)



