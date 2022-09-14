if __name__ == '__main__':
    # import nltk
    #
    # nltk.download('omw-1.4')

    import gensim
    from nltk.stem import PorterStemmer
    from nltk.stem import WordNetLemmatizer


    stemmer = PorterStemmer()
    lemmatize = WordNetLemmatizer()

    def lemmatize_stemming(text):
        return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


    def preprocess(text):
        result = []
        for token in gensim.utils.simple_preprocess(text):
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
                result.append(lemmatize_stemming(token))

        return result

    doc_list = [open("articol.txt").read()]
    doc_tokenized = [preprocess(doc) for doc in doc_list]

    # print(doc_tokenized)

    dictionary = gensim.corpora.Dictionary()

    BoW_corpus = [dictionary.doc2bow(doc, allow_update=True) for doc in doc_tokenized]

    # print(BoW_corpus)

    id_words = [[(dictionary[ID], count) for ID, count in line] for line in BoW_corpus]

    # print(id_words)

    lda_model = gensim.models.LdaMulticore(BoW_corpus,
                                           num_topics=1,
                                           id2word=dictionary,
                                           passes=10,
                                           workers=2)

    topics = lda_model.show_topics()

    # print(topics)

    print("\n".join(f" Topic ID {topics[i][0]} : {topics[i][-1]}" for i in range(len(topics))))