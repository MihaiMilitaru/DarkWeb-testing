if __name__ == '__main__':
    from transformers import pipeline

    classifier = pipeline("text-classification", model='bhadresh-savani/distilbert-base-uncased-emotion', top_k=3)
    prediction = classifier(open("articol.txt").read()[:540])
    print(prediction)