import nltk
from textblob import TextBlob
falsenouns = ["start", "begin", "collect", "turn", "towards", "forwards", "navigate", "stop"]

def combinelists(directions, nounphrases):
    for v in range(len(directions)):
        for i in nounphrases:
            for f in nltk.word_tokenize(i):
                if f.lower() == directions[v][0].lower():
                    directions[v][0] = i

    directions = [[d, t] for [d, t] in directions if d not in falsenouns]

    finaldirections = []
    for i, direction in enumerate(directions):
        if i == 0 or direction[0] != directions[i - 1][0]:
            finaldirections.append(direction)
    return finaldirections

def map(pos):
    if (pos == "NN" or pos == "NNP" or pos == "NNS" or pos == "NNPS"):
        temp = "N"
    else:
        temp = "V"
    return temp

def extract(userinput):
    text = TextBlob(userinput)
    nounphrases = text.noun_phrases
    print(nounphrases)
    rawDirections = []
    text = nltk.word_tokenize(userinput)
    for word, pos in nltk.pos_tag(text):
        temp = word.lower()
        if (temp == "forwards" or temp == "forward" or temp == "proceed"):
                    rawDirections.append(["straight", "V"])
        elif (temp == "left"):
            rawDirections.append(["left", "V"])
        elif (word == "right"):
            rawDirections.append(["right", "V"])
        elif (pos == "NN" or pos == "NNP" or pos == "NNS" or pos == "NNPS"):
                rawDirections.append([word, map(pos)])
    print(rawDirections)
    # nouns = [(word, map(pos) for word, pos in nltk.pos_tag(text) if (pos == "NN" or pos == "NNP" or pos == "NNS" or pos == "NNPS" or pos == "VB" or pos == "VBG" or pos == "VBD" or pos == "VBN" or pos == "VBP" or pos == "VBZ")]
    # print(nouns)
    combined = combinelists(rawDirections, nounphrases)
    print(combined)
    return combined

extract("first, go to the tree, then turn right, then find the trash can, turn left, then stop by the pole")