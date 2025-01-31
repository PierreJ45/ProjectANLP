import random
import langid
import pycountry
from utils import get_unicode


class Model:
    def __init__(self, labels):
        self.labels = labels

    def infer(self, text): ...

    def validate(self, text_list, label_list):
        """
        returns correct, total
        """
        correct = 0
        total = 0
        for text, label in zip(text_list, label_list):
            if self.infer(text) == label:
                correct += 1
            total += 1
        return correct, total


class RandomModel(Model):
    def __init__(self, labels, seed):
        super().__init__(labels)
        self.seed = seed

    def infer(self, text):
        random.seed(self.seed)
        return random.choice(self.labels)


class ConstantModel(Model):
    def __init__(self, labels, constant_label):
        super().__init__(labels)
        self.constant_label = constant_label

    def infer(self, text):
        return self.constant_label


class LangidModel(Model):
    def __init__(self, labels):
        super().__init__(labels)

    def infer(self, text):
        two_letters_code = langid.classify(text)[0]

        language = pycountry.languages.get(alpha_2=two_letters_code)

        if language:
            return language.alpha_3

        return None


class StatModel(Model):
    def __init__(self, labels, language_unicodes, seed):
        super().__init__(labels)
        self.language_unicodes = language_unicodes
        self.seed = seed

    def infer(self, text):
        possible_languages = set()

        for char in text:
            char_unicode = get_unicode(char)
            if char_unicode in self.language_unicodes:
                possible_languages.add(self.language_unicodes[char_unicode])

        if len(possible_languages) == 1:
            print("a")
            return list(possible_languages)[0]

        if len(possible_languages) == 0:
            return "tgk"

        random.seed(self.seed)
        print("c")
        return random.choice(list(possible_languages))
