import random
import langid
import pycountry
from utils import get_unicode
from abc import abstractmethod, ABC


class Model(ABC):
    def __init__(self, labels):
        self.labels = labels

    @abstractmethod
    def infer(self, text) -> str | None: ...

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
    def __init__(self, labels, unicode_languages, seed):
        super().__init__(labels)
        self.unicode_languages = (
            unicode_languages  # key: unicode value: set of languages
        )
        self.seed = seed

    def infer(self, text):
        possible_languages = set(self.labels)
        random.seed(self.seed)

        for char in text:
            aux_possible_languages_list = list(possible_languages)
            char_unicode = get_unicode(char)

            if char_unicode in self.unicode_languages:
                possible_languages = possible_languages.intersection(self.unicode_languages[char_unicode])

            if len(possible_languages) == 0:
                return random.choice(aux_possible_languages_list)

        return random.choice(list(possible_languages))
