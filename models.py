import random
import langid
import pycountry

from abc import abstractmethod, ABC

from getData import get_test_data
from utils import get_unicode, Unicode, Language


class Model(ABC):
    def __init__(self, labels: list[Language]):
        self.labels = labels

    @abstractmethod
    def infer(self, text: str) -> Language: ...

    def validate(
        self, text_list: list[str], label_list: list[Language]
    ) -> tuple[int, int]:
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

    def generate_submission(self, path: str = "submission.csv") -> None:
        test_df = get_test_data()
        test_df["Label"] = test_df["Text"].apply(self.infer)
        test_df.drop(columns=["Usage", "Text"], inplace=True)
        test_df.index += 1
        test_df.to_csv(path, index_label="ID")


class RandomModel(Model):
    def __init__(self, labels: list[Language], seed: int):
        super().__init__(labels)
        self.seed = seed

    def infer(self, text: str) -> Language:
        random.seed(self.seed)
        return random.choice(self.labels)


class ConstantModel(Model):
    def __init__(self, labels: list[Language], constant_label: Language):
        super().__init__(labels)
        self.constant_label = constant_label

    def infer(self, text: str) -> Language:
        return self.constant_label


class LangidModel(Model):
    def __init__(self, labels: list[Language]):
        super().__init__(labels)

    def infer(self, text: str) -> Language:
        two_letters_code = langid.classify(text)[0]

        language = pycountry.languages.get(alpha_2=two_letters_code)

        if language:
            return language.alpha_3

        return "None"


class StatModel(Model):
    def __init__(
        self,
        labels: list[Language],
        unicode_languages: dict[Unicode, set[Language]],
        seed: int,
    ):
        super().__init__(labels)
        self.unicode_languages = (
            unicode_languages  # key: unicode value: set of languages
        )
        self.seed = seed

    def infer(self, text: str) -> Language:
        possible_languages = set(self.labels)
        random.seed(self.seed)

        for char in text:
            aux_possible_languages_list = list(possible_languages)
            char_unicode = get_unicode(char)

            if char_unicode in self.unicode_languages:
                possible_languages = possible_languages.intersection(
                    self.unicode_languages[char_unicode]
                )

            if len(possible_languages) == 0:
                return random.choice(aux_possible_languages_list)

        return random.choice(list(possible_languages))


class StatModelOnlyIf1Language(Model):
    def __init__(
        self, labels: list[Language], unicode_languages: dict[Unicode, set[Language]]
    ):
        super().__init__(labels)
        self.unicode_languages = (
            unicode_languages  # key: unicode value: set of languages
        )

    def infer(self, text: str) -> Language:
        possible_languages = set(self.labels)

        for char in text:
            char_unicode = get_unicode(char)

            if char_unicode in self.unicode_languages:
                possible_languages = possible_languages.intersection(
                    self.unicode_languages[char_unicode]
                )

        if len(possible_languages) == 1:
            return possible_languages.pop()
        else:
            return "None"
