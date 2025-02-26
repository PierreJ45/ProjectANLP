# from getData import get_train_data
from deep_models import DeepModel
from utils import LABELS


model = DeepModel(LABELS)

# model.train(*get_train_data(removeNaNs=True))
print(model.infer("Hello, world!"))
