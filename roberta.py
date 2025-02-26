# from getData import get_train_data
from data import get_train_data
from deep_models import DeepModel

_, _, labels, _ = get_train_data()
model = DeepModel(labels)

# model.train(*get_train_data(removeNaNs=True))
print(model.infer("Hello, world!"))
