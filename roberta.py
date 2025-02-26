
from getData import get_train_data
from models import DeepModel
from utils import LABELS


model = DeepModel(LABELS)

model.train(*get_train_data(removeNaNs=True))
model.generate_submission()