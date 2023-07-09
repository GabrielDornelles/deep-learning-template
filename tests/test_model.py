import pytest
import torch
from models.resnet import ResNet18


@pytest.fixture(scope='session', name='model') # scope default is set to 'function', can also be 'module'
def setup_model():
    # Initialize your model once with this fixture, re-use it as you want
    model = ResNet18(num_classes=10)
    return model

@pytest.mark.parametrize("resolution", [(180, 50), (224, 224), (400, 400), (400, 100)])
def test_model_resolutions(resolution, model):
    dummy_input = torch.randn(1, 3, resolution[1], resolution[0]) # N C H W
    output = model(dummy_input)
    assert list(output.shape) == [1, model.num_classes]

# @pytest.fixture
# def setup_database():
#     db = connect_to_database()
#     # Create test data
#     create_test_data(db)
#     yield db  # provide the fixture value
#     # Tear down database connection
#     db.close()
