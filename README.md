# deep-learning-template

This project is mean to be used as a boilerplate. Re-use parts of it as you want. Feedback and upgrades are appreciated.

## Docker

Docker is a great tool, but here it's not meant to be used as a development tool. That's because everytime you make changes to your code you would need to rebuild the container, which takes sometime because of re-installing requirements.txt. Although there is some workarounds to solve this, I didn't implement any here, so keep in mind to develop with `virtualenv` and use `Docker` for your initial/final tests.

Build and run it:

```sh
docker build -t my_app_alias . 
docker run -it my_app_alias 
```

This will create a container named `my_app_alias` using the Dockerfile in the current directory.
## Tests

Automate your writting and create powerful tests with `Pytest`. In this repository we are not even scratching the surface of Pytest's potential, but we implement some basic automation:
```py
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
```

In the above code we set a pytest `Fixture` named `setup_model`, and change its default scope for session (default is function, which means everytime the fixture is called it runs the function), that way we only run the fixture at the starting of the session and keep it for whatever function uses it. Pytest is mostly used with fixtures setting up some data, coming from a DB or a known dataset, as we can yield data from those and apply teardown right after automatically:

```py
@pytest.fixture
def setup_database():
    db = connect_to_database()
    # Create test data
    create_test_data(db)
    yield db  # provide the fixture value
    # Tear down database connection
    db.close()

def test_function(setup_database):
    assert setup_database == 5 # assert yielded data from db is 5
```

We also create a test for model input resolution, and use `pytest.mark.parametrize` to reduce some copying and paste of code, automating the inputs for the function.

To run tests we simply run:

```sh
pytest
```

Pytest will run every file in the directory `tests` with the prefix `test`, inside those files, it runs every function with the prefix `test`.


Additionally, running `pytest -s` makes your print statements visible in the terminal.


## Pre-commit

Our pre-commit file applies a very simple code stylization, `flake8` and `black`.

TODO: write about it.


## CI

We implemented a very basic Continuous Integration pipeline that has 2 steps, `Lint` and `Test`. The `.github/worksflows/test.yml` ensures that the whenever there is a push or a pull request to the main branch, we check for code style (lint) and run the tests.

TODO: write about it.


