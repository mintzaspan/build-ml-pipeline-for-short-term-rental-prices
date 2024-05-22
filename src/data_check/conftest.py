import pytest
import pandas as pd
import wandb
import yaml


# Load config file
with open("../../config.yaml", "r") as f:
    config = yaml.safe_load(f)

# start wandb run
run = wandb.init(
    project=config["main"]["project_name"],
    group=config["main"]["experiment_name"],
    job_type='data_check')


def pytest_addoption(parser):
    parser.addoption("--csv", action="store")
    parser.addoption("--ref", action="store")
    parser.addoption("--kl_threshold", action="store")
    parser.addoption("--min_price", action="store")
    parser.addoption("--max_price", action="store")
    parser.addoption("--min_nights", action="store")
    parser.addoption("--max_nights", action="store")


@pytest.fixture(scope='session')
def data(request):
    run = wandb.init(job_type="data_tests", resume=True)

    # Download input artifact. This will also note that this script is using this
    # particular version of the artifact
    data_path = run.use_artifact(request.config.option.csv).file()

    if data_path is None:
        pytest.fail("You must provide the --csv option on the command line")

    df = pd.read_csv(data_path)

    return df


@pytest.fixture(scope='session')
def ref_data(request):
    run = wandb.init(job_type="data_tests", resume=True)

    # Download input artifact. This will also note that this script is using this
    # particular version of the artifact
    data_path = run.use_artifact(request.config.option.ref).file()

    if data_path is None:
        pytest.fail("You must provide the --ref option on the command line")

    df = pd.read_csv(data_path)

    return df


@pytest.fixture(scope='session')
def kl_threshold(request):
    kl_threshold = request.config.option.kl_threshold

    if kl_threshold is None:
        pytest.fail("You must provide a threshold for the KL test")

    return float(kl_threshold)


@pytest.fixture(scope='session')
def min_price(request):
    min_price = request.config.option.min_price

    if min_price is None:
        pytest.fail("You must provide min_price")

    return float(min_price)


@pytest.fixture(scope='session')
def max_price(request):
    max_price = request.config.option.max_price

    if max_price is None:
        pytest.fail("You must provide max_price")

    return float(max_price)


@pytest.fixture(scope='session')
def min_nights(request):
    min_nights = request.config.option.min_nights

    if min_nights is None:
        pytest.fail("You must provide min_nights")

    return float(min_nights)


@pytest.fixture(scope='session')
def max_nights(request):
    max_nights = request.config.option.max_nights

    if max_nights is None:
        pytest.fail("You must provide max_nights")

    return float(max_nights)
