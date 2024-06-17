import pytest

from fusion.authentication import FusionCredentials
from fusion.fusion import Fusion


@pytest.fixture()
def creds() -> FusionCredentials:
    return FusionCredentials.from_dict(
        {
            "resource": "JPMC:URI:RS-93742-Fusion-PROD",
            "application_name": "fusion",
            "root_url": "https://fusion.jpmorgan.com/api/v1/",
            "auth_url": "https://authe.jpmorgan.com/as/token.oauth2",
        }
    )


@pytest.fixture()
def client(creds: FusionCredentials) -> Fusion:
    client = Fusion(credentials=creds)
    return client
