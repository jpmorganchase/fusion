import os

import pytest

from fusion._fusion import FusionCredentials
from fusion.fusion import Fusion


@pytest.fixture()
def creds() -> FusionCredentials:
    return FusionCredentials.from_client_id(
        client_id=os.getenv("FUSION_CLIENT_ID"),
        client_secret=os.getenv("FUSION_CLIENT_SECRET"),
        resource="JPMC:URI:RS-93742-Fusion-PROD",
        auth_url="https://authe.jpmorgan.com/as/token.oauth2",
        proxies={},
        fusion_e2e=None,
    )


@pytest.fixture()
def client(creds: FusionCredentials) -> Fusion:
    client = Fusion(credentials=creds)
    return client
