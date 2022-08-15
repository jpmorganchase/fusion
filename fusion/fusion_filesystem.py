from fsspec.implementations.http import HTTPFileSystem
from fsspec.callbacks import _DEFAULT_CALLBACK
import logging


logger = logging.getLogger(__name__)


async def _ls_real(self, url, detail=True, **kwargs):
    # ignoring URL-encoded arguments
    clean_url = url
    if "http" not in url:
        url = f'{self.client_kwargs["root_url"]}catalogs/' + url
    kw = self.kwargs.copy()
    kw.update(kwargs)
    # logger.debug(url)
    session = await self.set_session()
    async with session.get(url, **self.kwargs) as r:
        self._raise_not_found_for_status(r, url)
        out = await r.json()

    out = [clean_url + f'/{x["identifier"]}' for x in out["resources"]]
    return out


class FusionHTTPFileSystem(HTTPFileSystem):
    """Fusion HTTP filesystem.
    """
    def __init__(self, *args, **kwargs):
        """
        Same signature as the fsspec HTTPFileSystem
        Args:
            *args:
            **kwargs:
        """
        super().__init__(*args, **kwargs)

    def _decorate_url(self, url):
        if "http" not in url:
            url = f'{self.client_kwargs["root_url"]}catalogs/' + url
        return url

    async def _ls_real(self, url, detail=True, **kwargs):
        # ignoring URL-encoded arguments
        clean_url = url
        if "http" not in url:
            url = f'{self.client_kwargs["root_url"]}catalogs/' + url
        kw = self.kwargs.copy()
        kw.update(kwargs)
        # logger.debug(url)
        session = await self.set_session()
        async with session.get(url, **self.kwargs) as r:
            self._raise_not_found_for_status(r, url)
            out = await r.json()

        out = [clean_url + f'/{x["identifier"]}' for x in out["resources"]]
        return out

    def ls(self, url, detail=True, **kwargs):
        """
        List resources.
        Args:
            url:
            detail:
            **kwargs:

        Returns:

        """
        url = self._decorate_url(url)
        ret = super().ls(url, detail=True, **kwargs)
        return [x.split(f'{self.client_kwargs["root_url"]}catalogs/')[-1] for x in ret]

    def exists(self, url, detail=True, **kwargs):
        """
        Check existence.
        Args:
            url:
            detail:
            **kwargs:

        Returns:

        """
        url = self._decorate_url(url)
        return super().exists(url, **kwargs)

    def cat(self, url, start=None, end=None, **kwargs):
        """
        Fetch paths' contents.
        Args:
            url:
            start:
            end:
            **kwargs:

        Returns:

        """
        url = self._decorate_url(url)
        return super().cat(url, start=None, end=None, **kwargs)

    def get(self, rpath, lpath, chunk_size=5 * 2 ** 20, callback=_DEFAULT_CALLBACK, **kwargs):
        """
        Copy file(s) to local.
        Args:
            rpath:
            lpath:
            chunk_size:
            callback:
            **kwargs:

        Returns:

        """
        url = self._decorate_url(lpath)
        raise NotImplementedError

    def put(self,
        lpath,
        rpath,
        chunk_size=5 * 2 ** 20,
        callback=_DEFAULT_CALLBACK,
        method="post",
        **kwargs):
        """
        Copy file(s) from local.
        Args:
            lpath:
            rpath:
            chunk_size:
            callback:
            method:
            **kwargs:

        Returns:

        """

        rpath = self._decorate_url(rpath)
        raise NotImplementedError