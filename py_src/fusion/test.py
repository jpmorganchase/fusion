from fusion.reports import Reports as _Reports, Report  # keep at top of file or import inside the method

def reports(self, *, reports: list[Report] | None = None):
    # Get the Fusion SDK object (ReportsWrapper) that already has client=self
    wrapper = super().reports()

    # Patch loaders so they inject client and return the SAME wrapper
    def _wrap_from_csv(file_path: str):
        loaded = _Reports.from_csv(file_path, client=self)
        wrapper.reports = loaded.reports
        for r in wrapper.reports:
            r.client = self
        return wrapper

    def _wrap_from_dataframe(df):
        loaded = _Reports.from_dataframe(df, client=self)
        wrapper.reports = loaded.reports
        for r in wrapper.reports:
            r.client = self
        return wrapper

    def _wrap_from_object(source):
        loaded = _Reports.from_object(source, client=self)
        wrapper.reports = loaded.reports
        for r in wrapper.reports:
            r.client = self
        return wrapper

    # Attach the patched methods to this instance
    wrapper.from_csv = _wrap_from_csv          # type: ignore[attr-defined]
    wrapper.from_dataframe = _wrap_from_dataframe  # type: ignore[attr-defined]
    wrapper.from_object = _wrap_from_object        # type: ignore[attr-defined]

    # If user passed initial Report objects, bind client to them
    if reports:
        wrapper.reports = reports
        for r in wrapper.reports:
            r.client = self

    return wrapper
