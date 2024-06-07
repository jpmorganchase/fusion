"""Command line environment for fusion."""

import argparse
import inspect

from fusion import Fusion

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fusion command line environment")
    methods = [
        method_name
        for method_name in dir(Fusion)
        if callable(getattr(Fusion, method_name)) and not method_name.startswith("_")
    ]

    args = inspect.signature(Fusion.__init__).parameters
    args = {param.name for param in args.values()}  # type: ignore

    for m in methods:
        met = getattr(Fusion, m)
        args = args.union({param.name for param in inspect.signature(met).parameters.values()})  # type: ignore

    args = args.difference({"self"})  # type: ignore

    for a in args:
        parser.add_argument("--" + a, default=None)

    parser.add_argument("--method", default=None)
    args = parser.parse_args()  # type: ignore
    kw = {}
    for k in ["root_url", "credentials", "download_folder", "log_level", "log_path"]:
        if vars(args)[k] is not None:
            kw[k] = vars(args)[k]

    client = Fusion(**kw)
    if vars(args)["method"] is not None:
        method = getattr(client, vars(args)["method"])
        kw_m = {}
        args_m = {param.name for param in inspect.signature(method).parameters.values()}.difference({"self"})
        for k in args_m:
            if vars(args)[k] is not None:
                v = vars(args)[k]
                if v == "True":
                    v = True
                if v == "False":
                    v = False

                kw_m[k] = v

        method(**kw_m)
