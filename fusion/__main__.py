import ast
import argparse
from fusion import Fusion


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fusion command line environment')
    methods = [method_name for method_name in dir(Fusion) if
               callable(getattr(Fusion, method_name)) and not method_name.startswith("_")]
    args = set(list(Fusion.__init__.__code__.co_varnames))
    for m in methods:
        met = getattr(Fusion, m)
        set(met.__code__.co_varnames)
        args = args.union(set(met.__code__.co_varnames))

    args = args.difference(set('self'))
    for a in args:
        parser.add_argument('--'+a, default=None)

    parser.add_argument('--method', default=None)
    args = parser.parse_args()
    kw = {}
    for k in ["root_url", "credentials", "download_folder", "log_level", "log_path"]:
        if vars(args)[k] is not None:
            kw[k] = vars(args)[k]

    client = Fusion(**kw)
    if vars(args)["method"] is not None:
        method = getattr(client, vars(args)["method"])
        kw_m = {}
        args_m = set(method.__code__.co_varnames).difference(set('self'))
        for k in args_m:
            if vars(args)[k] is not None:
                kw_m[k] = ast.literal_eval(vars(args)[k])

        method(**kw_m)