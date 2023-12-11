import launch
from importlib_metadata import version

MIN_VERSION = "0.24.0"


def install():
    needs_install = False
    if launch.is_installed("diffusers"):
        installed_version = tuple(version("diffusers").split("."))
        min_version = tuple(MIN_VERSION.split("."))
        if installed_version < min_version:
            needs_install = True
    else:
        needs_install = True

    if needs_install:
        try:
            launch.run_pip(f"install diffusers>={MIN_VERSION}", "requirements for SVD")
        except Exception as e:
            print(e)
            print("Can't install diffusers.")


install()
