import os

from modules import shared


def get_webui_setting(key, default):
    value = shared.opts.data.get(key, default)
    if not isinstance(value, type(default)):
        value = default
    return value


class FileManager:
    def __init__(self):
        self.update_models_dir()

    @property
    def models_dir(self) -> str:
        """
        Get models directory.

        Returns:
            str: svd models directory
        """
        self.update_models_dir()
        if not os.path.isdir(self._models_dir):
            os.makedirs(self._models_dir, exist_ok=True)
        return self._models_dir

    def update_models_dir(self) -> None:
        """Update models directory."""

        models_dir = get_webui_setting("svd_models_dir", "")
        if (len(models_dir) == 0) or (models_dir.isspace()):
            models_dir = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "models"
            )
        self._models_dir = models_dir


file_manager = FileManager()
