import json


class FileIOMixin:
    """Mixin for adding file loading and saving functionality to pydantic BaseModel."""
    @classmethod
    def load_from_file(cls, file_path: str):
        """Load parameters from a JSON file."""
        with open(file_path, "r") as file:
            data = json.load(file)
        return cls.model_validate(data)

    def save_to_file(self, file_path: str):
        """Save the model data to a JSON file."""
        with open(file_path, "w") as file:
            json.dump(self.model_dump(), file, indent=4)
