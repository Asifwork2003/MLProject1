import os
import sys
import dill
from src.exception import CustomException

def save_object(file_path, obj):
    """
    Save a Python object to a file using dill.

    Parameters:
    - file_path (str): The path where the object should be saved.
    - obj: The object to be saved.

    Raises:
    - CustomException: If there is an error during saving the object.
    """
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
