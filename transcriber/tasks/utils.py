
import os

def min_value_check(arg,value):
    if arg<=value:
        raise ValueError(f"{str(arg)} should be greater than or equal to {value}")
        
    return True
    
def path_check(path):

    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} does not exists")
    return True
