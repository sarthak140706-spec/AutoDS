import platform
import sys

import streamlit
import pandas
import numpy
import sklearn


def get_system_info():
    """
    Returns system and package information.
    """

    return {
        "Operating System": platform.system(),
        "OS Version": platform.version(),
        "Machine": platform.machine(),
        "Processor": platform.processor(),
        "Python Version": sys.version.split()[0],
        "Streamlit Version": streamlit.__version__,
        "Pandas Version": pandas.__version__,
        "NumPy Version": numpy.__version__,
        "Scikit-Learn Version": sklearn.__version__,
    }