import os                                                              # Importing the os module to interact with the operating system (for file and directory handling).
from pathlib import Path                                               # Importing Path from pathlib module for easy path manipulation in an object-oriented way.
import logging                                                         # Importing the logging module to log messages like info, warning, error, etc.

logging.basicConfig(level=logging.INFO,format='[%(asctime)s]: %(message)s:')  # Configuring the logging system to show log messages with timestamp and custom message format, set at INFO level.


list_of_files=[                                                        # Creating a list variable named list_of_files containing relative paths of files that need to be created for the project structure.
    "src/__init__.py",                                                 # File 1: __init__.py inside src folder – marks it as a Python package.
    "src/helper.py",                                                   # File 2: helper.py inside src – usually contains helper functions.
    "src/prompt.py",                                                   # File 3: prompt.py inside src – usually contains prompt-related logic for the application.
    ".env",                                                            # File 4: .env file – used to store environment variables like API keys.
    "setup.py",                                                        # File 5: setup.py – used for package setup or installation.
    "research/trials.ipynb",                                           # File 6: Jupyter notebook file for research experiments.
    "app.py",                                                          # File 7: app.py – main application script, often Flask or Streamlit entry point.
    "store_index.py",                                                  # File 8: store_index.py – might handle vector database or index storage logic.
    "static/.gitkeep",                                                 # File 9: .gitkeep inside static – ensures empty folders are tracked in Git.
    "templates/chat.html",                                             # File 10: HTML file for frontend chat interface inside templates directory.
    "testing.py"                                                       # File 11: testing.py – used for testing application functionalities.
]

for filepath in list_of_files:                                         # Starting a for loop to iterate through each file path in list_of_files.
    filepath=Path(filepath)                                            # Converting the file path string into a Path object for better handling of file system paths.
    filedir,filename=os.path.split(filepath)                           # Splitting the file path into directory path (filedir) and file name (filename) using os.path.split().

    if filedir != "":                                                  # Checking if the file path contains a directory (i.e., not an empty string).
        os.makedirs(filedir,exist_ok=True)                             # Create the directory (and intermediate ones if needed); does nothing if it already exists (exist_ok=True).
        logging.info(f"Creating directory; {filedir} for the file {filename}")  # Log an info message indicating that the directory is being created for the file.

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):  # Check if the file does not exist OR if the file exists but is empty (size 0).
        with open(filepath,'w') as f:                                   # Open the file in write mode ('w') – creates an empty file if it doesn’t exist.
            logging.info(f"Creating empty file : {filepath}")           # Log an info message stating that an empty file is being created.
            
    else:                                                               # Else condition: if the file already exists and is not empty.
        logging.info(f"{filename} is already created.")                 # Log a message that the file is already present and not recreated.
