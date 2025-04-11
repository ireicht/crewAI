import unicodedata
import datetime
import re
from pathlib import Path
import json




# Function to read JSON configuration file
def load_config(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except:
        print(f"File does not yet exist: {file_path}, trying to create it..")
        save_config(file_path,{"init_file":f"{file_path}"})
        print(f"file created: {file_path}")
        return load_config(file_path)


# Function to write dictionary to JSON configuration file
def save_config(file_path, config_dict):
    with open(file_path, 'w') as file:
        json.dump(config_dict, file, indent=4)


# Getter and Setter for BENCHMARK_SESSION_ID_MOD
def get_benchmark_session_id_mod():
    config = load_config('benchmark_tmp.json')
    return config.get('BENCHMARK_SESSION_ID_MOD', '')

def set_benchmark_session_id_mod(value):
    config = load_config('benchmark_tmp.json')
    config['BENCHMARK_SESSION_ID_MOD'] = value
    save_config('benchmark_tmp.json', config)

# Getter and Setter for BENCHMARK_BASE_PATH
def get_benchmark_base_path():
    config = load_config('benchmark_tmp.json')
    return config.get('BENCHMARK_BASE_PATH', '')

def set_benchmark_base_path(value):
    config = load_config('benchmark_tmp.json')
    config['BENCHMARK_BASE_PATH'] = value
    save_config('benchmark_tmp.json', config)

# Getter and Setter for BENCHMARK_LOG_FILE_PATH
def get_benchmark_log_file_path():
    config = load_config('benchmark_tmp.json')
    return config.get('BENCHMARK_LOG_FILE_PATH', '')

def set_benchmark_log_file_path(value):
    config = load_config('benchmark_tmp.json')
    config['BENCHMARK_LOG_FILE_PATH'] = value
    save_config('benchmark_tmp.json', config)

def reset_benchmark_tmp_file():
    config = load_config('benchmark_tmp.json')
    config = {}
    save_config('benchmark_tmp.json', config)

# Getter and Setter for BENCHMARK_SESSION_ID_MOD
def get_benchmark_crew_iteration():
    config = load_config('benchmark_tmp.json')
    return config.get('crew_iteration', '')

def set_benchmark_crew_iteration(value):
    config = load_config('benchmark_tmp.json')
    config['crew_iteration'] = value
    save_config('benchmark_tmp.json', config)

def append_finished_crew_iteration(value):
    config = load_config('benchmark_tmp.json')
    current_finished_crew_iterations = config.get('finished_crew_iteration',[])
    current_finished_crew_iterations.append(value)
    config['finished_crew_iteration'] = current_finished_crew_iterations
    save_config('benchmark_tmp.json', config)

def get_finished_crew_iterations() -> list :
    config = load_config('benchmark_tmp.json')
    return config.get('finished_crew_iteration',[])
     

def print_structured(json_input):
    def print_dict(d, indent=2):
        for key, value in d.items():
            if isinstance(value, dict):
                print(' ' * indent + f"{key}:")
                print_dict(value, indent + 2)
            elif isinstance(value, list):
                print(' ' * indent + f"{key}:")
                for item in value:
                    if isinstance(item, dict):
                        print_dict(item, indent + 4)
                    else:
                        print(' ' * (indent + 4) + str(item))
            else:
                # Preserve the original formatting of curly braces and quotes
                formatted_value = value.replace("{", "{{").replace("}", "}}").replace('"', '\\"')
                print(' ' * indent + f"{key}: {formatted_value}")
    
    print_dict(json_input)


def write_log(filename, message):
    """
    Writes a message with a timestamp to a specified log file.

    Parameters:
        filename (str): The name of the log file.
        message (str): The content of the message to be written.
    """
    try:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(filename, 'a') as log_file:
            log_file.write(f'[{timestamp}] {message}\n')
        print("Message successfully written to", filename)
    except Exception as e:
        print("An error occurred while writing to the log file:", str(e))

def sanitize_filename(value):#
    """
    Sanitize the filename to remove invalid characters, including special characters
    such as umlauts, accents, etc.
    """
    # Normalize unicode characters to their closest ASCII representation
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    
    # Replace specific characters (e.g., umlauts, sharp s)
    value = value.replace('ä', 'ae').replace('ö', 'oe').replace('ü', 'ue').replace('ß', 'ss')

    # Replace any other non-alphanumeric characters with '_'
    value = re.sub(r'[^a-zA-Z0-9_\-]', '_', value)
    
    return value


def get_variable_value(filepath: str, var_of_interest: str) -> str:
    # Define a regex pattern to match the variable assignment
    pattern = re.compile(rf'{re.escape(var_of_interest)}\s*=\s*(\d+)')
    
    try:
        with open(filepath, 'r') as file:
            for line in file:
                match = pattern.search(line)
                if match:
                    return match.group(1)  # Return the value found
    except FileNotFoundError:
        print(f"File not found: {filepath}")
    except Exception as e:
        print(f"An error occurred: {e}")

    return None  # Return None if the variable is not found


def update_file_variable(file_path, var_prefix, new_value):
    """
    Updates a variable in a file by modifying the line that starts with the specified prefix.
    If the line is not found, it appends the variable definition at the end of the file.

    Args:
        file_path (str or Path): The path to the file.
        var_prefix (str): The prefix of the variable line to update.
        new_value (str): The new value to assign to the variable.
    """
    # Convert file_path to a Path object (if it isn't already)
    file_path = Path(file_path)

    # Check if the file exists
    if not file_path.exists():
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")

    # Read the file content
    lines = file_path.read_text().splitlines(keepends=True)

    updated = False
    new_lines = []
    
    # Process each line and update the target variable line
    for line in lines:
        if line.startswith(var_prefix):
            new_lines.append(f"{var_prefix}{new_value}\n")
            updated = True
        else:
            new_lines.append(line)
    
    # Append the variable if it wasn't found
    if not updated:
        new_lines.append(f"{var_prefix}{new_value}\n")
    
    # Write the updated content back to the file
    file_path.write_text("".join(new_lines))

def format_duration(duration):
    # Format the duration in HH:MM:SS format
    hours, remainder = divmod(int(duration.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)
    formatted_duration = "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)
    return formatted_duration