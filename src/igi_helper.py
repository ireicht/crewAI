import unicodedata
import datetime
import re
import ast

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