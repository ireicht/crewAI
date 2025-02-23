import unicodedata
import re

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