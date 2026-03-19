# Read birthday.txt file as a dictionary

def load_birthdays(group='parivar', filepath="input_data/birthday.txt"):
    """
    Load birthdays from text file into a dictionary
    
    Args:
        group: Group name to load ('parivar' or 'vellapanti')
        filepath: Path to birthday.txt file
    
    Returns:
        Dictionary with {name: date} mappings
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            # Use eval to convert the string representation to actual dict
            # The file contains: 'parivar': {...}, 'vellapanti': {...}
            data = eval(f"{{{content}}}")
            birthdays = data.get(group, {})
        
        print(f"✓ Loaded {len(birthdays)} birthdays from {filepath} (group: {group})")
        return birthdays
    except FileNotFoundError:
        print(f"❌ Error: File '{filepath}' not found")
        return {}
    except Exception as e:
        print(f"❌ Error reading birthdays: {e}")
        return {}

# Test it
if __name__ == "__main__":
    birthdays_dict = load_birthdays()
    
    print("\n--- Birthday Dictionary ---")
    print(birthdays_dict)
    
    print("\n--- Individual Birthdays ---")
    for name, date in birthdays_dict.items():
        print(f"{name}: {date}")
    
    # Example: Check specific person's birthday
    print("\n--- Lookup Examples ---")
    person = "Preeti"
    if person in birthdays_dict:
        print(f"{person}'s birthday is on {birthdays_dict[person]}")
    
    person = "Mohan"
    if person in birthdays_dict:
        print(f"{person}'s birthday is on {birthdays_dict[person]}")
