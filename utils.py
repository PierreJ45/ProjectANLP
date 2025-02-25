from getData import get_train_data
import os

workspace_path = os.getcwd()
train_csv_path = os.path.join(workspace_path, 'train_submission.csv')
train_df_without_NaNs, _ = get_train_data(train_csv_path, seed=1, removeNaNs=True, validation_proportion=0)
LABELS = train_df_without_NaNs['Label'].unique().tolist()

def get_unicode(char):
    """
    Returns the Unicode code point of a given character.

    Args:
    char (str): A single character.

    Returns:
    str: The Unicode code point in the format 'U+XXXX'.
    """
    if len(char) != 1:
        raise ValueError("Input must be a single character.")

    return f"U+{ord(char):04X}"

def process_unicode(dataset, get_unicode):
    """
    Process the dataset to gather unique Unicode code points and 
    associate each with the corresponding language.
    
    Args:
    dataset (DataFrame): The dataset containing language labels and text.
    get_unicode (function): The function to get Unicode from character.
    
    Returns:
    set, dict: A set of all unique Unicode code points and a dict mapping languages to their Unicode code points.
    """
    # Initialize a set for all unique Unicode values
    all_unicodes = set()
    
    # Initialize a dictionary to store unicodes for each language
    language_unicodes = {}

    # Iterate through each row in the dataset
    for index, row in dataset.iterrows():

        # Get the language and the text
        language = row['Label']
        text = row['Text']  # Assuming 'Text' column contains the actual text

        # Initialize a set to store unicodes for this specific language
        language_unicode_set = set()
        
        # Iterate over each character in the text
        for char in text:
            # Get the Unicode for the character
            unicode = get_unicode(char)
            
            # Add the Unicode to the global set
            all_unicodes.add(unicode)
            
            # Add the Unicode to the language-specific set
            language_unicode_set.add(unicode)

        # Add the language-specific Unicode set to the dictionary
        if language not in language_unicodes:
            language_unicodes[language] = language_unicode_set
        else:
            language_unicodes[language].update(language_unicode_set)

    return all_unicodes, language_unicodes

def inverse_dictionary(dictionary):
    """
    dictionnary : keys = languages, values = set of all unicodes seen in that language

    output : keys = unicodes, values = set of languages in which they appear
    """
    res = {}

    for language in dictionary:
        for unicode in dictionary[language]:
            if unicode not in res:
                res[unicode] = set()
            res[unicode].add(language)

    return res