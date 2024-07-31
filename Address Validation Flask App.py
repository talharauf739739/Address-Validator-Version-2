    import pandas as pd
    import pickle
    import regex as re
    import xgboost as xgb
    from sklearn.preprocessing import LabelEncoder
    import joblib
    
    # Define the replacements for the address column
    replacements = {
        r'\bRd\b': 'Road',
        r'\bApt\b': 'Apartment',
        r'\bAPT\b': 'Apartment',
        r'\bapt\b': 'Apartment',
        r'\bApartments\b': 'Apartment',
        r'\bAjk\b': 'Azad Kashmir',
        r'\bAJK\b': 'Azad Kashmir',
        r'\bajk\b': 'Azad Kashmir',
        r'\bFlate\b': 'Flat',
        r'\bflate\b': 'Flat',
        r'\bChownk\b': 'Chowk',
        r'\bchownk\b': 'Chowk',
        r'\bCHOWNK\b': 'Chowk',
        r'\bChock\b': 'Chowk',
        r'\bchock\b': 'Chowk',
        r'\bCHOCK\b': 'Chowk',
        r'\bCollage\b': 'College',
        r'\bCOLLAGE\b': 'College',
        r'\bcollage\b': 'College',
        r'\bgali\b': 'Street',
        r'\bGali\b': 'Street',
        r'\bGALI\b': 'Street',
        r'\bNO\b': 'Number',
        r'\bNo\b': 'Number',
        r'\bno\b': 'Number',
        r'\bn\b': 'Number',
        r'\b#\b': 'Number',
        r'\bSEC\b': 'Sector',
        r'\bSec\b': 'Sector',
        r'\bsec\b': 'Sector',
        r'\bH\b': 'House',
        r'\bh\b': 'house',
        r'\bH(?![oO])\b': 'House',
        r'\bh(?![oO])\b': 'house',
        r'\bst\b': 'Street',
        r'\bSt\b': 'Street',
        r'\bST\b': 'Street',
        r'\bST(?![rR])\b': 'Street',
        r'\bSt(?![rR])\b': 'Street',
        r'\bst(?![rR])\b': 'Street',
        r'\b(H|h)(\d+)\b': r'House \2'  # New pattern to replace 'H' or 'h' followed by digits with 'House' followed by the digits
    }
    # Function to perform replacements (same as in training code)
    def replace_words(text):
        if isinstance(text, str):
            for pattern, replacement in replacements.items():
                text = re.sub(pattern, replacement, text)
            return text
        else:
            return text
    
    #Correct Specific Keywords, "House" and "Street", If Spell wrong, make correction
    #!pip install fuzzywuzzy
    import fuzzywuzzy
    from fuzzywuzzy import process
    
    # Define correct spellings
    #correct_house_spellings = {'word': '\bHouse\b|\bStreet\b|Colony|Factory|Pharmacy|Hospital|Floor|Department|Commercial|Phase', 'case': False}
    # Function to correct misspelled keywords
    def correct_spelling(address):
        corrected_address = address
        # Check for misspelled 'House'
        if 'House' not in address:
            closest_house_match = process.extractOne('House', [address])
            similarity_score_house = closest_house_match[1]  # Access the similarity score
            if similarity_score_house >= 90:  # Adjust threshold as needed
                corrected_address = corrected_address.replace(closest_house_match[0], 'House')
    
        # Check for misspelled 'Street'
        if 'Street' not in address:
            closest_street_match = process.extractOne('Street', [address])
            similarity_score_street = closest_street_match[1]  # Access the similarity score
            if similarity_score_street >= 90:  # Adjust threshold as needed
                corrected_address = corrected_address.replace(closest_street_match[0], 'Street')
    
        # Check for misspelled 'Street'
        if 'Colony' not in address:
            closest_street_match = process.extractOne('Colony', [address])
            similarity_score_street = closest_street_match[1]  # Access the similarity score
            if similarity_score_street >= 90:  # Adjust threshold as needed
                corrected_address = corrected_address.replace(closest_street_match[0], 'Colony')
    
        # Check for misspelled 'Street'
        if 'Factory' not in address:
            closest_street_match = process.extractOne('Factory', [address])
            similarity_score_street = closest_street_match[1]  # Access the similarity score
            if similarity_score_street >= 90:  # Adjust threshold as needed
                corrected_address = corrected_address.replace(closest_street_match[0], 'Factory')
    
        # Check for misspelled 'Street'
        if 'Pharmacy' not in address:
            closest_street_match = process.extractOne('Pharmacy', [address])
            similarity_score_street = closest_street_match[1]  # Access the similarity score
            if similarity_score_street >= 90:  # Adjust threshold as needed
                corrected_address = corrected_address.replace(closest_street_match[0], 'Pharmacy')
    
        # Check for misspelled 'Street'
        if 'Hospital' not in address:
            closest_street_match = process.extractOne('Hospital', [address])
            similarity_score_street = closest_street_match[1]  # Access the similarity score
            if similarity_score_street >= 90:  # Adjust threshold as needed
                corrected_address = corrected_address.replace(closest_street_match[0], 'Hospital')
    
        # Check for misspelled 'Street'
        if 'Floor' not in address:
            closest_street_match = process.extractOne('Floor', [address])
            similarity_score_street = closest_street_match[1]  # Access the similarity score
            if similarity_score_street >= 90:  # Adjust threshold as needed
                corrected_address = corrected_address.replace(closest_street_match[0], 'Floor')
    
        # Check for misspelled 'Street'
        if 'Department' not in address:
            closest_street_match = process.extractOne('Department', [address])
            similarity_score_street = closest_street_match[1]  # Access the similarity score
            if similarity_score_street >= 90:  # Adjust threshold as needed
                corrected_address = corrected_address.replace(closest_street_match[0], 'Department')
    
        # Check for misspelled 'Street'
        if 'Commercial' not in address:
            closest_street_match = process.extractOne('Commercial', [address])
            similarity_score_street = closest_street_match[1]  # Access the similarity score
            if similarity_score_street >= 90:  # Adjust threshold as needed
                corrected_address = corrected_address.replace(closest_street_match[0], 'Commercial')
    
        # Check for misspelled 'Street'
        if 'Phase' not in address:
            closest_street_match = process.extractOne('Phase', [address])
            similarity_score_street = closest_street_match[1]  # Access the similarity score
            if similarity_score_street >= 90:  # Adjust threshold as needed
                corrected_address = corrected_address.replace(closest_street_match[0], 'Phase')
        if 'Village' not in address:
            closest_street_match = process.extractOne('Village', [address])
            similarity_score_street = closest_street_match[1]  # Access the similarity score
            if similarity_score_street >= 90:  # Adjust threshold as needed
                corrected_address = corrected_address.replace(closest_street_match[0], 'Village')
    
        return corrected_address
    
    
    
    
    
    
    # Define the input data
    address = 'chak'
    area = 'E-8'
    city = 'Islamabad-E'
    API_Match = ''  # You need to set this value based on your API result
    
    # Concatenate 'address', 'province', and 'city' into a single string
    full_address = f"{address}, {area}, {city}"
    
    # Apply replacements to the 'full_address' string
    updated_address = replace_words(full_address)
    
    #Apply Spelling Correction On Keywords like "House" and "Street"
    updated_address = correct_spelling(updated_address)
    
    
    
    import re
    
    # Function to determine if address only contains digits or only a single word
    def is_incomplete_address(address):
        return address.isdigit() or len(address.split()) == 1
    
    # Apply the function to create the 'Feature_4' column
    def extract_feature_4(address):
        return is_incomplete_address(address)
    
    feature_num_1 = ['House', 'Street', 'Sector', 'Number', 'Muhallah', 'Mohallah', 'Office', 'Shop', 'Hospital', 'Block', 'Shop', 'Flat', 'Flats', 'Room', 'Rooms', 'Floor', 'Apartment']
    feature_num_3 = ['Mobile', 'Mobiles', 'Ward', 'Branch', 'Mullah', 'Plaza', 'Studio', 'Workshop', 'Canal', 'Chowk', 'Airport', 'Masjid', 'Mosque', 'Ltd', 'Limited', 'Mill', 'Mills', 'Mil', 'Gate', 'Pump', 'Station', 'Clinic', 'Self', 'Bus', 'Center', 'Post Office', 'Plot', 'Town', 'Colony', 'Bank', 'Workshop', 'Block', 'University', 'Collge', 'School', 'Garden', 'Town','\bChaki\b', 'Pharmacy', 'Factory', 'Room', 'Mall', 'Park', 'Building', 'Market', 'Tower', 'Store', 'Markaz', 'Department', 'Hostel', 'Hotel', 'Cafe', 'Bakery', 'Centre', 'Suite', 'Complex', 'Bakery', 'Commercial', 'Phase', 'Bazar', 'Gate', '\bAdda\b', '\badda\b','Near', 'Opposite', 'Road']
    
    # Define feature extraction functions
    def extract_feature_1(text):
        return (any(re.search(r'\b{}\b'.format(feature), text) for feature in feature_num_1))
    
    def extract_feature_3(text):
        return any(re.search(r'\b{}\b'.format(feature), text) for feature in feature_num_3)
    
    
    """# Define feature extraction functions (similar to the training code)
    def extract_feature_1(text):
        return re.search(r'(BLOCK|Block|block|SHOP|Shop|shop|SECTOR|Sector|sector|FLAT|Flat|flat|Room|room)', text) is not None"""
    
    # Function to extract feature 2 (same as in training code)
    def extract_feature_2(text):
        return 'True' in text
    
    """def extract_feature_3(text):
        return re.search(r'(House|HOUSE|house|Street|street|STREET|Number)', text) is not None"""
    
    
    #API Code Below
    #API CODE
    import requests
    import pandas as pd
    
    # Replace YOUR_API_KEY with your actual API key.
    #API_KEY = "YOUR_API_KEY"
    
    def validate_address(address):
        url = f"https://api.geoapify.com/v1/geocode/search?text={address}&limit=1&apiKey={API_KEY}"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                if data.get("features"):
                    result = data["features"][0]
                    latitude = result["geometry"]["coordinates"][1]
                    longitude = result["geometry"]["coordinates"][0]
                    return "Y", latitude, longitude
                else:
                    return "N", None, None
            else:
                return "N", None, None
        except requests.exceptions.RequestException:
            return "N", None, None
    
    def translate_lat_long_to_address(latitude, longitude):
        url = f"https://api.geoapify.com/v1/geocode/reverse?lat={latitude}&lon={longitude}&apiKey={API_KEY}"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                if data.get("features"):
                    result = data["features"][0]
                    return result["properties"]["formatted"]
                else:
                    return None
            else:
                return None
        except requests.exceptions.RequestException:
            return None
    def process_api(address, area, city):
      # Concatenate 'area' and 'city' fields with 'address' field
      formatted_address = f'{address}, {area}, {city}'
      # Apply the validate_address function to the formatted address and create the 'API_Output' variable
      API_Output, _, _ = validate_address(formatted_address)
      return API_Output
    
    """
    feature_num_1 = ['House', 'Street', 'Sector', 'Number', 'Muhallah', 'Mohallah', 'Office', 'Shop', 'Hospital', 'Block', 'Shop', 'Flat', 'Flats', 'Room', 'Rooms', 'Floor', 'Apartment']
    feature_num_3 = ['Mobile', 'Mobiles', 'Ward', 'Branch', 'Mullah', 'Plaza', 'Studio', 'Workshop', 'Canal', 'Chowk', 'Airport', 'Masjid', 'Mosque', 'Ltd', 'Limited', 'Mill', 'Mills', 'Mil', 'Gate', 'Pump', 'Station', 'Clinic', 'Self', 'Bus', 'Center', 'Post Office', 'Plot', 'Town', 'Colony', 'Bank', 'Workshop', 'Block', 'University', 'Collge', 'School', 'Garden', 'Town', r'\bChak\b', 'chak', 'chaki', r'\bChaki\b', 'Pharmacy', 'Factory', 'Room', 'Mall', 'Park', 'Building', 'Market', 'Tower', 'Store', 'Markaz', 'Department', 'Hostel', 'Hotel', 'Cafe', 'Bakery', 'Centre', 'Suite', 'Complex', 'Bakery', 'Commercial', 'Phase', 'Bazar', 'Gate', r'\bAdda\b', 'adda']
    
    """
    
    # Replace 'YOUR_API_KEY' with the actual API key provided by Geoapify
    API_KEY = '84404b3a566a4540884ec766842899b5"'
    
    API_Output = process_api(address, area, city)
    import re
    #if (API_Output == "Y") and (re.search(r'House|Street|Number', address, flags=re.IGNORECASE) is not None) and (re.search(r'BLOCK|SHOP|SECTOR|FLAT|ROOM', address, flags=re.IGNORECASE) is not None):
    if (API_Output == "Y") and (re.search(r'^(?!\d)', address, flags=re.IGNORECASE)) and (re.search(r'^(?!\s)', address, flags=re.IGNORECASE )) :
    #updated_address (re.search('r/d).~only_contains(/d), address)
        API_Match = True
    else:
        API_Match = False
    
    # Extract features
    featured_1 = (extract_feature_1(updated_address))
    featured_2 = extract_feature_2(str(API_Match))
    featured_3 = extract_feature_3(updated_address)
    featured_4 = extract_feature_4(address)
    
    # Create a DataFrame with the extracted features
    input_data = pd.DataFrame({
        'API': API_Match, # True or False
        'Featured_1': [featured_1],
        'Featured_2': [featured_2],
        'Featured_3': [featured_3],
        'Featured_4': [featured_4]
    })
    
    # Load the trained XGBoost model
    model_filename = 'G:/Office/PriceOye/AddressValidationApp/AddressValidationApp/models/Address_Validation_Model_Updated.joblib'  # Adjust the path as needed
    loaded_model = joblib.load(model_filename)
    
    # Make predictions using the loaded model
    y_pred_encoded = loaded_model.predict(input_data)
    
    """
    # Inverse transform the predicted labels to get the original textual class labels
    label_encoder = LabelEncoder()
    label_encoder.fit(training_data['Result'])  # Ensure training_data is available from training code
    predicted_labels = label_encoder.inverse_transform(y_pred_encoded)
    """
    
    def decode(y_pred_encoded):
      if y_pred_encoded==0:
        return 'Complete'
      else:
        return 'Incomplete'
    
    # Additional messages based on keywords in the address
    
    def additional_msg(y_pred_encoded):
    # Additional messages based on keywords in the address
      if y_pred_encoded[0] == 1:  # Assuming 'Incomplete' label is encoded as 1
        if re.search(r'\bChak\b', address, flags=re.IGNORECASE):
            message = "Please Mention Muhallah, Gali Number, OR Courier Office, In your Address"
        elif re.search(r'\bVillage\b', address, flags=re.IGNORECASE):
            message = "Please Mention City Address, Or Courier Office Address"
        else:
            message = "Please Mention Nearby Famous Place, Road, Or Building."
      else:
        message = "Proceed With Order."

      return message


    
    print("Predicted Label:", decode(y_pred_encoded))
    print("Note:", additional_msg(y_pred_encoded))
    #print("Note:", additional_msg(y_pred_encoded))
    
        
