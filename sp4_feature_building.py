## in this script, we read a whatsapp csv file that with "timestamp, sender, message, english_message" columns
## input file "data/whatsapp/parivar/parivar_google_translated.csv"
## and create more features of out it like datetime parsing, year, month, month name etc.
## also, we read birthday data from read_birthdays.py and match messages sent on birthdaydate . 
## output file has birthday_person column which is non-empty if message was sent on someone's birthday.
## output file "data/whatsapp/parivar/parivar_features.csv"
import pandas as pd
from utility.read_birthdays import load_birthdays

def add_features_to_whatsapp_data(folder_name, file_name):
    # Load the translated WhatsApp data
    

    df_csv = pd.read_csv(f"data/whatsapp/{folder_name}/{file_name}_google_translated.csv", encoding="utf-8-sig")
    #df_out = df_csv[~df_csv["message"].str.contains("<Media omitted>", na=False, regex=False)].reset_index(drop=True)

    # normalize weird spaces that can appear before AM/PM and parse
    ts = df_csv["timestamp"].astype(str).str.replace("\u202f", " ", regex=False).str.replace("\xa0", " ", regex=False).str.strip()
    df_csv["dt"] = pd.to_datetime(ts, format="%m/%d/%y, %I:%M %p", errors="coerce")

    # extract year, month (number) and month name
    df_csv["year"] = df_csv["dt"].dt.year

    df_csv["month"] = df_csv["dt"].dt.month
    #df_csv["month_name"] = df_csv["dt"].dt.month_name()
    df_csv["month_short"] = df_csv["dt"].dt.strftime("%b")
    df_csv["month_year"] = df_csv["dt"].dt.strftime("%b-%y").str.upper()  # e.g., 'SEP-24'

    # Extract day-month for birthday matching (DD-MMM format)
    df_csv["day_month"] = df_csv["dt"].dt.strftime("%d-%b").str.upper()  # e.g., '21-JAN'

    # Load birthday data and merge
    birthdays_dict = load_birthdays(group=folder_name)  # Pass folder name as group

    # Month name normalization mapping
    month_map = {
        'JANUARY': 'JAN', 'FEBRUARY': 'FEB', 'MARCH': 'MAR', 'APRIL': 'APR',
        'MAY': 'MAY', 'JUNE': 'JUN', 'JULY': 'JUL', 'AUGUST': 'AUG',
        'SEPTEMBER': 'SEP', 'OCTOBER': 'OCT', 'NOVEMBER': 'NOV', 'DECEMBER': 'DEC'
    }

    # Convert birthday dictionary to DataFrame
    birthday_data = []
    for name, date in birthdays_dict.items():
        # Normalize the date format to DD-MMM (uppercase)
        date_normalized = date.upper()
        
        # Split into day and month
        parts = date_normalized.split('-')
        if len(parts) == 2:
            day, month = parts
            
            # Ensure 2-digit day format (add leading zero if needed)
            if len(day) == 1:
                day = '0' + day
            
            # Normalize month to 3-letter abbreviation
            if len(month) > 3:
                # Handle full month names like 'APRIL', 'MARCH'
                month = month_map.get(month, month[:3])
            
            date_normalized = f"{day}-{month}"
            birthday_data.append({'birthday_date': date_normalized, 'birthday_person': name})

    # Only merge if we have birthday data
    if birthday_data:
        df_birthdays = pd.DataFrame(birthday_data)
        print(f"\nBirthday lookup table:")
        print(df_birthdays)

        # Merge with WhatsApp data
        df_csv = df_csv.merge(df_birthdays, how='left', left_on='day_month', right_on='birthday_date')
        print(f"\n✓ Merged birthday data")
    else:
        print(f"\n⚠️  No birthday data found for group '{folder_name}' - adding empty birthday_person column")
        df_csv['birthday_person'] = None
        df_csv['birthday_date'] = None
    
    # Show sample of merged data
    if 'birthday_person' in df_csv.columns:
        print(f"\n📊 Sample rows with birthday data:")
        sample_cols = ["timestamp", "sender", "day_month", "birthday_person"]
        print(df_csv[sample_cols].head(10).to_string())

    out_csv = f"data/whatsapp/{folder_name}/{file_name}_features.csv"
    df_csv.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n✅ Features saved to: {out_csv}")
    print(f"   Total messages: {len(df_csv)}")
    print(f"   Total columns: {len(df_csv.columns)}")
    print(f"   Columns: {df_csv.columns.tolist()}")


add_features_to_whatsapp_data("parivar", "parivar")