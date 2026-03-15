# Taking whatsapp chat.txt and converting it to a clean csv file where 
# no message is broken into multiple lines and storing in OUTPUT path
import re
from pathlib import Path
import pandas as pd

def f_txt_to_csv(folder_name, file_name):
    
    INPUT = Path(f"data/whatsapp/{folder_name}/{file_name}.txt")
    OUTPUT = Path(f"data/whatsapp/{folder_name}/{file_name}_clean.csv")
    # Regex: matches typical WhatsApp message-start like "10/1/24, 11:31 AM - "
    msg_start_re = re.compile(r'^\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}(?:\s*[APap][Mm])?\s*-\s+')

    lines = INPUT.read_text(encoding="utf-8").splitlines()
    messages = []
    current = None

    for line in lines:
        if msg_start_re.match(line):
            if current is not None:
                messages.append(re.sub(r'\s+', ' ', current).strip())
            current = line
        else:
            if current is None:
                current = line
            else:
                current += " " + line.strip()

    if current:
        messages.append(re.sub(r'\s+', ' ', current).strip())

    df = pd.DataFrame({"text": messages})
    df.to_csv(OUTPUT, index=True, encoding="utf-8-sig")
    print(f"Wrote {len(messages)} messages to {OUTPUT}")


