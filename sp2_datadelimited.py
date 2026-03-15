## in this script, we read a whatsapp csv file that was created in step 1, 
# and split each line into timestamp, sender, and message columns, 
# then save as a CSV file.
## then wherever send is blank, it means system message or unparsed line.

import re
from pathlib import Path
import pandas as pd

def clean_to_split(folder_name, file_name):
    # Set input/output paths based on folder and file names
    inp = Path(f"data/whatsapp/{folder_name}/{file_name}_clean.csv")
    out = Path(f"data/whatsapp/{folder_name}/{file_name}_split.csv")

    # Regex that captures timestamp, optional sender (before the colon) and the rest as message
    msg_re = re.compile(
        r'^(?P<timestamp>\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}(?:\s*[APap][Mm])?)\s*-\s*(?:(?P<sender>[^:]+):\s*)?(?P<message>.*)$'
    )

    # Read messages: accept CSV with column 'text' or plain text file (one message per line)
    if inp.suffix.lower() == ".csv":
        df_in = pd.read_csv(inp, encoding="utf-8-sig")
        if "text" in df_in.columns:
            lines = df_in["text"].astype(str).tolist()
        else:
            lines = df_in.iloc[:, 0].astype(str).tolist()
    else:
        lines = inp.read_text(encoding="utf-8").splitlines()

    rows = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        m = msg_re.match(line)
        if m:
            ts = m.group("timestamp").strip()
            sender = (m.group("sender") or "").strip()
            msg = m.group("message").strip()
        else:
            # fallback: no timestamp parsed -> put line into message column
            ts, sender, msg = "", "", line
        msg = " ".join(msg.split())  # collapse internal whitespace
        rows.append({"timestamp": ts, "sender": sender, "message": msg})

    df_out = pd.DataFrame(rows)
    # Replace empty or null sender with "system"
    df_out["sender"] = df_out["sender"].replace("", "system").fillna("system")
    out_csv = out.with_suffix('.csv')
    df_out.to_csv(out_csv, index=True, encoding="utf-8-sig")
    print(f"Wrote {len(df_out)} rows to {out_csv}")