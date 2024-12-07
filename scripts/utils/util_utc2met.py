from datetime import datetime

def utc_to_met(utc):
    utc = clean(utc)
    dateref = datetime(2017, 1, 1)
    
    date_formats = [
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H",
        "%Y-%m-%d"
    ]
    
    for date_format in date_formats:
        try:
            reqdate = datetime.strptime(utc, date_format)
            break
        except ValueError:
            continue
    else:
        raise ValueError("UTC string not in required format")

    MET = (reqdate - dateref).total_seconds()
    return MET

def clean(utc):
    return utc.replace('T', ' ')

# test the function 
utc_time = "2024-06-30T20:32:01.197"
print(utc_to_met(utc_time))
