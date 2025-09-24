from datetime import datetime

def time_to_decimal_hours(time_str):
    """Convert 'HH:MM:SS.sss' to decimal hours."""
    t = datetime.strptime(str(time_str), '%H:%M:%S.%f')
    return t.hour + t.minute/60 + t.second/3600 + t.microsecond/3600000000
