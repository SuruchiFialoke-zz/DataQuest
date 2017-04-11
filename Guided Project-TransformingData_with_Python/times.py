from dateutil.parser import parse
from datetime import datetime
import read

df = read.load_data()

def get_hour(time_str):
    parse(time_str)
    
    
    