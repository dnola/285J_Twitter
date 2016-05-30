import pandas # Works with data in table format
import glob # Lets you essentially do a wildcard filename search
import pickle # Lets you easily dump and load python objects to disk


### Data should be in folder in same directory as this file named 'Raw Data'
#

raw_data = None
for f in glob.glob('Raw Data/*.tsv'):
    try:
        d = pandas.read_csv(open(f,'rU'), sep="\t",
                            header=None,
                            names = ["tweet_id", "text", "user", "user_id",
                                     "user_location", "time","latitude",
                                     "longitude", "gps_precision"],
                            # warn_bad_lines=False,
                            error_bad_lines=False) # Skip lines that don't have
                                                   # enough columns
    except:
        print("Opening in rU mode failed... You are probably using Windows...")
        d = pandas.read_csv(open(f,'r',encoding='utf8'), sep="\t",
                        header=None,
                        names=["tweet_id", "text", "user", "user_id",
                               "user_location", "time", "latitude",
                               "longitude", "gps_precision"],
                        # warn_bad_lines=False,
                        error_bad_lines=False)
    try:
        raw_data = raw_data.append(d) # Grab the column containing an
    # actual tweet and turn it into a list - add it to the giant list
    except:
        raw_data=d

print(len(raw_data))

# Don't worry if this line gives you warnings. Pandas is whiny
raw_data[['latitude','longitude']] = raw_data[['latitude','longitude']].apply(pandas.to_numeric, errors='coerce') # force latitude and longitude to float
raw_data=raw_data.dropna() # drop any row with an NA in any column

print(len(raw_data))

pickle.dump(raw_data, open('pandas_data.pkl','wb'))

pickle.dump(raw_data['text'].tolist(), open('raw_text_data.pkl','wb'))
