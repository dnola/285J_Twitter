import pandas # Works with data in table format
import glob # Lets you essentially do a wildcard filename search
import pickle # Lets you easily dump and load python objects to disk

### Data should be in folder in same directory as this file named 'Raw Data'
#

raw_data = []
for f in glob.glob('Raw Data/*.tsv'):
    d = pandas.read_csv(open(f,'rU'), sep="\t",
                        header=None,
                        names = ["id", "text", "user", "unknown feature 1",
                                 "location", "time","maybe latitude?",
                                 "maybe longitude?", "unknown feature 2"],
                        # warn_bad_lines=False,
                        error_bad_lines=False) # Skip lines that don't have
                                               # enough columns


    raw_data+=d["text"].tolist() # Grab the column containing an
    # actual tweet and turn it into a list - add it to the giant list
    # not sure why there are multiple files...

# TODO: Figure out how many lines we are throwing away because of bad
# formatting - if we are throwing a lot away figure out why

pickle.dump(raw_data, open('raw_text_data.pkl','wb'))

# Note from David: This is literally 8 lines of code - Python has killer
# libraries - try doing this much work in 8 lines of C++ code!