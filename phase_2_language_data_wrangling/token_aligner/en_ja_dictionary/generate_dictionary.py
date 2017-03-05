"""
python generate_dictionary.py kanji_dict name_dict

TODO - REFACTOR BIG TIME
"""


import sys
import re


kanji_dict = open(sys.argv[1])
for entry in kanji_dict:
    if '#' in entry: continue
    entry = entry.strip()
    kanji = entry.split()[0]
    definitions = re.findall('{(.*?)}', entry)
    definitions = [re.sub('\(.+\)', '', x).strip() for x in definitions]
    for d in definitions:

        print '%s,%s' % (kanji, d)



# ex entry:
#  word [alternate word(;word2...)] /(n) donor/contributor/
name_dict = open(sys.argv[2])
i = 0
for entry in name_dict:
    if '#' in entry: continue
    ja_name = entry.split()[0]
    alt_name = re.findall('\[(.*?)\]', entry)
    if len(alt_name) > 0: # probably a better way to do this... 
        names = [ja_name,  alt_name[0] ]
    else:
        names = [ja_name]

    # /(o) African National Congress/ANC/   -->   ['African National Congress', 'ANC']
    # messy....
    defs = [x.strip() for y in re.findall('/(.*)/', entry) for x in y.split('/')]
    defs = [re.sub('\(.+\)', '', x).strip() for x in defs]

    for name in names:
        for d in defs:
            print '%s,%s' % (name, d)



# ex entry:
#  word(;word2;word3...) [alternate word(;word2...)] /(n) donor/contributor/EntL1714960X/
general_dict = open(sys.argv[3])
next(general_dict)    # avoid header
for entry in general_dict:
    ja_names = entry.split()[0].split(';')
    alt_name = re.findall('\[(.*?)\]', entry)
    if len(alt_name) > 0: # probably a better way to do this... 
        names = ja_names + alt_name[0].split(';')
    else:
        names = ja_names
        
    names = [re.sub('\(.+\)|{(.+)}', '', x).strip() for x in names]

    # /(o) African National Congress/ANC/   -->   ['African National Congress', 'ANC']
    # messy....
    defs = [x.strip() for y in re.findall('/(.*)/', entry) for x in y.split('/')][:-1]
    defs = [re.sub('\(.+\)|{(.+)}', '', x).strip() for x in defs]

    for name in names:
        for d in defs:
            if len(d) > 0:
                print '%s,%s' % (name, d)






