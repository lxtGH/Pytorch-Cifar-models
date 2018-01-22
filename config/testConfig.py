#!/usr/bin/python
import json

text = json.load(open("parameters.json",encoding='utf-8'))
print (type(text))
for key in text:
    print(key,text[key])
