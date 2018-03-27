#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 14:12:32 2018

@author: aditya
"""
import re

filename = "sumdata/bothtrain.txt"

with open(filename,"r") as f:
    data = f.read()
    
parts = data.split("\n")


full_text = []
text_summary = []

error_line = []

for part in parts:
    try:
        if len(part) > 0:
            full_text.append(part.split("\t")[0])
            text_summary.append(part.split("\t")[1])
    except:
        print("error")
    
def clean_data(line):
    line = line.split()
    line = [word.lower() for word in line]
    line = [re.sub("[,.-!?]"," ",word) for word in line] # Remove punctuation in the text
    line = [re.sub("[\(][a-zA-z0-9]*[\)]"," ", word) for word in line] # Remove words within brackets 
    line = [re.sub("[0-9]*[.]*[0-9]*"," ", word) for word in line] # Remove Numbers 
    
    return " ".join(line)


temp = []
for line in full_text:
    a = line[:50]
    if bool(re.search("[A-Z]{3,}[,][\s][a-zA-Z]{3,}[\(][A-Za-z][\)]",a)):
        temp.append(line)