from os import listdir
import string

# load doc into memory
def load_doc(filename):    
	# open the file as read only
    file = open(filename, encoding='utf-8', errors = 'ignore')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

# split a document into news story and highlights
def split_story(doc):
    # find first highlight
    index = doc.find('@highlight')
    # split into story and highlights
    story, highlights = doc[:index], doc[index:].split('@highlight')
    # strip extra white space around each highlight
    highlights = [h.strip() for h in highlights if len(h) > 0]
    return story, highlights

# clean a list of lines
def clean_lines(lines):
    cleaned = list()
    # prepare a translation table to remove punctuation
    table = str.maketrans('', '', string.punctuation)
    for line in lines:
        # strip source cnn office if it exists
        index = line.find('(CNN) --')
        if index > -1:
            line = line[index+len('(CNN) --'):]
        index = line.find('(CNN)--')
        if index > -1:
            line = line[index+len('(CNN)--'):]            
        index = line.find('(CNN)')
        if index > -1:
            line = line[index+len('(CNN)'):]            
        # tokenize on white space
        line = line.split()
        # store as string
        cleaned.append(' '.join(line))
    # remove empty strings
    cleaned = [c for c in cleaned if len(c) > 0]
    return cleaned
    
# load all stories in a directory
def load_stories(directory):
    stories = list()
    count = 0;
    for name in listdir(directory):
        filename = directory + '/' + name
        # load document
        doc = load_doc(filename)
        # split into story and highlights
        story, highlights = split_story(doc)
        # store
        stories.append({'story':story, 'highlights':highlights})
        count += 1
        if (count % 5000 == 0):
            print(count)
        
    return stories

# load stories

directory = 'cnn/stories/'
stories = load_stories(directory)
print('Loaded Stories %d' % len(stories)) 

directory = 'dailymail/stories/'
stories2 = load_stories(directory)
print('Loaded Stories2 %d' % len(stories2))

# clean stories

for example in stories:
    example['story'] = clean_lines(example['story'].split('\n'))
    example['highlights'] = clean_lines(example['highlights'])
print('Cleaned Stories %d' % len(stories))     
for example in stories2:
    example['story'] = clean_lines(example['story'].split('\n'))
    example['highlights'] = clean_lines(example['highlights'])  
print('Cleaned Stories2 %d' % len(stories2))

with open('cnnAllData.txt', 'w') as f:
    for example in stories:
        f.write(" ".join(example['story']) + "\t" + " ".join(example['highlights']) + "\n")

with open('dmAllData.txt', 'w') as g:
    for example in stories2:
        g.write(" ".join(example['story']) + "\t" + " ".join(example['highlights']) + "\n") 

print("done")        