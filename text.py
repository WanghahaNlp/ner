import re
pretrained = ["b1",1,2,4]
char = "b1"
a = any(x in pretrained for x in [char, char.lower(),re.sub('\d', '0', char.lower())])
print([char, char.lower(),re.sub('\d', '0', char.lower())])
print(a)