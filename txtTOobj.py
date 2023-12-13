import sys
f = open(sys.argv[1], "r")

Lines = f.readlines()
numofrows = Lines[0].split()

for i in range(int(numofrows[0])):
	print(("v "+Lines[i+1]).replace("    ", " "))

for i in range(int(numofrows[1])):
	print(("f "+Lines[i+int(numofrows[0])+1]).replace("    ", " "))

f.close()
