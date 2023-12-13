import sys
f = open(sys.argv[1], "r")

Lines = f.readlines()
numofrows = Lines[0].split()

for i in range(int(numofrows[0])):
	s = "v "+Lines[i+1]
	while "  " in s:
		s = s.replace("  ", " ")
	print(s)

for i in range(int(numofrows[1])):
	s = "f "+Lines[i+int(numofrows[0])+1]
	while "  " in s:
		s = s.replace("  ", " ")
	print(s)
	
f.close()

