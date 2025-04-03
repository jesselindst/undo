# Read the file
file_path = 'adv_1.txt' 


a = []
b = []

with open(file_path, 'r') as file:
    for line in file:
        cols = line.strip().split() 
        if len(cols) >= 2:  
            a.append(int(cols[0]))
            b.append(int(cols[1]))


a.sort()
b.sort()
sum = 0

for i in range(len(a)):
    sum = sum + abs(a[i] - b[i])
    

similarity_score = 0
for i in range(len(a)):
    cur_num = a[i]
    cur_entries = 0
    for j in range(len(b)):
        if cur_num == b[j]:
          cur_entries = cur_entries + 1
      
    similarity_score = cur_num * cur_entries + similarity_score
    
print(similarity_score)    
    
#print(sum)