import re
with open("adv_3.txt", 'r') as file:
  
  res = 0
  for row in file:
    pattern = "mul\(\d{1,3},\d{1,3}\)"
    x = re.findall(pattern, row)
    for i in x: 
      pair = eval(i[3:]) 
      res += pair[0]* pair[1]     
      
  print(res)
      
      
      
      
      