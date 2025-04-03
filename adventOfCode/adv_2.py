file_path = 'adv_2.txt'


def homogenity(l):
    state = 0  # 1 is decendign 2 is ascending
    for i in range(len(l) - 1):
        if l[i] < l[i + 1]:
            if state == 1:
                return False
            else:
                state = 2
        elif l[i] > l[i + 1]:
            if state == 2:
                return False
            else:
                state = 1
    return True


def dist(l):
    for i in range(len(l) - 1):
        n = abs(l[i] - l[i + 1])
        if n < 1 or n > 3:
            return False
    return True


counter = 0

with open(file_path, 'r') as file:
    for line in file:
        row = line.split()
        row_i = [int(i) for i in row]

        if homogenity(row_i) and dist(row_i) == True:
            counter += 1
        else:
          for i in range(len(row_i)):
            mod = row_i[:i] + row_i[i+1:]
            print(row_i, mod)
            if homogenity(mod) and dist(mod) == True:
              counter += 1
              break
            


print(counter)
