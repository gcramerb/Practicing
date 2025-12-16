

def circle_game(n):
    curr_pl = 0
    circle = [ i for i in range(1,n+1)]
    for k in range(1,n):
        pos = curr_pl + k
        del_pos = pos % len(circle)
        del circle[del_pos]
        curr_pl = del_pos
    return circle[0]
import heapq 
def schedule(tasks):
    data = []
    for idx,arriv,prio,dur in tasks:
        data.append([arriv,dur,prio,idx])
    tasks_sorted= sorted(tasks)
    res,minHeap = [],[]
    i,time = 0,tasks_sorted[0][0]
    while minHeap or i < len(tasks_sorted):
        while i<len(tasks_sorted) and time >= tasks_sorted[i][0]:
            heapq.heappush(minHeap,[tasks_sorted[i][1],tasks_sorted[i][2]])
            i+=1
        if not minHeap:
            time = tasks_sorted[i][0]
        else:
            procTime , idx = heapq.heappop(minHeap)
            time += procTime
            res.append(idx)
    return res 

def evaluate(expression):
    nums = []
    elements = expression.split(" ")
    elements.reverse()
    val = 0
    first_op = True 
    for ele in elements:
        if ele == "+":
            a = nums.pop()
            b = nums.pop()
            val = a + b
            nums.append(val)
        elif ele =="*":
            a = nums.pop()
            b = nums.pop()
            val = a + b
            nums.append(val)
        elif ele =="/":
            a = nums.pop()
            b = nums.pop()
            val = int(a/b)
            nums.append(int(val / b))
        elif ele == "-":
            a = nums.pop()
            b = nums.pop()
            val = a + b
            nums.append(val)
        else:
            nums.append(int(ele))
    return val[0]
def next_pos(cur_pos,grid,i,j):
    rows = len(grid)
    cols = len(grid[0])
    n_i,n_j= cur_pos[0] + i,cur_pos[1] + j 
    if n_i>= rows or n_i < 0:
        return cur_pos
    if n_j >= cols or n_j < 0:
        return cur_pos
    
    if grid[n_i][n_j] == 1:
        return cur_pos
    return [n_i,n_j]

def move_robot(grid, pos, instructions):
    end_pos =pos
    for inst in instructions:
        if inst == "U":
            i,j = 0,1
        elif inst == "D":
            i,j = 0,-1
        elif inst == "L":
            i,j = -1,0
        elif inst == "R":
            i,j = 1,0
        end_pos = next_pos(end_pos,grid,i,j)
    return end_pos

if __name__=="__main__":
    grid = [
        [0,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,0]
    ]
    pos = [1,0]
    inst = "RRDDLUUR"
    res = move_robot(grid,pos,inst)
    print(res)