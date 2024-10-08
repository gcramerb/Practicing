from typing import List, Optional
from collections import defaultdict
import heapq
class Solution1834:
    def __init__(self):
        self.solution = []
        self.to_process = []
    def process_task(self) ->int:
        next_avliable_time = -1 
        self.sort_to_process()
        _,init_time,_ = self.to_process[0]
        while next_avliable_time <= init_time:
            proc_time,init_time,idx  =  self.to_process.pop(0)
            self.solution.append(idx)
            next_avliable_time = init_time + proc_time
            if len(self.to_process) > 0:
                _,init_time,_ = self.to_process[0]
        return next_avliable_time
    def sort_to_process(self):
        self.to_process = sorted(self.to_process, key=lambda x: (x[0], x[2]))
    def getOrder(self, tasks: List[List[int]]) -> List[int]:
        for idx,data in enumerate(tasks):
            data.append(idx)
        tasks_sorted= sorted(tasks)
        next_avliable_time = -1 
        for task_init,proc_time,idx in tasks_sorted:
            self.to_process.append([proc_time,task_init,idx])
            if next_avliable_time <= task_init: 
                next_avliable_time = self.process_task()
        self.sort_to_process()
        while len(self.to_process) > 0: 
            proc_time,_,idx  =  self.to_process.pop(0)
            self.solution.append(idx)
        return self.solution
    def correctSolution(self,tasks: List[List[int]]):
        for idx,data in enumerate(tasks):
            data.append(idx)
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

def run1834():
    tasks = [[7,10],[7,12],[7,5],[7,4],[7,2]]
    sol = Solution1834()
    return  sol.getOrder(tasks)

class SolutionMock: 
    def process(self) -> bool:
        heapq.heapify(self.list_diffs)
        while len(self.list_diffs):
            old_diff = heapq.heappop(self.list_diffs)
            if old_diff <=self.init_bricks:
                self.init_bricks -= old_diff
                self.count +=1
            elif self.init_ladd >0:
                self.count +=1
                self.init_ladd -=1
            else:
                return True
        return False
    def furthestBuilding(self, heights: List[int], bricks: int, ladders: int) -> int:
        self.list_diffs = []
        self.count = 0
        self.init_bricks = bricks
        self.init_ladd = ladders
        for i in range(len(heights)-1):
            diff = heights[i+1] - heights[i]
            if diff <=0: # 7   
                self.count +=1
                continue
            if ladders > 0:
                ladders -= 1
                self.list_diffs.append(diff)
            elif diff <=bricks:
                bricks -=diff
                self.list_diffs.append(diff)
            else:
                self.list_diffs.append(diff)
                finished = self.process()
                if finished:
                    return self.count
                bricks = self.init_bricks
                ladders = self.init_ladd
        self.process()
        return self.count
    def furthestBuildingWorks(self, heights: List[int], bricks: int, ladders: int) -> int:
        max_heap = []
        for i in range(len(heights)-1):
            diff = heights[i+1] - heights[i]
            if diff <=0:
                continue
            bricks-= diff
            heapq.heappush(max_heap,-diff) #use as max heap
            if bricks <0:
                if ladders ==0:
                    return i
                ladders -=1
                bricks += -heapq.heappop(max_heap)      
        return len(heights)-1
class TreeNode:
     def __init__(self, x):
         self.val = x
         self.left = None
         self.right = None
class Solution863:
    def distanceK(self, root: TreeNode, target: TreeNode, k: int) -> List[int]:
        self.parent = {}
        self.set_parent(root,None)
        self.k = k
        self.visited = [] 
        self.sol = []
        self.dfs(target,k)
        return self.sol

    def set_parent(self, node:TreeNode = None,parent= None):
        if not node:
            return
        self.parent[node.val] = parent
        self.set_parent(node.right,node)
        self.set_parent(node.left,node)
    def dfs(self,node:TreeNode,k_i):
        if not node:
            return
        if node.val in self.visited:
            return 
        self.visited.append(node.val)
        if k_i == 0:
            self.sol.append(node.val)
            return
        self.dfs(node.right,k_i-1)
        self.dfs(node.left,k_i-1)
        self.dfs(self.parent[node.val],k_i-1)
def runMock():
    s = SolutionMock()
    heights = [1,5,1,2,3,4,10000]
    bricks = 4
    ladders = 1
    heights =[4,2,7,6,9,14,12]
    bricks = 5
    ladders = 1
    res = s.furthestBuilding(heights,bricks,ladders)
    return res

class Solution2976:
    def djk(self,node_source):
        nodes_to_proces = [(0,node_source)]
        heapq.heapify(nodes_to_proces)
        while nodes_to_proces:
            dist_j,node_j  = heapq.heappop(nodes_to_proces)
            if node_j in self.min_path[node_source]:
                continue       
            self.min_path[node_source][node_j] = dist_j
            for weight,neighbor  in self.graph[node_j]:
                dist_with_j = dist_j + weight
                heapq.heappush(nodes_to_proces, (dist_with_j,neighbor))

                    
    def get_minimum_graph(self,original,changed,cost):
        self.my_inf = sum(cost) +1 
        self.graph= defaultdict(list)
        self.min_path= defaultdict(lambda: defaultdict(lambda: self.my_inf))
        for ori_i,changed_i,cost_i in zip(original,changed,cost):
            self.graph[ori_i].append((cost_i,changed_i))
            #self.min_path[ori_i][changed_i] = cost_i



    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        if len(cost) < len(original):
            return -1
        self.cost = cost
        self.get_minimum_graph(original,changed,cost)
        for s in source:
            self.djk(s)
        output = 0
        for c_s,c_t in zip(source,target):
            if c_s == c_t:
                continue
            cost_i = self.min_path[c_s][c_t]
            if cost_i >= self.my_inf:
                return -1 
            output +=cost_i
        return output
def run2976():
    source = "abcd"
    target = "acbe"
    original = ["a","b","c","c","e","d"]
    changed =  ["b","c","b","e","b","e"]
    cost =     [ 2,  5,  5,  1,   2 , 20]
    s = Solution2976()
    awns = s.minimumCost(source,target,original,changed,cost)
    return awns
class Solution833:
    def findReplaceString(self, s: str, indices: List[int], sources: List[str], targets: List[str]) -> str:
        final_str = list(s)
        changes = [(i_,s_,t_) for i_,s_,t_ in zip(indices,sources,targets)]
        K = len(indices)
        changes = sorted(changes)
        for idx,source_i,target_i in changes:
            len_sub_string = len(source_i)
            if s[idx:idx + len_sub_string] == source_i:
                for k in range(len_sub_string):
                    final_str[idx + k] = ""
                final_str[idx] = target_i           
        return "".join(final_str)
def run833():
    sol = Solution833()
    s = "vmokgggqzp"
    indices = [3,5,1]
    sources = ["kg","ggq","mo"]
    targets = ["s","so","bfr"]
    res = sol.findReplaceString(s, indices, sources, targets )
    return res
def run1509():
    nums = [9,48,92,48,81,31]
    if len(nums) <= 3:
        return 0
    list_sort = sorted(nums)
    poiter_1 = 3
    poiter_2 = -4
    for i in range(3):
        diff_1 = list_sort[-1] - list_sort[poiter_1]  
        diff_2 = list_sort[poiter_2] - list_sort[0]
        if diff_1 < diff_2:
            list_sort =  list_sort[1:] 
        else:
            list_sort =  list_sort[:-1]
        poiter_1 -=1
        poiter_2 +=1
    return list_sort[-1] - list_sort[0]
class Solution253:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        rooms = []
        heapq.heapify(rooms)
        max_rooms = 0
        intervals.sort()
        for task_ini,task_end in intervals:
            heapq.heappush(rooms,(task_end,task_ini))
            while rooms:
                free_task_end,free_task_ini = heapq.heappop(rooms)
                if free_task_end > task_ini:
                    heapq.heappush(rooms,(free_task_end,free_task_ini))
                    break
            max_rooms = max(max_rooms,len(rooms))
        return max_rooms
def run253():
    intervals = [[0,30],[5,10],[15,20]]
    s= Solution253()
    out = s.minMeetingRooms(intervals)
    return out
import random
class Solution528:
    def __init__(self, w: List[int]):
        if len(w)==0:
            return
        self.N = len(w)
        self.agg_sum= [w[0]]
        for i in range(1,self.N):
            self.agg_sum.append(w[i] + self.agg_sum[i-1])

    def bin_search(self,val,low,high):
        if low == high:
            return low
        mid = (low+high)//2
        if val <= self.agg_sum[mid]:
            if mid ==0 or val > self.agg_sum[mid-1]:
                return mid
            else: 
                return self.bin_search(val,low,mid-1)
        return self.bin_search(val,mid+1,high)  

    def pickIndex(self) -> int:
        if len(self.agg_sum)  == 1:
            return 0
        val= random.uniform(0,self.agg_sum[-1])
        return self.bin_search(val,0,self.N-1)

class Solution554:
    def leastBricks(self, wall: List[List[int]]) -> int:
        edges = defaultdict(lambda:0)
        max_ = 0
        for row in wall:
            intersec = 0
            for i in range(len(row)-1): 
                brick = row[i]
                intersec+= brick
                edges[intersec] +=1
                max_ = max(max_,edges[intersec])
        return len(wall) - max_
def run554():
    inp = [[1,2,2,1],[3,1,2],[1,3,2],[2,4],[3,1,2],[1,3,1,1]]
    out = 2 
    s = Solution554()
    res = s.leastBricks(inp)
    return res == out
def run528():
    s = Solution528([1, 3])
    for _ in range(5):
        res = s.pickIndex()
        print(res)
    return res
class Solution2402:
    def get_next_room(self,start):
        self.time_avaliable = -1 
        self.next_room = -1 
        while len(self.rooms) > 0:
            next_room,time_avaliable = heapq.heappop(self.rooms)
            if time_avaliable <= start:
                self.time_avaliable = time_avaliable
                self.next_room = next_room
                break
            else:
                heapq.heappush(self.rooms_to_add_back,(time_avaliable,next_room))


    def mostBooked(self, n: int, meetings: List[List[int]]) -> int:
        meetings = sorted(meetings)
        self.rooms_used = defaultdict(lambda:0)
        self.rooms = [(i, 0) for i in range(n)]
        heapq.heapify(self.rooms)
        self.rooms_to_add_back =[]
        heapq.heapify(self.rooms_to_add_back)
        for start, end in meetings:
            self.get_next_room(start)
            delay = 0
            if self.next_room >= 0:
                self.time_avaliable = end
            else: 
                next_time_avaliable ,self.next_room = heapq.heappop(self.rooms_to_add_back)
                delay = next_time_avaliable - start
                self.time_avaliable = end + delay 
            self.rooms_used[self.next_room] += 1    
            heapq.heappush(self.rooms,(self.next_room,self.time_avaliable))
            while len(self.rooms_to_add_back):
                t,r = heapq.heappop(self.rooms_to_add_back)
                heapq.heappush(self.rooms,(r,t))
        max_ = 0
        awn = -1
        for i in range(n):
            if self.rooms_used[i] > max_:
                awn = i
                max_ = self.rooms_used[i]
        return awn

def run2402():
    s = Solution2402()
    n = 3
    meetings = [[1,20],[2,10],[3,5],[4,9],[6,8]]
    res = s.mostBooked(n,meetings)  
    print(res)
    return res

if __name__ == "__main__":
    
    res = run2402()
    print(res)