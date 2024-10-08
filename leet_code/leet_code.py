from typing import List, Optional
from collections import defaultdict
class Solution1235:
#1235
    def jobScheduling(self,startTime: List[int], endTime: List[int], profit: List[int]) -> int:
        intervals = sorted(zip(startTime,endTime,profit))
        cache = {}
        def dfs(i):
            if i == len(intervals):
                return 0
            if i in cache:
                return cache[i]
            #do not include
            res = dfs(i+1)
            #include
            j = i+1
            while j < len(intervals):
                if intervals[i][1] <= intervals[j][0]:
                    break
                j +=1
            cache[i] = max(res,intervals[i][2]+ dfs(j))
            return cache[i]
        return dfs(0)
    
def run1235():
    s = Solution1235()
    startTime = [1,2,3,4,6]
    endTime = [3,5,10,6,9]
    profit = [20,20,100,70,60]
    res = s.jobScheduling(startTime,endTime,profit)
    print(res)
    return res
#846
def isNStraightHand( hand: List[int], groupSize: int) -> bool:
    list_sorted = sorted(hand)
    N = len(list_sorted)
    if N % groupSize != 0:
        return False
    for i in range(0,N-groupSize,groupSize):
        for j in range(1,groupSize):
            if(list_sorted[i+j] != list_sorted[i] + 1):
                return False
    return True

#648
def replaceWords( dictionary: List[str], sentence: str) -> str:
    ans = ""
    curr_root = ""
    added = False
    for c in sentence:
        if c ==" ":
            if not added:
               ans +=curr_root # adding world without root
            ans += " "
            added = False
            curr_root = ""
            continue
        if added:
            continue
        curr_root +=c
        if curr_root in dictionary:
            ans +=curr_root
            curr_root = ""
            added = True
    if not added:
        ans +=curr_root # adding world without root
    return ans

#1026
from math import inf 
from collections import  deque   
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
def createNode(val):
    if val is None:
        return None
    else:
        return TreeNode(val)
def createTree(node_list) -> TreeNode:
    tree = TreeNode(node_list[0])
    fila = deque()
    add_left = True
    fila.append(tree)
    for i in range(1,len(node_list)):
        if len(fila) == 0:
            break
        val = node_list[i]
        node = createNode(val)
        prev_node = fila.popleft()
        if add_left:
            prev_node.left =  node
            add_left = False
            fila.append(prev_node)
        else:
            prev_node.right =  node
        if node:
            fila.append(node)
    return tree
class Solution1026:
    def printTree(self,node):
        curr_lvl = deque()
        next_lvl = deque()
        curr_lvl.append(node)
        while len(curr_lvl) != 0:
            curr_node = curr_lvl.popleft()
            if curr_node is None:
                print("N   ", end='')
                continue
            print(str(curr_node.val) + "  ", end='')
            next_lvl.append(curr_node.left)
            next_lvl.append(curr_node.right)
            if len(curr_lvl) == 0:
                print("")
                curr_lvl = next_lvl
                next_lvl= deque()
            




    def dfs_min_max(self, node:TreeNode):
        if node is None:
            return -1,inf
        max_sub1, min_sub1 = self.dfs_min_max(node.left)
        max_sub2, min_sub2 = self.dfs_min_max(node.right)
        return max(max_sub1,max_sub2,node.val),min(min_sub1,min_sub2,node.val)
        
    def dfs(self,node:TreeNode):
        if node is None:
            return -1
        max_sub, min_sub = self.dfs_min_max(node)
        curr_max = max(abs(node.val - min_sub),abs(node.val - max_sub))
        return max(curr_max,self.dfs(node.right),self.dfs(node.left))
        

    def maxAncestorDiff(self, root: Optional[TreeNode]) -> int:
        return self.dfs(root)

class Solution1110:
    def dfs(self,node, to_delete,is_root):
        if not node:
            return
        if node.val in to_delete:
            self.dfs(node.left,to_delete,True)
            self.dfs(node.right,to_delete,True)
        else:
            if node.left is not None:
                if node.left.val in to_delete:
                    self.dfs(node.left,to_delete,True)
                    node.left = None
                else:
                    self.dfs(node.left,to_delete,False)
            if node.right is not None:
                if  node.right.val in to_delete:
                    self.dfs(node.right,to_delete,True)
                    node.right = None 
                else:
                    self.dfs(node.right,to_delete,False)
            if is_root:
                self.ans.append(node)

        
    def delNodes(self, root: Optional[TreeNode], to_delete: List[int]) -> List[TreeNode]:
        self.ans = []
        nodes_to_delete = to_delete
        self.dfs(root,nodes_to_delete,True)
        return self.ans
class Solution2101:
    def maximumDetonation(self, bombs: List[List[int]]) -> int:
        N = len(bombs)
        bomb_exp_set = {}
        for i in range(N):
            bomb_exp_set[i] = set()
        for i in range(N):
            for j in range(i,N):
                if i == j:
                    bomb_exp_set[j].add(i)
                    continue
                bi_exp_bj, bj_exp_bi = self.bomb_1exp2(bombs[i],bombs[j])
                if bi_exp_bj:
                    bomb_exp_set[i].add(j)
                if bj_exp_bi:
                    bomb_exp_set[j].add(i)
        max_bomb_exp_count = 0
        for i in range(N):
            to_visit = bomb_exp_set[i].copy()
            while len(to_visit) >0:
                bomb_j = to_visit.pop()
                for k in bomb_exp_set[bomb_j]:
                    if k not in bomb_exp_set[i]:
                        bomb_exp_set[i].add(k)
                        to_visit.add(k)
            max_bomb_exp_count = max(len(bomb_exp_set[i]),max_bomb_exp_count)
        return max_bomb_exp_count
    def bomb_1exp2(self,bomb1,bomb2):
        bombs_dist = pow(pow(bomb1[0] -bomb2[0],2)  + pow(bomb1[1] -bomb2[1],2),.5)
        b1_b2 = False
        b2_b1 = False
        if bomb1[2] >= bombs_dist:
            b1_b2 = True
        if bomb2[2] >= bombs_dist:
            b2_b1 = True
        return b1_b2, b2_b1
class Solution945:
    def minIncrementForUnique(self, nums: List[int]) -> int:
        """
        the input has a set of unique part and another with duplicates
        get the maximun value of the part wwith 
        does not make sense to increment on the part that is already unique.
        """
        num_changes = 0
        nums = sorted(nums)
        num_i = nums[0]
        for i in range(1,len(nums)):
            num_i = nums[i]
            if nums[i-1] < num_i:
                continue
            diff = 0
            if nums[i-1] > num_i:
                diff = nums[i-1] - num_i
            num_i = num_i +1 + diff
            num_changes  = num_changes + 1 + diff
            nums[i] = num_i
        return num_changes
def run945():
    s= Solution945()
    res = s.minIncrementForUnique([1,2,3,1,2])
    #res = s.maximumDetonation([[1,2,3],[2,3,1],[3,4,2],[4,5,3],[5,6,4]])
    print(res)

class Solution2013:

    def __init__(self):
        self.map = {}
        

    def add(self, point: List[int]) -> None:
        point_tuple = (point[0],point[1])
        if point_tuple in self.map.keys():
           self.map[point_tuple] +=1
        else:
            self.map[point_tuple]  = 1
    def get_point_possibilities(self,square_side,point,point_2):
        possibilities = [ ] 
        point_3 = (point[0] + square_side ,point[1])
        point_4 = (point[0] + square_side ,point_2[1])
        possibilities.append((point_3,point_4))
        point_3 = (point[0] - square_side ,point[1])
        point_4 = (point[0] - square_side ,point_2[1])
        possibilities.append((point_3,point_4))
        return possibilities


    def count(self, point: List[int]) -> int:
        point_tuple = (point[0],point[1])
        count = 0
        for point_2 in self.map.keys():
            if point_2[0] == point_tuple[0]: # x value coordenate
                square_side = abs(point_2[1] - point[1])
                if square_side == 0:
                    continue
                possibilities = self.get_point_possibilities(square_side,point,point_2)
                for point_possible in possibilities:
                    if point_possible[0] in self.map.keys() and point_possible[1] in self.map.keys():
                        count += self.map[point_2] *self.map[point_possible[0]] * self.map[point_possible[1]]
        return count
def run2013():
    s = Solution2013()
class Solution1136:
    def __init__(self):
        self.map = {}
        self.graph = {}
        self.max_length = 0
    def getGraph(self, n,relations):
        self.graph = {i:[] for i in range(1,n+1)}
        for n1,n2 in relations:
            self.graph[n1].append(n2)   
    def dfs(self,node_i,visited):
        visited.add(node_i)
        if node_i in self.map.keys():
            return self.map[node_i]
        max_val = 0
        for node_son in self.graph[node_i]:
            if node_son in visited and node_son not in self.map.keys():
                return -1
            val = self.dfs(node_son,visited)
            if  val<0:
                return -1
            max_val = max(val,max_val)
        self.map[node_i] = max_val+1
        self.max_length = max(max_val+1,self.max_length)
        return max_val+1
    def minimumSemesters(self, n: int, relations: List[List[int]]) -> int:
        self.getGraph(n,relations)
        for i in range(1,n+1):
            visited = set()
            ans = self.dfs(i,visited)
            if ans < 0:
                return -1
        return self.max_length
def run1136():
    s = Solution1136()
    N = 3
    graph_list = [[1,2],[2,3],[3,1]]
    ans = s.minimumSemesters(N,graph_list)    
    print(ans)
class Solution1101:
    def earliestAcq(self, logs: List[List[int]], n: int) -> int:
        logs_sorted = sorted(logs)
        groups = {}
        for relation in logs_sorted:
            tmp,f1,f2 = relation
            set_f1 = set()
            set_f2 = set()
            if f1 in groups.keys():
                set_f1 = groups[f1]
            if f2 in groups.keys():
                set_f2 = groups[f2]
            curr_set = set_f1.union(set_f2)
            curr_set.add(f1)
            curr_set.add(f2)
            if(len(curr_set) == n):
                return tmp
            for k in curr_set:
                groups[k] = curr_set
        return -1
def run1101():
    s = Solution1101()
    n  = 6
    logs = [[20190101,0,1],[20190104,3,4],[20190107,2,3],[20190211,1,5],[20190224,2,4],[20190301,0,3],[20190312,1,2],[20190322,4,5]] 
    ans = s.earliestAcq(logs,n)
    print(ans)
class Solution2863:
    def maxSubarrayLength(self, nums: List[int]) -> int:
        new_array = [(nums[i],i) for i in range(len(nums))]
        new_array = sorted(new_array,reverse=True)
        long_size = 0
        min_index = len(nums) +1 
        for num,old_idx in new_array:
            min_index = min(min_index,old_idx)
            long_size = max(long_size, old_idx- min_index )
        return long_size +1 
def run2863():
    nums= [57,55,50,60,61,58,63,59,64,60,63]
    s = Solution2863()
    ans = s.maxSubarrayLength(nums)
    print(ans)
class Solution2316:
    def init_data_strc(self, num_nodes):
        self.num_nodes = num_nodes
        self.parent = list(range(num_nodes))
    def union(self,x,y):
        if self.parent[x] == self.parent[y]: return
        parent_x = self.find(x)
        parent_y = self.find(y)
        self.parent[parent_x] = parent_y

    def find(self,i):     
        if self.parent[i] == i:
            return i 
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i] 

    def countPairs(self, n: int, edges: List[List[int]]) -> int:
        count = 0
        self.init_data_strc(n)
        for x,y in edges:
            self.union(x,y)
        size = [0] * n
        for node in range(n):
            p = self.find(node)
            size[p] += 1
        res = 0
        for node in range(n):
            p = self.find(node)
            res += n - size[p]
        return res // 2
def run2316():
    s = Solution2316()
    s.countPairs(20,[[13,3],[10,1],[6,2],[7,8],[15,0],[0,2],[9,1],[7,11],[3,0],[3,5],[2,7],[6,17],[12,11],[6,16],[3,4],[14,9],[1,0],[18,2],[1,19]])
if __name__ == "__main__":
    #s = Solution1026()
    #s = Solution1110()
    #null = None
    #root = createTree([1,2,3,4,5,6,7])
    #root = s.createTree([8,3,10,1,6,null,14,null,null,4,7,13])
    #s.printTree(root)
    #res = s.delNodes(root,[3,5])
    run2316()
