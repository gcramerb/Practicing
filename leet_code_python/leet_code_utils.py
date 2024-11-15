from typing import List
from collections import deque
def bfs(root,graph):
    visited = set([root])
    to_visit = deque([root])
    while to_visit:
        node = to_visit.popleft()
        # fazer algo com o no.. 
        for neig in graph[node]:
            if neig not in visited:
                to_visit.append(neig)
                visited.add(neig)

def stack_usage():
    s = deque()
    s.append(1)
    s.append(2)
    s.append(3)
    s.append(4)
    while s:
        print(s.pop())
def queue_usage():
    q = deque()
    q.append(1)
    q.append(2)
    q.append(3)
    q.append(4)
    while q:
        print(q.popleft())

    
if __name__ == "__main__":
    print("hello world")
    queue_usage()