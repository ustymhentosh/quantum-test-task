def count_islands(grid):
    if not grid or not grid[0]:
        return 0

    n_rows, n_cols = len(grid), len(grid[0])
    
    parent = list(range(n_rows * n_cols))
    rank = [1] * n_rows * n_cols

    def find(n1):
        res = n1
        while res != parent[res]:
            parent[res] = parent[parent[res]]
            res = parent[res]
        return res

    def union(n1, n2):
        p1, p2 = find(n1), find(n2)

        if p1 == p2:
            return False

        if rank[p2] > rank[p1]:
            parent[p1] = p2
            rank[p2] += rank[p1]
        else:
            parent[p2] = p1
            rank[p1] += rank[p2]
        return True

    islands = 0

    for r in range(n_rows):
        for c in range(n_cols):
            if grid[r][c]:
                islands += 1
                
                # check right neighbor
                if c + 1 < n_cols and grid[r][c + 1]:
                    if union(r * n_cols + c, r * n_cols + (c + 1)):
                        islands -= 1
                        
                # check down neighbor
                if r + 1 < n_rows and grid[r + 1][c]:
                    if union(r * n_cols + c, (r + 1) * n_cols + c):
                        islands -= 1
    return islands

grid_1 = [[0, 1, 0], [0, 0, 0], [0, 1, 1]]
grid_2 = [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]]
grid_3 = [[0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 0, 1]]

print(count_islands(grid_1))
print(count_islands(grid_2))
print(count_islands(grid_3))