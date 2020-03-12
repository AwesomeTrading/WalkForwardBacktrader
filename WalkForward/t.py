n = 3
# print(range(10))
# for i in range(10):
#     print(i)
a = sum(
    # 1.0 / np.linalg.norm(np.subtract(points[i], points[j]))
    j
    for i in range(n)
    for j in range(n)
    if i > j
)


ia = []
ia.append([])
ia.append([])
c = 0
while c < n:
    for i in range(n):
        for j in range(n):
            if i > j:
                ia[0].append(i)
                ia[1].append(j)
    c += 1

ib = []
i = 0
while i < len(ia[1]):
    # print(ia[0][i], ia[1][i])
    ib.append(ia[0][i])
    ib.append(ia[1][i])
    i += 1
print(sum(ib))
