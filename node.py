# 使用networkX计算节点的pagerank
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# 创建有向图
G = nx.DiGraph()   
# 设置有向图的边集合
edges = [("A", "B"), ("A", "E"),("A", "D"), ("A", "F"), ("B", "C"), ("C", "E"), ("D", "C"), ("D", "A"), ("D", "E"),("E", "C"),("E", "B"),("F", "D")]
# 在有向图G中添加边集合
for edge in edges:
    G.add_edge(edge[0], edge[1])

# 有向图可视化
layout = nx.spring_layout(G)
nx.draw(G, pos=layout, with_labels=True)
plt.show()
# 计算简化模型的PR值
pr1 = nx.pagerank(G, alpha=1)
print("简化模型的PR值：", pr1)
a = np.array([[0, 0, 0, 1/3, 0,  0], 
			[1/4, 0, 0,  0,  1/2,0],
			[0,   1, 0, 1/3, 1/2,0],
			[1/4, 0, 0, 0,   0,  1],
            [1/4, 0, 1, 1/3, 0,  0],
            [1/4, 0, 0, 0,   0,  0]])

b = np.array([x for x in pr1.values()])

w = b
#简化模型迭代100次
def work(a, w):
	for i in range(100):
		w = np.dot(a, w)
		print(w)


if __name__ == '__main__':
    work(a, w)
	#random_work(a_leak, w, 4)
	#random_work(a, w, 4)
	
#print(work(a, w))
#print(random_work(a, w, 4))
#random_work(a_leak, w, 4)
#random_work(a_sink, w, 4)
pr2 = nx.pagerank(G, alpha=0.8)
print("随机模型的PR值：",pr2)

#随机模型迭代100次
b2 = np.array([x for x in pr1.values()])
w2=b2
def random_work(a, w2, n):
	d = 0.85
	for i in range(100):
		w = (1-d)/n + d*np.dot(pr2, w)
		print(w)
