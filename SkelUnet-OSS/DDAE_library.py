#!/usr/bin/env python
# coding: utf-8

# ##### All the code necessary  for Deep-Denoising Auto-Encoder 

# In[1]:


import numpy 
import matplotlib.pyplot as plt
from matplotlib import image

from sklearn.neighbors import KDTree
import networkx as nx
nx.__version__ # should be 2.1

from shapely.geometry import Polygon, Point, LineString, LinearRing, MultiPolygon

import numpy.linalg as LA
from queue import PriorityQueue

import csv


# In[2]:


def get_vertices(input_map):
    
    '''from a thini map save skeleton poits as vertex of a graph\n
    input_map: a binary image of 64x64px'''

    vertex=[]#empty list
    col=0
    for i in input_map:
        row=0
        for j in i:
            if j==0.0:
                vertex.append((row,col)) #col row
            row+=1
        col+=1
    return vertex


# In[3]:


def sample_vertices(vertex, n):
    
    '''return n samples from list vertexes \n
    n: number of samples to return'''
    samples=[]
    for i in range(n):
        indexes=numpy.random.randint(0,len(vertex),1)
#         print(indexes)
        samples.append(vertex[int(indexes)])
        vertex.pop(int(indexes))
        
    return samples #tiene el problema que pude devolver muestras repetidas, arreglado 110622


# In[4]:


def search_corners(img_map):
    
    '''Input: map .png of 64x64 px\n
    Output: array of pair with xy-coordinates\nv2.0 220822'''
    map_size=(64-1,64-1)
    corners=[]
#     corners_x=[]
#     corners_y=[]
    row=0
    for i in img_map:# i iteration on rows
        col=0
#         ref=img_map[row,col]

        for j in i: #j iteration on columns


            if j==0.0: #busca esquinas exteriores a los obstaculos

                if (row==0 and col==0) or (row==0 and col==map_size[1]) or (row==map_size[0] and col==0) or (row==map_size[0] and col==map_size[1]):
                    corners.append((row,col))

                if (col==0 or col==map_size[1]) and 0<row<map_size[0]:
                    if img_map[row-1,col]==1.0 or img_map[row+1,col]==1.0:
                        corners.append((row,col))
#                         corners_x.append(col)
#                         corners_y.append(row)

                if (row==0 or row==map_size[0]) and 0<col<map_size[1]:
                    if img_map[row,col-1]==1.0 or img_map[row,col+1]==1.0:
                        corners.append((row,col))
#                         corners_x.append(col)
#                         corners_y.append(row)

                if 0<row<map_size[0] and 0<col<map_size[1]:

                    P1=img_map[row-1,col-1]
                    P2=img_map[row-1,col]
                    P3=img_map[row-1,col+1]
                    P4=img_map[row,col-1]
                    P6=img_map[row,col+1]
                    P7=img_map[row+1,col-1]
                    P8=img_map[row+1,col]
                    P9=img_map[row+1,col+1]

                    if (P6==0.0 and P8==0.0) and (P1==1.0 and P2==1.0 and P4==1.0):
                        corners.append((row,col))
#                         corners_x.append(col)
#                         corners_y.append(row)
                    if (P4==0.0 and P8==0.0) and (P2==1.0 and P3==1.0 and P6==1.0):
                        corners.append((row,col))
#                         corners_x.append(col)
#                         corners_y.append(row)
                    if (P2==0.0 and P4==0.0) and (P6==1.0 and P8==1.0 and P9==1.0):
                        corners.append((row,col))
#                         corners_x.append(col)
#                         corners_y.append(row)
                    if (P2==0.0 and P6==0.0) and (P4==1.0 and P7==1.0 and P8==1.0):
                        corners.append((row,col))
#                         corners_x.append(col)
#                         corners_y.append(row)

                    if (P6==0.0 and P8==0.0) and (P9==1.0):
                        corners.append((row,col))
#                         corners_x.append(col)
#                         corners_y.append(row)
                    if (P4==0.0 and P8==0.0) and (P7==1.0):
                        corners.append((row,col))
#                         corners_x.append(col)
#                         corners_y.append(row)
                    if (P2==0.0 and P4==0.0) and (P1==1.0):
                        corners.append((row,col))
#                         corners_x.append(col)
#                         corners_y.append(row)
                    if (P2==0.0 and P6==0.0) and (P3==1.0):
                        corners.append((row,col))
#                         corners_x.append(col)
#                         corners_y.append(row)

            col+=1
        row+=1

#     print('(rows,columns)\n',corners)    
    return corners
    


# In[5]:


def check_function_v2(ref,e,H_seek,V_seek,img_map):
    '''v 2.0 220822'''
    
    #check for the correct direction
    check_test=False
    left_column=False
    right_column=False
    up_rows=False
    bottom_rows=False
    change_up=False
    change_bottom=False
    change_left=False
    change_right=False
    limit_row=False
    limit_column=False

    if H_seek==True:
        
#         print('H')
        
        if e[0]-1<0 or e[0]+1>63:
            limit_row=True
        else:
            limit_row=False
            
        if ref[1]<e[1]:#move forward to left
            
            if e[0]-1>=0: 
                for p in range(e[1]-ref[1]):#220822  
                    if p==1:
                        ref_point=img_map[e[0]-1,e[1]-p]
                    elif p>1 and img_map[e[0]-1,e[1]-p]!=ref_point and change_up==False:
                        change_up=True
                        
                    if img_map[e[0]-1,e[1]-p]==1.0 or img_map[e[0]-1,e[1]]==1.0 or img_map[e[0]-1,ref[1]]==1.0:# and p>1: 220822    
                        up_rows=True # at some point there are an occuped pixel 
                        
#                 if img_map[e[0]-1,e[1]]!=img_map[e[0]-1,ref[1]] and change_up==False: #ask for the limits
#                     change_up=True
 
            if e[0]+1<=63:
#                 print('rev bottom')
                for p in range(e[1]-ref[1]):
                    if p==1:
                        ref_point=img_map[e[0]+1,e[1]-p]
#                         print('ref_point:',ref_point)
                    elif p>1 and img_map[e[0]+1,e[1]-p]!=ref_point and change_bottom==False:
#                         print('change bottom')
                        change_bottom=True
                
                    if img_map[e[0]+1,e[1]-p]==1.0 or img_map[e[0]+1,e[1]]==1.0 or img_map[e[0]+1,ref[1]]==1.0:#and p>1:
                        bottom_rows=True
                        
#                 if img_map[e[0]+1,e[1]]!=img_map[e[0]+1,ref[1]] and change_up==False: #ask for the limits
#                     change_up=True

                                                
        elif ref[1]>e[1]:#move forward to right

            if e[0]-1>=0:
                for p in range(ref[1]-e[1]):
                    if p==1:
                        ref_point=img_map[e[0]-1,e[1]+p]
                    elif p>1 and img_map[e[0]-1,e[1]+p]!=ref_point and change_up==False:
                        change_up=True
                    
                    if img_map[e[0]-1,e[1]+p]==1.0 or img_map[e[0]-1,e[1]]==1.0 or img_map[e[0]-1,ref[1]]==1.0: #and p>1:
                        up_rows=True
                
#                 if img_map[e[0]-1,e[1]]!=img_map[e[0]-1,ref[1]] and change_up==False: #ask for the limits
#                     change_up=True

                        
            if e[0]+1<=63:
                for p in range(ref[1]-e[1]):
                    if p==1:
                        ref_point=img_map[e[0]+1,e[1]+p]
                    elif p>1 and img_map[e[0]+1,e[1]+p]!=ref_point and change_bottom==False:
                        change_bottom=True
                        
                    if img_map[e[0]+1,e[1]+p]==1.0 or img_map[e[0]+1,e[1]]==1.0 or img_map[e[0]+1,ref[1]]==1.0:#and p>1:
                        bottom_rows=True
                
#                 if img_map[e[0]+1,e[1]]!=img_map[e[0]+1,ref[1]] and change_up==False: #ask for the limits
#                     change_up=True

        
#         print('up_rows,bottom_rows,change_up,change_bottom,limit_row:',up_rows,bottom_rows,change_up,change_bottom,limit_row)
        if ((up_rows==True and bottom_rows==True) or (up_rows==False and bottom_rows==False)) and limit_row==False:
            check_test==False
        elif ((up_rows==True and bottom_rows==False) or (up_rows==False and bottom_rows==True)) and change_up==False and change_bottom==False: #and maybe dont be the border
            check_test=True
        elif (bottom_rows==False or up_rows==False) and limit_row==True and change_up==False and change_bottom==False:
            check_test=True

# *********************************************************************************************************************
                        
    elif V_seek==True:

#         print('V')
        if e[1]-1<0 or e[1]+1>63:
            limit_column=True
        else:
            limit_column=False

                    
        if ref[0]<e[0]:#move forward to down
#             print('r<e')
                                    
            if e[1]-1>=0:
#                 print('check left column')
                for p in range(e[0]-ref[0]):
                    if p==1:
                        ref_point=img_map[e[0]-p,e[1]-1]
                    elif p>1 and img_map[e[0]-p,e[1]-1]!=ref_point and change_left==False:
                        change_left=True
                        
                    if img_map[e[0]-p,e[1]-1]==1.0 or img_map[e[0],e[1]-1]==1.0 or img_map[ref[0],e[1]-1]==1.0:#and p>1:
                        left_column=True #find an one 

            if e[1]+1<=63:
#                 print('check right column')
                for p in range(e[0]-ref[0]):
                    if p==1:
                        ref_point=img_map[e[0]-p,e[1]+1]
                    elif p>1 and img_map[e[0]-p,e[1]+1]!=ref_point and change_right==False:
                        change_right=True

                    if img_map[e[0]-p,e[1]+1]==1.0 or img_map[e[0],e[1]+1]==1.0 or img_map[ref[0],e[1]+1]==1.0:#and p>1:
                        right_column=True
        
        elif ref[0]>e[0]:#move forward to up
#             print('r>e')
                        
            if e[1]-1>=0:                
                for p in range(ref[0]-e[0]):
                    if p==1:
                        ref_point=img_map[e[0]+p,e[1]-1]
                    elif p>1 and img_map[e[0]+p,e[1]-1]!=ref_point and change_left==False:
                        change_left=True
                    
                    if img_map[e[0]+p,e[1]-1]==1.0 or img_map[e[0],e[1]-1]==1.0 or img_map[ref[0],e[1]-1]==1.0:#and p>1:
                        left_column=True #find an one 
                        
            if e[1]+1<=63:                
                for p in range(ref[0]-e[0]):
                    if p==1:
                        ref_point=img_map[e[0]+p,e[1]+1]
                    elif p>1 and img_map[e[0]+p,e[1]+1]!=ref_point and change_right==False:
                        change_right=True

                    if img_map[e[0]+p,e[1]+1]==1.0 or img_map[e[0],e[1]+1]==1.0 or img_map[ref[0],e[1]+1]==1.0:#and p>1:
                        right_column=True

#         print('left_column,right_columns,change_left,change_right,limit_column:',left_column,right_column,change_left,change_right,limit_column)
        if ((left_column==True and right_column==True) or (left_column==False and right_column==False)) and limit_column==False:
            check_test==False
        elif ((left_column==True and right_column==False) or (left_column==False and right_column==True)) and change_left==False and change_right==False: #and maybe dont be the border
            check_test=True
        elif (left_column==False or right_column==False) and limit_column==True and change_up==False and change_bottom==False:
            check_test=True

    return check_test


# In[6]:


def make_polygon_v2(corners2,img_map, sample):
    '''Reorder corners to form a polygon\n v 2.0 220822'''    
    p=0
    all_polygons=[]
    break_while=False
    while len(corners2)>0 and break_while==False: #until the list is empty
        max_corners=len(corners2)
        p+=1
        free_polygon=[]
        H_seek=True
        V_seek=False 
        found=False
        check_test=False
        ref=corners2[0]
        free_polygon.append(ref)
        origin=corners2[0]
        for n in range(max_corners):#iterate the points 
            cota=63
            for e in corners2:
                if H_seek==True and ref[0]==e[0] and ref[1]!=e[1]:
                    dif=abs(ref[1]-e[1])
                    check_test=check_function_v2(ref,e,H_seek, V_seek,img_map)######check function
                    if dif<=cota and check_test==True:
                        cota=dif
                        hopeful=e
                        found=True
                elif V_seek==True and ref[1]==e[1] and ref[0]!=e[0]:
                    dif=abs(ref[0]-e[0])
                    check_test=check_function_v2(ref,e,H_seek, V_seek, img_map)######check function
                    if dif<=cota and check_test==True:
                        cota=dif
                        hopeful=e
                        found=True
                else:
                    pass
            
            if found==True:
                free_polygon.append(hopeful)
                corners2.remove(hopeful)
                ref=hopeful
                if V_seek==True:
                    H_seek=True
                    V_seek=False
                elif H_seek==True:
                    V_seek=True
                    H_seek=False
                found=False
            else:
                break_while=True
            if free_polygon.count(origin)==2:
#                 print('closed polygon')
                break     
        all_polygons.append(free_polygon)
        
        if break_while==True:                 
            print('CORNER NOT FOUND -ERROR! map:'+str(sample))
        
    return all_polygons


# In[7]:


def can_connect(p1, p2, polygon_list):
    '''220622 v1.0 (from example)'''
    y1,x1 = p1
    y2,x2 = p2
    
    line = LineString([(x1, y1), (x2, y2)])
    free = True
    
    for p in polygon_list:
        if line.crosses(p):
            free = False
    
    return free


# In[8]:


def graph(nodes, polygon_list):
#     nodes=muestras
    nodes.sort()
    m_tree=KDTree(numpy.array(nodes))#change after test
    g = nx.Graph()
    k=6#6#4#for some maps needs to be less
    for mil in nodes:
        mil_array = numpy.array(mil)
        distances, indexes = m_tree.query([mil_array], k) # find the nearest k milestones
        for dist, idx in zip(distances[0], indexes[0]):
            if dist > 0:
                if can_connect(mil, nodes[int(idx)], polygon_list):
                    g.add_edge(mil, nodes[int(idx)], weight=dist)#if is not itself
    return g


# In[9]:


def heuristic(n1, n2):
    # TODO: finish
    return LA.norm(numpy.array(n2) - numpy.array(n1))


# In[10]:


def a_star(graph, heuristic, start, goal):
    """Modified A* to work with NetworkX graphs."""
    
    path = []
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False
    
    while not queue.empty():
        item = queue.get()
        current_cost = item[0]
        current_node = item[1]
        
        if current_node == goal:
#             print('Found a path.')
            found = True
            break
            
        else:

            for next_node in graph[current_node]:
                cost = graph.edges[current_node, next_node]['weight']
                new_cost = current_cost + cost + heuristic(next_node, goal)
                    
                if next_node not in visited:
                    visited.add(next_node)          

                    queue.put((new_cost, next_node))
                    
                    branch[next_node] = (new_cost, current_node)
             
    path = []
    path_cost = 0
    
    if found:
        # retrace steps
        path = []
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
            
    return path[::-1], path_cost


# In[11]:


def random_map_point(img_map):
    
    '''Return a grap (networkx) with a new valid random node  \n
    Input:map, graph points\n
    Output:graph'''
    
    new_node=numpy.random.randint(low=0,high=64,size=2,dtype=int)
    
    while (img_map[new_node[1],new_node[0]]==1.0):
        new_node=numpy.random.randint(low=0,high=64,size=2,dtype=int)
    
            
    return new_node


# In[12]:


def add_to_graph(g,m_tree,point,nodes,polygon_list):
    '''add node and edge to graph if it is possible
    add nodes and polygon_list input on octuber 31/2022'''
    
    dst, idx = m_tree.query([point], 1)
    nearest_node=nodes[idx[0][0]]
    add=False
    
    if can_connect(point,nearest_node,polygon_list):
#         print('add:',point)
        g.add_edge(tuple(point), nearest_node, weight=dst)
        add=True
    else:
#         print('node do not add to graph')
        add=False
    return add, g

