# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
### Chern number

import numpy as np
from numpy import sqrt, cos, sin
import matplotlib.pyplot as mp
import time

def normalized_dot_product(v1, v2):
    import numpy as np
    product = np.dot(np.conj(v1).T, v2)
    return product / np.abs(product)


def Calcurate_line_curl(line_1, line_2):
    #line [점][0:eigvelue, 1:eigvector][energy_index]
    import numpy as np
    energy_num = len(line_1[0][0])
    result = [] # [E_0, E_1, E_2...]
    for E in range(energy_num): #고유벡터 갯수 = 에너지 스테이트 갯수
        curl = []
        for i in range(len(line_1)-1):
            temp = np.log(
                         normalized_dot_product(line_1[i][1][:][E], line_1[i+1][1][:][E])* #아래 가로
                         normalized_dot_product(line_1[i+1][1][:][E], line_2[i+1][1][:][E])* #우측 세로
                         normalized_dot_product(line_2[i+1][1][:][E], line_2[i][1][:][E])* # 상단 가로
                         normalized_dot_product(line_2[i][1][:][E], line_1[i][1][:][E]) #좌측 세로            
                            )
#             temp = np.log(#line_n [위치][eigvector(2x2)] [:][E] 열벡터 
#                          normalized_dot_product(line_1[i][1][:][E], line_1[i+1][1][:][E])* #아래 가로
#                          normalized_dot_product(line_1[i+1][1][:][E], line_2[i+1][1][:][E])/ #우측 세로
#                          normalized_dot_product(line_2[i][1][:][E], line_2[i+1][1][:][E])/ # 상단 가로
#                          normalized_dot_product(line_1[i][1][:][E], line_2[i][1][:][E]) #좌측 세로            
#                             )
            temp = -np.imag(temp)
            #print(normalized_dot_product(line_1[i][1][:][E], line_1[i+1][1][:][E]))
            curl.append(temp)
            
        result.append(curl)
    return result

def Calcurate_line_eigval(line_1, line_2):
    #line [점][0:eigvelue, 1:eigvector][energy_index]
    import numpy as np
    energy_num = len(line_1[0][0])
    result = [] # [E_0, E_1, E_2...]
    for E in range(energy_num): #고유벡터 갯수 = 에너지 스테이트 갯수
        curl = []
        
        for i in range(len(line_1)-1):
            temp = line_1[i][0][E]
            curl.append(temp)
            
        result.append(curl)
    return result

def Pauli_Ham (h0, hx, hy, hz):  # 2x2 Hamiltonian matrix by pauli matrix
    #pauli
    from numpy import array
    sig_0 = array([[1,0],[0,1]])
    sig_x = array([[0,1],[1,0]])
    sig_y = array([[0,-1j],[1j,0]])
    sig_z = array([[1,0],[0,-1]])
    H = sig_0*h0 + sig_x*hx + sig_y*hy + sig_z*hz
    return H

def Haldane_Graphene (M, t1, t2, phi, kx, ky):
    from numpy import sqrt, sin, cos, array
    from numpy.linalg import eigh, eig
    a = [0] #a[0]  Nearest Hoping
    a.append(array([sqrt(3)/2,1/2+1]))  #a[1]
    a.append(array([-sqrt(3)/2,1/2+1])) #a[2]
    a.append(array([0,-1+1]))           #a[3]
    b = [0] # Next Neareast Hoping
    b.append(array([sqrt(3)/2,3/2])) #b[1]
    b.append(array([sqrt(3)/2,-3/2])) #b[2]
    b.append(array([sqrt(3),0]))      #b[3]
    h_0, h_x, h_y, h_z = 0, 0, 0, M
    for i in range(1,4):
            h_0 += 2*t2*cos(phi)*(cos(kx*b[i][0]+ky*b[i][1]))
            h_x += t1*(cos(kx*a[i][0]+ky*a[i][1]))
            h_y += -t1*(sin(kx*a[i][0]+ky*a[i][1]))
            h_z += -2*t2*sin(phi)*(sin(kx*b[i][0]+ky*b[i][1]))
    H = Pauli_Ham(h_0, h_x, h_y, h_z)
    
    eigval, eigvec = eig(H)
    solted_indices = np.argsort(eigval)
    eigval = eigval[solted_indices]
    eigvec = eigvec[solted_indices]
    result = [eigval, eigvec]
    return result


tic = time.time()

M = 0
t1 = 1
t2 = 0.1
phi = np.pi/2
n = 2
kx = np.linspace(-1*np.pi, 1*np.pi, 100)
ky = np.linspace(-1*np.pi, 1*np.pi, 100)
KX, KY = np.meshgrid(kx, ky)

brillouin_basis = [0] #brillouin zone basis
brillouin_basis.append(2*np.pi*np.array([0, 2/3]))
brillouin_basis.append(2*np.pi*np.array([1/np.sqrt(3), -1/3]))

# eigenvalues = np.zeros((n, len(kx), len(ky)), dtype=complex)

# for ii, i in enumerate(ky):
#     for jj, j in enumerate(kx):
#          eigenvalues[:, jj, ii],_ = Haldane_Graphene(M,t1, t2, phi, i, j)

# Boundary condition to make a cell in Brillouin zone
lower_bound = np.array([0,0])
##### 참고용 ##### 
higher_bound = brillouin_basis[1]+brillouin_basis[2]

Num = 100
ALPHAS, BETAS = np.linspace(0,1, Num, endpoint = True), np.linspace(-0.0001, 1, Num, endpoint = False)

# # Condition
# condition = (k_grid @ brillouin_basis[1] >= lower_bound @ brillouin_basis[1]) & (k_grid @ brillouin_basis[1] <= higher_bound @ brillouin_basis[1]) & \
#             (k_grid @ brillouin_basis[2] >= lower_bound @ brillouin_basis[2]) & (k_grid @ brillouin_basis[2] <= higher_bound @ brillouin_basis[2])



n=2
field_strength = np.full((n, len(ALPHAS)-1, len(BETAS)-1), np.nan)
field_strength_eigval = np.full((n, len(ALPHAS)-1, len(BETAS)-1), np.nan)
line_eig=[]

for line_index, i in enumerate(ALPHAS):
    tmp_line_eig = []
    
    for j in BETAS:
        now_kx, now_ky = i*brillouin_basis[1]+j*brillouin_basis[2]
        tmpvec = Haldane_Graphene(M,t1, t2, phi, now_kx, now_ky)
        tmp_line_eig.append(tmpvec)# 현재 포인트의 결과를 추가
    
    line_eig.append(tmp_line_eig)  
    
    if line_index != 0 and line_index != len(ALPHAS)-1:
        field_strength[:,:,line_index] = Calcurate_line_curl(line_eig[0], line_eig[1])# 두 라인을 계산하는 함수
        field_strength_eigval[:,:,line_index] = Calcurate_line_eigval(line_eig[0], line_eig[1])# 두 라인을 계산하는 함수
        line_eig.pop(0)

ALPHAS, BETAS = np.meshgrid(ALPHAS, BETAS)

ALPHAS = ALPHAS[..., np.newaxis]* brillouin_basis[1]
BETAS = BETAS[..., np.newaxis]* brillouin_basis[2]

Chern_space = ALPHAS + BETAS + lower_bound
fig = mp.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')


#ax = fig.add_subplot(111, projection='3d')
#ax.plot_surface(KX[:-1, :-1], KY[:-1, :-1], field_strength[0, :, :], cmap= 'cool')
#ax.plot_surface(KX[:-1, :-1], KY[:-1, :-1], field_strength[1, :, :], cmap='spring')
ax.plot_surface(Chern_space[..., 0][:-1, :-1], Chern_space[..., 1][:-1, :-1], field_strength[0, :, :], cmap= 'cool')
ax.plot_surface(Chern_space[..., 0][:-1, :-1], Chern_space[..., 1][:-1, :-1], field_strength_eigval[0, :], cmap= 'cool')
#ax.plot_surface(Chern_space[..., 0][:-1, :-1], Chern_space[..., 1][:-1, :-1], field_strength_eigval[1, :], cmap= 'cool')
#mp.scatter(Chern_space[..., 0][:-1, :-1], Chern_space[..., 1][:-1, :-1], c=field_strength[1, :, :]*500, cmap = 'cool')
#mp,xlim()

#ax.plot_surface(KX, KY, field_strength[1, :, :], cmap = 'cool')

ax.set_xlabel('K_x')
ax.set_ylabel('K_y')
ax.set_zlabel('F.S')
ax.axis('equal')
ax.view_init(90,-90)
mp.title('Haldane model - Graphene')
mp.show()

#fig = mp.figure(figsize=(8,8))
#ax.plot_surface(KX, KY, eigenvalues[0], cmap = 'cool')
#ax.plot_surface(KX, KY, eigenvalues[1], cmap = 'cool')
#ax.view_init(90,-90)
#mp.scatter(Chern_space[..., 0], Chern_space[..., 1], color= 'red', s=0.1)
#mp.axis('equal')
print(1*np.nansum(field_strength[0][:])+0*np.nansum(field_strength[1]))

print(time.time() - tic)




# +
# ### Chern number

# import numpy as np
# from numpy import sqrt, cos, sin
# import matplotlib.pyplot as mp
# import time

# def normalized_dot_product(v1, v2):
#     import numpy as np
#     product = np.dot(np.conj(v1).T, v2)
#     return product / np.abs(product)
#     #return product


# def Calcurate_line_curl(point1, point2, point3, point4):
#     import numpy as np
#     energy_num = len(point1[0])
#     result = [] # [E_0, E_1, E_2...]
    
#     for E in range(energy_num): #고유벡터 갯수 = 에너지 스테이트 갯수
        
#         temp = np.log(
#                          normalized_dot_product(point1[1][E], point2[1][E])* #아래 가로
#                          normalized_dot_product(point2[1][E], point3[1][E])* #우측 세로
#                          normalized_dot_product(point3[1][E], point4[1][E])* # 상단 가로
#                          normalized_dot_product(point4[1][E], point1[1][E]) #좌측 세로            
#                             )
#         #print(normalized_dot_product(point2[1][E], point3[1][E]), point2[1][E], point3[1][E])
#         #print(point1[1][E], point2[1][E], point3[1][E], point4[1][E], '\n')
#         temp = -np.imag(temp)
#         result.append(temp)
        
#     return result

# def Calcurate_line_eigval(line_1, line_2):
#     #line [점][0:eigvelue, 1:eigvector][energy_index]
#     import numpy as np
#     energy_num = len(line_1[0][0])
#     result = [] # [E_0, E_1, E_2...]
#     for E in range(energy_num): #고유벡터 갯수 = 에너지 스테이트 갯수
#         curl = []
        
#         for i in range(len(line_1)-1):
#             temp = line_1[i][0][E]
#             curl.append(temp)
            
#         result.append(curl)
#     return result

# def Pauli_Ham (h0, hx, hy, hz):  # 2x2 Hamiltonian matrix by pauli matrix
#     #pauli
#     from numpy import array
#     sig_0 = array([[1,0],[0,1]])
#     sig_x = array([[0,1],[1,0]])
#     sig_y = array([[0,-1j],[1j,0]])
#     sig_z = array([[1,0],[0,-1]])
#     H = sig_0*h0 + sig_x*hx + sig_y*hy + sig_z*hz
#     return H

# def Haldane_Graphene (M, t1, t2, phi, kx, ky):
#     from numpy import sqrt, sin, cos, array
#     from numpy.linalg import eigh, eig
#     a = [0] #a[0]  Nearest Hoping
#     a.append(array([sqrt(3)/2,1/2+1]))  #a[1]
#     a.append(array([-sqrt(3)/2,1/2+1])) #a[2]
#     a.append(array([0,-1+1]))           #a[3]
#     b = [0] # Next Neareast Hoping
#     b.append(array([sqrt(3)/2,3/2])) #b[1]
#     b.append(array([sqrt(3)/2,-3/2])) #b[2]
#     b.append(array([sqrt(3),0]))      #b[3]
#     h_0, h_x, h_y, h_z = 0, 0, 0, M
#     for i in range(1,4):
#             h_0 += 2*t2*cos(phi)*(cos(kx*b[i][0]+ky*b[i][1]))
#             h_x += t1*(cos(kx*a[i][0]+ky*a[i][1]))
#             #h_y += -t1*(sin(kx*a[i][0]+ky*a[i][1]))
#             h_y += t1*(sin(kx*a[i][0]+ky*a[i][1]))
#             #h_z += M + 2*t2*sin(phi)*(sin(kx*b[i][0]+ky*b[i][1]))
#             h_z +=  -2*t2*sin(phi)*(sin(kx*b[i][0]+ky*b[i][1]))
#     H = Pauli_Ham(h_0, h_x, h_y, h_z)
    
#     eigval, eigvec = eig(H)
#     solted_indices = np.argsort(eigval)
#     eigval = eigval[solted_indices]
#     eigvec = eigvec[solted_indices]
#     result = [eigval, eigvec]
#     return result


# tic = time.time()

# M0 = 0.00
# t1 = 1.
# t2 = 0.1
# phi = np.pi/2
# n = 2

# Num = 10

# kx = np.linspace(-1*np.pi, 1*np.pi, 100)
# ky = np.linspace(-1*np.pi, 1*np.pi, 100)
# KX, KY = np.meshgrid(kx, ky)

# brillouin_basis = [0] #brillouin zone basis
# brillouin_basis.append(2*np.pi*np.array([0, 2/3]))
# brillouin_basis.append(2*np.pi*np.array([1/np.sqrt(3), -1/3]))

# # Boundary condition to make a cell in Brillouin zone
# lower_bound = np.array([0,0])

# ALPHAS1, BETAS1 = np.linspace(0,1, Num, endpoint = True), np.linspace(0., 1., Num, endpoint = True)

# field_strength = np.full((n, len(ALPHAS1)-1, len(BETAS1)-1), np.nan)
# field_strength_eigval = np.full((n, len(ALPHAS)-1, len(BETAS)-1), np.nan)

# ALPHAS1, BETAS1 = np.meshgrid(ALPHAS1, BETAS1)

# ALPHAS = ALPHAS1[..., np.newaxis]* brillouin_basis[1]
# BETAS = BETAS1[..., np.newaxis]* brillouin_basis[2]
# Chern_space = ALPHAS + BETAS + lower_bound
# mp.scatter(Chern_space[...,0], Chern_space[...,1]) #형태 확인
# Chern_space_x = Chern_space[..., 0]
# Chern_space_y = Chern_space[..., 1]

# for i in range(len(Chern_space[:-1])):
#     for j in range(len(Chern_space[0][:-1])):
#         now_kpoint1 = Chern_space[i, j]
#         now_kpoint2 = Chern_space[i+1,j]
#         now_kpoint3 = Chern_space[i+1,j+1]
#         now_kpoint4 = Chern_space[i,j+1]
#         #print(now_kpoint2, now_kpoint3)
        
#         point1 = Haldane_Graphene(M0, t1, t2, phi, now_kpoint1[0], now_kpoint1[1])
#         point2 = Haldane_Graphene(M0, t1, t2, phi, now_kpoint2[0], now_kpoint2[1])
#         point3 = Haldane_Graphene(M0, t1, t2, phi, now_kpoint3[0], now_kpoint3[1])
#         point4 = Haldane_Graphene(M0, t1, t2, phi, now_kpoint4[0], now_kpoint4[1])
# #        print(np.dot(point1[1][0], point4[1][0]))
#         field_strength[:, i, j] = Calcurate_line_curl(point1, point2, point3, point4)
        

        


# fig = mp.figure(figsize=(8,8))
# ax = fig.add_subplot(111, projection='3d')

# ax.plot_surface(Chern_space[..., 0][:-1, :-1], Chern_space[..., 1][:-1, :-1], field_strength[0, :, :], cmap= 'cool')

# ax.set_xlabel('K_x')
# ax.set_ylabel('K_y')
# ax.set_zlabel('F.S')
# ax.axis('equal')
# ax.view_init(30,60)
# #ax.set_zlim(-0.1,0.1)
# mp.title('Haldane model - Graphene')
# mp.show()

# print(1*np.nansum(field_strength[1])+0*np.nansum(field_strength[0]))

# print(time.time() - tic)



