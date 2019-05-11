#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
my_arr = np.arange(1000000)
my_list = list(range(1000000))
get_ipython().run_line_magic('time', 'for _ in range(10): my_arr2 = my_arr * 2')


# In[2]:


get_ipython().run_line_magic('time', 'for _ in range(10): my_list2 = [x * 2 for x in my_list]')


# In[5]:


data = np.random.randn(2, 3)
print(data)


# In[6]:


print(data * 10)


# In[7]:


print(data + data)


# In[8]:


data.shape


# In[9]:


data.dtype


# In[10]:


data1 = [6, 7.5, 8, 0, 1]
arr1 = np.array(data1)
print(arr1)


# In[11]:


data2 = [[1, 2, 3, 4], [5, 6, 7, 8]]
arr2 = np.array(data2)
print(arr2)


# In[12]:


arr2.ndim


# In[13]:


arr2.shape


# In[14]:


arr1.dtype


# In[15]:


arr2.dtype


# In[16]:


np.zeros(10)


# In[18]:


np.zeros((3, 6))


# In[20]:


np.empty((2, 3, 2))


# In[21]:


np.arange(15)


# In[22]:


arr1 = np.array([1, 2, 3], dtype=np.float64)
arr2 = np.array([1, 2, 3], dtype=np.int32)
arr1.dtype


# In[23]:


arr2.dtype


# In[24]:


arr = np.array([1, 2, 3, 4, 5])
arr.dtype


# In[26]:


float_arr = arr.astype(np.float64)
float_arr.dtype


# In[32]:


arr = np.array([3.7, -1.2, -2.6, 0.5, 12.9, 10.1])
arr


# In[28]:


arr.astype(np.int32)


# In[33]:


numeric_strings = np.array(['1.25', '-9.6', '42'], dtype = np.string_)
numeric_strings.astype(float)


# In[34]:


int_array = np.arange(10)
calibers = np.array([.22, .270, .357, .380, .44, .50], dtype = np.float64)
int_array.astype(calibers.dtype)


# In[35]:


empty_uint32 = np.empty(8, dtype='u4')
empty_uint32


# In[36]:


arr = np.array([[1., 2., 3.], [4., 5., 6]])
arr


# In[37]:


arr*arr


# In[38]:


arr - arr


# In[39]:


1/arr


# In[40]:


arr ** 0.5


# In[41]:


arr2 = np.array([[0., 4., 1.],[7., 2., 12.]])
arr2


# In[42]:


arr2 > arr1


# In[43]:


arr = np.arange(10)
arr


# In[44]:


arr[5]


# In[45]:


arr[5:8]


# In[47]:


arr[5:8]=12
arr


# In[50]:


arr_slice = arr[5:8]
arr_slice


# In[51]:


arr_slice


# In[52]:


arr_slice[1]=12345
arr


# In[53]:


arr_slice[:] = 64
arr


# In[55]:


arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
arr2d[2]


# In[56]:


arr2d[0][2]


# In[57]:


arr2d[0, 2]


# In[58]:


arr3d = np.array([[[1, 2, 3], [4, 5, 6]],[[7, 8, 9], [10, 11, 12]]])
arr3d


# In[59]:


arr3d[0]


# In[60]:


old_values = arr3d[0].copy()
arr3d[0] = 42
arr3d


# In[61]:


arr3d[0] = old_values
arr3d


# In[62]:


arr3d[1, 0]


# In[63]:


x = arr3d[1]
x


# In[64]:


x[0]


# In[65]:


arr


# In[66]:


arr[1:6]


# In[67]:


arr2d


# In[68]:


arr2d[:2]


# In[69]:


arr2d[:2, 1:]


# In[70]:


arr2d[1, :2]


# In[71]:


arr2d[:2, 2]


# In[72]:


arr2d[:, :1]


# In[73]:


arr2d[:2, 1:] = 0
arr2d


# In[81]:


names = np.array(['Bob', 'Joe', 'Will','Bob', 'Will', 'Joe', 'Joe'])
data = np.random.randn(7, 4)
names


# In[82]:


data


# In[83]:


names == 'Bob'


# In[84]:


data[names == 'Bob']


# In[85]:


data[names == 'Bob', 2:]


# In[86]:


data[names == 'Bob', 3]


# In[87]:


names != 'Bob'


# In[88]:


data[~(names == 'Bob')]


# In[89]:


cond = names == 'Bob'
data[~cond]


# In[91]:


mask = (names == 'Bob')|(names == 'Will')
mask


# In[92]:


data[mask]


# In[93]:


data[data<0]=0
data


# In[94]:


data[names != 'Joe'] = 7
data


# In[95]:


arr = np.empty((8, 4))
for i in range(8):
    arr[i] = i
arr


# In[96]:


arr[[4, 3, 0, 6]]


# In[97]:


arr[[-3, -5, -7]]


# In[98]:


arr = np.arange(32).reshape((8, 4))
arr


# In[99]:


arr[[1, 5, 7, 2], [0, 3, 1, 2]]


# In[100]:


arr[[1, 5, 7, 2]][:, [0, 3, 1, 2]]


# In[101]:


arr = np.arange(15).reshape((3, 5))
arr


# In[102]:


arr.T


# In[103]:


arr = np.random.randn(6, 3)
arr


# In[104]:


np.dot(arr.T, arr)


# In[105]:


arr = np.arange(16).reshape((2, 2, 4))
arr


# In[106]:


arr.transpose((1, 0, 2))


# In[107]:


arr


# In[108]:


arr.swapaxes(1, 2)


# In[109]:


arr = np.arange(10)


# In[110]:


arr


# In[111]:


np.sqrt(arr)


# In[112]:


np.exp(arr)


# In[113]:


x = np.random.randn(8)
y = np.random.randn(8)
x


# In[114]:


y


# In[115]:


np.maximum(x, y)


# In[116]:


arr = np.random.randn(7)*5
arr


# In[117]:


remainder, whole_part = np.modf(arr)
remainder


# In[118]:


whole_part


# In[119]:


arr


# In[121]:


np.sqrt(arr)


# In[122]:


np.sqrt(arr, arr)


# In[123]:


arr


# In[124]:


points = np.arange(-5, 5, 0.01)
xs, ys = np.meshgrid(points, points)
ys


# In[125]:


z = np.sqrt(xs ** 2 + ys ** 2)
z


# In[134]:


import matplotlib.pyplot as plt
plt.imshow(z, cmap=plt.cm.gray)
plt.title('image plot of $\sqrt{x^2 + y^2}$ for a grid of values')
plt.colorbar()


# In[135]:


plt.title('image plot of $\sqrt{x^2 + y^2}$ for a grid of values')  # 这里没有办法合并，只有将title移到上面才可以


# In[136]:


xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])
result = [(x if c else y)
         for x, y, c in zip(xarr, yarr, cond)]
result


# In[137]:


result = np.where(cond, xarr, yarr)
result


# In[138]:


arr = np.random.randn(4, 4)
arr


# In[139]:


arr > 0


# In[140]:


np.where(arr > 0, 2, -2)


# In[141]:


np.where(arr> 0, 2, arr)


# In[142]:


arr = np.random.randn(5, 4)
arr


# In[143]:


arr.mean()


# In[144]:


np.mean(arr)


# In[145]:


arr.sum()


# In[146]:


arr.mean(axis = 1)


# In[147]:


arr.sum(axis = 0)


# In[149]:


arr = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
arr


# In[150]:


arr.cumsum(axis = 0)


# In[151]:


arr.cumprod(axis = 1)


# In[152]:


arr = np.random.randn(100)
(arr > 0).sum()


# In[153]:


bools = np.array([False, False, True, False])
bools.any()


# In[154]:


bools.all()


# In[155]:


arr = np.random.randn(6)
arr


# In[156]:


arr.sort()


# In[157]:


arr


# In[158]:


arr = np.random.randn(5, 3)
arr


# In[159]:


arr.sort()
arr


# In[160]:


large_arr = np.random.randn(1000)
large_arr.sort()
large_arr[int(0.05*len(large_arr))]


# In[161]:


names = np.array(['Bob', 'Joe', 'Will','Bob', 'Will', 'Joe', 'Joe'])
np.unique(names)


# In[162]:


ints = np.array([3, 3, 3, 2, 2, 1, 1, 4, 4])
np.unique(ints)


# In[163]:


sorted(set(names))


# In[164]:


values = np.array([6, 0, 0, 3, 2, 5, 6])
np.in1d(values, [2, 3, 6])


# In[166]:


arr = np.arange(10)
np.save('some_array', arr)
np.load('some_array.npy')


# In[172]:


np.savez('array_archive.npz', a=arr, b=arr)


# In[173]:


arch = np.load('array_archive.npz')
arch['b']


# In[174]:


x = np.array([[1., 2., 3.], [4., 5., 6.]])
y = np.array([[6., 23.], [-1, 7], [8, 9]])
x


# In[175]:


y


# In[176]:


x.dot(y)


# In[177]:


np.dot(x, y)


# In[178]:


np.dot(x, np.ones(3))


# In[179]:


x @ np.ones(3)


# In[180]:


from numpy.linalg import inv, qr
X = np.random.randn(5, 5)
mat = X.T.dot(X)
inv(mat)  #求逆


# In[181]:


mat.dot(inv(mat))


# In[182]:


q, r = qr(mat)
r


# In[183]:


samples = np.random.normal(size=(4, 4))
samples


# In[184]:


from random import normalvariate
N = 1000000
get_ipython().run_line_magic('timeit', 'samples = [normalvariate(0, 1) for _ in range(N)]')


# In[190]:


get_ipython().run_line_magic('timeit', 'np.random.normal(size=N)')
np.random.seed(1234)
rng = np.random.RandomState(1234)
rng.randn(10)


# In[191]:


import random
position = 0
walk = [position]
steps = 1000
for i in range(steps):
    step = 1 if random.randint(0, 1) else -1
    position += step
    walk.append(position)
plt.plot(walk[:100])


# In[193]:


nsteps = 1000
draws = np.random.randint(0, 2, size=nsteps)
steps = np.where(draws > 0, 1, -1)
walk = steps.cumsum()
walk.min()


# In[194]:


walk.max()


# In[195]:


(np.abs(walk) >= 10).argmax()


# In[196]:


nwalks = 5000
nsteps = 1000
draws = np.random.randint(0, 2, size = (nwalks, nsteps))
steps = np.where(draws > 0, 1, -1)
walks = steps.cumsum(1)
walks


# In[197]:


walks.max()


# In[198]:


walks.min()


# In[200]:


hits30 = (np.abs(walks) >= 30).any(1)
hits30


# In[201]:


hits30.sum()


# In[203]:


crossing_times = (np.abs(walks[hits30]) >= 30).argmax(1)
crossing_times.mean()


# In[204]:


steps = np.random.normal(loc = 0, scale = 0.25, size = (nwalks, nsteps))

