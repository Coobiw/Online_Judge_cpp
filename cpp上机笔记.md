# STL的一些应用
## STL sort（#include<algorithm>)
sort(起始地址，终止地址，排序方式）

排序方式应该是一个返回bool型的排序函数的函数名（如果为true，则比较函数第一个参数会排在第二个参数前面，默认值default是升序方式）

	比如对a[n]排序
	sort(a,a+n)
## STL vector（#include<vector>)
### 初始化
C++的初始化方法很多，各种初始化方法有一些不同。

(1): vector<int> ilist1;

默认初始化，vector为空， size为0，表明容器中没有元素，而且
capacity 也返回 0，意味着还没有分配内存空间。这种初始化方式适用于元素个数未知，需要在程序中动态添加的情况。

(2): vector<int> ilist2(ilist);

vector<int> ilist2  = ilist; 

两种方式等价 ，ilist2 初始化为ilist 的拷贝，ilist必须与ilist2 类型相同，也就是同为int的vector类型，ilist2将具有和ilist相同的容量和元素

(3): vector<int> ilist = {1,2,3.0,4,5,6,7};

 vector<int> ilist {1,2,3.0,4,5,6,7};

ilist 初始化为列表中元素的拷贝，列表中元素必须与ilist的元素类型相容，本例中必须是与整数类型相容的类型，整形会直接拷贝，其他类型会进行类型转换。

(4): vector<int> ilist4(7);

默认值初始化，**ilist4中将包含7个元素**，每个元素进行缺省的值初始化，对于int，也就是被赋值为0，因此ilist4被初始化为包含7个0。当程序运行初期元素大致数量可预知，而元素的值需要动态获取的时候，可采用这种初始化方式。

(5):vector<int> ilist5(number,value);

指定值初始化，ilist5被初始化为**包含number个值为value的int**

### 一些方法
1. push_back(）在尾部添加一个元素
2. pop_back()从尾部删除一个元素
3. vector.begin(),vector.end()，返回开头/结尾的迭代器（指针），用sort排序的时候要这么用，不能直接用vector名
4. vector.size()和vector.length()返回长度
5. vector.insert(pos,number,value)
6. vector.erase(pos,number)

## STL stack(#include<stack>)
### 初始化
	stack<tyname> name
### 方法
	stack.empty()
	stack.size()
	stack.push()栈顶进
	stack.pop()栈顶出
	stack.top()获得栈顶元素
## STL queue（#include<queue>)
### 初始化
	queue<typename> name
### 方法
	queue.empty()
	queue.size()
	queue.push()队尾进
	queue.pop()队头出
	queue.front()获得头部元素
	queue.back()获得尾部元素
## STL priority_queue
### 初始化
	priority_queue<int> name # 大根堆
	priority_queue<int,vector<int>,greater<int>> # 小根堆
	priority_queue<Type, Container, Functional>

>Type 就是数据类型，Container 就是容器类型（Container必须是用数组实现的容器，比如vector,deque等等，但不能用 list。STL里面默认用的是vector），Functional 就是比较的方式，当需要用自定义的数据类型时才需要传入这三个参数，使用基本数据类型时，只需要传入数据类型，默认是大顶堆

### 方法
>- top 访问队头元素
>- empty 队列是否为空
>- size 返回队列内元素个数
>- push 插入元素到队尾 (并排序)
>- emplace 原地构造一个元素并插入队列
>- pop 弹出队头元素
## STL string(#include<string>,#include<cstring>)
### 一、初始化
1、默认初始化

	string s; //s是一个空串

2.使用字符串字面值初始化

	string s1=“hello world”; //拷贝初始化
	string s2(“hello world”); //直接初始化
	注意：s1、s2的内容不包括’\0’

3.使用其他字符串初始化

	string s2=s1; //拷贝初始化，s1是string类对象
	string s2(s1); //直接初始化，s1是string类对象

4.使用单个字符初始化

	string s(10, ‘a’); //直接初始化，s的内容是aaaaaaaaaa
### 二、方法
1.insert

	s.insert(pos,str)//在s的pos位置插入str
	s.insert(s.it,ch)在s的it指向位置前面插入一个字符ch，返回新插入的位置的迭代器

2.erase

	
	int main ()
	{
	  std::string str ("This is an example sentence.");
	  std::cout << str << '\n';
	                          // "This is an example sentence."
	  str.erase (10,8);       //            ^^^^^^^^
	  //直接指定删除的字符串位置第十个后面的8个字符
	  std::cout << str << '\n';
	                            // "This is an sentence."
	  str.erase (str.begin()+9);//           ^
	  //删除迭代器指向的字符
	  std::cout << str << '\n';
	                            // "This is a sentence."
	                            //       ^^^^^
	  str.erase (str.begin()+5, str.end()-9);
	  //删除迭代器范围的字符
	  std::cout << str << '\n';
	                            // "This sentence."
	  return 0;
	}

3.substr(pos,n)
	
	int main()
	{
	    ios::sync_with_stdio(false);
	    string s="abcdefg";
	
	    //s.substr(pos1,n)返回字符串位置为pos1后面的n个字符组成的串
	    string s2=s.substr(1,5);//bcdef
	
	    //s.substr(pos)//得到一个pos到结尾的串
	    string s3=s.substr(4);//efg
	
	    return 0;
	}

4.find(substring)

**find函数主要是查找一个字符串是否在调用的字符串中出现过，大小写敏感。**

**如果查找不到，返回std::npos.**

## STL map(#include<map>)
1.声明一个散列表映射： 

map<key的数据类型，value的数据类型> 映射表名

2.状态判断

.empty() 和.size()

3.元素增添和删除

增加：insert(pair<数据类型，数据类型>(key,value))

删除：erase(key)

4.访问：

①mymap[key] ②mymap.at(key) ③迭代器

5.元素操作

清空：clear()

查找：

①find(key) 

若找到则返回该元素迭代器（=指针），否则返回迭代器end()

②count(key)

count()方法返回值是一个整数，1表示有这个元素，0表示没有这个元素。

6.迭代器

	for(it = mymap.begin();it!=mymap.end();it++)
	访问key: it->first
	访问value: it->second
## STL functional里的function
### 概述
类模版std::function是一种通用、多态的函数封装。std::function的实例可以对**任何可以调用的目标实体**进行存储、复制、和调用操作，这些目标实体**包括普通函数、Lambda表达式、函数指针、以及其它函数对象等**。std::function对象是对C++中现有的可调用实体的一种类型安全的包裹（我们知道像函数指针这类可调用实体，是类型不安全的）。

通常std::function是一个函数对象类，它包装其它任意的函数对象，被包装的函数对象具有类型为T1, …,TN的N个参数，并且返回一个可转换到R类型的值。std::function使用 模板转换构造函数接收被包装的函数对象；特别是，闭包类型可以隐式地转换为std::function。

简单概述：通过std::function对C++中各种可调用实体（普通函数、Lambda表达式、函数指针、仿函数以及其它函数对象等）的封装，形成一个新的可调用的、统一的std::function对象。

### leetcode常用
	function<返回值类型(形参类型)> 函数名字 = [&](形参类型 形参名字)
	{
	
	}；（这里要有；）
# 算法上机应用笔记

## Hash Table哈希表（散列表）
### 定义：
散列表（Hash table，也叫哈希表），是**根据键（Key）**而**直接访问**在内存储存位置的数据结构。也就是说，它通过计算一个关于键值的函数，将所需查询的数据映射到表中一个位置来访问记录，这加快了查找速度。这个映射函数称做散列函数，存放记录的数组称做散列表。

散列函数：将关键字key映射为关键字所在地址key address。
即Hash(key) = Address。

### c++ map
#### 定义
相当于python里的字典，即建立key和value的字典，方便使用

#### STL map
1.声明一个散列表映射： 

map<key的数据类型，value的数据类型> 映射表名

2.状态判断

.empty() 和.size()

3.元素增添和删除

增加：insert(pair<数据类型，数据类型>(key,value))

删除：erase(key)

4.访问：

①mymap[key] ②mymap.at(key) ③迭代器

5.元素操作

清空：clear()

查找：

①find(key) 

若找到则返回该元素迭代器（=指针），否则返回迭代器end()

②count(key)

count()方法返回值是一个整数，1表示有这个元素，0表示没有这个元素。

6.迭代器

	for(it = mymap.begin();it!=mymap.end();it++)
	访问key: it->first
	访问value: it->second


#### 应用1：计数（数组中元素出现个数）
#### 应用2：前缀和+hash表 解决中缀和为某个值的个数
**前缀和：不包括当前索引的元素！是当前索引元素之前！**

一边建hash_map一边求结果！

sum[j]-sum[i] = goal，就可以用j做遍历求i(**因为nums里的数只有0和1，前缀和随索引是单调递增的，先遍历大的再遍历小的，小的已经存到hash_map里了就可以直接查找了！**）


	class Solution {
	public:
	    int numSubarraysWithSum(vector<int>& nums, int goal) {
	           int num[nums.size()+1];
	           memset(num,0,(nums.size()+1)*sizeof(int));
	
	           int i,j;
	           int sum = 0;
	           int result = 0;
	           for(j=0;j<nums.size();j++)
	           {
	               num[sum]+=1;
	               sum+=nums[j];
	               if(sum<goal)continue;
	               else
	               {
	                   result += num[sum-goal];
	               }
	           }
	           return result;
	           
	    }
	};


## 摩尔投票（Boyer-Moore 投票算法）
### 介绍
Boyer-Moore 投票算法的基本思想是：在每一轮投票过程中，从数组中删除两个不同的元素，直到投票过程无法继续，此时数组为空或者数组中剩下的元素都相等。

如果数组为空，则数组中不存在主要元素；

如果数组中剩下的元素都相等，则数组中剩下的元素可能为主要元素。

**Boyer-Moore 投票算法的步骤如下：**

维护一个候选主要元素candidate 和候选主要元素的出现次数count，初始时 candidate 为任意值，count=0；

遍历数组 nums 中的所有元素，遍历到元素 xx 时，进行如下操作：

如果 count=0，则将 xx 的值赋给 candidate，否则不更新 candidate 的值；

如果 x=candidate，则将 count 加 1，否则将 count 减 1。

**遍历结束之后，如果数组 nums 中存在主要元素，则candidate 即为主要元素，否则candidate 可能为数组中的任意一个元素。**

**由于不一定存在主要元素，因此需要第二次遍历数组**

验证 candidate 是否为主要元素。第二次遍历时，统计candidate 在数组中的出现次数，如果出现次数大于数组长度的一半，则 candidate 是主要元素，返回 canndidate，否则数组中不存在主要元素，返回 -1。

为什么当数组中存在主要元素时，Boyer-Moore 投票算法可以确保得到主要元素？

在 Boyer-Moore 投票算法中，遇到相同的数则将 count 加 1，遇到不同的数则将 count 减 1。根据主要元素的定义，主要元素的出现次数大于其他元素的出现次数之和，因此在遍历过程中，主要元素和其他元素两两抵消，最后一定剩下至少一个主要元素，此时 candidate 为主要元素，且 count≥1。

时间复杂度O(n) 空间复杂度O(1)

### 应用
求众数，求major number

	class Solution {
	public:
	    int majorityElement(vector<int>& nums) {
	        int major_num = nums[0];
	        int major_freq = 1;
	        for(int i=1;i<nums.size();i++)
	        {
	            if(nums[i]==major_num)major_freq++;
	            else
	            {
	                major_freq--;
	                if(major_freq<0)
	                {
	                    major_num = nums[i];
	                    major_freq = 1;
	                }
	            }
	        }
	        bool check;
	        int counter=0;
	        if(major_freq>=1)
	        {
	            for(int i=0;i<nums.size();i++)
	            {
	                if(nums[i]==major_num)counter++;
	            }
	            if(counter>(nums.size()-1)/2)return major_num;  
	        }
	        return -1;
	    }
	};

## 贪心策略 Greedy Strategy 

### 思想：
**总是选择当前状态下最优的策略，而不顾对后续产生的影响。**

**贪心选择：每一步贪心选出来的一定是原问题的最优解的一部分**

**最优子结构：每一步贪心完后会留下子问题，子问题的最优解与贪心选择出来的解可以合成原问题的最优解**

*应用方法：从初态开始考虑，进行贪心选择*

### 应用条件：

常用于求解最优化问题：但容易得到局部最优解

如果要得到全局最优：要求**具有最优子结构**

即每步贪心选择都满足**无后效性**：从前的状态不会影响之后的状态，之后的状态只由当前状态决定

**（类似于马尔可夫过程）**

### 应用1：简单贪心（类似于找钱问题）
	例题：夏日炎炎，小男孩 Tony 想买一些雪糕消消暑。
	
	商店中新到 n 支雪糕，用长度为 n 的数组 costs 表示雪糕的定价，其中 costs[i] 表示第 i 支雪糕的现金价格。Tony 一共有 coins 现金可以用于消费，他想要买尽可能多的雪糕。
	
	给你价格数组 costs 和现金量 coins ，请你计算并返回 Tony 用 coins 现金能够买到的雪糕的 最大数量 。
	
	注意：Tony 可以按任意顺序购买雪糕。

**贪心策略：每次买最便宜的，就可以买最多！**

代码：

	class Solution {
	public:
	    int maxIceCream(vector<int>& costs, int coins) {
	        priority_queue<int,vector<int>,greater<int>> my_queue;
	        for(int i=0;i<costs.size();i++)
	        {
	            my_queue.push(costs[i]);
	        }
	        int num=0;
	        while(!my_queue.empty())
	        {
	            coins -= my_queue.top();
	            my_queue.pop();
	            if(coins<0)break;
	            num++;
	        }
	        return num;
	    }
	};

### 应用2：区间贪心（类似于节目安排（最晚结束贪心））
#### 区间贪心问题描述
它是指当有多个不同的区间存在，且这些区间有可能相互重叠时，如何才能从众多区间中，选取**最多**的**两两互不相交**的区间。

## 图论
### 邻接矩阵表示法
vertex[n]表示结点

edge[n][n]表示边

题目中的一种代码：[https://leetcode-cn.com/problems/chuan-di-xin-xi/](https://leetcode-cn.com/problems/chuan-di-xin-xi/)

		int edge[n][n];
        int i;
        //for(i=0;i<n;i++)
        //   memset(edge[i],0,sizeof(int)*n);
        memset(edge,0,sizeof(int)*n*n);
        for(i=0;i<relation.size();i++)
        {
            edge[relation[i][0]][relation[i][1]] = 1;
        }

### 邻接表表示法（用向量！）
代码：

	vector<vector<int>> edges(n);
	        for (auto &edge : relation) {
	            int src = edge[0], dst = edge[1];
	            edges[src].push_back(dst);
	        }

### BFS（队列+visited数组）
注意：如果规定了走的步数（深度） 用BFS的话，可以一次pop出一层，push进下一层！（**每次对队列中元素个数size进行遍历**）

**这样队列中存放的一直是同一层次的元素**，代码如下：

	class Solution {
	public:
	    int numWays(int n, vector<vector<int>> &relation, int k) {
	        vector<vector<int>> edges(n);
	        for (auto &edge : relation) {
	            int src = edge[0], dst = edge[1];
	            edges[src].push_back(dst);
	        }
	
	        int steps = 0;
	        queue<int> que;
	        que.push(0);
	        while (!que.empty() && steps < k) {
	            steps++;
	            int size = que.size();
	            for (int i = 0; i < size; i++) {
	                int index = que.front();
	                que.pop();
	                for (auto &nextIndex : edges[index]) {
	                    que.push(nextIndex);
	                }
	            }
	        }
	
	        int ways = 0;
	        if (steps == k) {
	            while (!que.empty()) {
	                if (que.front() == n - 1) {
	                    ways++;
	                }
	                que.pop();
	            }
	        }
	        return ways;
	    }
	};

### DFS（栈+visited数组 / 递归（如果涉及到步数，建议递归））
递归实现含步数的DFS:

	class Solution {
	public:
	    int numWays(int n, vector<vector<int>>& relation, int k) {
	        int edge[n][n];
	        int i;
	        //for(i=0;i<n;i++)
	        //   memset(edge[i],0,sizeof(int)*n);
	        memset(edge,0,sizeof(int)*n*n);
	        for(i=0;i<relation.size();i++)
	        {
	            edge[relation[i][0]][relation[i][1]] = 1;
	        }
	        int answer=0;
	        function<void(int, int)> dfs = [&](int index, int steps) 
	        {
	            if(steps==k)
	            {
	                if(index == n-1) ++answer;
	                return;
	            }
	            for(int i=0;i<n;i++)
	            {
	                if(edge[index][i]==1)dfs(i,steps+1);
	            }
	
	        };
	
	        dfs(0,0);
	        return answer;  
	    }
	};



## 动态规划 Dynamic Programming
### 适用
分解成子问题，子问题之间不独立（如果独立直接递归分治了），存在重复运算，所以用dp数组来记录运算结果，避免重复运算

**是一种空间换时间的算法策略**

### 子数组和子序列的定义
子序列和子数组
子序列：由数组中不连续的元素组成的数组。eg：[1, 2, 3, 4]子序列可为[1, 4]

子数组：数组中连续元素组成的数组。eg：[1, 2, 3, 4]子数组可以为[1, 2, 3]
### 应用1：求最大子数组和
**题目描述：**

输入一个整型数组，数组中的一个或连续多个整数组成一个子数组。求所有子数组的和的最大值。

要求时间复杂度为O(n)。

**代码：**

	class Solution {
	public:
	    int maxSubArray(vector<int>& nums) {
	        int n=nums.size();
	        int dp[n]; //dp[i]代表以第i个数为结尾的最大子数组和
	        dp[0]=nums[0];
	        int maxium = dp[0];
	        int i;
	        for(i=1;i<n;i++)
	        {
	            dp[i]=max(dp[i-1]+nums[i],nums[i]);
	            if(dp[i]>maxium)maxium=dp[i];
	        }
	        return maxium;
	    }
	};
### 应用2：最长公共子序列
#### 问题描述与动态规划递推规律分析
问题描述：字符序列的子序列是指从给定字符序列中随意地（不一定连续）去掉若干个字符（可能一个也不去掉）后所形成的字符序列。令给定的字符序列X=“x0，x1，…，xm-1”，序列Y=“y0，y1，…，yk-1”是X的子序列，存在X的一个严格递增下标序列<i0，i1，…，ik-1>，使得对所有的j=0，1，…，k-1，有xij=yj。例如，X=“ABCBDAB”，Y=“BCDB”是X的一个子序列。

考虑最长公共子序列问题如何分解成子问题，设A=“a0，a1，…，am-1”，B=“b0，b1，…，bn-1”，并Z=“z0，z1，…，zk-1”为它们的最长公共子序列。不难证明有以下性质：

（1） 如果am-1=bn-1，**则**zk-1=am-1=bn-1，**且“z0，z1，…，zk-2”是“a0，a1，…，am-2”和“b0，b1，…，bn-2”的一个最长公共子序列**；

（2） 如果am-1!=bn-1，则若zk-1!=am-1，蕴涵“z0，z1，…，zk-1”是“a0，a1，…，am-2”和“b0，b1，…，bn-1”的一个最长公共子序列；

（3） 如果am-1!=bn-1，则若zk-1!=bn-1，蕴涵“z0，z1，…，zk-1”是“a0，a1，…，am-1”和“b0，b1，…，bn-2”的一个最长公共子序列。

这样，在找A和B的公共子序列时，如有am-1=bn-1，则进一步需要解决一个子问题，找“a0，a1，…，am-2”和“b0，b1，…，bm-2”的一个最长公共子序列；

如果am-1!=bn-1，则要解决两个子问题，找出“a0，a1，…，am-2”和“b0，b1，…，bn-1”的一个最长公共子序列和找出“a0，a1，…，am-1”和“b0，b1，…，bn-2”的一个最长公共子序列，再取两者中**较长者**作为A和B的最长公共子序列。

#### 公式总结
![](../1.png)

#### cpp代码
	class Solution {
	public:
	    int minOperations(vector<int>& target, vector<int>& arr) {
	        int m=target.size();
	        int n=arr.size();
	
	        int i,j;
	        int dp[m+1][n+1];
	        for(i=0;i<=m;i++)
	        {
	            for(j=0;j<=n;j++)
	            {
	                if(i==0 or j==0)dp[i][j]=0;
	                else if(target[i-1]==arr[j-1])
	                {
	                    dp[i][j]=dp[i-1][j-1]+1;
	                }
	                else
	                {
	                    dp[i][j]=max(dp[i-1][j],dp[i][j-1]);
	                }
	            }
	        }
	        std::cout<<dp[m][n]<<endl;
	        return m-dp[m][n];
	    }
	};

### 应用3：最长回文子序列
#### 回文子序列性质
1. 对于一个子序列而言，如果它是回文子序列，并且长度大于 22，那么将它首尾的两个字符去除之后，它仍然是个回文子序列。

2. 最长回文子序列问题可以转化为**逆序字符串与原字符串的最长相同子序列问题**

#### 根据性质2，直接编程实现
#### 根据性质1，进行动态规划分析
![](../3.png)

#### 动态规划cpp代码
	class Solution {
	public:
	    int longestPalindromeSubseq(string s) {
	        int n = s.length();
	        vector<vector<int>> dp(n, vector<int>(n));
	        for (int i = n - 1; i >= 0; i--) {
	            dp[i][i] = 1;
	            char c1 = s[i];
	            for (int j = i + 1; j < n; j++) {
	                char c2 = s[j];
	                if (c1 == c2) {
	                    dp[i][j] = dp[i + 1][j - 1] + 2;
	                } else {
	                    dp[i][j] = max(dp[i + 1][j], dp[i][j - 1]);
	                }
	            }
	        }
	        return dp[0][n - 1];
	    }
	};



### 应用4：等差数列划分（Leetcode）
#### 题目描述：
如果一个数列**至少有三个元素**，并且任意两个相邻元素之差相同，则称该数列为等差数列。

例如，[1,3,5,7,9]、[7,7,7,7] 和 [3,-1,-5,-9] 都是等差数列。
给你一个整数数组 nums ，返回数组 nums 中所有为等差数组的**子数组**个数。

**子数组是数组中的一个连续序列。**

 

示例 1：

输入：nums = [1,2,3,4]
输出：3
解释：nums 中有三个子等差数组：[1, 2, 3]、[2, 3, 4] 和 [1,2,3,4] 自身。

示例 2：

输入：nums = [1]
输出：0

#### 动态规划分析
**暴力方法的思路与算法**

考虑一个比较直观的做法：

我们枚举等差数列的最后两项nums[i−1] 以及nums[i]，那么等差数列的公差d即为 nums[i−1]−nums[i]；

随后我们使用一个指针 j 从 i−2 开始逆序地遍历数组的前缀部分 nums[0..i−2]：

如果 nums[j]−nums[j+1]=d，那么说明 nums[j],⋯,nums[i] 组成了一个长度至少为 3 的等差数列，答案增加 1；

否则更小的 j 也无法作为等差数列的首个位置了，我们直接退出遍历。

这个做法的时间复杂度是 O(n^2) 的，即枚举最后两项的时间复杂度为 O(n)，使用指针 j 遍历的时间复杂度也为 O(n)，相乘得到总时间复杂度 O(n^2)。对于一些运行较慢的语言，该方法可能会超出时间限制，因此我们需要进行优化。

**优化**

如果我们已经求出了 nums[i−1] 以及 nums[i] 作为等差数列的最后两项时，答案增加的次数 ti，那么能否快速地求出t(i+1)呢？

答案是可以的：

如果 nums[i]−nums[i+1]=d，那么在这一轮遍历中，j 会遍历到与上一轮相同的位置，答案增加的次数相同，并且额外多出了nums[i−1],nums[i],nums[i+1] 这一个等差数列，因此有：**t_{i+1} = t_i + 1**

如果 nums[i]−num[i+1] !=d，那么 j 从初始值 i−1 开始就会直接退出遍历，答案不会增加，因此有：**t_{i+1} = 0**


这样一来，我们通过上述简单的递推，即可在 O(1) 的时间计算等差数列数量的增量，总时间复杂度减少至 O(n)。

**空间复杂度优化**
因为只需要记录前一次的，而不需要记录全部的每一次，所以完全可以用O（1）的空间复杂度

![](../2.png)

#### 代码
	class Solution {
	public:
	    int numberOfArithmeticSlices(vector<int>& nums) {
	        int n = nums.size();
	        if(n<3)return 0;
	
	        int add=0;
	        int ans = 0;
	        for(int i=2;i<n;i++)
	        {
	            (nums[i]-nums[i-1]==nums[i-1]-nums[i-2])?ans+=++add:add=0;
	        }
	
	        return ans;
	    }
	};
## 双指针
### 应用1：用于有序数组
#### ①二分查找

#### ②两数和

### 应用2：用于数组和字符串的中间一段问题（双指针指示首尾）
#### 例题1：（连续多个字符的字符串删减）
给定一个由 n 个小写字母构成的字符串。

现在，需要删掉其中的一些字母，使得字符串中不存在连续三个或三个以上的 x。

请问，最少需要删掉多少个字母？

如果字符串本来就不存在连续的三个或三个以上 x，则无需删掉任何字母。

输入样例1：

6

xxxiii
输出样例1：

1

**思想**：

用索引i遍历字符串

一个指针找这一段字符串的第一个x，然后另一个指针找这一段字符串的最后一个x；便可计算x的数量进行删减。

之后i跳到最后一个x之后避免重复查找即可。

**代码：**

	#include<cstdio>
	#include<iostream>
	#include<string>
	#include<cstring>
	
	using namespace std;
	
	int main()
	{
	    int n;
	    cin>>n;
	    
	    string str;
	    cin>>str;
	    
	    int p1=0;
	    int p2=0;
	    int result=0;
	    
	    for(int i=0;i<n;i++)
	    {
	        if(str[i]=='x')
	        {
	            p1=i;
	            p2=p1;
	            while(str[p1]=='x')p1++;
	            if(p1-p2>2)result+=p1-p2-2;
	            i=p1;
	        }
	    }
	    
	    cout<<result<<endl;
	}



# 链表章节
## 1.两个链表首个公共结点问题
### 问题描述
输入两个链表，找出它们的第一个公共节点。

如下面的两个链表：

![](E:/笔记用图/上机笔记/1.png)

### 方法1：双指针各自遍历，浪漫相遇
**是很经典的相遇问题的解法**

**走过自己的路，再走一遍对方走的路，一定会相遇！**

我们使用两个指针 node1，node2 分别指向两个链表 headA，headB 的头结点，然后同时分别逐结点遍历，当 node1 到达链表 headA 的末尾时，重新定位到链表 headB 的头结点；当 node2 到达链表 headB 的末尾时，重新定位到链表 headA 的头结点。

这样，当它们相遇时，所指向的结点就是第一个公共结点。

	class Solution {
	public:
	    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
	        ListNode *node1 = headA;
	        ListNode *node2 = headB;
	        
	        while (node1 != node2) {
	            node1 = node1 != NULL ? node1->next : headB;
	            node2 = node2 != NULL ? node2->next : headA;
	        }
	        return node1;
	    }
	};

图解链接：[https://leetcode-cn.com/problems/liang-ge-lian-biao-de-di-yi-ge-gong-gong-jie-dian-lcof/solution/shuang-zhi-zhen-fa-lang-man-xiang-yu-by-ml-zimingm/](https://leetcode-cn.com/problems/liang-ge-lian-biao-de-di-yi-ge-gong-gong-jie-dian-lcof/solution/shuang-zhi-zhen-fa-lang-man-xiang-yu-by-ml-zimingm/)

复杂度分析:

时间复杂度：O(M+N)。

空间复杂度：O(1)。
### 方法2：尾部对齐
思路：公共结点以及之后的结点都是相同的，所以尾部对齐之后（**事实上是尾部对齐，然后调整长的链表的指针到短链表的头对齐位置，这样两个指针到尾部的距离相同**），就可以各自走相同的步数到达第一个相同结点。

直接给出我自己写的代码

	/**
	 * Definition for singly-linked list.
	 * struct ListNode {
	 *     int val;
	 *     ListNode *next;
	 *     ListNode(int x) : val(x), next(NULL) {}
	 * };
	 **/
	class Solution {
	public:
	    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
	        ListNode* temp1=headA;
	        ListNode* temp2=headB;
	
	        int length1=0;
	        int length2=0;
	
	        while(temp1 != NULL)
	        {
	            length1++;
	            temp1=temp1->next;
	        }
	
	        while(temp2 != NULL)
	        {
	            length2++;
	            temp2=temp2->next;
	        }
	
	        temp1=headA;
	        temp2=headB;
	
	        if(length1<=length2)
	        {
	            int counter=length2-length1;
	            while(counter)
	            {
	                temp2=temp2->next;
	                counter--;
	            }
	        }
	
	        else
	        {
	            int counter=length1-length2;
	            while(counter)
	            {
	                temp1=temp1->next;
	                counter--;
	            }
	        }
	
	        while(temp1 != temp2 && temp1!=NULL && temp2!=NULL)
	        {
	            temp1=temp1->next;
	            temp2=temp2->next;
	        }
	        if(temp1==NULL || temp2==NULL)return NULL;
	        return temp1;
	    }
	};

## 2.含随机指针random的链表的拷贝（Leetcode）
### 问题描述：
给你一个长度为 n 的链表，每个节点包含一个额外增加的随机指针 random ，该指针可以指向链表中的任何节点或空节点。

构造这个链表的 深拷贝。 深拷贝应该正好由 n 个 全新 节点组成，其中每个新节点的值都设为其对应的原节点的值。新节点的 next 指针和 random 指针也都应指向复制链表中的新节点，并使原链表和复制链表中的这些指针能够表示相同的链表状态。复制链表中的指针都不应指向原链表中的节点 。

例如，如果原链表中有 X 和 Y 两个节点，其中 X.random --> Y 。那么在复制链表中对应的两个节点 x 和 y ，同样有 x.random --> y 。

返回复制链表的头节点。

用一个由 n 个节点组成的链表来表示输入/输出中的链表。每个节点用一个 [val, random_index] 表示：

val：一个表示 Node.val 的整数。
random_index：随机指针指向的节点索引（范围从 0 到 n-1）；如果不指向任何节点，则为  null 。
你的代码 只 接受原链表的头节点 head 作为传入参数。

![](E:/笔记用图/上机笔记/2.png)

输入：head = [[7,null],[13,0],[11,4],[10,2],[1,0]]

输出：[[7,null],[13,0],[11,4],[10,2],[1,0]]

### Leetcode官方题解
本题要求我们对一个特殊的链表进行深拷贝。如果是普通链表，我们可以直接按照遍历的顺序创建链表节点。而本题中因为随机指针的存在，当我们拷贝节点时，「当前节点的随机指针指向的节点」可能还没创建，因此我们需要变换思路。**一个可行方案是，我们利用回溯的方式，让每个节点的拷贝操作相互独立。**对于**当前节点，我们首先要进行拷贝，然后我们进行「当前节点的后继节点」和「当前节点的随机指针指向的节点」拷贝**，拷贝完成后将创建的新节点的指针返回，即可完成当前节点的两指针的赋值。

具体地，我们**用哈希表记录每一个节点对应新节点的创建情况**。遍历该链表的过程中，我们**检查「当前节点的后继节点」和「当前节点的随机指针指向的节点」的创建情况**。如果这两个节点中的任何一个节点的新节点没有被创建，我们都立刻递归地进行创建。当我们拷贝完成，回溯到当前层时，我们即可完成当前节点的指针赋值。注意一个节点可能被多个其他节点指向，因此我们可能递归地多次尝试拷贝某个节点，为了防止重复拷贝，我们需要首先检查当前节点是否被拷贝过，如果已经拷贝过，我们可以直接从哈希表中取出拷贝后的节点的指针并返回即可。

在实际代码中，我们需要特别判断给定节点为空节点的情况。

复杂度分析

时间复杂度：O(n)，其中 n 是链表的长度。对于每个节点，我们至多访问其「后继节点」和「随机指针指向的节点」各一次，均摊每个点至多被访问两次。

空间复杂度：O(n)，其中 n 是链表的长度。为哈希表的空间开销。


### 个人理解
带了random指针的链表，其实已经**趋向于一种有向图了**，但特殊的又是不看random，它就是一个链表

而遍历图的时候，会**用visited数组记录访问情况，以免重复访问**，这其实就是一种用哈希表的思想。

类比visited数组：此题中，是要对列表进行拷贝，**有了random指针，就会有重复创建结点的问题** 

**所以用哈希表记录下原链表结点与拷贝后结点的映射关系，以免重复创建，浪费时间（空间换时间）**

### 官方代码
因为用到了递归回溯思想，对每个结点的判断、创建操作都是相同的，且独立的，不会存在重复计算（**因为用了哈希表映射**），所以可以**直接用递归写代码！！！**

	class Solution {
	public:
	    unordered_map<Node*, Node*> cachedNode;
	
	    Node* copyRandomList(Node* head) {
	        if (head == nullptr) {
	            return nullptr;
	        }
	        if (!cachedNode.count(head)) {
	            Node* headNew = new Node(head->val);
	            cachedNode[head] = headNew;
	            headNew->next = copyRandomList(head->next);
	            headNew->random = copyRandomList(head->random);
	        }
	        return cachedNode[head];
	    }
	};

