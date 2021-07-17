# STL的一些应用
## STL vector
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

查找：find(key)

若找到则返回该元素迭代器（=指针），否则返回迭代器end()

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

## 字符串处理
### STL string
#### 一、初始化
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
#### 二、方法
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

## 动态规划 Dynamic Programming
### 适用
分解成子问题，子问题之间不独立（如果独立直接递归分治了），存在重复运算，所以用dp数组来记录运算结果，避免重复运算

**是一种空间换时间的算法策略**
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

