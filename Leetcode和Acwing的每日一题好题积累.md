# Leetcode和Acwing的好题

## 题目1：变位词组（7.18leetcode）
链接：[https://leetcode-cn.com/problems/group-anagrams-lcci/](https://leetcode-cn.com/problems/group-anagrams-lcci/)

### 题目叙述：
编写一种方法，对字符串数组进行排序，将所有变位词组合在一起。变位词是指字母相同，但排列不同的字符串。

注意：本题相对原题稍作修改

示例:

输入: ["eat", "tea", "tan", "ate", "nat", "bat"],
输出:
[
  ["ate","eat","tea"],
  ["nat","tan"],
  ["bat"]
]
说明：

所有输入均为小写字母。
不考虑答案输出的顺序。

### 思路
字母相同而排序不同，则直接按升序排序后，完全相同，然后进行hash table计数（记住在未排序前的索引即可）

### 个人cpp代码
	class Solution {
	public:
	    vector<vector<string>> groupAnagrams(vector<string>& strs) {
	        vector<string>copy_strs = strs;
	        int n=strs.size();
	        int i;
	        map<string,vector<int>> counter_map;
	        vector<int>init_vector;
	        for(i=0;i<n;i++)
	        {
	            sort(strs[i].begin(),strs[i].end());
	            //cout<<strs[i]<<endl;
	            //cout<<(counter_map.find(strs[i])==counter_map.end())<<endl;
	            if(counter_map.find(strs[i])==counter_map.end())
	            {
	                if(i!=0)init_vector.pop_back();
	                init_vector.push_back(i);
	                counter_map.insert(pair<string,vector<int>>(strs[i],init_vector));
	            }
	            else
	            {
	                counter_map[strs[i]].push_back(i);
	            }
	        }
	        //cout<<counter_map["abt"][0];
	        vector<vector<string>>result_group;
	        vector<string> strv1;
	        for(map<string,vector<int>>::iterator it=counter_map.begin();it!=counter_map.end();it++)
	        {
	            for(int j=0;j<it->second.size();j++)
	            {
	                int index;
	                index = it->second[j];
	                strv1.push_back(copy_strs[index]);
	            }
	            result_group.push_back(strv1);
	            strv1.clear();
	        }

        return result_group;
    }
	};

## 题目2：检查是否区域内所有整数都被覆盖
### 题目描述：
给你一个二维整数数组 ranges 和两个整数 left 和 right 。每个 ranges[i] = [starti, endi] 表示一个从 starti 到 endi 的 闭区间 。

如果闭区间 [left, right] 内每个整数都被 ranges 中 至少一个 区间覆盖，那么请你返回 true ，否则返回 false 。

已知区间 ranges[i] = [starti, endi] ，如果整数 x 满足 starti <= x <= endi ，那么我们称整数x 被覆盖了。

 

示例 1：

输入：ranges = [[1,2],[3,4],[5,6]], left = 2, right = 5

输出：true

解释：2 到 5 的每个整数都被覆盖了：

- 2 被第一个区间覆盖。
- 3 和 4 被第二个区间覆盖。
- 5 被第三个区间覆盖。

[https://leetcode-cn.com/problems/check-if-all-the-integers-in-a-range-are-covered/](https://leetcode-cn.com/problems/check-if-all-the-integers-in-a-range-are-covered/)
### 最优方法：差分数组
#### 思路与算法
![](E:/笔记用图/上机笔记/3.png)

#### 代码：
	class Solution {
	public:
	    bool isCovered(vector<vector<int>>& ranges, int left, int right) {
	        vector<int> diff(52, 0);   // 差分数组
	        for (auto&& range: ranges) {
	            ++diff[range[0]];
	            --diff[range[1]+1];
	        }
	        // 前缀和
	        int curr = 0;
	        for (int i = 1; i <= 50; ++i) {
	            curr += diff[i];
	            if (i >= left && i <= right && curr <= 0) {
	                return false;
	            }
	        }
	        return true;
	    }
	};

#### 复杂度分析
复杂度分析：（注意，这里面都是l 不是1！）

时间复杂度：O(n+l)，其中 n 为ranges 的长度，l 为 diff 的长度。初始化 diff 数组的时间复杂度为 O(l)，遍历 ranges 更新差分数组的时间复杂度为 O(n)，求解前缀和并判断是否完全覆盖的时间复杂度为 O(l)。

空间复杂度：O(l)，即为 diff 的长度。

### 方法2（个人方法） 区间左端点排序
#### 思路
对区间按左端点排序。

排序后遍历ranges，判断left第一次落在li，right第一次落在ri。

然后，从li遍历到ri，要求：

每次记录下最右边的位置 mostright

如果下一个的左端点位置 比最右位置mostright 大了 2或者更多，则说明找不到，否则则可以找到

#### 代码
	class Solution {
	public:
	    bool isCovered(vector<vector<int>>& ranges, int left, int right) {
	        function <bool(vector<int>,vector<int>)> left_sort = [&] (vector<int> x,vector<int> y)
	        {
	            if(x[0]<y[0])return true;
	            return false;
	        };
	
	        sort(ranges.begin(),ranges.end(),left_sort);
	        int n=ranges.size();
	        int i;
	        int li=-1;
	        int ri=-1;
	        for(i=0;i<n;i++)
	        {
	            for(int j=0;j<=1;j++)
	                cout<<ranges[i][j]<<" ";
	            cout<<endl;
	        }
	        for(i=0;i<n;i++)
	        {
	            if(left>=ranges[i][0] and left<=ranges[i][1])
	            {
	                li=i;
	                break;
	            }
	        }
	        for(i=0;i<n;i++)
	        {
	            if(right>=ranges[i][0] and right<=ranges[i][1])
	            {
	                ri=i;
	                break;
	            }
	        }
	        cout<<li<<";"<<ri<<endl;
	        if(li==-1 or ri==-1)return false;
	        if(li==ri)return true;
	        int most_right=0;
	        for(i=li;i<ri;i++)
	        {
	            if(ranges[i][1]>most_right)most_right=ranges[i][1];
	            if(ranges[i+1][0]-most_right>1)return false;
	        }
	        return true;
	    }
	};

